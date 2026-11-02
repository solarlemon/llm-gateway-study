#include "kv_cache.hpp"
#include <algorithm>
#include <limits>
#include <stdexcept>

namespace mini
{
    namespace
    {
        // 带溢出检查的乘法：a*b 若超出 size_t 上限则抛 overflow_error
        std::size_t multiply(std::size_t a, std::size_t b)
        {
            if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a)
                throw std::overflow_error("KV shape or byte size overflows size_t");
            return a * b;
        }
    }

    // ---------- 构造：校验 → 算尺寸（全程防溢出）→ 一次性分配 ----------
    KVCache::KVCache(KVShape shape) : shape_(shape)
    {
        // 四个维度必须为正
        if (!shape.layers || !shape.capacity || !shape.kv_heads || !shape.head_dim)
            throw std::invalid_argument("KV dimensions must be positive");

        // 每层宽度 = kv_heads * head_dim
        layer_width_ = multiply(shape.kv_heads, shape.head_dim);
        // 每层跨度 = capacity * layer_width_
        layer_stride_ = multiply(shape.capacity, layer_width_);
        // 总元素数 = layers * layer_stride_
        const auto elements = multiply(shape.layers, layer_stride_);
        // 总字节 = elements * sizeof(float) * 2（K 和 V 各一份）
        allocated_bytes_ = multiply(multiply(elements, sizeof(float)), 2);

        // 所有尺寸乘法均已在分配前完成；resize 失败时 vector 自会回收
        keys_.resize(elements);
        values_.resize(elements);
        written_.resize(shape.layers, 0); // 每层一个「本 token 已写」标记
    }

    // ---------- 有效字节数：只读，无需再防溢出 ----------
    std::size_t KVCache::valid_bytes() const noexcept
    {
        // 构造时已校验 layers*capacity*layer_width_ 不溢出，
        // 而 valid_len_ <= capacity，故本乘积必然安全
        return shape_.layers * valid_len_ * layer_width_ * sizeof(float) * 2;
    }

    // ---------- 写入当前 token 在某一层的 KV ----------
    void KVCache::write_layer(std::size_t layer, const std::vector<float> &key,
                              const std::vector<float> &value)
    {
        // 1. 层号越界
        if (layer >= shape_.layers)
            throw std::out_of_range("KV layer out of range");
        // 2. 缓存已满
        if (valid_len_ == shape_.capacity)
            throw std::out_of_range("KV capacity exhausted");
        // 3. 载荷尺寸必须恰好等于每层宽度
        if (key.size() != layer_width_ || value.size() != layer_width_)
            throw std::invalid_argument("KV layer payload has incorrect size");
        // 4. 同一 token 的同一层不能写两次
        if (written_[layer])
            throw std::logic_error("KV layer already written for current token");

        // 5. 位置固定为当前 token 的 valid_len_，写入 K 与 V，并打标记
        const auto offset = layer * layer_stride_ + valid_len_ * layer_width_;
        std::copy(key.begin(), key.end(), keys_.begin() + offset);
        std::copy(value.begin(), value.end(), values_.begin() + offset);
        written_[layer] = 1;
    }

    // ---------- 读取入口：边界 + 可读性校验，返回扁平下标 ----------
    std::size_t KVCache::readable_index(std::size_t layer, std::size_t position,
                                        std::size_t head, std::size_t component) const
    {
        // 边界检查
        if (layer >= shape_.layers || position >= shape_.capacity ||
            head >= shape_.kv_heads || component >= shape_.head_dim)
            throw std::out_of_range("KV coordinate out of range");

        // 可读性检查：
        //   position <  valid_len_               → 历史已提交，任意层可读
        //   position == valid_len_ && written_   → 当前 token 该层已写，可读
        //   position == valid_len_ && !written_  → 当前 token 该层未写，不可读
        //   position >  valid_len_               → 未来位置，不可读
        if (position > valid_len_ || (position == valid_len_ && !written_[layer]))
            throw std::out_of_range("KV position has no valid data");

        // 扁平下标：[layer, position, head, component]
        return layer * layer_stride_ + position * layer_width_ + head * shape_.head_dim + component;
    }

    // ---------- 两个 getter：逻辑全部委托给 readable_index ----------
    float KVCache::key(std::size_t layer, std::size_t position,
                       std::size_t head, std::size_t component) const
    {
        return keys_[readable_index(layer, position, head, component)];
    }
    
    float KVCache::value(std::size_t layer, std::size_t position,
                         std::size_t head, std::size_t component) const
    {
        return values_[readable_index(layer, position, head, component)];
    }

    // ---------- 提交当前 token：要求所有层都已写 ----------
    void KVCache::commit_token()
    {
        // 容量检查
        if (valid_len_ == shape_.capacity)
            throw std::out_of_range("KV capacity exhausted");
        // 任何一层没写就不允许提交（保证前缀完整）
        if (std::find(written_.begin(), written_.end(), 0) != written_.end())
            throw std::logic_error("KV token requires all layers before commit");

        ++valid_len_;     // 本 token 正式成为历史前缀
        rollback_token(); // 清标记，为下一个 token 的写入复位
    }

    // ---------- 回滚当前 token：只清标记，不动 valid_len_ 与已写数据 ----------
    void KVCache::rollback_token() noexcept
    {
        std::fill(written_.begin(), written_.end(), 0);
    }

    // ---------- 重置整个请求：valid_len_ 归零 + 清标记，保留分配 ----------
    void KVCache::reset() noexcept
    {
        valid_len_ = 0;
        rollback_token();
    }
} // namespace mini