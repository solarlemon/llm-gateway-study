#pragma once
#include <cstddef>
#include <vector>

namespace mini
{
    // 缓存形状：层数、最大位置数、KV 头数、每头维度
    struct KVShape
    {
        std::size_t layers, capacity, kv_heads, head_dim;
    };

    // 一个请求独占一个缓存。所有访问由所属 worker 串行执行（类内不加锁）。
    // K 存储经过 RoPE 的 key，V 存储对应 value；
    // 布局均为 [layer, position, head, component]。
    class KVCache
    {
    public:
        // 构造：按 shape 一次性预分配 keys_/values_/written_，
        // 计算 layer_width_、layer_stride_、allocated_bytes_；valid_len_ = 0
        explicit KVCache(KVShape shape);

        // 缓存与请求身份绑定，禁用拷贝与移动
        KVCache(const KVCache &) = delete;
        KVCache &operator=(const KVCache &) = delete;
        KVCache(KVCache &&) = delete;
        KVCache &operator=(KVCache &&) = delete;

        // ---------- 只读访问器 ----------
        const KVShape &shape() const noexcept { return shape_; }                  // 形状
        std::size_t valid_len() const noexcept { return valid_len_; }             // 已提交 token 数
        std::size_t allocated_bytes() const noexcept { return allocated_bytes_; } // 预分配总量
        std::size_t valid_bytes() const noexcept;                                 // 有效部分字节数

        // ---------- 写入阶段 ----------
        // 每层一次，写入「当前 token」在该层的全部 KV heads。
        // 目标位置固定为 valid_len（当前 token 的位置）；禁止覆盖已提交的历史位置；尺寸/层号错误时不改变任何状态。
        void write_layer(std::size_t layer, const std::vector<float> &key,
                         const std::vector<float> &value);

        // ---------- 读取（attention 用） ----------
        // 通过 readable_index 做边界 + 可读性检查后取值；
        // 检查可阻止读取未写入或旧请求残留的数据。
        float key(std::size_t layer, std::size_t position,
                  std::size_t head, std::size_t component) const;
        float value(std::size_t layer, std::size_t position,
                    std::size_t head, std::size_t component) const;

        // ---------- 提交/回滚阶段 ----------
        // 全部层都写入成功后才调用：valid_len_ += 1，本 token 成为历史前缀。
        // 提交前已写入的层可在当前位置做 attention（供当前 token 计算用）。
        void commit_token();

        // 部分层失败时放弃本 token：清 written_ 标记，valid_len_ 不变；
        // 已提交的前缀仍有效。noexcept 保证回滚自身不抛。
        void rollback_token() noexcept;

        // 只重置有效状态（valid_len_ = 0、清 written_），保留已分配内存；
        // 访问检查会阻止读到旧请求留下的数据。
        void reset() noexcept;

    private:
        // 把 (layer, position, head, component) 映射为扁平下标，
        // 同时做边界与可读性校验（越界/未写入/不可读则报错）
        std::size_t readable_index(std::size_t layer, std::size_t position,
                                   std::size_t head, std::size_t component) const;

        KVShape shape_;                          // 形状副本
        std::size_t layer_width_, layer_stride_, // 每层宽度、层间跨度
            allocated_bytes_;                    // 预分配总字节
        std::vector<float> keys_, values_;       // 扁平存储：[layer, position, head, comp]
        std::vector<unsigned char> written_;     // 每层「本 token 已写」标记
        std::size_t valid_len_ = 0;              // 已提交 token 数
    };
} // namespace mini