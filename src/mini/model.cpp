// Layout and math reference: Andrej Karpathy's llama2.c (MIT), see THIRD_PARTY.md.
#include "model.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace mini
{
    namespace
    {
        // 编译期强制 IEEE754 FP32：RoPE/softmax 的数值语义依赖它
        static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559,
                      "mini model requires IEEE754 FP32");

        // 取消检查：cancel 为空则跳过；取消时抛 runtime_error
        void check_cancel(const Cancel &cancel)
        {
            if (cancel && cancel())
                throw std::runtime_error("mini_cancelled");
        }

        // 矩阵-向量乘：w 是 rows × x.size() 的扁平行主序
        std::vector<float> matvec(const std::vector<float> &w, const std::vector<float> &x, int rows)
        {
            std::vector<float> out(rows);
            for (int r = 0; r < rows; ++r)
            {
                float sum = 0;
                for (std::size_t c = 0; c < x.size(); ++c)
                    sum += w[r * x.size() + c] * x[c];
                out[r] = sum;
            }
            return out;
        }

        // RMSNorm：x * (1/sqrt(mean(x²)+eps)) * weight
        std::vector<float> norm(const std::vector<float> &x, const std::vector<float> &weight)
        {
            float sum = 0;
            for (float v : x)
                sum += v * v;
            const float scale = 1.0f / std::sqrt(sum / static_cast<float>(x.size()) + 1e-5f);
            auto out = x;
            for (std::size_t i = 0; i < x.size(); ++i)
                out[i] = x[i] * scale * weight[i];
            return out;
        }

        // residual 残差原地累加
        void residual(std::vector<float> &x, const std::vector<float> &delta)
        {
            for (std::size_t i = 0; i < x.size(); ++i)
                x[i] += delta[i];
        }

        // RoPE：两两配对旋转；频率按 (i % head_dim)/head_dim 计算
        void rope(std::vector<float> &x, std::size_t position, int head_dim)
        {
            for (std::size_t i = 0; i < x.size(); i += 2)
            {
                const float frequency = 1.0f / std::pow(10000.0f, static_cast<float>(i % head_dim) / head_dim);
                const float angle = static_cast<float>(position) * frequency;
                const float a = x[i], b = x[i + 1];
                x[i] = a * std::cos(angle) - b * std::sin(angle);
                x[i + 1] = a * std::sin(angle) + b * std::cos(angle);
            }
        }
    }

    // ---------- 构造：加载 legacy llama2.c checkpoint ----------
    Model::Model(const std::string &checkpoint)
    {
        // 端序必须是 little-endian
        const std::uint32_t endian_probe = 1;
        if (*reinterpret_cast<const unsigned char *>(&endian_probe) != 1)
            throw std::runtime_error("mini checkpoint requires little-endian host");

        // 打开并取大小
        std::ifstream input(checkpoint, std::ios::binary | std::ios::ate);
        if (!input)
            throw std::runtime_error("cannot open mini checkpoint");
        const std::streamoff bytes = input.tellg();
        // 至少 28 字节头；上限 256MB；浮点区必须 4 字节对齐
        if (bytes < 28 || bytes > 256 * 1024 * 1024 || (bytes - 28) % 4 != 0)
            throw std::runtime_error("invalid mini checkpoint byte size");

        // 读 7 个 int32 头
        input.seekg(0);
        std::int32_t header[7];
        input.read(reinterpret_cast<char *>(header), sizeof(header));
        if (!input || header[5] == std::numeric_limits<std::int32_t>::min())
            throw std::runtime_error("invalid mini header");

        // header[5] 符号编码：>0 共享 classifier/embedding，<0 不共享
        const bool shared = header[5] > 0;
        config_ = {header[0], header[1], header[2], header[3], header[4],
                   std::abs(header[5]), header[6]};
        const auto &c = config_;

        // 维度范围与注意力形状校验
        for (int value : {c.dim, c.hidden_dim, c.layers, c.heads, c.kv_heads, c.vocab, c.capacity})
            if (value <= 0 || value > 65536)
                throw std::runtime_error("unsupported mini dimension");
        if (c.layers > 128 || c.dim % c.heads || c.heads % c.kv_heads || (c.dim / c.heads) % 2)
            throw std::runtime_error("invalid mini attention shape");

        // 一次性读入浮点区
        std::vector<float> data(static_cast<std::size_t>(bytes - 28) / sizeof(float));
        input.read(reinterpret_cast<char *>(data.data()),
                   static_cast<std::streamsize>(data.size() * sizeof(float)));
        if (!input)
            throw std::runtime_error("truncated mini checkpoint");
        if (!std::all_of(data.begin(), data.end(), [](float x)
                         { return std::isfinite(x); }))
            throw std::runtime_error("nonfinite mini weights");

        // 按固定顺序切张量
        std::size_t offset = 0;
        auto take = [&](std::size_t count)
        {
            if (count > data.size() - offset) // 减法避免溢出
                throw std::runtime_error("truncated mini tensor");
            std::vector<float> result(data.begin() + offset, data.begin() + offset + count);
            offset += count;
            return result;
        };
        const std::size_t d = c.dim, h = c.hidden_dim,
                          kd = d * c.kv_heads / c.heads;

        embedding_ = take(static_cast<std::size_t>(c.vocab) * d);
        layers_.resize(c.layers);
        for (auto &l : layers_)
            l.attention_norm = take(d);
        for (auto &l : layers_)
            l.q = take(d * d);
        for (auto &l : layers_)
            l.k = take(kd * d); // GQA：kd × d
        for (auto &l : layers_)
            l.v = take(kd * d);
        for (auto &l : layers_)
            l.projection = take(d * d);
        for (auto &l : layers_)
            l.ffn_norm = take(d);
        for (auto &l : layers_)
            l.gate = take(h * d);
        for (auto &l : layers_)
            l.down = take(d * h);
        for (auto &l : layers_)
            l.up = take(h * d);
        final_norm_ = take(d);
        // 文件里的两份 RoPE 频率表跳过（前向实时计算）
        take(static_cast<std::size_t>(c.capacity) * (d / c.heads));
        // classifier：共享则复用 embedding_，否则单独读
        classifier_ = shared ? embedding_ : take(static_cast<std::size_t>(c.vocab) * d);
        if (offset != data.size())
            throw std::runtime_error("unexpected mini checkpoint trailing data");
    }

    // ---------- 由 Config 推导缓存形状 ----------
    // 把 Config 映射成 KVShape。head_dim = dim / heads
    KVShape Model::cache_shape() const noexcept
    {
        return {static_cast<std::size_t>(config_.layers),
                static_cast<std::size_t>(config_.capacity),
                static_cast<std::size_t>(config_.kv_heads),
                static_cast<std::size_t>(config_.dim / config_.heads)};
    }

    // ---------- 缓存形状校验：capacity 允许更小，其余必须精确匹配 ----------
    void Model::validate_cache(const KVCache &cache) const
    {
        const auto expected = cache_shape();
        const auto &actual = cache.shape();
        if (actual.layers != expected.layers || actual.capacity > expected.capacity ||
            actual.kv_heads != expected.kv_heads || actual.head_dim != expected.head_dim)
            throw std::invalid_argument("mini cache/model shape mismatch");
    }

    // ---------- 单 token 完整前向->计算注意力 ----------
    Logits Model::forward_token(TokenId token, KVCache &cache, const Cancel &cancel) const
    {
        validate_cache(cache); // 形状匹配
        if (token < 0 || token >= config_.vocab)
            throw std::out_of_range("mini token out of range");
        const auto position = cache.valid_len(); // 当前 token 的位置
        if (position >= cache.shape().capacity)
            throw std::out_of_range("mini context exhausted");

        // 维度推导
        const int d = config_.dim, head_dim = d / config_.heads;
        const int kv_dim = head_dim * config_.kv_heads,
                  sharing = config_.heads / config_.kv_heads;

        // 嵌入查表：取 token 对应的 d 维向量
        std::vector<float> x(embedding_.begin() + token * d,
                             embedding_.begin() + (token + 1) * d);
        try
        {
            for (std::size_t layer = 0; layer < layers_.size(); ++layer)
            {
                check_cancel(cancel); // 逐层检查取消
                const auto &weights = layers_[layer];

                // ---- attention 子层 ----
                auto normalized = norm(x, weights.attention_norm);
                // QKV 投影：q = W_q · normalized（d 维）、k/v = W_k/W_v · normalized（kv_dim 维）
                auto q = matvec(weights.q, normalized, d);      // d 维
                auto k = matvec(weights.k, normalized, kv_dim); // kv_dim 维
                auto v = matvec(weights.v, normalized, kv_dim);
                rope(q, position, head_dim);
                rope(k, position, head_dim); // v 不做 RoPE

                // 写入当前层当前位置的 K/V（提交前该层可读）
                cache.write_layer(layer, k, v);

                // 多头 attention（含当前 token，因为已写入）
                std::vector<float> attention(d, 0);
                for (int head = 0; head < config_.heads; ++head)
                {
                    const int kv_head = head / sharing; // GQA 映射
                    std::vector<float> scores(position + 1);
                    // 对历史 t ∈ [0, position] 算 score = q·k / sqrt(head_dim)
                    for (std::size_t t = 0; t <= position; ++t)
                    {
                        float dot = 0;
                        for (int i = 0; i < head_dim; ++i)
                            dot += q[head * head_dim + i] * cache.key(layer, t, kv_head, i);
                        scores[t] = dot / std::sqrt(static_cast<float>(head_dim));
                    }
                    // 数值稳定 softmax
                    const float maximum = *std::max_element(scores.begin(), scores.end());
                    float sum = 0;
                    for (auto &score : scores)
                    {
                        score = std::exp(score - maximum);
                        sum += score;
                    }
                    for (auto &score : scores)
                        score /= sum;
                    // 加权求和 value，累加到 attention[head*head_dim + i]
                    for (std::size_t t = 0; t <= position; ++t)
                        for (int i = 0; i < head_dim; ++i)
                            attention[head * head_dim + i] +=
                                scores[t] * cache.value(layer, t, kv_head, i);
                }
                residual(x, matvec(weights.projection, attention, d));

                // ---- SwiGLU FFN 子层 ----
                normalized = norm(x, weights.ffn_norm);
                auto gate = matvec(weights.gate, normalized, config_.hidden_dim);
                auto up = matvec(weights.up, normalized, config_.hidden_dim);
                for (std::size_t i = 0; i < gate.size(); ++i)
                    gate[i] = (gate[i] / (1.0f + std::exp(-gate[i]))) * up[i]; // silu(gate)*up
                residual(x, matvec(weights.down, gate, d));
            }

            // 最终 norm + 分类器 → logits
            auto logits = matvec(classifier_, norm(x, final_norm_), config_.vocab);
            check_cancel(cancel);
            // 所有层与最终输出都成功，才提交本 token
            cache.commit_token();
            return logits;
        }
        catch (...)
        {
            // 任何失败：放弃整个请求的缓存状态，再重抛
            cache.reset();
            throw;
        }
    }

    // ---------- 预填充：逐 token 前向整个 prompt ----------
    Logits Model::prefill(const std::vector<TokenId> &ids, KVCache &cache,
                          std::size_t chunk_size, const Cancel &cancel) const
    {
        validate_cache(cache);
        // 只能在空 cache 上 prefill，且输入非空、chunk_size > 0
        if (ids.empty() || chunk_size == 0 || cache.valid_len() != 0)
            throw std::invalid_argument(
                "prefill requires nonempty input, positive chunk and empty cache");
        if (ids.size() > cache.shape().capacity)
            throw std::out_of_range("mini prompt too long");
        for (auto id : ids)
            if (id < 0 || id >= config_.vocab)
                throw std::out_of_range("mini token out of range");

        Logits logits;
        for (std::size_t offset = 0; offset < ids.size();)
        {
            const auto count = std::min(chunk_size, ids.size() - offset);
            for (std::size_t i = 0; i < count; ++i)
                logits = forward_token(ids[offset + i], cache, cancel);
            offset += count;
        }
        return logits; // 最后一个 token 的 logits
    }

    // ---------- 单步解码：要求已有前缀 ----------
    Logits Model::decode(TokenId token, KVCache &cache, const Cancel &cancel) const
    {
        if (!cache.valid_len())
            throw std::invalid_argument("decode requires prefilled cache");
        return forward_token(token, cache, cancel);
    }

    // ---------- 对照实现：临时缓存 + 完整 prefill ----------
    Logits Model::forward_full(const std::vector<TokenId> &ids) const
    {
        KVCache temporary(cache_shape());
        return prefill(ids, temporary);
    }

    // ---------- 贪心：argmax，空 logits 拒绝 ----------
    TokenId Model::greedy(const Logits &logits)
    {
        if (logits.empty())
            throw std::invalid_argument("empty mini logits");
        return static_cast<TokenId>(
            std::max_element(logits.begin(), logits.end()) - logits.begin());
    }
} // namespace mini