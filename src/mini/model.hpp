#pragma once
#include "kv_cache.hpp" // KVShape / KVCache
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace mini
{
    using TokenId = std::int32_t;         // 词表 id
    using Logits = std::vector<float>;    // 位置上的输出分布
    using Cancel = std::function<bool()>; // 取消谓词：true 表示应中断

    // 模型超参
    struct Config
    {
        int dim, hidden_dim, layers, heads, kv_heads, vocab, capacity;
        //  dim       : 嵌入/隐藏维度
        //  hidden_dim: FFN 中间维度
        //  layers    : Transformer 层数
        //  heads     : 注意力头数（Q）
        //  kv_heads  : KV 头数（< heads 即 GQA/MQA）
        //  vocab     : 词表大小
        //  capacity  : 最大上下文 token 数
    };

    // 单层权重（Llama pre-norm 结构）
    struct Layer
    {
        std::vector<float> attention_norm, q, k, v, projection; // attention 子层
        std::vector<float> ffn_norm, gate, down, up;            // SwiGLU FFN 子层
    };

    // 权重只读共享。调用方为每个请求建立独立 KVCache，
    // 所有工作区均为调用内局部变量 —— 因此本类可多线程共享。
    class Model
    {
    public:
        // 从 checkpoint 加载权重与 config_
        explicit Model(const std::string &checkpoint);

        // 只读配置
        const Config &config() const noexcept { return config_; }

        // 由 Config 推导 KVShape{layers, capacity, kv_heads, dim/heads}
        KVShape cache_shape() const noexcept;

        // 预填充：处理整个 prompt（ids），按 chunk_size 分块前向整个 prompt，
        // 逐 token 写入 cache，返回最后位置的 logits。
        Logits prefill(const std::vector<TokenId> &ids, KVCache &cache,
                       std::size_t chunk_size = 8, const Cancel &cancel = {}) const;

        // 解码：对单个新 token 前向，写入 cache，返回该 token 位置的 logits，供下一步贪心采样。
        Logits decode(TokenId token, KVCache &cache, const Cancel &cancel = {}) const;

        // 对照实现：不使用缓存，从空状态重算完整前缀。用于验证增量路径（prefill/decode）的数值正确性。
        Logits forward_full(const std::vector<TokenId> &ids) const;

        // 贪心采样：静态函数，取 logits 的 argmax，不依赖模型状态。
        static TokenId greedy(const Logits &logits);

    private:
        // 单 token 完整前向：embedding → N×(attn+FFN) → final norm → classifier。
        // 内部逐层 write_layer，最后 commit_token。
        Logits forward_token(TokenId token, KVCache &cache, const Cancel &cancel) const;

        // 校验传入 cache 的形状与 cache_shape() 一致，防止写越界。
        void validate_cache(const KVCache &cache) const;

        // ---- 只读权重 ----
        Config config_{};                                        // 超参
        std::vector<float> embedding_, final_norm_, classifier_; // 嵌入表/最终 norm/输出投影
        std::vector<Layer> layers_;                              // 各层权重
    };
} // namespace mini