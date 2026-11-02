#include "model.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

void require(bool value)
{
    if (!value)
        throw std::runtime_error("model assertion failed");
}
template <class E, class F>
void rejects(F action)
{
    try
    {
        action();
    }
    catch (const E &)
    {
        return;
    }
    throw std::runtime_error("expected exception");
}
void equal(const mini::Logits &a, const mini::Logits &b)
{
    require(a.size() == b.size());
    for (std::size_t i = 0; i < a.size(); ++i)
        require(std::abs(a[i] - b[i]) < 1e-5f);
}
int main(int argc, char **argv)
{
    try
    {
        if (argc != 2)
            throw std::runtime_error("model_test MODEL");
        const mini::Model model(argv[1]);
        const auto shape = model.cache_shape();
        require(model.config().dim == 64 && shape.layers == 5 && shape.kv_heads == 4);
        std::vector<mini::TokenId> prefix{1, 12, 47, 123, 5};
        mini::KVCache a(shape), b(shape);
        const auto baseline = model.prefill(prefix, a, 1);
        equal(baseline, model.prefill(prefix, b, 2));
        require(a.valid_len() == 5 && a.valid_bytes() == 6400);
        const auto first = mini::Model::greedy(baseline);
        const auto next_logits = model.decode(first, a);
        prefix.push_back(first);
        equal(next_logits, model.forward_full(prefix));
        require(a.valid_len() == 6 && b.valid_len() == 5);
        for (int i = 0; i < 8; ++i)
        {
            const mini::TokenId fixed = 30 + i;
            prefix.push_back(fixed);
            equal(model.decode(fixed, a), model.forward_full(prefix));
        }
        // 两请求交错只读同一个模型，各自 KV 独立。
        equal(model.decode(first, b), next_logits);
        a.reset();
        int checks = 0;
        rejects<std::runtime_error>([&]
                                    { model.prefill({1, 2, 3}, a, 2, [&]
                                                    { return ++checks == 3; }); });
        require(a.valid_len() == 0);
        rejects<std::out_of_range>([&]
                                   { a.key(0, 0, 0, 0); });
        equal(model.prefill({1, 8, 9}, a), model.forward_full({1, 8, 9}));
        rejects<std::invalid_argument>([&]
                                       { model.prefill({1}, a); });
        a.reset();
        rejects<std::invalid_argument>([&]
                                       { model.prefill({}, a); });
        rejects<std::invalid_argument>([&]
                                       { model.prefill({1}, a, 0); });
        rejects<std::out_of_range>([&]
                                   { model.prefill({model.config().vocab}, a); });
        require(a.valid_len() == 0);
        mini::KVCache small({shape.layers, 2, shape.kv_heads, shape.head_dim});
        model.prefill({1, 9}, small);
        rejects<std::out_of_range>([&]
                                   { model.decode(10, small); });
        require(small.valid_len() == 2);
        mini::KVCache wrong({1, 2, 1, 2});
        rejects<std::invalid_argument>([&]
                                       { model.prefill({1}, wrong); });
        std::cout << "PASS real-model prefill/decode logits, GQA KV, chunks, isolation, cancellation and bounds\n";
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
