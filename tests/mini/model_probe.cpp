#include "model.hpp"
#include "json.hpp"
#include <iostream>
#include <sstream>

// ID 入口专注核对真实模型计算。参数为 checkpoint、逗号分隔 ID、最大输出数。
int main(int argc, char** argv) {
    try {
        if (argc < 3 || argc > 4) throw std::invalid_argument("Usage: mini_cli MODEL ID,ID,... [MAX_OUTPUT=0]");
        mini::Model model(argv[1]);
        std::vector<mini::TokenId> prompt;
        std::stringstream input(argv[2]);
        std::string item;
        while (std::getline(input,item,',')) {
            std::size_t used = 0;
            const int id = std::stoi(item,&used);
            if (used != item.size()) throw std::invalid_argument("invalid token ID");
            prompt.push_back(id);
        }
        std::size_t used = 0;
        const std::string count_text = argc == 4 ? argv[3] : "0";
        const int maximum = std::stoi(count_text,&used);
        if (used != count_text.size() || maximum < 0 || maximum > model.config().capacity)
            throw std::invalid_argument("invalid output budget");
        if (prompt.size()+static_cast<std::size_t>(maximum) > static_cast<std::size_t>(model.config().capacity))
            throw std::out_of_range("prompt/output budget exceeds context");
        mini::KVCache cache(model.cache_shape());
        auto logits = model.prefill(prompt,cache,2);
        nlohmann::json report = {{"prompt_ids",prompt},{"prefill_logits",logits}};
        std::vector<mini::TokenId> output;
        std::string reason = maximum ? "length" : "prefill_only";
        for (int i = 0; i < maximum; ++i) {
            const auto next = mini::Model::greedy(logits);
            if (next == 2) { reason = "eos"; break; } // stories260K 配套 tokenizer 的 EOS。
            output.push_back(next);
            if (i+1 < maximum) logits = model.decode(next,cache);
        }
        report["output_ids"] = output;
        report["finish_reason"] = reason;
        report["valid_len"] = cache.valid_len();
        report["kv_valid_bytes"] = cache.valid_bytes();
        report["kv_allocated_bytes"] = cache.allocated_bytes();
        std::cout << report.dump() << '\n';
        return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
