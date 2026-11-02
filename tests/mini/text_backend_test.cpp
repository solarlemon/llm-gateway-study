#include "mini_backend.hpp"
#include <iostream>
#include <stdexcept>

void check(bool value, int line)
{
    if (!value)
        throw std::runtime_error("mini text assertion failed at line " + std::to_string(line));
}
#define require(value) check((value), __LINE__)
template <class Error, class Action>
void rejects(Action action)
{
    try
    {
        action();
    }
    catch (const Error &)
    {
        return;
    }
    throw std::runtime_error("expected mini text exception");
}
int main(int argc, char **argv)
{
    try
    {
        if (argc != 3)
            throw std::invalid_argument("mini_text_test MODEL TOKENIZER");
        mini::Tokenizer tokenizer(argv[2], 512);
        for (const std::string text : {"", "Once upon a time", "Hello, world!", "你好，世界！", "a\n\nb", " 😀 "})
        {
            auto ids = tokenizer.encode(text);
            std::string decoded;
            for (std::size_t i = 1; i < ids.size(); ++i)
                decoded += tokenizer.piece(ids[i - 1], ids[i]);
            require(text == decoded);
        }
        rejects<std::invalid_argument>([&]
                                       { tokenizer.encode(std::string("a\0b", 3)); });
        rejects<std::invalid_argument>([&]
                                       { tokenizer.encode("\xe4"); });
        MiniBackend engine(argv[1], argv[2]);
        using Json = nlohmann::json;
        Json request = {{"messages", Json::array({{{"role", "user"}, {"content", "Once upon a time"}}})}, {"max_tokens", 32}};
        Json stats;
        std::string emitted;
        auto baseline = engine.generate(request, [&](const std::string &piece)
                                        { emitted += piece; }, []
                                        { return false; }, stats);
        require(baseline["text"] == emitted && emitted.size() > 10 && stats["kv_cleared"] == true);
        auto stop_request = request;
        const auto prefix = emitted.substr(0, 3);
        stop_request["stop"] = prefix;
        auto stopped = engine.generate(stop_request, [](const std::string &) {}, []
                                       { return false; }, stats);
        require(stopped["text"] == "" && stopped["finish_reason"] == "stop" && stats["kv_cleared"] == true);
        rejects<std::runtime_error>([&]
                                    { engine.generate(request, [](const std::string &)
                                                      { throw std::runtime_error("output failed"); }, []
                                                      { return false; }, stats); });
        require(stats["kv_cleared"] == true);
        int checks = 0;
        rejects<std::runtime_error>([&]
                                    { engine.generate(request, [](const std::string &) {}, [&]
                                                      { return stats.contains("prompt_tokens") && ++checks == 3; }, stats); });
        require(stats["kv_cleared"] == true);
        auto recovered = engine.generate(request, [](const std::string &) {}, []
                                         { return false; }, stats);
        require(recovered["token_ids"] == baseline["token_ids"]);
        request["temperature"] = 0.5;
        rejects<MiniError>([&]
                           { engine.generate(request, [](const std::string &) {}, []
                                             { return false; }, stats); });
        std::cout << "PASS tokenizer UTF8/roundtrip, real text, stop, callback failure, cancellation and recovery\n";
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
