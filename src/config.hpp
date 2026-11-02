#pragma once
#include <cstdlib>
#include <stdexcept>
#include <string>

struct Config
{
    // ---- 默认值 ----
    int port = 8081;
    int concurrency = 2, queue_capacity = 4, http_threads = 16;
    int queue_ms = 5000, request_ms = 120000;
    int stream_buffer = 65536, response_limit = 4 * 1024 * 1024;
    int metrics_window = 1024;
    std::string request_log = "build/requests.jsonl";
    std::string mini_model_path, mini_tokenizer_path; // 默认空，由 HOME 推导

    // 读取整数环境变量：未设置返回 fallback；
    // 必须整串是数字、且在 [minimum, maximum] 内，否则抛 runtime_error
    static int env(const char *name, int fallback, int minimum, int maximum)
    {
        const char *raw = std::getenv(name);
        if (!raw)
            return fallback; // 未设置 → 默认
        std::string s(raw);
        size_t used = 0;
        long value = std::stol(s, &used);                           // 非数字/超 long 会抛，向上冒泡
        if (used != s.size() || value < minimum || value > maximum) // 整串消费 + 范围
            throw std::runtime_error(std::string("invalid configuration: ") + name);
        return static_cast<int>(value);
    }

    static Config load()
    {
        Config c;

        // 1. 后端模式：只允许未设置或 "mini"
        if (auto p = std::getenv("BACKEND_MODE"))
        {
            if (std::string(p) != "mini")
                throw std::runtime_error("only BACKEND_MODE=mini is supported");
        }

        // 2. 模型路径：HOME 推导，再被显式环境变量覆盖
        if (auto p = std::getenv("HOME"))
        {
            c.mini_model_path = std::string(p) + "/ModelWeights/stories260K/stories260K.bin";
            c.mini_tokenizer_path = std::string(p) + "/ModelWeights/stories260K/tok512.bin";
        }
        if (auto p = std::getenv("MINI_MODEL_PATH"))
            c.mini_model_path = p;
        if (auto p = std::getenv("MINI_TOKENIZER_PATH"))
            c.mini_tokenizer_path = p;

        // 3. 数值字段：逐项读取 + 范围校验
        c.port = env("GATEWAY_PORT", c.port, 1, 65535);
        c.concurrency = env("MAX_ACTIVE", c.concurrency, 1, 32);
        c.queue_capacity = env("QUEUE_CAPACITY", c.queue_capacity, 0, 128);
        c.http_threads = env("HTTP_THREADS", c.http_threads, 4, 256);
        c.queue_ms = env("QUEUE_TIMEOUT_MS", c.queue_ms, 1, 600000);
        c.request_ms = env("REQUEST_TIMEOUT_MS", c.request_ms, 1, 600000);
        c.stream_buffer = env("STREAM_BUFFER_BYTES", c.stream_buffer, 1024, 1048576);
        c.response_limit = env("RESPONSE_LIMIT_BYTES", c.response_limit, 1024, 16777216);
        c.metrics_window = env("METRICS_WINDOW", c.metrics_window, 16, 65536);

        // 4. 日志路径：可被环境变量覆盖
        if (auto p = std::getenv("REQUEST_LOG_PATH"))
            c.request_log = p;

        // 5. 跨字段一致性：HTTP 线程必须容纳「执行中 + 排队中 + 2 个管理余量」
        if (c.http_threads < c.concurrency + c.queue_capacity + 2)
            throw std::runtime_error("HTTP_THREADS must be >= MAX_ACTIVE + QUEUE_CAPACITY + 2");

        return c;
    }
};