#pragma once
#include <cstdlib>
#include <stdexcept>
#include <string>

struct Config {
    int port = 8081, upstream_port = 8080;
    std::string upstream_host = "127.0.0.1";
    int concurrency = 2, queue_capacity = 4, http_threads = 16;
    int queue_ms = 5000, request_ms = 120000, idle_ms = 30000;
    int stream_buffer = 65536, response_limit = 4 * 1024 * 1024;
    static int env(const char *name, int fallback, int minimum, int maximum) {
        const char *raw = std::getenv(name);
        if (!raw) return fallback;
        std::string s(raw); size_t used = 0;
        long value = std::stol(s, &used);
        if (used != s.size() || value < minimum || value > maximum)
            throw std::runtime_error(std::string("invalid configuration: ") + name);
        return static_cast<int>(value);
    }
    static Config load() {
        Config c;
        c.port = env("GATEWAY_PORT", c.port, 1, 65535);
        c.upstream_port = env("UPSTREAM_PORT", c.upstream_port, 1, 65535);
        if (auto p = std::getenv("UPSTREAM_HOST")) c.upstream_host = p;
        c.concurrency = env("MAX_ACTIVE", c.concurrency, 1, 32);
        c.queue_capacity = env("QUEUE_CAPACITY", c.queue_capacity, 0, 128);
        c.http_threads = env("HTTP_THREADS", c.http_threads, 4, 256);
        c.queue_ms = env("QUEUE_TIMEOUT_MS", c.queue_ms, 1, 600000);
        c.request_ms = env("REQUEST_TIMEOUT_MS", c.request_ms, 1, 600000);
        c.idle_ms = env("UPSTREAM_IDLE_MS", c.idle_ms, 1, 600000);
        c.stream_buffer = env("STREAM_BUFFER_BYTES", c.stream_buffer, 1024, 1048576);
        c.response_limit = env("RESPONSE_LIMIT_BYTES", c.response_limit, 1024, 16777216);
        if (c.http_threads < c.concurrency + c.queue_capacity + 2)
            throw std::runtime_error("HTTP_THREADS must be >= MAX_ACTIVE + QUEUE_CAPACITY + 2");
        return c;
    }
};
