#pragma once
#include "config.hpp"
#include "task.hpp"

inline void forward_request(const Config &config, const std::shared_ptr<Task> &task) {
    auto client = std::make_shared<httplib::Client>(config.upstream_host, config.upstream_port);
    client->set_connection_timeout(2, 0);
    client->set_read_timeout(std::chrono::milliseconds(config.idle_ms));
    client->set_write_timeout(2, 0);
    auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(task->deadline - Clock::now()).count();
    client->set_max_timeout(static_cast<time_t>(std::max<int64_t>(1, remaining)));
    { std::lock_guard<std::mutex> lock(task->mu); task->client = client; }

    httplib::Request request;
    request.method = "POST";
    request.path = "/v1/chat/completions";
    request.body = task->body;
    request.set_header("Content-Type", "application/json");
    request.set_header("X-Request-ID", task->id);
    request.response_handler = [task](const httplib::Response &response) {
        std::lock_guard<std::mutex> lock(task->mu);
        task->status = response.status;
        task->type = response.get_header_value("Content-Type", "application/json");
        if (task->stream && response.status >= 200 && response.status < 300 &&
            task->type.find("text/event-stream") == std::string::npos) {
            task->failure = "invalid_upstream_content_type";
            return false;
        }
        task->headers = true;
        task->cv.notify_all();
        return !task->cancelled;
    };
    request.content_receiver = [task, config](const char *data, size_t length, size_t, size_t) {
        std::unique_lock<std::mutex> lock(task->mu);
        if (task->cancelled) return false;
        if (task->stream && task->status >= 200 && task->status < 300) {
            // Only recognize the chat endpoint's terminal event; do not reserialize
            // JSON or UTF-8. Bound the parser's partial line independently of FIFO.
            for (size_t i = 0; i < length; ++i) {
                if (data[i] == '\n') {
                    if (!task->sse_line.empty() && task->sse_line.back() == '\r') task->sse_line.pop_back();
                    if (task->sse_line == "data: [DONE]" || task->sse_line == "data:[DONE]") task->sse_done = true;
                    task->sse_line.clear();
                } else {
                    if (task->sse_line.size() >= 65536) { task->failure = "upstream_sse_line_too_large"; return false; }
                    task->sse_line.push_back(data[i]);
                }
            }
            // Byte-transparent SSE bridge; network chunks are NOT tokens/events.
            // Split a large upstream callback so one chunk never exceeds capacity.
            for (size_t offset = 0; offset < length;) {
                size_t n = std::min(length - offset, static_cast<size_t>(config.stream_buffer));
                task->cv.wait_until(lock, task->deadline, [&] {
                    return task->cancelled || task->buffered + n <= static_cast<size_t>(config.stream_buffer);
                });
                if (task->cancelled || Clock::now() >= task->deadline) return false;
                task->chunks.emplace_back(data + offset, n);
                task->buffered += n;
                task->peak_buffered = std::max(task->peak_buffered, task->buffered);
                offset += n;
                task->cv.notify_all();
            }
        } else {
            if (task->response.size() + length > static_cast<size_t>(config.response_limit)) {
                task->failure = "upstream_response_too_large";
                return false;
            }
            task->response.append(data, length);
        }
        return true;
    };
    bool already_cancelled;
    { std::lock_guard<std::mutex> lock(task->mu); already_cancelled = task->cancelled; }
    if (already_cancelled) {
        std::string reason;
        { std::lock_guard<std::mutex> lock(task->mu); reason = task->failure; }
        task->reject(504, reason); return;
    }
    auto result = client->send(request);
    {
        std::unique_lock<std::mutex> lock(task->mu);
        if (result && task->stream && task->status >= 200 && task->status < 300 && !task->sse_done && !task->cancelled)
            task->failure = "upstream_incomplete_stream";
        if (!result || task->cancelled) {
            if (task->failure.empty()) task->failure = "upstream_io_error";
            if (!task->headers) {
                task->status = task->failure == "request_timeout" ? 504 : 502;
                task->response = error_body(task->failure, task->id);
                task->type = "application/json";
            } else if (!task->stream || task->status >= 300) {
                task->status = task->failure == "request_timeout" ? 504 : 502;
                task->response = error_body(task->failure, task->id);
                task->type = "application/json";
            }
        }
        task->headers = task->done = true;
        task->cv.notify_all();
        // Streaming execution slot remains held until the provider drains or cancels.
        if (task->stream && task->status >= 200 && task->status < 300) {
            if (!task->cv.wait_until(lock, task->deadline, [&] { return task->delivered || task->cancelled; })) {
                lock.unlock(); task->cancel("request_timeout");
            }
        }
    }
}
