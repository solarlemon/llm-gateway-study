#include "src/scheduler.hpp"
#include "src/upstream.hpp"
#include <csignal>
#include <pthread.h>

namespace {
void respond_error(httplib::Response &response, int status, const std::string &code, const std::string &id) {
    response.status = status;
    response.set_content(error_body(code, id), "application/json");
}

// Validate the supported chat subset. Token/context limits remain backend-owned.
bool valid_request(const Json &j) {
    if (!j.is_object() || !j.contains("messages") || !j["messages"].is_array() || j["messages"].empty()) return false;
    for (const auto &message : j["messages"]) {
        if (!message.is_object() || !message.contains("role") || !message["role"].is_string() ||
            !message.contains("content") || !message["content"].is_string()) return false;
        const auto role = message["role"].get<std::string>();
        if (role != "system" && role != "user" && role != "assistant") return false;
    }
    if (j.contains("stream") && !j["stream"].is_boolean()) return false;
    if (j.contains("max_tokens") && (!j["max_tokens"].is_number_integer() ||
        j["max_tokens"].get<double>() < 1 || j["max_tokens"].get<double>() > 4096)) return false;
    return true;
}

// Runs only while the HTTP handler owns the request socket. Never saved in Task.
bool wait_for_task(const httplib::Request &request, const std::shared_ptr<Task> &task,
                   const std::function<bool()> &predicate) {
    std::unique_lock<std::mutex> lock(task->mu);
    while (!predicate()) {
        task->cv.wait_for(lock, 20ms);
        lock.unlock();
        if (request.is_connection_closed()) task->cancel("client_disconnected");
        if (Clock::now() >= task->deadline) task->cancel("request_timeout");
        lock.lock();
        if (task->cancelled) return false;
    }
    return true;
}
}

int main() {
    try {
        Config config = Config::load();
        // Signal handling is synchronous on a dedicated thread, not an unsafe handler.
        sigset_t signals;
        sigemptyset(&signals); sigaddset(&signals, SIGINT); sigaddset(&signals, SIGTERM); sigaddset(&signals, SIGUSR1);
        pthread_sigmask(SIG_BLOCK, &signals, nullptr);
        httplib::Server server;
        Scheduler scheduler(config, [&](const auto &task) { forward_request(config, task); });
        server.new_task_queue = [config] { return new httplib::ThreadPool(config.http_threads, config.http_threads, config.http_threads); };
        server.set_payload_max_length(1024 * 1024);
        server.set_read_timeout(5, 0); server.set_write_timeout(2, 0);
        server.set_keep_alive_max_count(1);
        std::atomic<unsigned long> serial{0};
        server.Get("/health", [](const httplib::Request &, httplib::Response &res) { res.set_content("ok\n", "text/plain"); });
        server.Get("/stats", [&](const httplib::Request &, httplib::Response &res) { res.set_content(scheduler.stats().dump(), "application/json"); });
        server.Post("/v1/chat/completions", [&](const httplib::Request &req, httplib::Response &res) {
            const auto id = std::to_string(++serial);
            res.set_header("X-Request-ID", id);
            auto body = Json::parse(req.body, nullptr, false);
            if (body.is_discarded() || !valid_request(body)) { respond_error(res, 400, "invalid_request", id); return; }
            if (!body.contains("max_tokens")) body["max_tokens"] = 128;
            auto task = std::make_shared<Task>(id, body.dump(), body.value("stream", false), config.queue_ms, config.request_ms);
            scheduler.submit(task);
            if (!wait_for_task(req, task, [&] { return task->headers; })) {
                std::string reason;
                { std::lock_guard<std::mutex> lock(task->mu); reason = task->failure; }
                respond_error(res, 504, reason, id); return;
            }
            int status;
            { std::lock_guard<std::mutex> lock(task->mu); status = task->status; }
            if (!task->stream || status < 200 || status >= 300) {
                if (!wait_for_task(req, task, [&] { return task->done; })) {
                    respond_error(res, 504, "request_cancelled", id); return;
                }
                std::lock_guard<std::mutex> lock(task->mu);
                res.status = task->status; res.set_content(task->response, task->type);
                if (res.status == 503) res.set_header("Retry-After", "1");
                return;
            }
            res.status = status;
            res.set_header("Cache-Control", "no-cache"); res.set_header("X-Accel-Buffering", "no");
            res.set_chunked_content_provider("text/event-stream",
                [task](size_t, httplib::DataSink &sink) {
                    for (;;) {
                        if (Clock::now() >= task->deadline) { task->cancel("request_timeout"); return false; }
                        std::string chunk, failure;
                        bool done = false, cancelled = false;
                        {
                            std::unique_lock<std::mutex> lock(task->mu);
                            if (!task->cv.wait_for(lock, 250ms, [&] { return !task->chunks.empty() || task->done || task->cancelled; })) {
                                lock.unlock();
                                // SSE comment heartbeat reveals an idle downstream disconnect.
                                if (!sink.write(": gateway heartbeat\n\n", 21)) { task->cancel("client_disconnected"); return false; }
                                continue;
                            }
                            cancelled = task->cancelled;
                            if (!task->chunks.empty()) {
                                chunk = std::move(task->chunks.front()); task->chunks.pop_front();
                                task->buffered -= chunk.size(); task->cv.notify_all();
                            } else { done = task->done; failure = task->failure; }
                        }
                        if (cancelled) return false;
                        if (!chunk.empty()) {
                            if (!sink.write(chunk.data(), chunk.size())) { task->cancel("client_disconnected"); return false; }
                        } else if (done) {
                            if (!failure.empty()) {
                                auto error = "\n\nevent: error\ndata: " + error_body(failure, task->id) + "\n\n";
                                if (!sink.write(error.data(), error.size())) { task->cancel("client_disconnected"); return false; }
                            }
                            sink.done();
                            { std::lock_guard<std::mutex> lock(task->mu); task->delivered = true; task->cv.notify_all(); }
                            return true;
                        }
                    }
                },
                [task](bool success) {
                    if (!success) task->cancel("downstream_write_failed");
                    std::lock_guard<std::mutex> lock(task->mu);
                    task->delivered = true; task->cv.notify_all();
                });
        });
        std::thread signal_thread([&] {
            int received = 0;
            sigwait(&signals, &received);
            if (received != SIGUSR1) { scheduler.stop(); server.stop(); }
        });
        std::cout << "Gateway on 127.0.0.1:" << config.port << " -> " << config.upstream_host << ':' << config.upstream_port << std::endl;
        bool ok = server.listen("127.0.0.1", config.port);
        scheduler.stop();
        pthread_kill(signal_thread.native_handle(), SIGUSR1); signal_thread.join();
        return ok ? 0 : 1;
    } catch (const std::exception &e) { std::cerr << e.what() << std::endl; return 1; }
}
