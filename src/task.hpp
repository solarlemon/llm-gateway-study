#pragma once
#include "httplib.h"
#include "json.hpp"
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <string>

using Clock = std::chrono::steady_clock;
using namespace std::chrono_literals;
using Json = nlohmann::json;
inline std::string error_body(const std::string &code, const std::string &id) {
    return Json{{"error", {{"code", code}, {"message", code}}}, {"request_id", id}}.dump();
}

// Owned by scheduler, worker and response provider. No raw request/socket retained.
struct Task {
    std::string id, body;
    bool stream;
    Clock::time_point received = Clock::now(), started = received, queue_deadline, deadline;
    std::mutex mu;
    std::condition_variable cv;
    bool headers = false, done = false, delivered = false, worker_released = false;
    bool cancelled = false;
    int status = 502;
    std::string type = "application/json", failure, response;
    std::string sse_line;
    bool sse_done = false;
    std::deque<std::string> chunks;
    size_t buffered = 0, peak_buffered = 0;
    std::shared_ptr<httplib::Client> client;

    Task(std::string request_id, std::string request_body, bool streaming, int queue_ms, int total_ms)
        : id(std::move(request_id)), body(std::move(request_body)), stream(streaming) {
        queue_deadline = received + std::chrono::milliseconds(queue_ms);
        deadline = received + std::chrono::milliseconds(total_ms);
    }
    void cancel(const std::string &reason) {
        std::shared_ptr<httplib::Client> current;
        {
            std::lock_guard<std::mutex> lock(mu);
            if (!cancelled) { cancelled = true; failure = reason; }
            current = client;
        }
        cv.notify_all();
        // cpp-httplib Client::stop interrupts an in-flight socket operation.
        // Repeated watchdog calls cover a cancel/send-start race.
        if (current) current->stop();
    }
    void reject(int code, const std::string &reason) {
        std::lock_guard<std::mutex> lock(mu);
        status = code; failure = reason;
        response = error_body(reason, id); headers = done = true;
        worker_released = true;
        cv.notify_all();
    }
};
