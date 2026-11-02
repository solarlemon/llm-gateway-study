#pragma once
#include "json.hpp"
#include "metrics.hpp"
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
inline std::string error_body(const std::string &code, const std::string &id)
{
    return Json{{"error", {{"code", code}, {"message", code}}}, {"request_id", id}}.dump();
}

// 共享任务是线程间交接点；可变状态由 mu 保护，cv 负责唤醒，真正的条件保存在字段中。
struct Task
{
    std::string id, body;
    bool stream;
    // 接收时间和开始处理时间、排队截止时间（超过拒绝）、总处理截止时间
    Clock::time_point received = Clock::now(), started = received, queue_deadline, deadline;

    std::mutex mu;
    std::condition_variable cv;
    // headers：响应头可用；done：生产结束；delivered：发送路径已收尾。三者不是同一个时刻。
    bool headers = false, done = false, delivered = false, worker_released = false;
    bool cancelled = false;

    // 响应数据
    int status = 500;
    std::string type = "application/json", failure, response;
    size_t sse_bytes = 0, first_content_end = 0, forwarded_bytes = 0;

    // 追踪与缓冲数据
    std::shared_ptr<RequestTrace> trace;
    std::deque<std::string> chunks;
    size_t buffered = 0, peak_buffered = 0;

    Task(std::string request_id, std::string request_body, bool streaming, int queue_ms, int total_ms,
         std::shared_ptr<RequestTrace> observation)
        : id(std::move(request_id)), body(std::move(request_body)), stream(streaming), trace(std::move(observation))
    {
        received = trace->received;
        trace->work_pending(stream);
        queue_deadline = received + std::chrono::milliseconds(queue_ms);
        deadline = received + std::chrono::milliseconds(total_ms);
    }

    // 发布协作式取消并唤醒缓冲生产者/消费者，模型在 token 或层之间检查状态。
    void cancel(const std::string& reason) {
        {
            std::lock_guard<std::mutex> lock(mu);
            if (!cancelled) { cancelled=true; failure=reason; }
            trace->fail(failure);
        }
        cv.notify_all();
    }

    void reject(int code, const std::string &reason)
    {
        std::lock_guard<std::mutex> lock(mu);
        status = code;
        failure = reason;
        trace->fail(reason);
        trace->worker_finished();
        response = error_body(reason, id);
        headers = done = true;
        worker_released = true;
        cv.notify_all();
    }
};
