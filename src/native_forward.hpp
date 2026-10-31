#pragma once
#include "llama_backend.hpp"
#include "task.hpp"
#include "config.hpp"

// 【原生适配层】本函数在调度 worker 中运行；不建立到模型服务的 HTTP 连接。
// 输入是配置、共享引擎引用和请求 Task；输出写入 Task，而非直接操作下游 socket。
inline void forward_native(const Config &config, LlamaBackend &engine, const std::shared_ptr<Task> &task)
{
    Json stats;
    double backpressure_ms = 0;
    // generate 通过此回调查询断连/超时。它是协作式检查，不能立即中断正在执行的 GPU kernel。
    auto cancelled = [&]
    {
        if (Clock::now() >= task->deadline)
            task->cancel("request_timeout"); // 超时主动取消
        std::lock_guard<std::mutex> lock(task->mu);
        return task->cancelled;
    };

    // 把完整 SSE 帧拆成不超过容量的字节块入队；帧边界与队列块边界可以不同。
    // frame 是序列化后的 SSE 字节，不是模型 token；content 用于标记首个有效回答事件。
    // 此闭包同步执行于生成 worker，cv 等待释放 Task 锁，让 HTTP provider 能出队腾出容量。
    auto push = [&](const std::string &frame, bool content)
    {
        std::unique_lock<std::mutex> lock(task->mu);
        if (task->cancelled)
            throw NativeError("native_cancelled", 499);
        task->headers = true;
        task->status = 200;
        task->type = "text/event-stream";
        task->sse_bytes += frame.size();
        // 记录首个有效回答事件的累计字节数，供后续计算 TTFT（首次内容时间）。
        if (content && !task->first_content_end)
            task->first_content_end = task->sse_bytes;
        task->cv.notify_all();
        for (size_t offset = 0; offset < frame.size();)
        {
            const size_t n = std::min(frame.size() - offset, static_cast<size_t>(config.stream_buffer));
            auto before = Clock::now();

            // 等待缓冲空位会阻塞当前生成回调，从而把慢客户端压力传回生成循环；不会无限追加 chunks。
            task->cv.wait_until(lock, task->deadline, [&]
                                { return task->cancelled || task->buffered + n <= static_cast<size_t>(config.stream_buffer); });
            backpressure_ms += std::chrono::duration<double, std::milli>(Clock::now() - before).count();
            if (task->cancelled || Clock::now() >= task->deadline)
                // 抛 NativeError 让控制流穿透 generate，在函数末尾的 catch 统一处理
                throw NativeError("native_cancelled", 499);
            task->chunks.emplace_back(frame.data() + offset, n);
            task->buffered += n;
            task->peak_buffered = std::max(task->peak_buffered, task->buffered);
            offset += n;
            // 消费者出队后会在 Task::cancel 或 SSE provider 回调里 notify_all，唤醒生产者继续入队。
            task->cv.notify_all();
        }
    };

    // SSE 以 data: 加 JSON 和空行组成事件；content 放增量文本，结束标记另行发送。
    auto chunk = [&](Json delta, Json finish, Json extra = Json::object())
    {
        Json value = {{"id", "native-" + task->id},
                      {"object", "chat.completion.chunk"},
                      {"choices", Json::array({{{"index", 0},
                                                {"delta", std::move(delta)},
                                                {"finish_reason", std::move(finish)}}})}};
        value.update(extra);
        return "data: " + value.dump() + "\n\n";
    };

    try
    {
        // 【进入生成核心】参数依次为请求 JSON、文字输出回调、取消检查回调、阶段统计引用。
        // generate 是同步调用：每产生可发送文字就调用此 lambda；只有完整生成结束才返回 result。
        // 流式请求在回调里立即入队；非流式请求只打点，之后使用 result 中累计的完整文本。
        auto result = engine.generate(Json::parse(task->body), [&](const std::string &piece)
                                      {
            task->trace->native_content();
            if (task->stream) push(chunk({{"content",piece}},nullptr),true); }, cancelled, stats);

        if (cancelled())
            // 生成结束时再查一次取消——防止取消恰好发生在最后一次检查之后。
            throw NativeError("native_cancelled", 499);
        stats["backpressure_wait_ms"] = backpressure_ms;
        task->trace->native_finished(stats);
        task->trace->backend({{"usage", result["usage"]}});
        if (task->stream)
        {
            push(chunk(Json::object(), result["finish_reason"], {{"usage", result["usage"]}, {"native_timings", stats}}), false);
            // 先发送 finish_reason/usage/阶段计时，再发送协议结束标记；入队成功并不等于客户端已收到。
            push("data: [DONE]\n\n", false);
            stats["backpressure_wait_ms"] = backpressure_ms;
            task->trace->native_finished(stats);
        }
        else
        {
            // 非流式路径：生成结束后把完整回答封装成单个 JSON，再交给 main.cpp 的普通响应路径。
            Json response = {{"id", "native-" + task->id}, {"object", "chat.completion"}, {"choices", Json::array({{{"index", 0}, {"message", {{"role", "assistant"}, {"content", result["text"]}}}, {"finish_reason", result["finish_reason"]}}})}, {"usage", result["usage"]}, {"native_timings", stats}};
            auto body = response.dump();
            if (body.size() > static_cast<size_t>(config.response_limit))
                throw NativeError("native_response_too_large");
            std::lock_guard<std::mutex> lock(task->mu);
            task->status = 200;
            task->type = "application/json";
            task->response = std::move(body);
        }
    }
    catch (const std::exception &e)
    {
        // 生成、文本编码或回压等待抛出的 C++ 异常在适配层转换成任务错误；不会触发 HTTP 后端重试。
        stats["backpressure_wait_ms"] = backpressure_ms;
        task->trace->native_finished(stats);
        const auto native = dynamic_cast<const NativeError *>(&e);
        std::lock_guard<std::mutex> lock(task->mu);
        if (task->failure.empty())
            task->failure = native ? native->code : "native_internal_error";
        // 尚未公布响应头时可以设置错误状态；流已开始时只能沿 failure 交给 provider 发送流内错误。
        if (!task->headers)
        {
            task->status = task->failure == "request_timeout" ? 504 : native ? native->status
                                                                             : 500;
            if (task->status == 499)
                task->status = 504;
            task->type = "application/json";
            task->response = error_body(task->failure, task->id);
        }
        task->trace->fail(task->failure);
    }

    std::unique_lock<std::mutex> lock(task->mu);
    // 标记生产结束并唤醒消费者；流式分支仍等待 delivered/取消，不能把生产结束当成发送结束。
    task->headers = task->done = true; // 生产结束，唤醒 provider 做最后消费
    task->cv.notify_all();
    if (task->stream && task->status == 200 && !task->cancelled)
    {
        // 流式 200 且未取消时，Worker 继续持有锁等待 delivered：
        // 即执行配额（Worker 槽位）一直占用到下游把缓冲排空，防止慢客户端在内存里无限堆积。
        if (!task->cv.wait_until(lock, task->deadline, [&]
                                 { return task->delivered || task->cancelled; }))
        {
            lock.unlock();
            task->cancel("request_timeout");
        }
    }
}
