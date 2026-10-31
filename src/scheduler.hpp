#pragma once
#include "config.hpp"
#include "task.hpp"
#include <algorithm>
#include <functional>
#include <iostream>
#include <map>
#include <thread>
#include <vector>

// 两条显式队列：ready 已预留执行槽位；waiting 是有界 FIFO。
// 网络 I/O 与 Task 取消永远发生在调度器锁之外。
class Scheduler
{
    Config config;
    std::mutex mu;                                              // 保护队列、active、running、closing
    std::condition_variable wake;                               // worker 等待新任务 / 看门狗周期唤醒
    bool closing = false;                                       // 关闭中
    size_t active = 0;                                          // 已占执行槽位数 == ready.size() + running.size()
    std::deque<std::shared_ptr<Task>> ready, waiting;           // 就绪（占槽）/ 等待（不占槽）
    std::map<std::string, std::shared_ptr<Task>> running;       // 执行中的任务，按 id
    std::vector<std::thread> workers;                           // 固定 Worker 池
    std::thread watchdog;                                       // 周期检查超时/取消
    std::function<void(const std::shared_ptr<Task> &)> execute; // 注入的执行函数
    std::atomic<unsigned long> accepted{0}, rejected{0}, expired{0}, finished{0};
    std::atomic<size_t> peak_buffer{0};

    // 判断任务是否已失效。在调度器锁内调用；内部只短暂取 t->mu，无阻塞 I/O。
    bool stale(const std::shared_ptr<Task> &t)
    {
        {
            std::lock_guard<std::mutex> lock(t->mu);
            if (t->cancelled)
            {
                // 已取消：推进到终态并唤醒等待者
                t->done = t->headers = t->worker_released = true;
                t->trace->worker_finished();
                t->cv.notify_all();
                return true;
            }
        }
        // 排队超时或总期限已过
        if (Clock::now() >= std::min(t->queue_deadline, t->deadline))
        {
            t->reject(504, "queue_timeout");
            ++expired;
            return true;
        }
        return false;
    }

    // 清理 waiting 中的失效任务（erase-remove 惯用法）
    void prune()
    {
        waiting.erase(std::remove_if(waiting.begin(), waiting.end(),
                                     [&](const auto &t)
                                     { return stale(t); }),
                      waiting.end());
    }

    // 在配额内把 waiting 队首提升到 ready，每提升一个 ++active（预留槽位）
    void promote()
    {
        prune();
        while (active < static_cast<size_t>(config.concurrency) && !waiting.empty())
        {
            ready.push_back(waiting.front());
            waiting.pop_front();
            ++active;
        }
        wake.notify_all();
    }

    // 任务收尾，归还配额：锁 t->mu 做状态收尾与日志构造；锁外 CAS 峰值；锁 mu 归还配额并 promote
    void complete(const std::shared_ptr<Task> &t)
    {
        size_t peak;
        std::string log;
        {
            // 分三段，锁顺序严格：t->mu → 释放 → mu
            std::lock_guard<std::mutex> lock(t->mu);
            peak = t->peak_buffered;
            t->client.reset(); // 释放 Client，避免析构阻塞
            t->worker_released = true;
            t->cv.notify_all();
            t->trace->fail(t->failure);
            t->trace->worker_finished();
            // 构造 JSON 日志：请求 ID、上游状态、失败原因、排队时长（started - received）、worker 生命周期（now - started）、峰值缓冲
            log = Json{{"request_id", t->id}, {"upstream_status", t->status}, {"failure", t->failure}, {"queue_ms", std::chrono::duration_cast<std::chrono::milliseconds>(t->started - t->received).count()}, {"worker_lifetime_ms", std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - t->started).count()}, {"peak_stream_buffer_bytes", peak}}.dump();
        }
        // 锁外 CAS 更新全局峰值缓冲
        size_t observed = peak_buffer.load();
        while (observed < peak && !peak_buffer.compare_exchange_weak(observed, peak))
        {
        }
        std::lock_guard<std::mutex> lock(mu);
        running.erase(t->id);
        // --active + promote() 是配额流转的核心：归还一个槽位后立刻尝试提升等待者。
        --active;
        ++finished;
        std::cout << log << std::endl;
        if (!closing)
            promote(); // 归还槽位后立即尝试提升等待者
    }

    void worker()
    {
        while (true)
        {
            std::shared_ptr<Task> t;
            {
                std::unique_lock<std::mutex> lock(mu);
                // 等待任务
                wake.wait(lock, [&]
                          { return closing || !ready.empty(); });
                if (ready.empty())
                    return; // closing 且无任务 → 退出
                // ready 队首出队，放入 running
                t = ready.front();
                ready.pop_front();
                running[t->id] = t;
            }
            // RAII 管理执行配额：任何退出路径（含异常）都 complete 归还 active。
            // 流式 execute 会等 delivered 或取消，因此生成结束不会立即放行下一个请求。
            struct SlotLease
            {
                Scheduler &owner;
                std::shared_ptr<Task> task;
                ~SlotLease() { owner.complete(task); } // 析构时释放资源归还配额
            } lease{*this, t};
            {
                std::lock_guard<std::mutex> lock(t->mu);
                t->started = Clock::now();
                t->trace->execution_started();
            }
            try
            {
                if (!stale(t))
                    execute(t); // 出队到执行间可能被取消/超时，再判一次
            }
            catch (const std::exception &e)
            {
                // 异常兜底：转 500，不泄漏配额（SlotLease 析构会 complete）
                t->cancel("internal_error");
                std::lock_guard<std::mutex> lock(t->mu);
                t->status = 500;
                t->headers = t->done = true;
                t->response = error_body("internal_error", t->id);
                t->cv.notify_all();
                std::cerr << "worker exception: " << e.what() << std::endl;
            }
        }
    }

public:
    Scheduler(Config c, std::function<void(const std::shared_ptr<Task> &)> fn) : config(c), execute(std::move(fn))
    {
        for (int i = 0; i < config.concurrency; ++i)
            workers.emplace_back([this]
                                 { worker(); });
        // 看门狗：20ms 周期检查 running 的 deadline/取消；锁内快照，锁外 cancel
        watchdog = std::thread([this]
                               {
            while (true) {
                std::vector<std::shared_ptr<Task>> snapshot;
                { std::unique_lock<std::mutex> lock(mu);
                  wake.wait_for(lock, 20ms);
                  if (closing && active == 0) return;   // 关闭且全部收尾 → 退出
                  prune();
                  for (const auto &entry : running) snapshot.push_back(entry.second); }
                for (const auto &t : snapshot) {
                    bool cancelled;
                    { std::lock_guard<std::mutex> lock(t->mu); cancelled = t->cancelled; }
                    // 重复 cancel 覆盖 cancel/send-start 竞态
                    if (Clock::now() >= t->deadline) t->cancel("request_timeout");
                    else if (cancelled) t->cancel("cancelled");
                }
            } });
    }

    bool submit(const std::shared_ptr<Task> &t)
    {
        std::lock_guard<std::mutex> lock(mu);
        if (closing)
        {
            t->reject(503, "shutting_down");
            ++rejected;
            return false;
        }
        prune(); // 清理 waiting 中的失效任务
        if (active < static_cast<size_t>(config.concurrency) && waiting.empty())
        {
            ++active;
            ready.push_back(t); // 有配额且无人排队 → 直通，预留槽位
        }
        else if (waiting.size() < static_cast<size_t>(config.queue_capacity))
        {
            waiting.push_back(t); // 排队（不占槽）
        }
        else
        {
            // 队列满：拒绝，返回 503
            t->reject(503, "queue_full");
            ++rejected;
            return false;
        }
        t->trace->enqueued();
        ++accepted;
        wake.notify_all();
        return true;
    }

    // 返回调度器状态统计
    Json stats()
    {
        std::lock_guard<std::mutex> lock(mu);
        return {{"active", active}, {"queued", waiting.size()}, {"accepted", accepted.load()}, {"rejected", rejected.load()}, {"queue_expired", expired.load()}, {"finished", finished.load()}, {"peak_stream_buffer_bytes", peak_buffer.load()}, {"stream_buffer_limit", config.stream_buffer}, {"max_active", config.concurrency}, {"queue_capacity", config.queue_capacity}};
    }

    // 停止准入并取消所有任务
    void stop()
    {
        std::vector<std::shared_ptr<Task>> snapshot;
        {
            std::lock_guard<std::mutex> lock(mu);
            if (closing)
                return; // 幂等
            closing = true;
            for (auto &t : waiting)
                t->reject(503, "shutting_down"); // 未执行直接拒绝
            waiting.clear();
            for (auto &t : ready)
                snapshot.push_back(t);
            for (auto &entry : running)
                snapshot.push_back(entry.second);
        }
        for (auto &t : snapshot)
            t->cancel("shutting_down"); // 锁外取消（可能阻塞）
        wake.notify_all();
    }

    // 析构：先停止准入并取消任务，再 join 所有后台线程，确保它们不再使用随后销毁的对象。
    ~Scheduler()
    {
        stop();
        if (watchdog.joinable())
            watchdog.join();
        for (auto &thread : workers)
            if (thread.joinable())
                thread.join();
    }
};