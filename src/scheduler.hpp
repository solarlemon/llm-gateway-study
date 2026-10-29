#pragma once
#include "config.hpp"
#include "task.hpp"
#include <algorithm>
#include <functional>
#include <iostream>
#include <map>
#include <thread>
#include <vector>

// Two explicit queues: ready reserves an execution slot; waiting is bounded FIFO.
// Network I/O and Task cancellation always occur outside the scheduler mutex.
class Scheduler {
    Config config;
    std::mutex mu;
    std::condition_variable wake;
    bool closing = false;
    size_t active = 0;
    std::deque<std::shared_ptr<Task>> ready, waiting;
    std::map<std::string, std::shared_ptr<Task>> running;
    std::vector<std::thread> workers;
    std::thread watchdog;
    std::function<void(const std::shared_ptr<Task>&)> execute;
    std::atomic<unsigned long> accepted{0}, rejected{0}, expired{0}, finished{0};
    std::atomic<size_t> peak_buffer{0};

    // Called with scheduler mutex held. No blocking I/O here.
    bool stale(const std::shared_ptr<Task> &t) {
        { std::lock_guard<std::mutex> lock(t->mu);
          if (t->cancelled) { t->done = t->headers = t->worker_released = true; t->cv.notify_all(); return true; } }
        if (Clock::now() >= std::min(t->queue_deadline, t->deadline)) {
            t->reject(504, "queue_timeout"); ++expired; return true;
        }
        return false;
    }
    void prune() {
        waiting.erase(std::remove_if(waiting.begin(), waiting.end(), [&](const auto &t) { return stale(t); }), waiting.end());
    }
    void promote() {
        prune();
        while (active < static_cast<size_t>(config.concurrency) && !waiting.empty()) {
            ready.push_back(waiting.front()); waiting.pop_front(); ++active;
        }
        wake.notify_all();
    }
    void complete(const std::shared_ptr<Task> &t) {
        size_t peak;
        std::string log;
        {
            std::lock_guard<std::mutex> lock(t->mu);
            peak = t->peak_buffered;
            t->client.reset(); t->worker_released = true; t->cv.notify_all();
            log = Json{{"request_id", t->id}, {"upstream_status", t->status}, {"failure", t->failure},
                       {"queue_ms", std::chrono::duration_cast<std::chrono::milliseconds>(t->started - t->received).count()},
                       {"worker_lifetime_ms", std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - t->started).count()},
                       {"peak_stream_buffer_bytes", peak}}.dump();
        }
        size_t observed = peak_buffer.load();
        while (observed < peak && !peak_buffer.compare_exchange_weak(observed, peak)) {}
        std::lock_guard<std::mutex> lock(mu);
        running.erase(t->id); --active; ++finished;
        std::cout << log << std::endl;
        if (!closing) promote();
    }
    void worker() {
        while (true) {
            std::shared_ptr<Task> t;
            {
                std::unique_lock<std::mutex> lock(mu);
                wake.wait(lock, [&] { return closing || !ready.empty(); });
                if (ready.empty()) return;
                t = ready.front(); ready.pop_front(); running[t->id] = t;
            }
            // Execution reservation is released on every exit path, including exceptions.
            struct SlotLease {
                Scheduler &owner; std::shared_ptr<Task> task;
                ~SlotLease() { owner.complete(task); }
            } lease{*this, t};
            { std::lock_guard<std::mutex> lock(t->mu); t->started = Clock::now(); }
            try {
                if (!stale(t)) execute(t);
            } catch (const std::exception &e) {
                t->cancel("internal_error");
                std::lock_guard<std::mutex> lock(t->mu);
                t->status = 500; t->headers = t->done = true;
                t->response = error_body("internal_error", t->id); t->cv.notify_all();
                std::cerr << "worker exception: " << e.what() << std::endl;
            }
        }
    }
public:
    Scheduler(Config c, std::function<void(const std::shared_ptr<Task>&)> fn) : config(c), execute(std::move(fn)) {
        for (int i = 0; i < config.concurrency; ++i) workers.emplace_back([this] { worker(); });
        watchdog = std::thread([this] {
            while (true) {
                std::vector<std::shared_ptr<Task>> snapshot;
                { std::unique_lock<std::mutex> lock(mu);
                  wake.wait_for(lock, 20ms);
                  if (closing && active == 0) return;
                  prune(); for (const auto &entry : running) snapshot.push_back(entry.second); }
                for (const auto &t : snapshot) {
                    bool cancelled;
                    { std::lock_guard<std::mutex> lock(t->mu); cancelled = t->cancelled; }
                    if (Clock::now() >= t->deadline) t->cancel("request_timeout");
                    else if (cancelled) t->cancel("cancelled");
                }
            }
        });
    }
    bool submit(const std::shared_ptr<Task> &t) {
        std::lock_guard<std::mutex> lock(mu);
        if (closing) { t->reject(503, "shutting_down"); ++rejected; return false; }
        prune();
        if (active < static_cast<size_t>(config.concurrency) && waiting.empty()) {
            ++active; ready.push_back(t);
        } else if (waiting.size() < static_cast<size_t>(config.queue_capacity)) waiting.push_back(t);
        else { t->reject(503, "queue_full"); ++rejected; return false; }
        ++accepted; wake.notify_all(); return true;
    }
    Json stats() {
        std::lock_guard<std::mutex> lock(mu);
        return {{"active", active}, {"queued", waiting.size()}, {"accepted", accepted.load()},
                {"rejected", rejected.load()}, {"queue_expired", expired.load()}, {"finished", finished.load()},
                {"peak_stream_buffer_bytes", peak_buffer.load()}, {"stream_buffer_limit", config.stream_buffer},
                {"max_active", config.concurrency}, {"queue_capacity", config.queue_capacity}};
    }
    void stop() {
        std::vector<std::shared_ptr<Task>> snapshot;
        { std::lock_guard<std::mutex> lock(mu);
          if (closing) return;
          closing = true;
          for (auto &t : waiting) t->reject(503, "shutting_down");
          waiting.clear();
          for (auto &t : ready) snapshot.push_back(t);
          for (auto &entry : running) snapshot.push_back(entry.second); }
        for (auto &t : snapshot) t->cancel("shutting_down");
        wake.notify_all();
    }
    ~Scheduler() {
        stop();
        if (watchdog.joinable()) watchdog.join();
        for (auto &thread : workers) if (thread.joinable()) thread.join();
    }
};
