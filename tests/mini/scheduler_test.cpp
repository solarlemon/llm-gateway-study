#include "scheduler.hpp"
#include <stdexcept>

void require(bool value) { if (!value) throw std::runtime_error("scheduler assertion failed"); }
template<class Predicate> void eventually(Predicate predicate) {
    const auto end = Clock::now()+2s;
    while (!predicate()) {
        if (Clock::now() >= end) throw std::runtime_error("scheduler test timed out");
        std::this_thread::sleep_for(1ms);
    }
}
int main() {
    Config config;
    config.concurrency=1; config.queue_capacity=1; config.queue_ms=100; config.request_ms=2000;
    Metrics metrics(16,"");
    std::mutex mu;
    std::condition_variable cv;
    bool release=false;
    std::atomic<int> calls{0};
    Scheduler scheduler(config,[&](const auto& task){
        ++calls;
        if (task->id == "a") {
            std::unique_lock<std::mutex> lock(mu);
            cv.wait(lock,[&]{return release;});
        }
        std::lock_guard<std::mutex> lock(task->mu);
        task->headers=task->done=true; task->status=200; task->cv.notify_all();
    });
    auto make = [&](const std::string& id){
        return std::make_shared<Task>(id,"{}",false,config.queue_ms,config.request_ms,
                                      std::make_shared<RequestTrace>(metrics,id));
    };
    auto a=make("a"), b=make("b"), c=make("c");
    // 显式屏障使 A 占住槽位，B 排队，C 必然遇到容量上限。
    require(scheduler.submit(a));
    eventually([&]{return calls==1;});
    require(scheduler.submit(b)); require(!scheduler.submit(c));
    { std::lock_guard<std::mutex> lock(c->mu); require(c->status==503 && c->failure=="queue_full"); }
    c->trace->response_finished(true,503);
    eventually([&]{ std::lock_guard<std::mutex> lock(b->mu); return b->done; });
    { std::lock_guard<std::mutex> lock(b->mu); require(b->status==504 && b->failure=="queue_timeout"); }
    require(calls==1); // 过期任务未执行。
    b->trace->response_finished(true,504);
    { std::lock_guard<std::mutex> lock(mu); release=true; }
    cv.notify_all();
    eventually([&]{return scheduler.stats()["active"]==0;});
    a->trace->response_finished(true,200);
    auto d=make("d"); require(scheduler.submit(d));
    eventually([&]{return scheduler.stats()["active"]==0;});
    d->trace->response_finished(true,200);
    require(calls==2 && scheduler.stats()["queued"]==0);
    auto rows=metrics.summary();
    require(rows["terminal"]==4 && rows["inflight"]==0);
    require(rows["states"]["rejected"]==1 && rows["states"]["queue_timeout"]==1);
    std::cout << "PASS scheduler queue capacity, expired-never-executed, quota recovery and terminal metrics\n";
}
