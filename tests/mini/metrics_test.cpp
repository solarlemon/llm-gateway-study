#include "metrics.hpp"
#include <iostream>
#include <stdexcept>

void require(bool value)
{
    if (!value)
        throw std::runtime_error("metrics assertion failed");
}

int main()
{
    Metrics metrics(4, "");
    RequestTrace trace(metrics, "1");
    trace.work_pending(true);
    trace.enqueued();
    trace.execution_started();
    trace.mini_content();
    trace.content_forwarded();
    trace.mini_finished({{"prefill_wall_ms", 2.0}});
    trace.response_finished(true, 200);
    require(metrics.summary()["terminal"] == 0);
    trace.worker_finished();
    trace.worker_finished();
    trace.response_finished(true, 200);
    require(metrics.summary()["terminal"] == 1);
    auto row = metrics.get("1");
    require(row["state"] == "success" && row["mini_timings"]["prefill_wall_ms"] == 2.0);
    require(row["ttft_mini_ms"].get<double>() <= row["ttft_forwarded_ms"].get<double>());
    RequestTrace failed(metrics, "2");
    failed.work_pending(true);
    failed.fail("mini_internal_error");
    failed.worker_finished();
    failed.response_finished(true, 200);
    require(metrics.get("2")["state"] == "failed");
    RequestTrace invalid(metrics, "3");
    invalid.response_finished(true, 400);
    require(metrics.get("3")["state"] == "invalid");

    // 直接输入已知样本，验证窗口淘汰后仍保留累计终态，P95 使用 nearest-rank。
    Metrics window(3, "");
    for (int value = 1; value <= 4; ++value)
    {
        window.begin();
        window.record({{"request_id", std::to_string(value)}, {"state", "success"}, {"queue_ms", value}, {"mini_timings", {{"prefill_wall_ms", value}}}});
    }
    auto summary = window.summary();
    require(summary["terminal"] == 4 && summary["window_count"] == 3 && summary["inflight"] == 0);
    require(summary["latency_ms"]["queue_ms"]["success"]["p50"] == 3);
    require(summary["latency_ms"]["queue_ms"]["success"]["p95"] == 4);
    require(summary["mini_timing_window"]["prefill_wall_ms"]["count"] == 3);
    require(window.get("1").is_null());
    std::cout << "PASS metrics exactly-once, failed SSE, invalid request, bounded window and quantiles\n";
}
