#pragma once
#include "json.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <deque>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

using MetricClock = std::chrono::steady_clock; // 单调时钟，不受系统时间调整影响
using MetricJson = nlohmann::json;

class Metrics
{
    std::mutex mu;   // 保护全部状态
    size_t capacity; // 内存窗口容量
    uint64_t received = 0, terminal = 0, log_errors = 0;
    std::map<std::string, uint64_t> counts; // 按 state 计数
    std::deque<MetricJson> records;         // 有界窗口：push_back + pop_front
    std::ofstream log;                      // JSONL 追加日志
    std::string run_id;                     // 进程运行 id

    // 最近秩分位数：排序后取 ceil(q*n)-1，不插值；空输入返回 null 分位
    static MetricJson distribution(std::vector<double> values)
    {
        if (values.empty())
            return {{"count", 0}, {"p50", nullptr}, {"p95", nullptr}, {"max", nullptr}};
        std::sort(values.begin(), values.end());
        auto p = [&](double q)
        { return values[static_cast<size_t>(std::ceil(q * values.size())) - 1]; };
        return {{"count", values.size()}, {"p50", p(.5)}, {"p95", p(.95)}, {"max", values.back()}};
    }

public:
    // 构造：生成 run_id；path 非空则建目录、追加打开日志，失败直接抛
    Metrics(size_t retained, const std::string &path) : capacity(retained)
    {
        run_id = std::to_string(std::chrono::duration_cast<std::chrono::microseconds>(
                                    std::chrono::system_clock::now().time_since_epoch())
                                    .count());
        if (!path.empty())
        {
            const auto parent = std::filesystem::path(path).parent_path();
            if (!parent.empty())
                std::filesystem::create_directories(parent);
            log.open(path, std::ios::app);
            if (!log)
                throw std::runtime_error("cannot open REQUEST_LOG_PATH: " + path);
        }
    }

    // 请求进入计数
    void begin()
    {
        std::lock_guard<std::mutex> lock(mu);
        ++received;
    }

    // 发布一条终态：打 run_id、计数、写日志（flush）、入窗口并裁剪
    void record(MetricJson row)
    {
        std::lock_guard<std::mutex> lock(mu);
        row["run_id"] = run_id;
        ++terminal;
        ++counts[row["state"].get<std::string>()];
        if (log.is_open())
        {
            log << row.dump() << '\n';
            log.flush();
            if (!log)
                ++log_errors;
        }
        records.push_back(std::move(row)); // 移动后 row 失效，故日志必须先写
        while (records.size() > capacity)
            records.pop_front();
    }

    // 返回窗口副本
    MetricJson recent()
    {
        std::lock_guard<std::mutex> lock(mu);
        return MetricJson(records);
    }

    // 倒序查找 request_id（最近发布优先）
    MetricJson get(const std::string &id)
    {
        std::lock_guard<std::mutex> lock(mu);
        for (auto it = records.rbegin(); it != records.rend(); ++it)
            if ((*it)["request_id"] == id)
                return *it;
        return nullptr;
    }

    // 汇总：总览 + 延迟分布（all/success 双套）+ mini 计时窗口
    MetricJson summary()
    {
        std::lock_guard<std::mutex> lock(mu);
        MetricJson latency = MetricJson::object();
        for (const std::string name : {"queue_ms", "ttft_mini_ms", "ttft_forwarded_ms", "gateway_e2e_ms"})
        {
            std::vector<double> all, success;
            for (const auto &row : records)
            {
                if (row.contains(name) && row[name].is_number())
                {
                    all.push_back(row[name].get<double>());
                    if (row["state"] == "success")
                        success.push_back(row[name].get<double>());
                }
            }
            latency[name] = {{"all", distribution(all)}, {"success", distribution(success)}};
        }
        MetricJson backend = MetricJson::object();
        for (const std::string name : {"prefill_wall_ms", "decode_wall_ms", "sampling_ms"})
        {
            std::vector<double> values;
            for (const auto &row : records)
            {
                // 只统计成功请求且 mini_timings 结构完整
                if (row["state"] == "success" && row["mini_timings"].is_object() &&
                    row["mini_timings"].contains(name) && row["mini_timings"][name].is_number())
                    values.push_back(row["mini_timings"][name].get<double>());
            }
            backend[name] = distribution(values);
        }
        return {{"run_id", run_id}, {"received", received}, {"terminal", terminal}, {"inflight", received - terminal}, {"states", counts}, {"log_write_errors", log_errors}, {"window_count", records.size()}, {"window_capacity", capacity}, {"quantile_method", "nearest_rank"}, {"latency_ms", latency}, {"mini_timing_window", backend}};
    }
};

// worker 和响应发送都结束后只发布一条终态。校验拒绝请求默认不需要 worker。
class RequestTrace
{
    Metrics &metrics;
    std::mutex mu;           // 保护本对象
    std::string id, failure; // 请求 id、首个失败原因
    bool work_done = true, response_done = false, emitted = false;
    bool delivered = false, stream = false;
    int response_status = 0;
    MetricJson usage = nullptr, mini_timings = nullptr;
    std::optional<MetricClock::time_point> first_mini, mini_end, queued, queue_end;
    std::optional<MetricClock::time_point> started, first_forwarded, response_end;

    // 有则算毫秒，无则返回 JSON null（区分「没发生」与「耗时 0」）
    static MetricJson elapsed(const std::optional<MetricClock::time_point> &to,
                              MetricClock::time_point from)
    {
        return to ? MetricJson(std::chrono::duration<double, std::milli>(*to - from).count())
                  : MetricJson(nullptr);
    }

    // 只有 work_done && response_done 且未发布过，才构造终态行；幂等
    std::optional<MetricJson> publish_locked()
    {
        if (emitted || !work_done || !response_done)
            return {};
        emitted = true;

        // 按优先级推导 state（顺序敏感）
        std::string state = "success";
        if (failure == "invalid_request" || response_status == 400)
            state = "invalid";
        else if (failure == "queue_full")
            state = "rejected";
        else if (failure == "queue_timeout")
            state = "queue_timeout";
        else if (failure == "request_timeout")
            state = "execution_timeout";
        else if (failure == "shutting_down")
            state = "shutdown";
        else if (failure == "client_disconnected" ||
                 failure == "downstream_write_failed")
            state = "cancelled";
        else if (!failure.empty() || response_status >= 400)
            state = "failed";
        else if (!delivered)
            state = "cancelled";

        // queue_ms 三元逻辑：入队后开始执行→到开始；入队但未开始→到 queue_end；未入队→null
        MetricJson queue_duration = nullptr;
        if (queued)
            queue_duration = elapsed(started ? started : queue_end, *queued);

        return MetricJson{
            {"request_id", id}, {"state", state}, {"failure", failure}, {"stream", stream}, {"http_status", response_status}, {"response_written", delivered}, {"queue_ms", queue_duration}, {"ttft_mini_ms", elapsed(first_mini, received)}, {"ttft_forwarded_ms", elapsed(first_forwarded, received)}, {"mini_complete_ms", elapsed(mini_end, received)}, {"gateway_e2e_ms", elapsed(response_end, received)}, {"usage", usage}, {"backend_mode", "mini"}, {"timing_source", "mini/app_wall"}, {"mini_timings", mini_timings}};
    }

public:
    const MetricClock::time_point received = MetricClock::now(); // 请求进入时刻（所有延迟基准）

    // 构造即计入 received
    RequestTrace(Metrics &value, std::string request_id)
        : metrics(value), id(std::move(request_id))
    {
        metrics.begin();
    }

    // 首个 mini 内容时刻（只记一次）
    void mini_content()
    {
        std::lock_guard<std::mutex> lock(mu);
        if (!first_mini)
            first_mini = MetricClock::now();
    }

    // mini 结束时刻（只记一次）+ 后端 stats
    void mini_finished(const MetricJson &value)
    {
        std::lock_guard<std::mutex> lock(mu);
        if (!mini_end)
            mini_end = MetricClock::now();
        mini_timings = value;
    }

    void set_usage(const MetricJson &value)
    {
        std::lock_guard<std::mutex> lock(mu);
        usage = value;
    }

    // 标记需要 worker：work_done 置 false，记录是否流式
    void work_pending(bool streaming)
    {
        std::lock_guard<std::mutex> lock(mu);
        work_done = false;
        stream = streaming;
    }

    void enqueued()
    {
        std::lock_guard<std::mutex> lock(mu);
        queued = MetricClock::now();
    }

    void execution_started()
    {
        std::lock_guard<std::mutex> lock(mu);
        started = MetricClock::now();
    }

    // 首个转发时刻（只记一次）
    void content_forwarded()
    {
        std::lock_guard<std::mutex> lock(mu);
        if (!first_forwarded)
            first_forwarded = MetricClock::now();
    }

    // 只记录首个非空失败原因
    void fail(const std::string &reason)
    {
        if (reason.empty())
            return;
        std::lock_guard<std::mutex> lock(mu);
        if (failure.empty())
            failure = reason;
    }

    // worker 结束：置 work_done、补 queue_end、尝试发布（锁外 record）
    void worker_finished()
    {
        std::optional<MetricJson> row;
        {
            std::lock_guard<std::mutex> lock(mu);
            work_done = true;
            if (!queue_end)
                queue_end = MetricClock::now();
            row = publish_locked();
        }
        if (row)
            metrics.record(std::move(*row));
    }

    // 响应结束：幂等；置 response_done/delivered/status/end，尝试发布（锁外 record）
    void response_finished(bool success, int status)
    {
        std::optional<MetricJson> row;
        {
            std::lock_guard<std::mutex> lock(mu);
            if (response_done)
                return;
            response_done = true;
            delivered = success;
            response_status = status;
            response_end = MetricClock::now();
            row = publish_locked();
        }
        if (row)
            metrics.record(std::move(*row));
    }
};