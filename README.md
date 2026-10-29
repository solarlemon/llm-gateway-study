# C++ 本地推理网关

完成“网络通信与资源管理”“并发调度与过载保护”：HTTP/SSE、参数校验、独立转发 worker、有界 FIFO 等待队列、截止时间及断连清理。推理由独立 llama-server 执行，性能与 AI Infra 实验继续按主文档推进。

## 构建与启动

在 WSL Ubuntu2204 运行：

```bash
cd ~/cpp_code/llm-gateway-study
bash scripts/build.sh
```

依赖复用同级 llama.cpp 中的 cpp-httplib 和 nlohmann/json，无需额外安装 Drogon。可用 `cmake -S . -B build -DLLAMA_DIR=/其他路径/llama.cpp` 指定位置。现有 llama.cpp 来自 ZIP，没有 Git commit；真实模型验证结果保存依赖文件 SHA256。

终端一（保留前台，Ctrl+C 停止）：

```bash
cd ~/cpp_code/llm-gateway-study
bash scripts/start_model.sh
```

默认使用你的 0.5B Q4_K_M 模型、CUDA 卸载、总上下文 2048、2 个 slot。本机后端日志显示单 slot 上下文为 1024。脚本设置 CUDA 动态库路径；可通过 `MODEL_PATH`、`LLAMA_DIR` 和 `UPSTREAM_PORT` 覆盖配置。

终端二：

```bash
cd ~/cpp_code/llm-gateway-study
./build/gateway
```

终端三：

```bash
curl http://127.0.0.1:8081/health
curl http://127.0.0.1:8081/stats
curl http://127.0.0.1:8081/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"你好，请介绍线程池。"}],"max_tokens":64,"stream":false}'
curl -N http://127.0.0.1:8081/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"用三句话介绍线程池。"}],"max_tokens":128,"stream":true}'
```

`/health` 仅检查网关，模型就绪要查上游 `/health`。所有服务默认只绑定本机。

## 请求行为

| 场景 | 行为 |
| --- | --- |
| 参数非法 | 400，不提交任务 |
| 等待队列满/服务关闭 | 503，常规响应附 Retry-After |
| 排队超时 | 504，过期任务不转发 |
| 执行截止时间 | 中断上游；已发流时关闭流 |
| 上游连接失败、截断或非流式体过大 | 502 |
| 上游业务错误，例如 429 | 保留原状态和正文 |
| SSE 中途失败/缺少 `[DONE]` | 追加 error 事件，不能修改已经发送的 HTTP 状态 |
| 客户端断开 | 取消等待任务或中断对应上游 |
| 客户端长时间不读 | 有界缓冲及 2 秒 socket 写超时，随后取消 |
| SIGINT/SIGTERM | 停止准入、取消等待/执行任务、等待线程退出 |

支持文本 messages 子集，role 为 system/user/assistant，content 必须是字符串。max_tokens 默认 128，允许 1–4096。精确上下文/token 校验仍由模型后端完成，不按字符数估算。

SSE 字节透明转发，不把网络块当作 token；增量识别结束标记，单行解析缓存最多 64 KiB。流式 FIFO 默认最多 64 KiB，另有一个发送中的块、HTTP 库和内核缓冲，因此不是总进程内存上限。静默期间每 250 ms 发送 SSE 注释心跳，以发现下游断连。

执行配额由 RAII SlotLease 管理；流式上游完成后仍等待下游发送完成或取消。HTTP handler 返回不代表请求生命周期结束。

## 配置

| 环境变量 | 默认值 | 含义 |
| --- | --- | --- |
| GATEWAY_PORT | 8081 | 网关端口 |
| UPSTREAM_HOST / UPSTREAM_PORT | 127.0.0.1 / 8080 | 上游 |
| MAX_ACTIVE | 2 | 执行槽，含正在发送的流式任务 |
| QUEUE_CAPACITY | 4 | 显式等待队列容量，允许 0 |
| QUEUE_TIMEOUT_MS | 5000 | 排队上限 |
| REQUEST_TIMEOUT_MS | 120000 | 接收完整请求后的任务总预算 |
| UPSTREAM_IDLE_MS | 30000 | 上游读取无数据超时 |
| HTTP_THREADS | 16 | 固定 HTTP 工作线程数 |
| STREAM_BUFFER_BYTES | 65536 | 每个流式任务 FIFO 字节上限 |
| RESPONSE_LIMIT_BYTES | 4194304 | 非流式/错误响应体上限 |

HTTP_THREADS 至少为 MAX_ACTIVE + QUEUE_CAPACITY + 2；为正常负载下的健康检查和拒绝处理留出余量，但不是独立管理通道。

```bash
MAX_ACTIVE=1 QUEUE_CAPACITY=2 QUEUE_TIMEOUT_MS=1000 ./build/gateway
```

## 验证

```bash
python3 tests/integration.py
python3 tests/real_model_smoke.py
```

第一项无需 GPU，使用可控 mock 测试 FIFO、队列满、超时、SSE、慢客户端、取消和退出。第二项自动在独立端口启动真实模型和网关，只清理自己创建的进程。

暂不运行模型时，可用 `python3 mock_backend.py --port 8080` 手动启动同一套可控后端，再运行网关学习请求流转。

结果在 `build/integration-results.json`、`build/real-model-results.json`；日志在同目录。它们证明功能正确性，不代表性能提升。验收记录见 `docs/网络与调度验收.md`。

## 学习顺序

1. `main.cpp`：参数、提交、响应头等待、非流式与 SSE 两条路径、退出。
2. `src/task.hpp`：共享所有权、取消标志、完成和交付阶段。
3. `src/scheduler.hpp`：ready 预留槽、waiting 有界 FIFO、watchdog、SlotLease。
4. `src/upstream.hpp`：HTTP 回调、有界缓冲、SSE 结束检查、上游中断。
5. `tests/integration.py`：屏障控制队列、故障注入、配额恢复。
6. `项目主文档-C++推理网关.md`：后续 Prefill/Decode/KV Cache 实验和简历范围。

## 必须理解的边界

本项目是有容量上限的**阻塞 HTTP 桥接**，不是自研 epoll Reactor 或事件驱动异步服务。HTTP 线程等待/发送，独立 worker 发起上游请求。框架线程和内部任务队列均有限；超过框架容量时可能直接关闭连接，不能保证所有超额连接收到 JSON 503，也不保证任意连接洪峰下健康检查及时返回。

FIFO 指调度顺序，多个 worker 的实际网络到达顺序受线程调度影响。取消上游后 GPU 结束时刻由后端检查粒度决定；本机日志观察到取消与 slot 释放，不代表立即抢占 CUDA 计算。

watchdog 约每 20 ms 检查，网络中断还受连接/写超时影响，不是严格实时终止。SIGTERM 主要覆盖已提交任务；未完整发送请求的连接受 5 秒读取超时约束。

原始代码及说明备份位于 `backup-20261029-192422/`。
