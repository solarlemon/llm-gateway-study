#!/usr/bin/env python3
"""Deterministic integration tests. Standard library only; no GPU needed.
Each gateway is owned by this process and shut down in finally/context exit.
"""
import concurrent.futures
import contextlib
import http.client
import json
import os
from pathlib import Path
import select
import socket
import struct
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = Path(__file__).resolve().parents[1]

def eventually(fn, timeout=5):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        try:
            value = fn()
            if value:
                return value
        except (OSError, http.client.HTTPException):
            pass
        time.sleep(0.02)
    raise AssertionError("condition did not become true")

def available_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]

class Backend:
    def __init__(self, port=0):
        self.mu = threading.Lock()
        self.arrivals = []
        self.disconnected = set()
        self.gates = {}
        owner = self
        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"
            def log_message(self, *args):
                pass
            def closed(self):
                ready, _, _ = select.select([self.connection], [], [], 0)
                if not ready:
                    return False
                try:
                    return self.connection.recv(1, socket.MSG_PEEK) == b""
                except OSError:
                    return True
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                spec = body.get("test", {})
                name = spec.get("name", "quick")
                with owner.mu:
                    owner.arrivals.append(name)
                    gate = owner.gates.setdefault(name, threading.Event())
                try:
                    if spec.get("hold"):
                        while not gate.wait(0.01):
                            if self.closed():
                                with owner.mu: owner.disconnected.add(name)
                                return
                    status = spec.get("status", 200)
                    self.send_response(status)
                    self.send_header("Connection", "close")
                    if body.get("stream") and status == 200:
                        self.send_header("Content-Type", "application/json" if spec.get("wrong_type") else "text/event-stream")
                        if spec.get("truncate"):
                            self.send_header("Content-Length", "100000")
                        self.end_headers()
                        if spec.get("silence"):
                            while not self.closed(): time.sleep(0.01)
                            with owner.mu: owner.disconnected.add(name)
                            return
                        if spec.get("flood"):
                            for _ in range(100000):
                                self.wfile.write(b"data: " + b"x" * 8192 + b"\n\n")
                                self.wfile.flush()
                        else:
                            frames = [b": mock comment\n\n", 'data: {"choices":[{"delta":{"content":"你"}}]}\n\n'.encode(),
                                      b'data: {"choices":[{"delta":{"content":"hello"}}]}\n\n', b"data: [DONE]\n\n"]
                            if spec.get("truncate") or spec.get("incomplete"): frames = frames[:2]
                            for frame in frames:
                                # Split across arbitrary byte boundaries, including UTF-8.
                                for offset in range(0, len(frame), 7):
                                    self.wfile.write(frame[offset:offset+7])
                                    self.wfile.flush()
                                time.sleep(0.12)
                    else:
                        payload = json.dumps({"name": name, "padding": "x" * spec.get("bytes", 0)}).encode()
                        self.send_header("Content-Type", "application/json")
                        self.send_header("Content-Length", str(len(payload) + (100 if spec.get("truncate") else 0)))
                        self.end_headers()
                        self.wfile.write(payload)
                        self.wfile.flush()
                except (OSError, ConnectionError):
                    with owner.mu: owner.disconnected.add(name)
                finally:
                    self.close_connection = True
        self.server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.port = self.server.server_port
    def release(self, name):
        with self.mu: self.gates[name].set()
    def seen(self, name):
        with self.mu: return name in self.arrivals
    def close(self):
        with self.mu:
            for gate in self.gates.values(): gate.set()
        self.server.shutdown()
        self.server.server_close()

class Gateway:
    def __init__(self, backend, **settings):
        self.port = available_port()
        env = dict(os.environ, GATEWAY_PORT=str(self.port), UPSTREAM_PORT=str(backend.port),
                   MAX_ACTIVE="1", QUEUE_CAPACITY="2", HTTP_THREADS="12", QUEUE_TIMEOUT_MS="2000",
                   REQUEST_TIMEOUT_MS="8000", STREAM_BUFFER_BYTES="1024", RESPONSE_LIMIT_BYTES="4096")
        env.update({k: str(v) for k, v in settings.items()})
        self.log = open(ROOT / "build" / "integration-gateway.log", "a")
        self.process = subprocess.Popen([str(ROOT / "build" / "gateway")], env=env, stdout=self.log, stderr=self.log)
        try:
            eventually(lambda: self.get("/health")[0] == 200)
        except BaseException:
            self.process.terminate(); self.process.wait(timeout=5); self.log.close(); raise
    def get(self, path):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        try:
            c.request("GET", path)
            r = c.getresponse()
            return r.status, r.read()
        finally: c.close()
    def stats(self): return json.loads(self.get("/stats")[1])
    def body(self, name, **spec):
        return {"messages": [{"role": "user", "content": "hello"}], "max_tokens": 8, "test": dict(name=name, **spec)}
    def post(self, name="quick", **spec):
        return self.raw_post(self.body(name, **spec))
    def raw_post(self, body):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=12)
        try:
            c.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
            r = c.getresponse()
            return r.status, r.read()
        finally: c.close()
    def stream(self, name, **spec):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        body = self.body(name, **spec); body["stream"] = True
        c.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
        return c, c.getresponse()
    def raw_socket(self, name, stream=False, **spec):
        body = self.body(name, **spec); body["stream"] = stream
        payload = json.dumps(body).encode()
        s = socket.create_connection(("127.0.0.1", self.port), timeout=5)
        s.sendall(f"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {len(payload)}\r\n\r\n".encode() + payload)
        return s
    def close(self):
        self.process.terminate()
        try: code = self.process.wait(timeout=6)
        except subprocess.TimeoutExpired:
            self.process.kill(); self.process.wait(); raise AssertionError("gateway shutdown stalled")
        self.log.close()
        assert code == 0, f"gateway exit {code}"
    def __enter__(self): return self
    def __exit__(self, *args): self.close()

def reset_socket(s):
    s.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0)); s.close()

def run():
    backend = Backend()
    passed = []
    def record(name): passed.append(name); print("PASS", name, flush=True)
    try:
        with Gateway(backend) as gateway, concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
            assert gateway.raw_post({})[0] == 400
            assert gateway.raw_post({"messages": [], "stream": "yes"})[0] == 400
            assert gateway.post()[0] == 200
            assert gateway.post("rate_limit", status=429)[0] == 429
            record("validation_nonstream_upstream_status")

            a = pool.submit(gateway.post, "fifo-a", hold=True)
            eventually(lambda: backend.seen("fifo-a"))
            b = pool.submit(gateway.post, "fifo-b", hold=True)
            eventually(lambda: gateway.stats()["queued"] == 1)
            c = pool.submit(gateway.post, "fifo-c", hold=True)
            eventually(lambda: gateway.stats()["queued"] == 2)
            assert gateway.post("overflow")[0] == 503
            assert not backend.seen("overflow")
            backend.release("fifo-a"); assert a.result()[0] == 200
            eventually(lambda: backend.seen("fifo-b")); assert not backend.seen("fifo-c")
            backend.release("fifo-b"); assert b.result()[0] == 200
            eventually(lambda: backend.seen("fifo-c")); backend.release("fifo-c"); assert c.result()[0] == 200
            eventually(lambda: gateway.stats()["active"] == 0)
            record("fifo_queue_capacity_and_503")

            assert gateway.post("oversized", bytes=10000)[0] == 502
            assert gateway.post("truncated", truncate=True)[0] == 502
            assert gateway.post("after-failure")[0] == 200
            record("upstream_failure_and_response_limit_release")

            conn, response = gateway.stream("sse")
            started = time.monotonic(); content_times = []; data = []
            while True:
                line = response.readline()
                if not line: break
                data.append(line)
                if line.startswith(b"data:"): content_times.append(time.monotonic() - started)
            conn.close()
            combined = b"".join(data).decode()
            assert '你' in combined and "hello" in combined and "[DONE]" in combined
            assert content_times[-1] - content_times[0] >= 0.15, content_times
            eventually(lambda: gateway.stats()["active"] == 0)
            record("incremental_sse_fragmented_utf8")

            conn, response = gateway.stream("broken-sse", truncate=True)
            data = response.read(); conn.close()
            assert b"event: error" in data and b"upstream_io_error" in data
            eventually(lambda: gateway.stats()["active"] == 0)
            record("stream_midflight_upstream_error")

            conn, response = gateway.stream("incomplete-sse", incomplete=True)
            data = response.read(); conn.close()
            assert b"upstream_incomplete_stream" in data
            conn, response = gateway.stream("wrong-sse-type", wrong_type=True)
            assert response.status == 502
            assert b"invalid_upstream_content_type" in response.read()
            conn.close()
            record("sse_terminal_marker_and_content_type")

            s = gateway.raw_socket("cancel-running", hold=True)
            eventually(lambda: backend.seen("cancel-running")); reset_socket(s)
            eventually(lambda: "cancel-running" in backend.disconnected)
            eventually(lambda: gateway.stats()["active"] == 0)
            record("nonstream_disconnect_interrupts_upstream")

            a = pool.submit(gateway.post, "queue-holder", hold=True)
            eventually(lambda: backend.seen("queue-holder"))
            s = gateway.raw_socket("cancel-queued")
            eventually(lambda: gateway.stats()["queued"] == 1)
            reset_socket(s)
            eventually(lambda: gateway.stats()["queued"] == 0)
            assert not backend.seen("cancel-queued")
            backend.release("queue-holder"); assert a.result()[0] == 200
            record("queued_disconnect_never_forwarded")

            s = gateway.raw_socket("flood", stream=True, flood=True)
            s.recv(256); reset_socket(s)
            eventually(lambda: gateway.stats()["active"] == 0)
            assert gateway.stats()["peak_stream_buffer_bytes"] <= 1024
            assert gateway.post("after-flood")[0] == 200
            record("stream_disconnect_and_bounded_buffer")

            s = gateway.raw_socket("silent-stream", stream=True, silence=True)
            s.recv(256); reset_socket(s)
            eventually(lambda: "silent-stream" in backend.disconnected)
            eventually(lambda: gateway.stats()["active"] == 0)
            record("silent_stream_disconnect_via_heartbeat")

        with Gateway(backend, QUEUE_TIMEOUT_MS=150) as gateway, concurrent.futures.ThreadPoolExecutor() as pool:
            a = pool.submit(gateway.post, "expiry-holder", hold=True)
            eventually(lambda: backend.seen("expiry-holder"))
            code, payload = gateway.post("expired")
            assert code == 504 and b"queue_timeout" in payload
            assert not backend.seen("expired")
            backend.release("expiry-holder"); assert a.result()[0] == 200
            record("queue_deadline_never_forwarded")

        with Gateway(backend, REQUEST_TIMEOUT_MS=250) as gateway:
            assert gateway.post("deadline", hold=True)[0] == 504
            eventually(lambda: "deadline" in backend.disconnected)
            eventually(lambda: gateway.stats()["active"] == 0)
            record("total_deadline_interrupts_idle_upstream")

        with Gateway(backend, MAX_ACTIVE=2, QUEUE_CAPACITY=0) as gateway:
            a = gateway.raw_socket("two-a", hold=True)
            b = gateway.raw_socket("two-b", hold=True)
            eventually(lambda: backend.seen("two-a") and backend.seen("two-b"))
            assert gateway.stats()["active"] == 2
            assert gateway.post("no-wait")[0] == 503
            reset_socket(a); reset_socket(b)
            eventually(lambda: gateway.stats()["active"] == 0)
            record("two_execution_slots_and_zero_queue")

        with Gateway(backend) as gateway:
            s = gateway.raw_socket("slow-reader", stream=True, flood=True)
            # Do not read: bounded application buffer plus finite socket write timeout.
            eventually(lambda: backend.seen("slow-reader"))
            eventually(lambda: gateway.stats()["active"] == 0, timeout=6)
            assert gateway.stats()["peak_stream_buffer_bytes"] <= 1024
            s.close()
            record("slow_client_write_timeout_and_bounded_buffer")

        # A missing backend must fail promptly and release its slot.
        absent = type("Absent", (), {"port": available_port()})()
        with Gateway(absent) as gateway:
            assert gateway.post()[0] == 502
            eventually(lambda: gateway.stats()["active"] == 0)
            record("connect_failure_release")

        gateway = Gateway(backend)
        active = gateway.raw_socket("shutdown-active", hold=True)
        eventually(lambda: backend.seen("shutdown-active"))
        waiting = gateway.raw_socket("shutdown-waiting")
        eventually(lambda: gateway.stats()["queued"] == 1)
        gateway.close(); active.close(); waiting.close()
        assert not backend.seen("shutdown-waiting")
        record("shutdown_running_and_queued")
    finally:
        backend.close()
    result = {"passed": len(passed), "cases": passed, "note": "correctness tests, not performance benchmark"}
    (ROOT / "build" / "integration-results.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))

if __name__ == "__main__": run()
