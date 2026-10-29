#!/usr/bin/env python3
"""Start only owned processes on temporary ports; exercise the actual CUDA model."""
import json
import hashlib
import os
from pathlib import Path
import subprocess
import time
import urllib.request
from integration import ROOT, Gateway, available_port, eventually, reset_socket

def run():
    def gpu_memory():
        return subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True).strip()
    idle_memory = gpu_memory()
    model_port = available_port()
    env = dict(os.environ, UPSTREAM_PORT=str(model_port))
    log_path = ROOT / "build" / "real-model-server.log"
    with log_path.open("w") as log:
        backend = subprocess.Popen(["bash", str(ROOT / "scripts" / "start_model.sh"), "--verbose"], env=env, stdout=log, stderr=log)
        def ready():
            if backend.poll() is not None:
                raise RuntimeError("model startup failed; inspect " + str(log_path))
            with urllib.request.urlopen(f"http://127.0.0.1:{model_port}/health", timeout=1) as response:
                return response.status == 200
        try:
            eventually(ready, timeout=60)
            loaded_memory = gpu_memory()
            target = type("Target", (), {"port": model_port})()
            with Gateway(target, MAX_ACTIVE=2, REQUEST_TIMEOUT_MS=30000) as gateway:
                payload = {"messages": [{"role": "user", "content": "Say hello in one sentence."}], "max_tokens": 32, "stream": False}
                status, body = gateway.raw_post(payload)
                result = json.loads(body)
                assert status == 200, result
                assert result["choices"][0]["message"]["content"]
                payload["stream"] = True
                import http.client
                conn = http.client.HTTPConnection("127.0.0.1", gateway.port, timeout=30)
                conn.request("POST", "/v1/chat/completions", json.dumps(payload), {"Content-Type": "application/json"})
                response = conn.getresponse()
                assert response.status == 200
                frames = []
                while True:
                    line = response.readline()
                    if not line: break
                    if line.startswith(b"data: "):
                        frames.append(line.decode().strip())
                conn.close()
                assert "data: [DONE]" in frames
                assert any('"content"' in frame for frame in frames)
                eventually(lambda: gateway.stats()["active"] == 0)
                # Gateway cancellation only: model-side cancellation is inspected in backend log.
                socket = gateway.raw_socket("real-cancel", stream=True)
                socket.recv(1024); reset_socket(socket)
                eventually(lambda: gateway.stats()["active"] == 0)
                summary = {"nonstream_status": status, "usage": result.get("usage"),
                           "stream_data_frames": len(frames), "stream_done": True,
                           "gateway_cancel_recovered": True, "gateway_stats": gateway.stats(),
                           "model": "Qwen2.5-0.5B-Instruct Q4_K_M", "backend_log": str(log_path),
                           "gpu_memory_mib_before": idle_memory, "gpu_memory_mib_model_loaded": loaded_memory}
                # The existing llama.cpp directory is a ZIP snapshot, not a Git checkout.
                summary["dependency_sha256"] = {name: hashlib.sha256((ROOT.parent / "llama.cpp" / name).read_bytes()).hexdigest()
                                                for name in ["vendor/cpp-httplib/httplib.h", "vendor/cpp-httplib/httplib.cpp", "vendor/nlohmann/json.hpp"]}
                (ROOT / "build" / "real-model-results.json").write_text(json.dumps(summary, indent=2))
                print(json.dumps(summary, indent=2))
        finally:
            backend.terminate()
            try: backend.wait(timeout=10)
            except subprocess.TimeoutExpired:
                backend.kill(); backend.wait()

if __name__ == "__main__": run()
