#!/usr/bin/env bash
set -euo pipefail
export PATH="/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
llama_dir="${LLAMA_DIR:-$HOME/cpp_code/llama.cpp}"
model_path="${MODEL_PATH:-$HOME/ModelWeights/Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q4_k_m.gguf}"
exec "$llama_dir/build/bin/llama-server" -m "$model_path" -ngl 99 -c 2048 -np 2 --host 127.0.0.1 --port "${UPSTREAM_PORT:-8080}" "$@"
