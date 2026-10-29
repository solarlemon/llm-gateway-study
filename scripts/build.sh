#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cmake -S "$project_dir" -B "$project_dir/build" -DCMAKE_BUILD_TYPE=Debug
cmake --build "$project_dir/build" -j 4
