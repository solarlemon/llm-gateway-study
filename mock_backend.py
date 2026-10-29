#!/usr/bin/env python3
"""Manual mock backend, shared with the deterministic integration suite."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / "tests"))
from integration import Backend

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    backend = Backend(args.port)
    print(f"Mock backend on 127.0.0.1:{backend.port}; Ctrl+C to stop", flush=True)
    try:
        backend.thread.join()
    except KeyboardInterrupt:
        pass
    finally:
        backend.close()
