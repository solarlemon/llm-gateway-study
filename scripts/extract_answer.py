#!/usr/bin/env python3
"""Extract choice 0 text from chat SSE; put the merged answer on log line 1.

Existing log: python3 scripts/extract_answer.py llm-answer.log
Live stream: curl -N ... | python3 scripts/extract_answer.py --output llm-answer.log
The HTTP/SSE protocol is unchanged. Original SSE events follow the summary.
"""
import argparse
import io
import json
import os
from pathlib import Path
import sys
import tempfile

MARKER = "--- 原始 SSE ---\n"

class Extractor:
    def __init__(self, emit=lambda text: None):
        self.emit = emit
        self.parts = []
        self.data = []
        self.event = "message"
        self.done = False
        self.errors = []

    def dispatch(self):
        data = "\n".join(self.data)
        event = self.event
        self.data = []; self.event = "message"
        if not data or self.done:
            return
        if data.strip() == "[DONE]":
            self.done = True
            return
        try:
            payload = json.loads(data)
            if not isinstance(payload, dict):
                raise ValueError("SSE data is not a JSON object")
            if event == "error" or "error" in payload:
                self.errors.append(json.dumps(payload, ensure_ascii=False))
                return
            choices = payload.get("choices", [])
            if not isinstance(choices, list):
                raise ValueError("choices is not an array")
            for choice in choices:
                if not isinstance(choice, dict) or choice.get("index", 0) != 0:
                    continue
                delta = choice.get("delta", {})
                content = delta.get("content") if isinstance(delta, dict) else None
                if isinstance(content, str):
                    self.parts.append(content)
                    self.emit(content)
        except (ValueError, TypeError) as exc:
            self.errors.append(str(exc))

    def feed(self, line):
        line = line.rstrip("\r\n")
        if not line:
            self.dispatch()
        elif not line.startswith(":"):
            field, sep, value = line.partition(":")
            if value.startswith(" "):
                value = value[1:]
            if field == "data" and sep:
                self.data.append(value)
            elif field == "event" and sep:
                self.event = value

    def finish(self):
        if self.data:
            self.dispatch()
        if not self.done:
            self.errors.append("未收到 [DONE]，回答可能不完整")
        return "".join(self.parts)

def format_log(answer, raw, errors):
    # Keep the first physical line intact even when the model emits newlines.
    summary = json.dumps(answer, ensure_ascii=False)[1:-1]
    status = "完成" if not errors else "异常或不完整：" + json.dumps(errors, ensure_ascii=False)
    return "回答：" + summary + "\n状态：" + status + "\n\n" + MARKER + raw

def atomic_write(path, text):
    path = Path(path)
    # Same-directory replacement: an interruption cannot leave a half-written log.
    fd, temp = tempfile.mkstemp(prefix=".answer-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as file:
            file.write(text)
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", help="已有 SSE 日志；默认原地整理")
    parser.add_argument("--output", help="输出日志路径；管道输入时必填")
    args = parser.parse_args()
    if not args.input and not args.output:
        parser.error("管道模式需要 --output")
    target = args.output or args.input
    def emit(text):
        sys.stdout.write(text); sys.stdout.flush()
    extractor = Extractor(emit)
    raw = []
    if args.input:
        content = Path(args.input).read_text(encoding="utf-8")
        if content.startswith("回答：") and MARKER in content:
            content = content.split(MARKER, 1)[1]
        source = io.StringIO(content)
    else:
        source = sys.stdin
    for line in source:
        raw.append(line)
        extractor.feed(line)
    answer = extractor.finish()
    atomic_write(target, format_log(answer, "".join(raw), extractor.errors))
    print()
    print("日志已保存：" + str(target), file=sys.stderr)
    if extractor.errors:
        print("；".join(extractor.errors), file=sys.stderr)
        return 1
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
