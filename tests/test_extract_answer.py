import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "extract_answer.py"
spec = importlib.util.spec_from_file_location("extract_answer", SCRIPT)
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)

class ExtractTests(unittest.TestCase):
    def parse(self, raw):
        extractor = module.Extractor()
        for line in raw.splitlines(keepends=True): extractor.feed(line)
        return extractor.finish(), extractor
    def frame(self, content):
        return "data: " + json.dumps({"choices": [{"index": 0, "delta": {"content": content}}]}, ensure_ascii=False) + "\n\n"
    def test_chinese_null_comments_and_done(self):
        raw = ": heartbeat\r\n\r\n" + self.frame(None) + self.frame("线程") + self.frame("池。") + "data: [DONE]\n\n"
        answer, parser = self.parse(raw)
        self.assertEqual(answer, "线程池。"); self.assertFalse(parser.errors)
    def test_newline_answer_stays_on_first_line(self):
        raw = self.frame("第一行\n第二行") + "data: [DONE]\n\n"
        answer, parser = self.parse(raw)
        output = module.format_log(answer, raw, parser.errors)
        self.assertEqual(output.splitlines()[0], "回答：第一行\\n第二行")
        self.assertTrue(output.endswith(raw))
    def test_incomplete_and_error(self):
        _, parser = self.parse(self.frame("部分回答") + 'event: error\ndata: {"error":"broken"}\n\n')
        self.assertEqual(len(parser.errors), 2)
    def test_multiline_data_and_other_choice(self):
        raw = 'data: {"choices":\ndata: [{"index":1,"delta":{"content":"skip"}}, {"index":0,"delta":{"content":"ok"}}]}\n\ndata: [DONE]\n\n'
        answer, parser = self.parse(raw)
        self.assertEqual(answer, "ok"); self.assertFalse(parser.errors)
    def test_file_and_pipeline_modes(self):
        raw = self.frame("你好") + "data: [DONE]\n\n"
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "answer.log"
            result = subprocess.run([sys.executable, str(SCRIPT), "--output", str(target)], input=raw, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout, "你好\n")
            first = target.read_text()
            result = subprocess.run([sys.executable, str(SCRIPT), str(target)], text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(target.read_text(), first)

if __name__ == "__main__": unittest.main()
