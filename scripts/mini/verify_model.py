"""Offline real-checkpoint comparison against a content-pinned upstream C forward."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True, type=Path)
parser.add_argument('--reference-dir', required=True, type=Path)
parser.add_argument('--build-dir', default='build', type=Path)
args = parser.parse_args()
root = Path(__file__).resolve().parents[2]
build = args.build_dir.resolve()
reference = args.reference_dir.resolve()
manifest = json.loads((reference/'manifest.json').read_text(encoding='utf-8-sig'))
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
assert sha(args.model) == manifest['model_sha256']
assert sha(reference/'run.c') == manifest['reference_sha256']
driver = build/'mini_reference'
subprocess.run(['cc','-O0','-I'+str(reference),str(root/'tests/mini/reference_driver.c'),'-lm','-o',str(driver)],check=True)
cli = build/'mini_cli'
def run(path, ids, count=None):
    command = [str(path),str(args.model.resolve()),','.join(map(str,ids))]
    if count is not None:
        command.append(str(count))
    return json.loads(subprocess.check_output(command,text=True))
rows = []
cases = [[1], [1,12,47,123,5], [1]+[(i*37+11)%512 for i in range(15)], [1]+[(i*17+9)%512 for i in range(62)]]
# 独立参考检查所有 logits，同时检查固定前缀后逐次反馈的语义。
cases += [[1,12,47,123,5]+list(range(30,30+n)) for n in range(1,9)]
for ids in cases:
    expected = run(driver,ids)
    actual = run(cli,ids,0)['prefill_logits']
    errors = [abs(a-b) for a,b in zip(actual,expected)]
    assert len(actual) == len(expected) == 512
    assert all(e <= 1e-4+1e-4*abs(b) for e,b in zip(errors,expected)), (len(ids),max(errors))
    assert max(range(512),key=actual.__getitem__) == max(range(512),key=expected.__getitem__)
    rows.append({'input_tokens':len(ids),'max_absolute_error':max(errors)})
# 对比真实 greedy 生成序列，包含 EOS 及上限语义。
prompt = [1,12,47,123,5]
expected_output = []
for _ in range(8):
    logits = run(driver,prompt+expected_output)
    token = max(range(len(logits)),key=logits.__getitem__)
    if token == 2:
        break
    expected_output.append(token)
generated = run(cli,prompt,8)
assert generated['output_ids'] == expected_output
one = run(cli,prompt,1)
assert one['valid_len'] == len(prompt)
with tempfile.TemporaryDirectory(dir=build) as temporary:
    bad = Path(temporary)/'truncated.bin'
    bad.write_bytes(args.model.read_bytes()[:128])
    assert subprocess.run([str(cli),str(bad),'1'],capture_output=True).returncode != 0
report = {'reference':'upstream llama2.c C forward; FP32','manifest':manifest,'cases':rows,
          'greedy_output_ids':expected_output,'generation_valid_len':generated['valid_len']}
(build/'mini-model-results.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
