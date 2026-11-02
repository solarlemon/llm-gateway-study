"""Owned CPU mini gateway, no upstream model process. JSON result is saved per run."""
import argparse
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
import http.client
import json
import os
from pathlib import Path
import socket
import struct
import subprocess
import time

parser = argparse.ArgumentParser()
parser.add_argument('--build-dir',type=Path,default=Path('build'))
parser.add_argument('--model',type=Path,required=True)
parser.add_argument('--tokenizer',type=Path,required=True)
parser.add_argument('--reference-dir',type=Path,required=True)
args = parser.parse_args()
root = Path(__file__).resolve().parents[2]
build = args.build_dir.resolve()
def eventually(check):
    end = time.monotonic()+10
    while time.monotonic() < end:
        try:
            result = check()
            if result:
                return result
        except (OSError, http.client.HTTPException):
            pass
        time.sleep(.01)
    raise AssertionError('condition did not complete')

class Client:
    def __init__(self, port): self.port = port
    def call(self, path, payload=None):
        connection = http.client.HTTPConnection('127.0.0.1',self.port,timeout=10)
        try:
            connection.request('GET' if payload is None else 'POST',path,
                               None if payload is None else json.dumps(payload),{'Content-Type':'application/json'})
            response = connection.getresponse()
            data = response.read()
            return response.status, data if path == '/health' else json.loads(data)
        finally: connection.close()

@contextmanager
def gateway(**overrides):
    with socket.socket() as available:
        available.bind(('127.0.0.1',0)); port = available.getsockname()[1]
    env = dict(os.environ,BACKEND_MODE='mini',GATEWAY_PORT=str(port),MAX_ACTIVE='2',QUEUE_CAPACITY='2',
               HTTP_THREADS='12',MINI_MODEL_PATH=str(args.model.resolve()),MINI_TOKENIZER_PATH=str(args.tokenizer.resolve()),
               REQUEST_TIMEOUT_MS='10000',REQUEST_LOG_PATH=str(build/'mini-http-requests.jsonl'))
    env.update({name:str(value) for name,value in overrides.items()})
    with (build/'mini-http.log').open('a') as log:
        process = subprocess.Popen([str(build/'gateway')],env=env,stdout=log,stderr=log)
        client = Client(port)
        try:
            eventually(lambda: process.poll() is None and client.call('/health')[0] == 200)
            yield client
        finally:
            process.terminate()
            try: process.wait(timeout=10)
            except subprocess.TimeoutExpired: process.kill(); process.wait()

checks = []
cli = build/'mini_text_cli'
def cli_result(text):
    return json.loads(subprocess.check_output([str(cli),str(args.model),str(args.tokenizer),text,'32'],text=True))
reference = build/'mini_tokenizer_reference'
subprocess.run(['cc','-O0','-I'+str(args.reference_dir.resolve()),str(root/'tests/mini/tokenizer_reference.c'),'-lm','-o',str(reference)],check=True)
for text in ['', 'Once upon a time', 'Hello, world!', '你好，世界！', 'a\n\nb', ' 😀 ']:
    expected = json.loads(subprocess.check_output([str(reference),str(args.tokenizer),text],text=True))
    assert expected == cli_result(text)['prompt_ids']
checks.append('tokenizer_matches_upstream_for_english_utf8_and_whitespace')

prompt = {'messages':[{'role':'user','content':'Once upon a time'}],'max_tokens':32,'temperature':0}
with gateway() as client:
    baseline = cli_result('Once upon a time')
    code, body = client.call('/v1/chat/completions',prompt)
    assert code == 200 and body['choices'][0]['message']['content'] == baseline['text']
    assert body['usage'] == baseline['usage'] and body['mini_timings']['kv_cleared']
    checks.append('real_nonstream_matches_text_cli')
    second = dict(prompt,messages=[{'role':'user','content':'A little cat'}])
    second_expected = cli_result('A little cat')['text']
    with ThreadPoolExecutor(max_workers=2) as pool:
        responses = list(pool.map(lambda request: client.call('/v1/chat/completions',request),[prompt,second]))
    assert [entry[0] for entry in responses] == [200,200]
    assert [entry[1]['choices'][0]['message']['content'] for entry in responses] == [baseline['text'],second_expected]
    checks.append('two_workers_share_weights_with_independent_KV')
    connection = http.client.HTTPConnection('127.0.0.1',client.port,timeout=10)
    connection.request('POST','/v1/chat/completions',json.dumps(dict(prompt,stream=True)),{'Content-Type':'application/json'})
    response = connection.getresponse(); assert response.status == 200
    text = ''; done = False; timings = None; frames = 0
    for line in iter(response.readline,b''):
        if not line.startswith(b'data: '): continue
        value = line[6:].decode().strip()
        if value == '[DONE]': done = True; continue
        event = json.loads(value)
        for choice in event.get('choices',[]):
            piece = choice.get('delta',{}).get('content','')
            if piece: frames += 1; text += piece
        timings = event.get('mini_timings',timings)
    connection.close()
    assert done and text == baseline['text'] and frames > 1 and timings['kv_cleared']
    checks.append('SSE_incremental_text_and_DONE_match_CLI')
    code, stopped = client.call('/v1/chat/completions',dict(prompt,stop=baseline['text'][:3]))
    assert code == 200 and stopped['choices'][0]['message']['content'] == ''
    for invalid in [dict(prompt,temperature=.5),dict(prompt,stop=''),dict(prompt,max_tokens=513),
                    dict(prompt,messages=[{'role':'user','content':' x'*1000}])]:
        assert client.call('/v1/chat/completions',invalid)[0] == 400
    assert client.call('/v1/chat/completions',prompt)[0] == 200
    checks.append('stop_validation_context_limit_and_recovery')
    # 真实生成过程中收到首个 SSE 内容后 RST，检查取消与 worker 槽位回收。
    connection = http.client.HTTPConnection('127.0.0.1',client.port,timeout=10)
    connection.request('POST','/v1/chat/completions',json.dumps(dict(prompt,stream=True,max_tokens=256)),{'Content-Type':'application/json'})
    response = connection.getresponse()
    while True:
        line = response.readline()
        assert line
        if b'"content"' in line: break
    raw = response.fp.raw._sock
    raw.setsockopt(socket.SOL_SOCKET,socket.SO_LINGER,struct.pack('ii',1,0))
    response.close(); connection.close()
    eventually(lambda: client.call('/stats')[1]['active'] == 0)
    rows = eventually(lambda: client.call('/requests')[1])
    assert any(row['state'] == 'cancelled' for row in rows)
    assert client.call('/v1/chat/completions',prompt)[0] == 200
    checks.append('stream_disconnect_releases_slots_and_next_request_recovers')
    eventually(lambda: client.call('/metrics')[1]['inflight'] == 0)
    rows = client.call('/requests')[1]
    assert all(row['backend_mode'] == 'mini' and row['timing_source'] == 'mini/app_wall' for row in rows)
    assert any(row['mini_timings'] and row['mini_timings'].get('kv_cleared') for row in rows)
    checks.append('mini_request_observation_and_terminal_states')
with gateway(REQUEST_TIMEOUT_MS=1) as client:
    code, body = client.call('/v1/chat/completions',dict(prompt,messages=[{'role':'user','content':'Once '*100}],max_tokens=32))
    assert code == 504, (code,body)
    eventually(lambda: client.call('/stats')[1]['active'] == 0)
    checks.append('execution_deadline_cancels_real_CPU_prefill')
(build/'mini-http-results.json').write_text(json.dumps({'passed':len(checks),'checks':checks,'sample':baseline['text']},indent=2))
print(json.dumps({'passed':len(checks),'checks':checks,'sample':baseline['text']},indent=2))
