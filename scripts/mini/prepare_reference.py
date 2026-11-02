"""Download and content-verify the small checkpoint and independent C reference."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

REVISION = '0bd21da7698eaf29a0d7de3992de8a46ef624add'
MODEL_SHA = 'b0a507e7ad0f626624f17112325e66691f9076d622e1d3274d103d00299f2696'
SOURCE_SHA = '9c4f2d5c6ae01b71726d1cc37530d71e60bff0ec7cc012565f16a43c1ca658bd'
SOURCE_URL = 'https://raw.githubusercontent.com/karpathy/llama2.c/master/run.c'
TOKENIZER_SHA = '037cb335abb25d1fa9e8ecae30ed2a3a8ace9302862ebcdc05d51a6bbb10c312'
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=Path, required=True)
parser.add_argument('--tokenizer', type=Path)
parser.add_argument('--reference-dir', type=Path, default=Path('.cache/mini-reference'))
args = parser.parse_args()

def fetch(path, url, expected):
    if path.exists():
        data = path.read_bytes()
    else:
        with urllib.request.urlopen(url, timeout=60) as response:
            data = response.read()
    digest = hashlib.sha256(data).hexdigest()
    if digest != expected:
        raise RuntimeError(f'Content hash mismatch: {path}; expected {expected}, got {digest}')
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        temporary = path.with_suffix(path.suffix+'.partial')
        temporary.write_bytes(data)
        temporary.replace(path)

fetch(args.model,f'https://huggingface.co/karpathy/tinyllamas/resolve/{REVISION}/stories260K/stories260K.bin',MODEL_SHA)
tokenizer = args.tokenizer or args.model.with_name('tok512.bin')
fetch(tokenizer,f'https://huggingface.co/karpathy/tinyllamas/resolve/{REVISION}/stories260K/tok512.bin',TOKENIZER_SHA)
fetch(args.reference_dir/'run.c',SOURCE_URL,SOURCE_SHA)
manifest = {'model_revision':REVISION,'model_sha256':MODEL_SHA,
            'source_url':SOURCE_URL,'reference_sha256':SOURCE_SHA,'tokenizer_sha256':TOKENIZER_SHA}
(args.reference_dir/'manifest.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps(manifest,indent=2))
