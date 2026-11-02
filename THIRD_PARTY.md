# mini backend reference and attribution

The checkpoint format and Transformer forward mathematics in src/mini/model.cpp
were adapted with reference to Andrej Karpathy's llama2.c, MIT licensed.
The original copyright and license are preserved in tests/mini/third_party/llama2.c-LICENSE.
The test adapter tests/mini/reference_driver.c includes the downloaded original run.c.
The production mini_core target does not link this reference implementation.

Upstream source: https://github.com/karpathy/llama2.c

Reference source SHA256: 9c4f2d5c6ae01b71726d1cc37530d71e60bff0ec7cc012565f16a43c1ca658bd

Checkpoint: https://huggingface.co/karpathy/tinyllamas/tree/0bd21da7698eaf29a0d7de3992de8a46ef624add/stories260K

The model repository declares the MIT license. Weights are downloaded separately
into ModelWeights; scripts verify their SHA256. The reference source URL uses master,
but downloads must match the recorded content hash. Changed upstream content is
rejected. Existing verified files can be reused offline.

Numerical validation currently uses upstream C FP32 forward and fresh-prefix
recomputation. PyTorch reference validation is pending.

The BPE and byte fallback conventions in src/mini/tokenizer.cpp reference the same MIT source. tok512.bin is taken from the same pinned model revision; its hash is verified by prepare_reference.py.

## Standalone HTTP and JSON dependencies

third_party/cpp-httplib contains the previously used cpp-httplib 0.62.0 split source,
MIT licensed by Yuji Hirose. Its original LICENSE is included.
Upstream: https://github.com/yhirose/cpp-httplib

third_party/nlohmann contains JSON 3.12.0, MIT licensed by Niels Lohmann.
The version-matched LICENSE.MIT is included.
Upstream: https://github.com/nlohmann/json

third_party/manifest.json records the vendored source hashes. The CMake build uses
these local copies. The numerical reference described above is an independent
small-model test program with its own license, not a running model service.
