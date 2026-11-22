# Experimental handwritten LLaMA-3.1-8B

This benchmark runs the handwritten decoder with the Poseidon GPU API. It does
not require the DSL runtime. The build is opt-in (`POSEIDON_BUILD_LLAMA_MANUAL`
is OFF by default). From the repository root, for SM80:

```sh
cmake -S . -B build-llama -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=80 -DPOSEIDON_BUILD_GPU=ON \
  -DPOSEIDON_BUILD_LLAMA_MANUAL=ON
cmake --build build-llama --target llama31_poseidon_manual llama31_poseidon_dual --parallel 2
```

The single-GPU runner accepts a pinned input manifest, a modulus fixture,
model weights and index, embeddings, and a reference output. Verify the
payloads independently before running; the runner checks the manifest
*declarations* against the expected pins in `source_manual_io.hpp`. Only the
public modulus fixture is included here. Use an unused report path.

```text
llama31_poseidon_manual --run[-v2] PLAN FIXTURE WEIGHTS INDEX EMBEDDINGS REFERENCE LAYERS VALID_TOKENS MANIFEST REPORT
llama31_poseidon_dual --gpus 2 --plan manual-fast-v1 --fixture FIXTURE --weights WEIGHT_DIR --index INDEX --embeddings EMBEDDINGS --reference REFERENCE --report REPORT
```

`--run` accepts the V1 single-layer payload. `--run-v2` accepts the V2 weight
shards for a one- or 32-layer run. The retained numerical/performance evidence
for this optimization is for **Block 0**, not acceptance of a 32-layer run.
Historical artifacts and host-specific launch scripts remain on
`perf/llama-manual-node2-20261116` (`f0c4cfbc`).

The LLaMA precision paths pass their quotient scale floor explicitly when
constructing EvalMod plans. They do not change the BERT precision profile or
set a process-wide EvalMod floor.
