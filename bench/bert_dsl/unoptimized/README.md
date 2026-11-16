# Original-prefix baseline (scoped implementation)

`generate_raw.py` reuses **read-only** `../model.py`, `../backend.py`, and the independent dense oracle from `../generate.py`. The main workload is fixed to T128/H768/F3072/12×64/one complete post-LN encoder block and byte-identical shared weights/input. `--tiny` is CPU implementation testing only.

This is not an absolutely optimization-free algorithm: original rectangular BSGS, within-matmul baby-rotation reuse, model-local expression reuse, zero-rotation elision, and original intra-CT head/key packing remain part of the original DSL. This round's extra arithmetic optimizations are disabled. `RawGraph.node` shares public constants only; `RawGraph.binary` never CSEs ciphertext arithmetic or folds nested masks. Plain-left add/mul operand ordering is required by the physical CP ABI, not operation elimination.

`RawLowering` uses `strategy='stage', fuse=False`, evicts alignment/Boot reuse before each request, and retains real canonical-scale MulCP(1)/Rescale, MulCC/ReLin/Rescale, native L16/scale30 → canonical L15 normalization, essential pre-Boot plaintext1-scale16 lift, and necessary low-budget guard Boots. No decrypt/re-encrypt. Public content/encoding deduplication, host-only PT setup cache and last-use release are storage/lifetime infrastructure, not removal of ciphertext computation.

Example CPU generation (large graph: **use remote CPU**, not the 30GiB local machine):

```sh
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /opt/conda/bin/python generate_raw.py \
  --source-directory source --assets assets --output-dir raw-prefix128
./verify-plan raw-prefix128/model.runtime-plan.json raw-prefix128/operator-spec.json
/opt/conda/bin/python audit_plan.py raw-prefix128
```

`assets` must contain shared weights/input and `native-build/profile-scale30.json` from `test-results/bert-a800-20261114/`; the generator asserts all shared SHA256 bindings. Remote read-only source snapshots and all outputs must live under `/root/private_data/bert-dsl-20261115/unoptimized/`. The standalone `verify-plan` links existing runtime-reader/verifier static libraries; it does not rebuild or alter the established native runner.

`audit_plan.py` validates the full physical plan on CPU: every MulCP is immediately followed by Rescale (or essential native Boot lift), every MulCC by ReLin/Rescale, every Rescale has actual-prime-correct scale, every Boot uses a real unity plaintext at scale2^16, full dimensions, native Boot count, payload liveness, and artifact hash. OperatorSpec cost entries are explicit **placeholders**, never empirical timing predictions.

`run_window.py` launches only the existing runner, managed allocator/pool78000MiB, binary rotation keys, OMP/BLAS4, CPUs32–35, resident plaintext on/GPU/host-only, 4GiB bundle budget and 4GiB native workspace. It enforces an absolute deadline by signaling only its own process group. Default zero warmups/one iteration is explicitly **cold**. Host cache admission must be selected from the full CPU audit; it is not a GPU-forward admission test.

Results and exact limitations are in `test-results/bert-a800-20261115/unoptimized/`. Do not interpret a partial run, time-to-resource-failure, isolated Boot, or FFN timing as a complete baseline measurement.
