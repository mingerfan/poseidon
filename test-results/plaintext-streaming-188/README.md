# Plaintext streaming on 188Server

This directory records implementation and validation of RuntimePlan V3,
local Fence, lazy bundle reads, CPU/GPU worker batches and compiler budgeting.
Large generated IR/plans/bundles and Nsight databases stay outside Git.

- DaCapo: 8/8 tests, including a 20,000-operation scheduling case. A further
  regression checks that explicit Fence selects V3 and survives CSE/DCE.
- ckks-runtime: 5/5 executables. The Release suite includes two 20,000-transfer
  runs, each with 625 fences, deferred outputs and per-batch reclamation checks.
- Poseidon: 19/19 GPU API tests, including online uploads, online Encode,
  GPU0/GPU3 shared weights across batches, retained ciphertext and CPU-worker Boot.
- MLP: eight combinations of one/four GPUs, immediate/prefetched preparation,
  sequential/per-device workers. All agree with eager CKKS within 1.55e-6,
  with unchanged level/scale. Each run executes 121 encodes, 100 blob reads and
  five fences. `compile_mlp.py`, `run_mlp.sh`, `run_memory.sh` reproduce the cases.
- Three measured iterations per combination return active GPU allocations to
  the warmup baseline and pinned transfer staging to zero. Reports separate
  active allocation increments, baseline, pool reservation and CUDA free bytes.
  The warmup baseline combines keys and parameter caches; individual key and
  workspace categories are not separately instrumented. Host RSS is in `.time`.
- `run_trace.sh` and `analyze_trace.py` measure three online NVTX ranges with
  Nsight. Neither schedule shows same-GPU H2D/kernel overlap on this small model.
  There is no demonstrated prefetch speedup. The existing readiness waits remain.

The generic memory runner uses deterministic one-hot inputs. Its finite/repeat
checks are distinct from the MLP fixture oracle. `multi-io.log` covers multiple
inputs and outputs for the full-model smoke path.

Server model artifacts and raw traces:
`/home/xuming/poseidon-plaintext-streaming-models`.
Full Qwen24 artifacts:
`/home/xuming/poseidon-qwen-dacapo-20261010-IUOMH1/artifacts/qwen24-plaintext-streaming`.
The full compiler experiment uses the original native bytecode, not diagnostic
MLIR, with the b60fb1d compiler. The later explicit-Fence export regression does
not change the streaming pass output. `run_full_compile.py` records the command,
limits and RSS. `run_full_smoke.py` waits for export completion, then attempts
one sequential local four-GPU execution with a 600-second wall limit and a
384-GiB address-space limit. It is not a complete-model numerical oracle.

`*.streaming.json` bounds batch-start live RNS objects plus new allocations and
configured cumulative workspace reservations. It excludes key/parameter storage,
unmeasured temporary workspace, encoder workspace, pool reservation and
fragmentation; it must not be read as a total GPU memory guarantee. Full compile,
actual device fit and complete numerical success are reported separately.

The complete Qwen24 compile/export succeeded in 1,248.74 seconds with 93.26 GiB
peak RSS; see `qwen24-compiler-report.json`. The generated V3 JSON is
8,756,201,335 bytes. Its `.streaming.json` estimates 22,353 batches and
1,325,109 encodes. This full experiment predates the extra overflow guards and
`distinct_encode_definitions`/`reencodes` counters added in DaCapo `e473cfb`.
Those follow-up changes passed `compiler-final-tests.log` without repeating the
full-model compilation.

The full runtime smoke reached its 600-second wall limit and was terminated by
the supervisor (603.84 seconds including cleanup, exit -15). Peak Host RSS was
15.01 GiB; all four device-memory samples remained zero and no output artifact
was produced. See `qwen24-runtime-smoke-report.json` and
`qwen24-runtime-samples.jsonl`. This does not establish GPU fit or full-model
numerical correctness, and is not evidence of a CUDA OOM. The unchanged strict
JSON callback parser scans the parent array on every object end, a separate
startup scalability issue; no function-level profile was captured in this run.
