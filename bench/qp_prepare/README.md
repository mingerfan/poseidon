# Exact QP preparation regression probe

Link an **already built, source-matched** PoseidonGPU library (no Runtime dependency):

```sh
GMP_ROOT="$D/tools/gmp" CUDA_PATH=/usr/local/cuda \
  bash bench/qp_prepare/build.sh "$D/build" "$D/qp_prepare"
# Local system-spdlog build may need: EXTRA_LIBS='-lspdlog -lfmt'
```

The build script never rebuilds the library or executes the probe. Use the existing
`experiments/gpu_keyload/build_native.sh "$D"` for `native_profile`; optional
post-timing fingerprints use:

```sh
"$D/native_profile" PUBLIC_PRIMES.txt RESULT.json HASH_RESULT.json
python -c 'import json,sys; a,b=map(lambda p:json.load(open(p)),sys.argv[1:]); assert a["public_fingerprint"]==b["public_fingerprint"] and a["public_inventory"]==b["public_inventory"]; print("public constants/metadata fingerprints agree")' reference-hash.json optimized-hash.json
```

`public_fingerprint` hashes all public words plus metadata; it is diagnostic FNV-1a-64,
not collision-free bytewise proof. `key_inventory` never hashes random key payloads
and is excluded from this fingerprint. `validation_s` is outside `native_profile_s`;
do not compare complete timing-bearing JSON files for equality.

```sh
# CPU-only oracle/NTT/period self-check: no CUDA initialization or GPU work.
"$D/qp_prepare" --cpu-only
# GPU runs require explicit execution on an authorized, idle GPU.
POSEIDON_GPU_QP_PREPARE=reference "$D/qp_prepare"
POSEIDON_GPU_QP_PREPARE=optimized POSEIDON_GPU_QP_PREPARE_THREADS=4 "$D/qp_prepare"
# In-process direct comparison: reference, optimized requested 1 and 4 workers.
POSEIDON_GPU_KEYLOAD_TRACE=1 "$D/qp_prepare" --compare > qp.stdout 2> qp.stderr
python3 bench/qp_prepare/check_failure_trace.py qp.stderr
```

24 configurations / 264 integer-polynomial cases: N=1024/4096; actual Q-prefix
levels 1/2/4 of four-prime contexts; 28/30/32-bit and mixed Q/P primes; P=2/3.
32-bit prime-pair products exceed 99% of 2^64. Cases cover zero, signed small coefficients
(scale 1024), positive/negative P-divisible integers, exact full-Q centered
boundaries, coefficients beyond q0*q1/2 forcing Q4 full CRT, and lazy Q-NTT
residues in **[0,2q)** (the reference INTT contract, including HEXL). The mixed
case puts a 32-bit prime before a 30-bit prime to exercise Garner subtraction
when q0 > q1. Additional aliases +/-q0*q1 and a last-coefficient/last-Q-limb-only
alias force full-Q verification and fallback.

GMP centers the input integers modulo full Q and independently reduces into every
Q/P prime before CPU NTT. GPU verification compares every full Q/P word, then
uploads eleven diagonals through a single-matrix compressed group and expands the
actual compact allocation using its bit-reversed indexing (ordinary order for
period=N). Proved oracle periods include 1, 8 and N; owner-offset/period plan
references and metadata are checked too. `--compare` retains exact word snapshots,
not just hashes. It also checks the optimized path with equivalent NTT words in
[2q,4q), against the canonical reference/GMP result, never sending these wider
lazy inputs to the original reference INTT. Exceptions cover non-NTT, unknown parms_id, bad shape, invalid
worker settings, and a later invalid diagonal with other tasks in flight; every
round subsequently performs a complete valid upload to test recovery. Requested
worker counts may be clamped by CPU affinity/cgroup quota; this test does not claim
four physical workers when the host disallows them. The trace checker returns 77
(SKIP), rather than claiming fault-branch coverage, if the effective limit is 1;
otherwise it verifies injected launch failures, serial fallback and later async
slot recovery. The test also runs two concurrent callers with separate nondefault
CUDA streams; it does not assert overlap, worker high-watermark or an aggregate
memory bound. GPU regression restores each caller's stream during teardown.

For an additional real GPU encrypt → Boot → decrypt numerical smoke, build
`bootstrap_smoke.cpp` against a CUDA architecture supported by the target GPU:

```sh
GMP_ROOT="$D/tools/gmp" CUDA_PATH=/usr/local/cuda \
  bash bench/qp_prepare/bootstrap_smoke_build.sh "$D/build" "$D/bootstrap_smoke"
OMP_NUM_THREADS=4 POSEIDON_GPU_QP_PREPARE=reference "$D/bootstrap_smoke"
OMP_NUM_THREADS=4 POSEIDON_GPU_QP_PREPARE=optimized "$D/bootstrap_smoke"
```

The N4096/Q50/P8 `sec_level=none` fixture is deliberately insecure and only a
numerical smoke, not a production parameter recommendation or R20 test. It checks
an uncompressed QP Boot; `qp_prepare --compare` exercises compressed and threaded
preparation separately.

The completed A800 first-start A/B and exact regression results are documented in
`docs/gpu-qp-first-start-fix-20261115.md` (paths relative to the worktree root).
The environment variable `POSEIDON_GPU_QP_TEST_ASYNC_FAILURE=1` is reserved for
this regression's injected OS-thread-creation failure and must remain unset in
normal runs.
