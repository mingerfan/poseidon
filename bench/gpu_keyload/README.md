# Standalone Poseidon GPU key-load microbench

This directory is an independent CMake entry point. It uses
`add_subdirectory(../..)` and links `poseidon::gpu`/`poseidon_shared`, **not**
ckks-runtime, its JSON library, or the legacy benchmark suite. No tests or key
generation run automatically during configure/build.

## Build

Prerequisites: CMake >= 3.26.4, C++/CUDA compiler, CUDA toolkit compatible with
this checkout's RMM/CCCL, GMP C/C++ development files, and the vendored GPU
stack documented in `../../third_party/README.md`. A800 is SM80.

From the repository root:

```sh
cmake -S bench/gpu_keyload -B bench/gpu_keyload/build-omp-on \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=80 \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
  -DPOSEIDON_BUILD_DEPS=OFF -DPOSEIDON_USE_MSGSL=OFF \
  -DPOSEIDON_USE_ZLIB=OFF -DPOSEIDON_USE_ZSTD=OFF \
  -DPOSEIDON_USE_SPDLOG=OFF -DPOSEIDON_USE_OPENMP=ON
cmake --build bench/gpu_keyload/build-omp-on --target poseidon_gpu_keyload -j 4
```

Add `-DGMP_ROOT=/prefix`, `-DCMAKE_CUDA_COMPILER=/cuda/bin/nvcc`, or
`-DCUDAToolkit_ROOT=/cuda` as necessary. CMake 4.x may require the policy-minimum
flag above for older bundled dependencies. No dependency/compression option is
forced: caller-supplied `-DPOSEIDON_...` settings are honored. When root spdlog
discovery is OFF, this harness supplies the already-configured RMM header-only
spdlog/fmt target to the CPU library, whose sources include spdlog unconditionally.
An alternative, matching the server's existing configuration, is:

```sh
# Replace the BUILD_DEPS/SPDLOG settings above with:
-DPOSEIDON_BUILD_DEPS=ON -DPOSEIDON_USE_SPDLOG=ON \
-DFETCHCONTENT_SOURCE_DIR_SPDLOG="$PWD/third_party/spdlog"
# Keep MSGSL, ZLIB, ZSTD OFF to avoid their downloads.
```

For an OpenMP build-option comparison, configure a **separate** build directory
with `-DPOSEIDON_USE_OPENMP=OFF`. Poseidon still globally uses `-fopenmp` and has
some unguarded pragmas (including Galois keygen), so OFF does **not** mean all
key generation is serial. JSON records the build option, `_OPENMP`, runtime max
threads, a separately observed probe-team size, affinity, OMP environment and
cgroup CPU quota. The probe is not a measurement of every internal keygen team.

## Run

```sh
BIN=bench/gpu_keyload/build-omp-on/bin/poseidon_gpu_keyload
# Small smoke command for a GPU machine; not run automatically:
OMP_NUM_THREADS=1 "$BIN" 8 4 2 2 smoke.json --check full --parameter-mode sparse

# Requested large shape; choose rotation_count and OMP threads explicitly:
OMP_NUM_THREADS=7 OMP_DYNAMIC=FALSE \
  "$BIN" 16 50 25 8 q50-p25-k8-omp7.json --bits 28 --check full
```

Positional arguments are exactly `logN q_count p_count rotation_count output.json`;
or use named `--logN`, `--q-count`, `--p-count`, `--rotation-count`, `--json`.
Use zero or five positional arguments; do not mix positional shape arguments
with their named equivalents. Other flags can accompany either form.

Defaults: `logN=16` (N=65536), Q=50, P=25, 16 rotation keys, bits=28,
`gpu-keyload.json`, device 0, parameter mode `none`, check mode `sample`.
`--device N` is a CUDA-visible index. All Q and P primes have `--bits` bits.
Parameters are CKKS, `sec_level_type::none`, hwt=192, logSlots=logN-1,
q0_level=0; steps are unique direct left rotations `1..rotation_count`, without
conjugation or implicit power-of-two keys. JSON includes the exact prime values.
**These are benchmark parameters/keys, not a production security configuration.**
PRNG seeds come from Poseidon's normal system entropy; configurations and
sampling are repeatable, but generated secret material is not bit-identical.

### Parameter full vs sparse: use separate processes

```sh
OMP_NUM_THREADS=7 "$BIN" 16 50 25 8 full.json \
  --check full --parameter-mode full
OMP_NUM_THREADS=7 "$BIN" 16 50 25 8 sparse.json \
  --check full --parameter-mode sparse --parameter-q-counts 1,9,34,50
python3 bench/gpu_keyload/compare_parameters.py full.json sparse.json
```

Modes:
- `none`: skip GpuParameterData entirely (default).
- `full`: use the normal constructor/build path, including the key-context
  level and every q-only chain level.
- `sparse`: build only requested q-only levels. Default selection is 1,9,34,50;
  sort/deduplicate/filter to `1 <= q <= Q`. An empty filtered selection is an
  argument error. Q32 therefore selects 1,9 by default.
- `top`: only the q-only Q level (convenience mode).

`--gpu-parameters` aliases `--parameter-mode`; `all` aliases `full`.
Each GPU stage records before/after `cudaMemGetInfo` and signed
`device_used_delta_bytes` (= free_before - free_after), with queries **outside**
timing. These global-device deltas can be perturbed by other processes and are
not a replacement for owner-allocation byte counts.

Parameter JSON reports level count, total owned bytes, and actual allocation
bytes of all seven root families, summed and per level/shard:
`ntt_tables`, `intt_tables`, `fourstep_intt_tables`, `intt_tables_normalized`,
`montgomery_ntt_tables`, `montgomery_intt_tables`,
`montgomery_intt_tables_normalized`. q-only levels may still carry P parameter
roots; `root_limb_count_including_parameter_p` makes that explicit. Other RNS
constants/fused tables count toward total parameter bytes, not the seven-family
subtotal. Fused-table/cache settings are recorded in the environment.

Root fingerprints are downloaded/computed in a separate, untimed-for-upload
`gpu_parameters_fingerprint` stage. `--check full` hashes every uint32 word;
`sample` hashes first/middle/last 16-word windows. The comparison script requires
**full** fingerprints from both runs and matches the seven families at every
sparse level. This is a 64-bit hash comparison, not a collision-free bytewise
comparison or an independent CPU mathematical oracle.

## What the measurements mean

Each stage prints `begin` and completion/failure with wall milliseconds to
**flushed stderr**. Final JSON is written on success and caught failures;
argument/output-path errors exit before keygen. The output directory must exist.
Exit codes: 0 success, 1 runtime/check failure, 2 CLI/output failure. SIGKILL/OOM
cannot produce a final report; the last flushed stage identifies progress.

Stages: CUDA/context initialization; CPU prime/context construction; ordinary
secret+public keygen; relin generation; Galois generation; relin upload; Galois
upload; mandatory relin/Galois equality checks; optional parameter build/upload;
optional parameter fingerprints. GPU stages drain work before the clock starts
and synchronize before stopping. This is a single cold setup run, not a
steady-state average. Per-stage times exclude teardown; total wall includes it.

**Key upload is NOT pure PCIe DMA:** it includes RMM/cudaMalloc allocation,
CPU uint64-to-uint32 packing, copies and synchronizations. It does not generate
keys, construct GPU parameters, pre-rotate keys, or perform correctness readback.
Parameter timing similarly includes host NTT/RNS table creation and optional
cache I/O, not just transfers. Native inverse-pre-rotation is not separately
exposed by the public uploader and is intentionally not replicated here.

CPU key counts distinguish nonempty keys from sparse Galois slots (often N);
GPU `metadata_key_slot_count` is likewise not an uploaded-key count. CPU byte
counts are coefficient payload/capacity for SK/PK/relin/Galois, excluding
metadata/context/scratch and memory retained in CPU pools. GPU key counts are
live owning fields plus any device pointer tables; host views do not count.
Normal uploads here use standard, full-Q keys only, not per-level copies.

Checks are outside upload timing and cannot be disabled. They validate every
key/digit/component mapping, Q/P shape, shard bounds and full nonoverlapping
logical coverage. Sample mode reads first/middle/last plus one fixed mixed
coefficient of **every Q/P limb of every shard**. Full mode compares every
uploaded uint32 residue to its CPU uint64 source with <=1 MiB readback scratch.
Field offsets, coefficient ranges and CPU P offsets are honored; pointer tables
are not mistaken for coefficient data. No rotate/relin mathematical operation
check is performed.

Limits: logN=8..17; `1 <= P <= Q`, `Q+P <= 256`, bits=2..31 (the prime finder
can still reject shapes with insufficient NTT primes); `rotation_count < N/2`.
Default Q50/P25 uses two HYBRID digits: one key has about 150 MiB CPU payload /
75 MiB GPU payload. Default 16 rotations therefore means about 2.49 GiB CPU and
1.25 GiB GPU evaluation-key payload including relin, before context/scratch and
parameters. Full parameter tables can consume multiple additional GiB.

## Local validation performed

Configured and fully compiled the standalone executable with CMake 4.4.5,
GCC 16.2.1, CUDA 13.4, SM80, bundled RMM, and dependency/compression discovery
OFF. `--help`, invalid arguments and no-visible-device error JSON were checked;
no CPU keys were generated, no GPU benchmark ran, and no remote was accessed.
Successful upload/correctness and full/sparse runtime results remain for the
GPU host. Source provenance includes configure-time HEAD and SHA256(main.cpp).
