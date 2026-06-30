# Poseidon Multi-GPU Benchmarks

This directory contains optional multi-GPU transfer benchmarks. They are not
part of the default build.

## Quick Run

Use the helper scripts for the normal workflow:

```bash
bench/mgpu/run_memcpy.sh
bench/mgpu/run_memcpy_extended.sh
bench/mgpu/run_nccl.sh
```

Both scripts configure, build, and run the corresponding benchmark. Their
defaults use the benchmark's normal CKKS size:

- `run_memcpy.sh`: source device `0`, destination device `1`, `degree=32768`,
  `iterations=50`, `warmup=5`.
- `run_memcpy_extended.sh`: source device `0`, destination device `1`,
  `degree=65536`, levels `L1..L40`, ciphertext counts `1..32`,
  `iterations=50`, `warmup=5`, with a timestamped CSV log.
- `run_nccl.sh`: devices `0,1,2,3,4,5,6,7`, root device `0`, `degree=32768`,
  `iterations=50`, `warmup=5`.

Override script defaults with environment variables:

```bash
DEVICES=0,1,2,3,4,5,6,7 ITERATIONS=50 WARMUP=5 DEGREE=32768 \
  bench/mgpu/run_nccl.sh

SOURCE_DEVICE=0 DESTINATION_DEVICE=1 ITERATIONS=50 WARMUP=5 DEGREE=32768 \
  bench/mgpu/run_memcpy.sh

SOURCE_DEVICE=0 DESTINATION_DEVICE=1 DEGREE=65536 MIN_LEVEL=1 MAX_LEVEL=40 \
  MIN_COUNT=1 MAX_COUNT=32 bench/mgpu/run_memcpy_extended.sh
```

For a very small smoke run, explicitly override the size:

```bash
DEGREE=1024 ITERATIONS=2 WARMUP=1 bench/mgpu/run_memcpy.sh
DEGREE=1024 MIN_LEVEL=1 MAX_LEVEL=2 MIN_COUNT=1 MAX_COUNT=2 \
  ITERATIONS=2 WARMUP=1 bench/mgpu/run_memcpy_extended.sh
DEVICES=0,1 DEGREE=1024 ITERATIONS=2 WARMUP=1 bench/mgpu/run_nccl.sh
```

Extra command-line arguments are appended to the benchmark invocation:

```bash
MODES=cuda_peer_broadcast,nccl_broadcast bench/mgpu/run_nccl.sh --p-count 1
MODES=object_loop,async_object_loop,contiguous_buffer bench/mgpu/run_memcpy.sh
bench/mgpu/run_memcpy.sh --modes object_loop,copy_objects
```

## Manual Build

Existing CUDA peer CKKS object-copy benchmark:

```bash
cmake -S . -B build-mgpu-bench \
  -DPOSEIDON_BUILD_MGPU=ON \
  -DPOSEIDON_BUILD_MGPU_BENCH=ON \
  -DPOSEIDON_BUILD_MGPU_CUDA_COMM=ON \
  -DPOSEIDON_BUILD_MGPU_GPU_OBJECTS=ON
cmake --build build-mgpu-bench --target poseidon_mgpu_ckks_transfer_bench -j2
```

NCCL comparison benchmark:

```bash
cmake -S . -B build-mgpu-nccl-bench \
  -DPOSEIDON_BUILD_MGPU=ON \
  -DPOSEIDON_BUILD_MGPU_BENCH=ON \
  -DPOSEIDON_BUILD_MGPU_GPU_OBJECTS=ON \
  -DPOSEIDON_BUILD_MGPU_NCCL_COMM=ON
cmake --build build-mgpu-nccl-bench --target poseidon_mgpu_nccl_transfer_bench -j2
```

The default build does not require NCCL. NCCL is linked only when
`POSEIDON_BUILD_MGPU_NCCL_COMM=ON`. The NCCL benchmark also enables
`POSEIDON_BUILD_MGPU_GPU_OBJECTS=ON` because it constructs real GPU CKKS
ciphertext objects and communicates their device fields.

## CUDA Peer CKKS Transfer Bench

```bash
./build-mgpu-bench/bin/poseidon_mgpu_ckks_transfer_bench \
  --source-device 0 --destination-device 1
```

Useful smoke run:

```bash
./build-mgpu-bench/bin/poseidon_mgpu_ckks_transfer_bench \
  --source-device 0 --destination-device 1 \
  --iterations 2 --warmup 1 --degree 1024
```

Supported CUDA peer transfer modes:

- `object_loop`: synchronous loop over `copy_object`.
- `object_loop_e2e`: end-to-end object loop. Each timed iteration materializes
  destination GPU ciphertexts, allocates their destination device buffers,
  creates object-copy requests, then runs a synchronous loop over
  `copy_object`. Source ciphertext construction and deterministic source data
  filling stay outside the timed region.
- `copy_objects`: calls the production batch API. This is a correctness API
  first and currently uses the same object-loop policy.
- `async_object_loop`: experimental P2P path that queues every object copy with
  `cudaMemcpyPeerAsync` on one benchmark-owned stream, then synchronizes that
  stream once at the end of the timed operation. Cross-device runs require CUDA
  peer access; the synchronous modes still keep their host-staging fallback.
- `contiguous_buffer`: raw single-buffer transfer with the same total payload
  size as the ciphertext batch, using `CudaPeerComm::copy_buffer`. This is a
  pure transfer upper bound, not a ciphertext end-to-end path.

Select a subset with `--modes`, for example:

```bash
./build-mgpu-bench/bin/poseidon_mgpu_ckks_transfer_bench \
  --source-device 0 --destination-device 1 \
  --modes object_loop,object_loop_e2e,contiguous_buffer
```

The CUDA peer benchmark defaults to ciphertext counts `1,5,10` and levels
`L4,L8,L12,L16,L20`. Override the case grid with either explicit lists or
closed ranges:

```bash
./build-mgpu-bench/bin/poseidon_mgpu_ckks_transfer_bench \
  --source-device 0 --destination-device 1 \
  --degree 65536 \
  --min-level 1 --max-level 40 \
  --min-count 1 --max-count 32 \
  --modes contiguous_buffer,object_loop,object_loop_e2e \
  --log bench/mgpu/logs/memcpy_extended.csv

./build-mgpu-bench/bin/poseidon_mgpu_ckks_transfer_bench \
  --source-device 0 --destination-device 1 \
  --levels 1,4,8,16,32,40 \
  --counts 1,2,4,8,16,32 \
  --log bench/mgpu/logs/memcpy_subset.jsonl \
  --log-format jsonl
```

`degree=65536` in this benchmark means a synthetic CKKS-shaped transfer
payload. The benchmark allocates `GpuCiphertextData` with the requested
`degree`, `q_count`, component count, and `p_count`; it does not create a full
Poseidon CKKS context or prove arithmetic correctness for that degree. Before
running the case grid, the benchmark preflights the largest selected case and
fails fast if the source/destination allocations do not fit. It does not
silently fall back to a smaller degree.

CSV and JSONL logs include one row per result with timestamp, hostname, visible
CUDA device list, selected source/destination devices, peer-access status,
shape, mode, byte counts, iteration counts, latency statistics, GB/s, and a
`synthetic_shape` marker. Stdout still prints rows as the benchmark progresses.

## NCCL Transfer Comparison Bench

By default, `poseidon_mgpu_nccl_transfer_bench` uses devices
`0,1,2,3,4,5,6,7` and root device `0`.

```bash
./build-mgpu-nccl-bench/bin/poseidon_mgpu_nccl_transfer_bench
```

If fewer than 8 GPUs are visible, override the device list:

```bash
./build-mgpu-nccl-bench/bin/poseidon_mgpu_nccl_transfer_bench \
  --devices 0,1 --iterations 2 --warmup 1 --degree 1024
```

Supported modes:

- `cuda_peer_broadcast`
- `cuda_peer_gather`
- `cuda_peer_sendrecv`
- `nccl_broadcast`
- `nccl_gather`
- `nccl_sendrecv`

Select a subset with `--modes`, for example:

```bash
./build-mgpu-nccl-bench/bin/poseidon_mgpu_nccl_transfer_bench \
  --devices 0,1 \
  --modes cuda_peer_broadcast,nccl_broadcast,cuda_peer_sendrecv,nccl_sendrecv
```

By default, both multi-GPU benchmarks report:

- `ct_bytes`: actual GPU payload bytes in one constructed CKKS ciphertext,
  measured from the object's GPU field allocation.
- `total_bytes`: `ct_bytes * count`, measured from the constructed ciphertext
  batch rather than printed from a standalone formula.
- `xfer_bytes`: effective transfer volume used for GB/s. The CUDA peer CKKS
  transfer benchmark uses `total_bytes`; NCCL broadcast and gather use
  `total_bytes * (rank_count - 1)`; `cuda_peer_sendrecv` and `nccl_sendrecv`
  use `total_bytes`.

These byte counts exclude CPU-side object metadata, shared pointers, vectors,
serialization headers, `parms_id`, and scale values because those are not part
of the GPU communication payload.

Latency is measured with host wall-clock time around the transfer operation
and an explicit synchronization before and after each timed iteration. The
CUDA peer object-copy benchmark therefore includes all work in the selected
mode. The NCCL benchmark uses NCCL stream synchronization for collective
modes. Point-to-point send/recv modes synchronize only the root and target
devices.

Gather pre-fills the root's own slot before timing. The timed gather payload
and `xfer_bytes` therefore count only data received from non-root ranks, which
matches the CUDA peer gather loop and NCCL in-place gather semantics.

Output includes mode, devices, root, ciphertext count, level, `q_count`,
`ct_bytes`, `total_bytes`, `xfer_bytes`, latency statistics, aggregate GB/s,
and `speedup_vs_cuda_peer`. The GB/s value is total timed payload divided by
latency; it is not per-device average bandwidth. NCCL broadcast speedup is
relative to `cuda_peer_broadcast`; NCCL gather speedup is relative to
`cuda_peer_gather`; NCCL send/recv speedup is relative to
`cuda_peer_sendrecv`.

Each mode fills deterministic GPU data before timing and validates sampled
payload words after timing. A single-GPU machine cannot produce real NCCL
multi-GPU performance data.
