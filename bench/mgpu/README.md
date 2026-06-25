# Poseidon Multi-GPU Benchmarks

This directory contains optional multi-GPU transfer benchmarks. They are not
part of the default build.

## Quick Run

Use the helper scripts for the normal workflow:

```bash
bench/mgpu/run_memcpy.sh
bench/mgpu/run_nccl.sh
```

Both scripts configure, build, and run the corresponding benchmark. Their
defaults use the benchmark's normal CKKS size:

- `run_memcpy.sh`: source device `0`, destination device `1`, `degree=32768`,
  `iterations=50`, `warmup=5`.
- `run_nccl.sh`: devices `0,1,2,3,4,5,6,7`, root device `0`, `degree=32768`,
  `iterations=50`, `warmup=5`.

Override script defaults with environment variables:

```bash
DEVICES=0,1,2,3,4,5,6,7 ITERATIONS=50 WARMUP=5 DEGREE=32768 \
  bench/mgpu/run_nccl.sh

SOURCE_DEVICE=0 DESTINATION_DEVICE=1 ITERATIONS=50 WARMUP=5 DEGREE=32768 \
  bench/mgpu/run_memcpy.sh
```

For a very small smoke run, explicitly override the size:

```bash
DEGREE=1024 ITERATIONS=2 WARMUP=1 bench/mgpu/run_memcpy.sh
DEVICES=0,1 DEGREE=1024 ITERATIONS=2 WARMUP=1 bench/mgpu/run_nccl.sh
```

Extra command-line arguments are appended to the benchmark invocation:

```bash
MODES=cuda_peer_broadcast,nccl_broadcast bench/mgpu/run_nccl.sh --p-count 1
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
- `nccl_broadcast`
- `nccl_gather`
- `nccl_sendrecv`

Select a subset with `--modes`, for example:

```bash
./build-mgpu-nccl-bench/bin/poseidon_mgpu_nccl_transfer_bench \
  --devices 0,1 \
  --modes cuda_peer_broadcast,nccl_broadcast,nccl_sendrecv
```

The benchmark runs ciphertext counts `1,5,10` across levels
`L4,L8,L12,L16,L20`. Both multi-GPU benchmarks report:

- `ct_bytes`: actual GPU payload bytes in one constructed CKKS ciphertext,
  measured from the object's GPU field allocation.
- `total_bytes`: `ct_bytes * count`, measured from the constructed ciphertext
  batch rather than printed from a standalone formula.
- `xfer_bytes` in the NCCL benchmark: effective transfer volume used for GB/s.
  Broadcast and gather use `total_bytes * (rank_count - 1)`;
  `nccl_sendrecv` uses `total_bytes`.

These byte counts exclude CPU-side object metadata, shared pointers, vectors,
serialization headers, `parms_id`, and scale values because those are not part
of the GPU communication payload.

Output includes mode, devices, root, ciphertext count, level, `q_count`,
`ct_bytes`, `total_bytes`, latency statistics, GB/s, and
`speedup_vs_cuda_peer`. NCCL broadcast speedup is relative to
`cuda_peer_broadcast`; NCCL gather speedup is relative to `cuda_peer_gather`.
`nccl_sendrecv` reports `n/a` because it is point-to-point and does not share
the same broadcast/gather baseline.

Each mode fills deterministic GPU data before timing and validates sampled
payload words after timing. A single-GPU machine cannot produce real NCCL
multi-GPU performance data.
