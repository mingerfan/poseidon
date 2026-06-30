# Multi-GPU P2P Transfer Optimization And Benchmark Plan

> Status: OUTDATED.
>
> This document is historical planning material. The main CUDA peer-access
> caching, async P2P mode, and memcpy benchmark extension work has already been
> implemented, and the final implementation intentionally differs from parts of
> this plan. Use the current source and benchmark docs as the source of truth:
>
> - `src/poseidon/mgpu/comm/cuda_peer_comm.*`
> - `bench/mgpu/ckks_transfer_bench.cpp`
> - `bench/mgpu/run_memcpy_extended.sh`
> - `bench/mgpu/README.md`
>
> Important differences from this plan:
>
> - `split_buffer` was not added because it only isolates raw multi-copy
>   overhead and does not answer the desired end-to-end object-copy latency
>   question.
> - The extended memcpy benchmark now includes `object_loop_e2e`, which times
>   destination materialization/allocation plus the `copy_object` loop.
> - Extended memcpy logging and sweep CLI support were added to the existing
>   CKKS transfer benchmark and `run_memcpy_extended.sh`.
> - The extended NCCL benchmark plan below has not been implemented as part of
>   the current work.
> - `copy_objects` remains present for existing comparison, but extended memcpy
>   defaults do not depend on it.

## Scope

This plan covers the next focused work for single-node multi-GPU point-to-point
transfer performance. It stays inside the mgpu communication and benchmark
layers. It must not add ad hoc copies to evaluators or operator handlers.

Primary targets:

- `src/poseidon/mgpu/comm/cuda_peer_comm.*`
- `bench/mgpu/ckks_transfer_bench.cpp`
- `bench/mgpu/nccl_transfer_bench.cpp`
- `bench/mgpu/README.md`
- optional helper scripts under `bench/mgpu/`

## 1. Cache CUDA Peer Access State

Problem:

- `CudaPeerComm::copy_buffer` currently checks peer access and attempts peer
  enablement on each object copy.
- For `count > 1`, this fixed host-side CUDA runtime overhead is paid once per
  ciphertext, which distorts small-message and multi-object benchmark results.

Plan:

1. Add a small peer-access cache inside `CudaPeerComm`.
2. Cache per `(destination_device, source_device)`:
   - whether peer access is supported;
   - whether peer access has already been enabled by this backend instance.
3. Keep same-device copies on the direct device-to-device path.
4. Keep unsupported peer pairs on the host-staging fallback path.
5. Add tests or a CUDA smoke path that confirms repeated copies do not require
   repeated peer enablement.

Acceptance criteria:

- Repeated `copy_object` calls for the same device pair do not call
  `cudaDeviceEnablePeerAccess` after the first successful enable.
- Existing host-staging fallback behavior is preserved when peer access is not
  available.
- Existing `CudaPeerComm::can_access_peer` public behavior is unchanged.

## 2. Keep `copy_objects` As A Correctness API

Problem:

- Batch object copy should not carry unproven aggregation logic by default.
- Current measurements do not show enough benefit to justify extra temporary
  buffers, extra local device-to-device copies, benchmark modes, and API
  surface.
- `copy_objects` is primarily a correctness API; optimization strategy should
  stay simple until a measured bottleneck justifies more complexity.

Plan:

1. Let `CudaPeerComm` use the base `GpuObjectCopyBackend::copy_objects`
   object-loop implementation.
2. Remove experimental aggregate-buffer object-copy code from the CUDA peer
   backend.
3. Keep the memcpy benchmark comparison limited to:
   - object loop;
   - production `copy_objects`.
4. Document that `copy_objects` is a correctness API first.
5. Reintroduce optimized batch strategy selection only after benchmark data
   shows a clear bottleneck and the added complexity is justified.

Acceptance criteria:

- Production batch object copies no longer regress badly versus the equivalent
  loop of `copy_object` for small and medium ciphertexts.
- The CUDA peer backend exposes no aggregate-buffer object-copy helper.
- The benchmark exposes only the production `copy_objects` policy and the
  explicit object loop for comparison.

## 3. Add Async P2P Submission Path

Problem:

- Current point-to-point loops use synchronous copy calls and synchronize around
  the whole operation.
- Multi-object copies pay repeated host submission overhead and may not expose
  enough queueing to the copy engine.

Plan:

1. Add an experimental async memcpy path using `cudaMemcpyPeerAsync`.
2. Use one explicitly owned stream per relevant device pair or benchmark case.
3. Submit all ciphertext copies, then synchronize once at the end of the timed
   operation.
4. Keep synchronous `copy_object` behavior as the default API until the async
   path is proven stable.
5. Add benchmark modes that isolate:
   - synchronous object loop;
   - async object loop with one final sync;
   - contiguous single-buffer transfer.

Acceptance criteria:

- Async mode validates copied payloads.
- Benchmark output clearly distinguishes sync and async modes.
- No evaluator/runtime code starts depending on the experimental async path
  without a separate integration decision.

## Further Benchmark Plan

Add two extended transfer benchmarks or benchmark modes:

- extended memcpy transfer benchmark;
- extended NCCL transfer benchmark.

Both benchmarks should use the same case grid:

- degree starts at `65536`;
- levels `L1` through `L40`;
- ciphertext counts `1` through `32`;
- default component count remains `2` unless overridden;
- `p_count` remains configurable.

Required output:

1. Print every result row to stdout as the run progresses.
2. Write every result row to a log file.
3. Prefer CSV or JSONL for machine parsing.
4. Include enough metadata in the log header or sidecar fields:
   - timestamp;
   - hostname if available;
   - visible device list;
   - selected source, destination, root, and target devices;
   - CUDA peer access result for the tested pair;
   - NCCL version if available;
   - degree, level, `q_count`, ciphertext count, component count, and `p_count`;
   - ciphertext bytes, total bytes, effective transfer bytes;
   - mode name;
   - warmup and timed iteration counts;
   - average, min, and max latency in milliseconds;
   - computed GB/s.

Memcpy benchmark modes:

- raw contiguous `cudaMemcpyPeer` transfer;
- raw split transfer with `count` separate chunks;
- current object `copy_object` loop;
- default `copy_objects` policy;
- async P2P loop, after step 3.

NCCL benchmark modes:

- `nccl_sendrecv` with one ciphertext per send/recv call;
- `nccl_sendrecv` using an aggregated contiguous buffer where possible;
- existing broadcast and gather modes for reference;
- CUDA peer sendrecv baseline in the same executable for direct comparison.

Implementation notes:

- Add CLI options rather than hard-coding the extended sweep:
  - `--degree 65536`;
  - `--min-level 1`;
  - `--max-level 40`;
  - `--min-count 1`;
  - `--max-count 32`;
  - `--log <path>`;
  - `--append-log`;
  - `--log-format csv|jsonl`.
- Keep the current smaller benchmark defaults or scripts for fast smoke tests.
- Add new scripts for the large 8-GPU run, for example:
  - `bench/mgpu/run_memcpy_extended.sh`;
  - `bench/mgpu/run_nccl_extended.sh`.
- The extended scripts should create a timestamped log path by default and echo
  it before launching the benchmark.
- Do not require NCCL for the memcpy-only build.

## Suggested Execution Order

1. Implement peer-access caching.
2. Keep `copy_objects` as the object-loop correctness API.
3. Extend memcpy benchmark modes and logging.
4. Add async P2P benchmark mode.
5. Extend NCCL benchmark modes and logging.
6. Run smoke cases on a small degree.
7. Run full 8-GPU extended sweeps:
   - degree `65536`;
   - levels `1..40`;
   - counts `1..32`;
   - stdout plus log capture.
8. Analyze per-message overhead separately from large contiguous bandwidth.

## Open Questions

- Whether the extended memcpy and NCCL sweeps should be separate binaries or
  modes inside the existing benchmark binaries.
- Whether the large run should test only one source/target pair or all ordered
  device pairs in the 8-GPU machine.
- Whether logs should be CSV only, JSONL only, or both.
- Whether benchmark output should include percentile latency once iteration
  counts are high enough.
