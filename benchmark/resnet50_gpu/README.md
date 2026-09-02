# Poseidon GPU ResNet50

This target is the native single-GPU ResNet50 implementation for the
`lfy-bootstrap-high-precision` branch. CPU code is restricted to setup,
encoding, key/matrix generation, weight loading, and final decrypt/validation.
All homomorphic tensor operations execute through `poseidon::gpu::GpuEvaluator`.

The implementation uses the Trident ImageNet ResNet50 topology (3/4/6/3
bottleneck blocks), a 32768-slot CKKS packing at N=65536, a `2^40`
application scale, and a higher `2^45` GPU EvalMod bootstrap scale. Bootstrap
returns directly to the `2^40` application scale.

Build and run:

```bash
CUDA_VISIBLE_DEVICES=0 ./benchmark/resnet50_gpu/run.sh
```

`run.sh` creates one log per invocation under `output/`. The filename contains
the local start time; build output, runtime diagnostics, the final result,
total inference time, and process status are all stored in that single file:

```text
benchmark/resnet50_gpu/output/resnet50_gpu_YYYYMMDD_HHMMSS.log
```

Override the directory with `POSEIDON_GPU_RESNET50_OUTPUT_DIR=/path/to/output`,
or the complete path with `POSEIDON_GPU_RESNET50_LOG_FILE=/path/to/file.log`.
Every `--infer` run also emits `inference_total_elapsed_ms`, seconds, hours,
and a human-readable `inference_total_elapsed_hms` value. The timer starts
before weight/input loading and includes CKKS setup, key/bootstrap
initialization, encrypted inference, validation, and final decryption.

Run the encrypted GPU correctness smoke suite:

```bash
CUDA_VISIBLE_DEVICES=0 POSEIDON_NTT_ALGO=fourstep \
  ./benchmark/resnet50_gpu/run.sh --smoke
```

Run a full encrypted ImageNet inference (image 0):

```bash
CUDA_VISIBLE_DEVICES=0 POSEIDON_NTT_ALGO=fourstep \
  ./benchmark/resnet50_gpu/run.sh --infer 0
```

The optional final argument limits execution to a prefix of bottleneck blocks.
`--infer 0 0` validates the real stem against a CPU reference and
`--infer 0 1` performs full per-stage validation of the first bottleneck.
Set `POSEIDON_GPU_RESNET50_VALIDATE_BLOCKS=1` to compare every encrypted
intermediate and the final 1000 logits with the plaintext polynomial network.

## GPU-only timing

For a short prefix whose expanded CKKS plaintexts fit device memory, first
prepare the complete input/model cache and then time an identical replay:

```bash
CUDA_VISIBLE_DEVICES=2 POSEIDON_NTT_ALGO=fourstep \
  ./benchmark/resnet50_gpu/run.sh --gpu-only 0 0
```

For the full 16-block network use the bounded-memory layer-wise mode:

```bash
CUDA_VISIBLE_DEVICES=2 POSEIDON_NTT_ALGO=fourstep \
  ./benchmark/resnet50_gpu/run.sh --gpu-staged 0
```

Each operator is prepared outside its measured interval. The identical replay
uses GPU-resident encoded operands, keys, bootstrap matrices and live
activation; its expanded plaintext cache is released before preparing the next
operator. `staged_gpu_only_elapsed_seconds` is the sum of these synchronized,
transfer-free replay intervals. It is not an uninterrupted whole-network wall
time, because all ResNet50 expanded NTT operands cannot coexist on a 32 GiB
V100.

Validate the production `k=16` global-pooling/FC head independently:

```bash
CUDA_VISIBLE_DEVICES=0 POSEIDON_NTT_ALGO=fourstep \
  ./benchmark/resnet50_gpu/run.sh --head-check 0
```

The physical modulus representation is intentionally different from the CPU
chain because GPU residues are `uint32_t`. The optimized application profile
is `Q36/P18/dnum=2`: q0 uses two physical primes, the verified bootstrap prefix
ends at Q34, and two application primes extend refreshed results to Q36. One
logical application level consumes two physical 32-bit Q primes. The
14-level `[15,15,27]` ReLU therefore consumes 28 physical primes and returns at
Q8. The logical application and EvalMod scales remain `2^40` and `2^45`.

Implemented and numerically tested on V100:

- CKKS encode/encrypt/upload and download/decrypt boundaries;
- GPU plaintext multiplication, rescale, add-plain and level-aligned add;
- GPU square, HYBRID relinearization and logical two-prime rescale;
- GPU Galois rotation and power-of-two composed rotations;
- Trident-compatible multiplexed CHW packing;
- encrypted im2col 7x7 stem and multiplexed GPU Conv2d+BN;
- GPU batch normalization, residual addition, average/global pooling and FC;
- the Trident `[15,15,27]` polynomial ReLU;
- reusable Q34 GPU bootstrap with Q36 application-chain restoration;
- Trident ImageNet weight/input loading and the full 3/4/6/3 runner.

Current Q36/40-bit profile validated on physical GPU 2:

- comprehensive CKKS/tensor/pooling/FC smoke: passed;
- real image-0 stem Conv+BN maximum error: below `1.1e-5`;
- real image-0 stem ReLU+pool maximum error: `2.11e-6`;
- real first-bottleneck output maximum error: below `1.2e-4`;
- first-bottleneck output: `Q8`, `log2(scale)=40`;
- fully preloaded stem GPU-only time: `24.07 s`;
- layer-wise stem plus first Bottleneck GPU-only time: `189.41 s`.

The old Q53/51-bit full-network validation baseline was:

- all-slot bootstrap error: `4.32e-8`;
- real image-0 stem error: below `9e-9`;
- real first-bottleneck output error: `3.54e-7`;
- full image-0 last-block output error: `2.85e-6`;
- full image-0 1000-logit maximum error: `4.75e-5`;
- image-0 plaintext/GPU prediction: `62 / 62` (match; dataset label is `65`).

The remaining first-block hotspots are the `64 -> 256` 1x1 expansion and its
projection shortcut. Their measured GPU-only times are about `63.5 s` and
`50.5 s`; further material speedup requires a packed diagonal/batched-channel
1x1 implementation rather than additional CPU/GPU transfer removal.

CPU activity is limited to setup, plaintext packing/encoding, key generation,
weight loading, reference diagnostics, and final decryption. Every operation
on encrypted model data is dispatched through the GPU runtime.
