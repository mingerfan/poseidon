# Poseidon GPU ResNet50

This target is the native single-GPU ResNet50 implementation for the
`gs/feat-application` work. CPU code is restricted to setup,
encoding, key/matrix generation, weight loading, and final decrypt/validation.
All homomorphic tensor operations execute through `poseidon::gpu::GpuEvaluator`.

The implementation uses the Trident ImageNet ResNet50 topology (3/4/6/3
bottleneck blocks), a 32768-slot CKKS packing at N=65536, a `2^40`
application scale, and a higher `2^55` GPU EvalMod bootstrap scale. Bootstrap
returns directly to the `2^40` application scale. The stem restores the standard
3x3/stride-2/pad-1 maximum pooling required by the checkpoint. The final global
average pooling remains part of standard ResNet50.

Weights and input data default to `data/resnet50` in this benchmark
directory. Set `POSEIDON_TRIDENT_RESNET50_ROOT` to override that directory.

Build and run:

```bash
CUDA_VISIBLE_DEVICES=2 ./benchmark/resnet50_gpu/run.sh
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
CUDA_VISIBLE_DEVICES=2 ./benchmark/resnet50_gpu/run.sh --smoke
```

Run a full encrypted ImageNet inference (image 0):

```bash
CUDA_VISIBLE_DEVICES=2 ./benchmark/resnet50_gpu/run.sh --infer 0
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
CUDA_VISIBLE_DEVICES=2 ./benchmark/resnet50_gpu/run.sh --gpu-only 0 0
```

For the full 16-block network use the bounded-memory layer-wise mode:

```bash
CUDA_VISIBLE_DEVICES=2 ./benchmark/resnet50_gpu/run.sh --gpu-staged 0
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
CUDA_VISIBLE_DEVICES=2 ./benchmark/resnet50_gpu/run.sh --head-check 0
```

The native50 precision profile uses `Q57/P29/dnum=2`, scale-55 C2S/EvalMod,
scale-45 S2C, and the original `q0/32` preparation. The degree-61 internal
approximation uses ratio 512 and S2C attenuation by 16; coefficient folding
restores that gain and compensates the actual integer preparation multiplier.
One additional physical rescale follows the final double-angle step. Surplus
Q limbs are dropped to return actual `Q31/scale40`, preserving the Q6/scale40
bootstrap input contract. Application moduli, weights and `[15,15,27]` neural
ReLU remain unchanged. The user authorized restoring standard stem maxpool on
2026-10-19 after CPU controls established the average-pool/checkpoint mismatch.

Maximum pooling uses eight comparisons per window. Its separate comparator
sharpens the existing sign approximation with four cubic refinements; it does
not replace the network's neural ReLU. Intermediate sign values undergo real
scale normalization before bootstrap. Invalid border candidates are masked
out, matching negative-infinity padding rather than introducing padded zeros.
The CPU candidate predicts 65 for image 0; its pool approximation differs from
exact maximum pooling by at most 0.0001282959 on the tested stem input.

The maxpool binary passed GPU acceptance on 2026-10-19: smoke, the original
bootstrap gate, the real stem, prefix2, and the full 16-block network, with
per-stage validation enabled. Bootstrap maximum error was 8.28058e-7 <= 1e-6
at actual Q31/scale40. All 167 full-network stage checks passed. Full-image
CPU/GPU maximum logit error was 0.00014363; CPU prediction, GPU prediction,
and the true label were all 65. GPU stem error against exact maximum pooling
was 0.000128233 <= the additional 2e-4 gate; error against the matching
polynomial CPU pooling reference was 6.91304e-6.

Full inference made 351 network bootstrap calls plus 64 pool-comparison and
64 sign-refresh calls (479 total); the standalone bootstrap test used one
separate call. Peak sampled device usage was 23354 MiB; full-process host
max RSS was 19026320 KiB. Full elapsed time was 12028.96 seconds, including
setup and reference validation, not GPU-only timing. This is a single-image
result, not test-set accuracy or whole-network ciphertext bitwise equality.
See [the maxpool validation record](validation/20261019-maxpool-native-result.json)
for audited binary/source origins, log hashes, parameters and resources.

Only this executable defines `POSEIDON_GPU_NATIVE50_PRECISION_BOOTSTRAP`.
Without that definition, shared runtime/config/uploader/kernel source selects
the original implementation, including the profile used by native ResNet18.
The target also links a centered CPU ModDown override for reference validation;
it does not change the shared CPU library. This is a native50 fix, not a general
P9/fourstep validation or a ResNet50 DSL result.

The security parameters remain N=65536, sparse secret Hamming weight 192 and
`sec_level_type::none`. Extending the bootstrap chain changes log2(Q) from
1544.216 to 1768.069 and log2(QP) from 2343.954 to 2695.707. No unchanged or
certified security-bit level is claimed. Device and host memory must be
measured independently of raw key-size estimates.

The earlier average-pool binary passed numerical acceptance on 2026-10-18;
those results below do not validate the new maxpool binary. Its sequence was
`--smoke`, `--bootstrap-smoke` (maximum error <= `1e-6`, Q31/scale40), then
`--infer 0 2` and `--infer 0`, both with
`POSEIDON_GPU_RESNET50_VALIDATE_BLOCKS=1`. A failure stops later acceptance
stages. That earlier average-pool binary passed all stages, including the original
bootstrap gate at `8.65125e-7` with actual Q31/scale40 and all 166 full-network
per-stage validation points. Full image-0 maximum logit error was `0.000193206`;
CPU/GPU predictions were both `62`, while the dataset label was `65` (miss).
This single-image result is not test-set accuracy. Full inference used 351
bootstrap pack calls, separate from the standalone bootstrap gate.

See [the validation record](validation/20261018-native-result.json) for binary
and source hashes, original gates, parameters, log hashes and resources.
Peak sampled device memory was 23354 MiB; full-process host max RSS was
18998968 KiB. Full elapsed time (11640.49 seconds) includes setup, diagnostics
and reference validation and is not a GPU-only timing measurement.
The eight tested same-ciphertext CPU/GPU rotation paths were bitwise equal;
whole-network numerical agreement is established by the original error gates,
not a whole-network ciphertext bitwise comparison.

Implemented and numerically tested on V100:

- CKKS encode/encrypt/upload and download/decrypt boundaries;
- GPU plaintext multiplication, rescale, add-plain and level-aligned add;
- GPU square, HYBRID relinearization and logical two-prime rescale;
- GPU Galois rotation and power-of-two composed rotations;
- Trident-compatible multiplexed CHW packing;
- encrypted im2col 7x7 stem and multiplexed GPU Conv2d+BN;
- GPU batch normalization, residual addition, average/global pooling and FC;
- the Trident `[15,15,27]` polynomial ReLU;
- reusable GPU double-hoist S2C-first bootstrap;
- Trident ImageNet weight/input loading and the full 3/4/6/3 runner.

Historical Q36/40-bit measurements on physical GPU 2 (not current acceptance):

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
