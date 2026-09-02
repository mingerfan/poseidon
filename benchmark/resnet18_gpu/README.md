# Poseidon GPU ResNet18

This benchmark implements the Trident ImageNet ResNet18 polynomial network on
one NVIDIA GPU. It uses the Trident `2/2/2/2` BasicBlock topology, pretrained
weights and CHW inputs. CPU work is limited to setup, plaintext encoding,
weight/input loading, key generation, optional reference diagnostics and final
decryption. Every operation on encrypted model data uses Poseidon's
`GpuEvaluator`.

The implementation reuses the CKKS, bootstrap, polynomial ReLU and multiplexed
tensor GPU layer from `benchmark/resnet50_gpu`. ResNet18 has its own 40/45-bit
logical scale profile in this directory.

## Build and checks

```bash
./benchmark/resnet18_gpu/run.sh --topology-check
./benchmark/resnet18_gpu/run.sh --weights-check
CUDA_VISIBLE_DEVICES=0 ./benchmark/resnet18_gpu/run.sh --smoke
```

## Encrypted inference

Run image 0 through all eight BasicBlocks:

```bash
CUDA_VISIBLE_DEVICES=0 POSEIDON_NTT_ALGO=fourstep \
  ./benchmark/resnet18_gpu/run.sh --infer 0
```

The optional final argument limits execution to a prefix of BasicBlocks.
`--infer 0 0` validates the encrypted stem, while `--infer 0 1` validates the
first complete BasicBlock. To validate every encrypted intermediate and the
final logits against the plaintext polynomial network, run:

```bash
CUDA_VISIBLE_DEVICES=0 POSEIDON_NTT_ALGO=fourstep \
POSEIDON_GPU_RESNET18_VALIDATE_BLOCKS=1 \
  ./benchmark/resnet18_gpu/run.sh --infer 0
```

The production `7x7x512` global-pooling and `512x1000` FC head can be checked
independently with `--head-check 0`.

## GPU-only timing

For a prefix whose expanded CKKS model plaintexts fit GPU memory, run an
untimed preparation pass followed by an identical transfer-free replay:

```bash
CUDA_VISIBLE_DEVICES=2 POSEIDON_NTT_ALGO=fourstep \
  ./benchmark/resnet18_gpu/run.sh --gpu-only 0 0
CUDA_VISIBLE_DEVICES=2 POSEIDON_NTT_ALGO=fourstep \
  ./benchmark/resnet18_gpu/run.sh --gpu-only 0 1
```

`gpu_only_preloaded_elapsed_seconds` starts after inputs, evaluation keys,
bootstrap matrices and encoded operands are resident on the device, and ends
after the encrypted FC output (or requested prefix) is synchronized. Final
D2H, decryption and prediction checks are outside the interval.

Expanded NTT plaintexts for all eight blocks do not fit simultaneously on a
32 GiB V100: one fully cached block already reaches about 30 GiB. For the full
network use layer-wise prepare/replay timing:

```bash
CUDA_VISIBLE_DEVICES=2 POSEIDON_NTT_ALGO=fourstep \
  ./benchmark/resnet18_gpu/run.sh --gpu-staged 0
```

Each operator is prepared outside its measured interval, replayed with no CPU
encoding or H2D operand transfer, then its expanded plaintexts are released
before the next operator. `staged_gpu_only_elapsed_seconds` is the sum of those
GPU-only replay intervals. It is intentionally different from uninterrupted
whole-network wall time, but is the executable transfer-free measurement for a
model whose expanded HE operands exceed device capacity.

## Logs

Every invocation creates one timestamped file containing build output, runtime
diagnostics, final results, total elapsed time and exit status:

```text
benchmark/resnet18_gpu/output/resnet18_gpu_YYYYMMDD_HHMMSS.log
```

Override the output directory with `POSEIDON_GPU_RESNET18_OUTPUT_DIR`, or the
complete filename with `POSEIDON_GPU_RESNET18_LOG_FILE`.

## Network and parameters

- input: ImageNet `3x224x224`, divided by Trident's boundary `20`;
- stem: encrypted `7x7 stride-2 Conv+BN`, polynomial ReLU, `3x3` average pool;
- stages: `2/2/2/2` BasicBlocks with widths `64/128/256/512`;
- projection: `1x1 stride-2 Conv+BN` at layer2/3/4 block0;
- nonlinear path: 16 reusable GPU bootstraps plus polynomial `[15,15,27]` ReLU;
- head: encrypted global average pool and `512x1000` fully connected layer;
- CKKS: `N=65536`, 32768 slots, Q36/P18, `dnum=2`;
- application operations: logical scale `2^40`;
- bootstrap: Q34 prefix, EvalMod scale `2^45`, output scale `2^40`;
- one ciphertext multiplication level consumes two physical 32-bit Q primes;
- the 14-level `[15,15,27]` ReLU consumes 28 physical Q primes.

Weights and input data default to `Trident/resnet18`. Set
`POSEIDON_TRIDENT_RESNET18_ROOT` to override that directory.

## Current V100 validation

On physical GPU 2, the new profile produced:

- scalar smoke error: `2.74e-8`;
- encrypted stem Conv+BN error: `9.33e-7`;
- stem ReLU+pool error: `1.74e-7`;
- first BasicBlock output error: `8.73e-5`;
- bootstrap output: `Q36`, `log2(scale)=40`;
- stem-only fully preloaded GPU time: `11.84 s`;
- stem plus one block fully preloaded GPU time: `43.36 s`;
- stem plus one block layer-wise GPU time: `36.35 s`;
- stem through the first stride-2 projection block (three BasicBlocks)
  layer-wise GPU time: `147.56 s`.

The three-block staged run also verifies cache eviction/reuse across blocks and
the `64 -> 128` stride-2 main branch plus its `1x1` projection shortcut.

The previous Q53/2^51 full validation is retained below as the baseline:

Image 0 was executed through all eight BasicBlocks with
`POSEIDON_GPU_RESNET18_VALIDATE_BLOCKS=1` on a 32 GiB V100:

- completed BasicBlocks: `8/8`;
- completed reusable GPU bootstraps: `16/16`;
- encrypted stem Conv+BN maximum error: `5.31e-10`;
- layer4.1 final activation maximum error: `3.25e-6`;
- 1000-logit maximum error: `4.31e-5`;
- plaintext/GPU/true label: `65 / 65 / 65`;
- end-to-end elapsed time: `3352.961 s` (`55 min 52.961 s`).

Trident's independent `resnet18_plain 0 0` reference also predicts label `65`
with the same average-pool stem.

The corresponding timestamped log is
`output/resnet18_gpu_20260818_232659.log`.
