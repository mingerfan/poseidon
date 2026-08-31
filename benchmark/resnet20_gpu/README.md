# Poseidon GPU ResNet20

中文运行步骤、参数说明和常见问题见 [README_RUN.md](README_RUN.md)。

This benchmark ports Trident's encrypted CIFAR-10 ResNet20 inference to one
NVIDIA GPU. CPU work is limited to setup, parameter/input loading, plaintext
encoding, key generation, optional reference diagnostics and final decryption;
encrypted model operations use Poseidon's `GpuEvaluator`.

The implementation matches Trident's network semantics:

- input: CIFAR-10 `3x32x32`, divided by boundary `40`;
- stem: `3x3` Conv+BN and polynomial ReLU;
- stages: `3/3/3` BasicBlocks with widths `16/32/64`;
- stage transitions: stride-2 convolution plus Option-A shortcut (even-pixel
  selection and symmetric zero-channel padding);
- nonlinear path: 18 GPU bootstraps plus polynomial `[15,15,27]` ReLU;
- head: encrypted global average pool and `64x10` fully connected layer.

The packed tensor layout uses the largest power-of-two page count that fits the
CKKS slots. CIFAR tensors have a 1024-slot page, so all stages use one
ciphertext pack instead of inheriting ImageNet's fixed two-page layout. Spatial
convolution rotations are cached once per input pack and kernel position. The
unused CIFAR pages are filled with 2/4/8 Trident-style input replicas, allowing
one compact plaintext product to evaluate several output channels at once.
Power-of-two page reductions use a logarithmic rotation tree. Encoded
convolution selectors and repeated scalar plaintexts are cached in device
memory by modulus level, and the encrypted-one input used by polynomial ReLU
is encrypted/uploaded once and cloned on the GPU. Batch-normalization scales
are folded into convolution weights, so cached selectors are reusable across
layers with the same packed shape.

Convolution, Option-A downsampling, average pooling and global average pooling
use lazy rescaling: plaintext products at the same level are accumulated by
Poseidon's fused GPU `multiply_plain_accumulate` kernel, and the completed
stage is rescaled once. The second, preloaded pass reuses the encoded operands
from device memory, so these fused loops do not encode or upload plaintexts.
The logical level schedule is unchanged. For example, a regular layer-1
replicated convolution reduces plaintext-product rescale calls from 88
(`8 groups x 9 kernels + 16 selectors`) to 9 (`8 groups + 1 output pack`).

## CKKS modulus and scale profile

The GPU implementation follows the current CPU logical profile: application
operations use scale `2^40`, EvalMod inside bootstrap uses `2^45`, and each
bootstrap returns directly at `2^40`. Since the CUDA RNS backend stores each
residue in `uint32`, CPU 40/45-bit logical primes are represented by multiple
physical primes no larger than 32 bits.

The full application chain has Q36 (1132 total bits), P18 with 32-bit P primes,
and `dnum=2`. Bootstrap raises only to the verified Q34 prefix and basis-extends
the refreshed result to Q36. A ciphertext multiplication removes one physical
prime and then uses one scale-correction prime, so the 14-level `[15,15,27]`
ReLU consumes 28 physical Q primes while preserving scale `2^40`. The scale and
bootstrap-prefix settings are fields in `gpu_config.h`; bootstrap runtime code
does not hard-code Q34 or a native output scale.

## Build and checks

Run directly from this directory:

```bash
cd benchmark/resnet20_gpu
./run.sh --topology-check
./run.sh --weights-check
CUDA_VISIBLE_DEVICES=0 ./run.sh --smoke
CUDA_VISIBLE_DEVICES=0 ./run.sh --shortcut-check
```

## Encrypted inference

```bash
CUDA_VISIBLE_DEVICES=0 POSEIDON_NTT_ALGO=fourstep \
  ./run.sh --infer 0
```

To measure execution after the input ciphertext, encoded model operands,
evaluation keys and bootstrap constants are already resident on the GPU, run:

```bash
CUDA_VISIBLE_DEVICES=2 POSEIDON_NTT_ALGO=fourstep \
  ./run.sh --gpu-only 0
```

At startup this generates one direct Galois key for each of the 235 rotations
required by the fixed complete ResNet20 topology. All keys are generated and
uploaded before the first network operation. It then performs one untimed pass
to populate the device-resident input/model cache. The final pass measures the
encrypted stem through encrypted FC output.
Final D2H transfer, decryption and prediction checking are outside the reported
`gpu_only_preloaded_elapsed_seconds` interval.

The current ResNet20 topology has 235 distinct steps and 1,831 logical rotation
calls. Power-of-two composition requires 8,181 GPU
key switches per inference; direct keys reduce this to 1,831 (77.6% fewer).
With the checked-in Q36/P18/dnum=2 configuration, a physical 32-GiB GPU 2 run
with lazy plaintext accumulation reported `8.77687 s`, predicted class `3`,
and reproduced the direct-key warmup
logits exactly (`preloaded_replay_max_logit_error=0`). An earlier validation
run comparing direct and composed rotation paths produced the same prediction.
The preceding direct-key implementation reported `8.86537 s`; this single
matched-card measurement is 1.0% lower overall. Individual regular 3x3
convolutions fell from roughly `92-132 ms` to `62-110 ms`, while Bootstrap
remains the dominant cost. The matching power-of-two-key image-0 run took
`15.4695 s`.

The tradeoff is initialization time and memory: generating 235 full-chain keys
is outside the measured interval and peak device use observed during setup was
about 30.0 GiB on a 32-GiB card. The timed interval keeps the input ciphertext,
encoded model operands, bootstrap constants, and direct rotation keys resident
on the GPU; it does not transfer rotation keys between CPU and GPU.

The optional final argument limits execution to a prefix of the nine
BasicBlocks. `--infer 0 0` checks the encrypted stem, and `--infer 0 1` runs
the first complete block. Set `POSEIDON_GPU_RESNET20_VALIDATE_BLOCKS=1` to
decrypt and compare all block outputs with the plaintext polynomial network.

The encrypted pooling/FC head can be checked independently:

```bash
CUDA_VISIBLE_DEVICES=0 ./run.sh --head-check 0
```

Weights, ReLU coefficients, and CIFAR-10 test inputs are included under
`data/resnet20`, so the benchmark no longer depends on the external
`Trident/resnet20` directory. `POSEIDON_TRIDENT_RESNET20_ROOT` can still
override the weight and test-input location when comparing another data copy.

The GPU CKKS runtime, multiplexed tensor operations, ReLU implementation, and
configuration sources are also local to this directory. Building this target
does not read sources from the sibling `resnet50_gpu` benchmark. It still uses
the Poseidon source tree at `../..`, the bundled `third_party/rmm`, and the
installed CUDA toolkit; set `POSEIDON_ROOT` at CMake configure time if this
benchmark directory is relocated relative to Poseidon.

## V100 performance

For image 0 on a 32 GiB V100, full block-by-block validation now completes in
`198.451 s`: plaintext and GPU predictions are both class `3`, and the final
maximum logit error is `4.1754e-4`. The same validation took `236.869 s` before
the device-resident plaintext cache, so the cache removes `38.418 s` (`16.22%`)
without changing the encrypted network.

Nsight Systems runs on physical GPU 2 show the same effect under profiler
overhead: elapsed time falls from `240.382 s` to `205.029 s`. Host-to-device
traffic falls from `35,650.725 MB` in `6,548` copies to `27,953.653 MB` in
`5,365` copies: `21.59%` fewer bytes and `18.07%` fewer transfers. H2D GPU time
falls from `4.294 s` to `3.244 s`; device-to-device traffic and homomorphic
kernel counts stay essentially unchanged, as expected.

The immediately preceding adaptive-page version took `331.739 s`, while the
original fixed-two-page implementation took `1009.878 s`. Relative to the
original, the current validated implementation is `5.09x` faster and uses
`80.35%` less elapsed time.
