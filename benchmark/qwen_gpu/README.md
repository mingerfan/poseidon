# Poseidon GPU Qwen2.5-0.5B

This benchmark ports Trident's CKKS Qwen2.5-0.5B inference path to the
Poseidon CUDA backend. Model structure, checkpoint loading, approximation
intervals, per-feature SiLU calibration, and bootstrap value scales come
directly from `Trident/qwen`; homomorphic arithmetic is performed on
`gpu::GpuCiphertextData` through `gpu::GpuEvaluator`.

## Implemented path

- Official BF16 `model.safetensors` loading through Trident's loader.
- One ciphertext per token and one ciphertext per 1024-feature chunk.
- GPU ciphertext add/subtract, ciphertext/plaintext multiply, relinearized
  ciphertext multiply, rotations, rescale, modulus alignment, and bootstrap.
- BSGS diagonal Linear for Hidden, Qwen KV, MLP, and LM-shaped matrices.
- Balanced low-depth Chebyshev evaluation for degree-15 RMSNorm and degree-31
  feature-calibrated SiLU.
- A complete single-token causal decoder layer: Input RMSNorm, encrypted V
  projection, exact one-key GQA reduction, output projection, both residuals,
  Post-Attention RMSNorm, Gate/Up, SiLU, SwiGLU, Down, and calibrated refreshes.
- Multi-token Q/K/V, split-half RoPE, causal GQA score products, two-token
  Sigmoid attention, online Sigmoid/Softplus Softmax for longer rows, and an
  encrypted per-layer KV Cache.
- Greedy autoregressive generation: encrypted prompt prefill, persistent
  per-layer GPU KV caches, then one encrypted token per decode step. The exact,
  Trident polynomial, and GPU CKKS paths must choose the same next token.
- A 24-layer driver with encrypted Final RMSNorm. Like Trident's
  `qwen_he_stack`, the refreshed final hidden state is decrypted by the client
  before the tied LM Head and argmax validation.
- Timestamped, single-file logs including per-stage and total elapsed time.
- Bounded 32-diagonal Linear preparation batches. CPU CKKS encoding is
  parallelized and each encoded/uploaded diagonal is reused across every
  token in the batch, without retaining a full dense layer in GPU memory.

For a one-token causal row, softmax has one element and is exactly 1. Q/K,
RoPE, maximum, exponential, and reciprocal therefore do not affect the result;
the attention output is the GQA-repeated encrypted V tensor. The multi-token
driver uses the complete Q/K/RoPE/online-attention path.

## Build and test

The default checkpoint location is:

```text
Trident/qwen/pretrained_parameters/Qwen2.5-0.5B
```

Check the checkpoint without creating a GPU runtime:

```bash
./benchmark/qwen_gpu/run.sh --model-info
```

Validate all 24 calibrated layer schedules and the GPU parameter shape:

```bash
./benchmark/qwen_gpu/run.sh --topology-check
```

Run the GPU primitive suite:

```bash
CUDA_VISIBLE_DEVICES=0 \
POSEIDON_NTT_ALGO=fourstep \
./benchmark/qwen_gpu/run.sh --smoke
```

Linear plaintext preparation uses up to eight CPU encoding threads by
default. Override the count and enable per-Linear timing with:

```bash
POSEIDON_GPU_QWEN_ENCODE_THREADS=16 \
POSEIDON_GPU_QWEN_PROFILE_LINEAR=1 \
./benchmark/qwen_gpu/run.sh --smoke
```

The profile reports the number of prepared plaintext diagonals, GPU products,
cross-token reuse factor, and preparation/multiply/rotation/addition times.

The suite covers tensor round-trip, baby- and giant-step Linear, degree-15
SiLU, and degree-15 RMSNorm. A validated V100 run produced maximum errors of
`2.48e-8`, `3.64e-8`, `3.69e-8`, and `3.90e-6`, respectively.

Run one calibrated real decoder layer (token 9707 is the position-zero token
used by Trident's calibration):

```bash
CUDA_VISIBLE_DEVICES=0 \
POSEIDON_NTT_ALGO=fourstep \
POSEIDON_GPU_QWEN_VALIDATE=1 \
./benchmark/qwen_gpu/run.sh --infer-token 9707 1
```

Run all 24 layers and the final client-side logits check:

```bash
CUDA_VISIBLE_DEVICES=0 \
POSEIDON_NTT_ALGO=fourstep \
./benchmark/qwen_gpu/run.sh --infer-token 9707 24
```

Run a calibrated multi-token prefill (the first two IDs from Trident's default
validation prompt):

```bash
CUDA_VISIBLE_DEVICES=0 \
POSEIDON_NTT_ALGO=fourstep \
POSEIDON_GPU_QWEN_VALIDATE=1 \
./benchmark/qwen_gpu/run.sh --infer-ids 9707,11 1
```

Run encrypted greedy generation. This example prefills three tokens, generates
two tokens, and executes all 24 decoder layers per step:

```bash
CUDA_VISIBLE_DEVICES=0 \
POSEIDON_NTT_ALGO=fourstep \
./benchmark/qwen_gpu/run.sh --generate-ids 9707,11,847 2 24
```

The normal generation path skips the CPU exact/polynomial reference preflight
and does not stop on reference-tolerance differences. Its output is marked
`validation_status=UNVERIFIED`. Set `POSEIDON_GPU_QWEN_VALIDATE=1` only when an
explicit, strict reference comparison is required.

The generation syntax is:

```text
--generate-ids ID,ID,... MAX_NEW_TOKENS [MAX_LAYERS] [MODEL_DIRECTORY]
```

To inspect greedy tokens even when a numerical tolerance is exceeded, run the
exact/polynomial reference preflight without allocating a GPU runtime:

```bash
POSEIDON_GPU_QWEN_ALLOW_TOLERANCE_MISS=1 \
POSEIDON_GPU_QWEN_PREFLIGHT_ONLY=1 \
./benchmark/qwen_gpu/run.sh --generate-ids 9707,11,847 4 24
```

This mode only relaxes numerical tolerance checks. An exact/polynomial argmax
mismatch remains a hard failure, as does an operator input outside a calibrated
polynomial interval.

`POSEIDON_GPU_QWEN_VALIDATE=1` enables the CPU exact/polynomial preflight and
decrypts intermediate tensors for development comparison against Trident's
stage traces. With validation disabled, the decoder, attention, RoPE,
nonlinear operators, and KV-cache execution path contains no mock or
server-side decryption; the final hidden-state decryption and LM Head are the
client boundary.

To validate only the first GPU layer while retaining the full 24-layer CPU
reference and generation calibration, set:

```bash
POSEIDON_GPU_QWEN_DIAGNOSTIC_LAYER_LIMIT=1
```

`--bootstrap-scale-check` sweeps representative Attention delta scales without
running a decoder layer.

To run in the background:

```bash
cd /home/guoshuai/github/poseidon/poseidon
nohup env CUDA_VISIBLE_DEVICES=0 POSEIDON_NTT_ALGO=fourstep \
  ./benchmark/qwen_gpu/run.sh --infer-token 9707 24 \
  >/dev/null 2>&1 &
echo $!
```

Logs are written to:

```text
benchmark/qwen_gpu/output/qwen_gpu_YYYYMMDD_HHMMSS.log
```

Override the build, output, or exact log path with
`POSEIDON_GPU_QWEN_BUILD_DIR`, `POSEIDON_GPU_QWEN_OUTPUT_DIR`, or
`POSEIDON_GPU_QWEN_LOG_FILE`.

## Parameter profile

The GPU backend uses the high-precision profile already validated by the GPU
ResNet benchmarks: `N=65536`, 32768 slots, application scale `2^51`, a Q53
application chain, and a Q34 high-precision GPU bootstrap raised back to Q53.
Physical primes are at most 32 bits because the CUDA RNS backend stores
residues in `uint32_t`; multiple physical primes implement each high-precision
logical level.

## Current validation status

The CUDA primitive suite, checkpoint loader, and all 24 approximation schedules
pass. The full 24-layer, two-step CPU preflight for prompt `9707,11,847`
selects tokens `829,374` on both the exact and polynomial paths; every layer's
cache grows from three to four tokens on the decode step.

A real V100 multi-token layer initially exposed excessive error in Attention:
Trident's non-power-of-two delta bootstrap scale produced `O(1)` refresh error
on the CUDA bootstrap. Rounding the external scale up to a power of two with a
minimum of four reduced delta-bootstrap error from `2.62` to `2.41e-7`, and
Attention output error from `0.0166159` to `7.38033e-6`. The complete GPU layer
then passed with `ckks_vs_polynomial_max_error=1.4995e-5` against tolerance
`0.0100686`. The validated one-layer run took 59 minutes 35 seconds on one
V100 before Linear preparation optimization.

After changing Linear to prepare 32 diagonals per bounded batch, parallelize
CPU CKKS encoding, and reuse every uploaded diagonal across three tokens, a
validated layer-0 prefill completed in 9 minutes 40 seconds including setup;
the recorded GPU stage total was about 6.4 minutes versus 56.1 minutes before
the change. Query Projection fell from about 204 seconds to 15.5 seconds,
Gate Projection from about 1015 seconds to 73.5 seconds, and the final layer
error remained `1.50223e-5`. Full 24-layer generation remains to be timed and
token-validated.
