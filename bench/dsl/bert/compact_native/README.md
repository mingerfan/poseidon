# Exact compact Poseidon dense Linear

This directory is an independent bench-only implementation. It does not change
Poseidon, the existing runner/backend, or their CMake. No Cerium backend is used.

## Integration contract

Include `compact_linear.h`; link target `bert::compact_linear` (add this directory
with `add_subdirectory`, or link its built `libbert_compact_linear.a` plus the
existing GPU/CPU libraries). Public interface:

```cpp
CompactLinearEngine engine(context, shared_ptr<const GaloisKeys>, device);
CompactLinearEngine::required_rotation_steps(); // 46 direct CPU keygen steps
CompactLinearEngine::required_binary_rotation_steps(); // 13 signed binary keys
engine.prepare(spec, float32_rowmajor_out_in, float32_bias);
auto output = engine.run(spec.id, output_tile, vector<const GpuCiphertextData*>);
```

`MatrixSpec`: `id, input_width, output_width, tile=256, baby=16, tokens=128,
input_level, input_scale_log2, output_level, output_scale_log2, rescale_count`.
Physical `L` means **Q count L+1**. Require output_level=input_level-rescale_count.
The GPU engine accepts N>=1024/tokens>=2 (the intended N65536/tokens128 case).
Counts1,2 and larger use real `GpuEvaluator::rescale_many`. Plaintext scale is
computed from the actual dropped primes, input scale, and requested output scale;
there is no scale-metadata rewrite. 56-bit encoding is supported.

Each distinct physical matrix/call spec needs its own prepared id. Prepare in
**descending input_level** order to upload the cropped owning keys only once.
The key owner retains highest prepared input Q and all context P. Prefix views
are not additional owning crops. Statistics report actual key and parameter bytes.

Caller synchronizes inputs, which must remain immutable and alive through an
output0,...,last sequence. Output0 unconditionally rebuilds babies, even if input
addresses are unchanged. Later outputs must use the same matrix/input objects and
strict sequence. A new output0 may interrupt an old sequence. Final output frees
the invocation cache; `reset_sequence()` explicitly abandons it. One engine is not
concurrently callable. Returned genuine GPU ciphertext is synchronized and ready
for a BaseApi result; the helper neither constructs nor fakes runtime ReadyEvents.

## Arithmetic and storage

For each input/output256 tile, all511 masked diagonals d=-255,...,255 are encoded
from the original independent float32 weights. No sparse/random/repeated weight
substitute is used. Negative BSGS split is floor-based: baby0..15 and giants
-256,-240,...,240. Diagonals are pre-rotated by -giant. Baby rotations are shared
across every output tile. A custom Q-only CUDA kernel accumulates **all input
tiles** for one giant group, then Poseidon performs only one giant rotation.
One final real rescale and a genuine full encoded bias complete each output.

Encoding retains only one full CPU plaintext at a time. Every full-N residue in
every Q limb is independently reconstructed and checked before exact periodic
storage is admitted. The full-period fallback stays in ordinary NTT order; smaller
periods use `reverse_bits(j,logN)&(period-1)`. Upload is additionally read back and
compared. Stats count compact residues plus full bias, full-equivalent GPU bytes,
CP terms, baby/giant rotation counts, actual fallback rotations, and compression
fallbacks. All original float32 weights and biases remain owned and reported.

The private stream is scoped around all allocations, arithmetic and destruction.
No host transfer/synchronization occurs per MAC. Cold allocator/NTT/rescale scratch
growth may synchronize internally; returned results synchronize once explicitly.
Use the existing validated pool, not the unvalidated cuda_async allocator.

## Build/test without any GPU work

`COMPACT_BUILD_GPU=OFF` builds the standard-C++ layout test, CPU-only Poseidon
encoding/compression check, and a genuine encrypted MulCP/Rescale(1 or2)/bias
physical-scale check. To avoid all Poseidon dependencies, also set
`COMPACT_BUILD_SETUP_CHECK=OFF`.

The GPU target can be **compiled and linked without running a GPU**. The updated
build defaults to source dependencies; explicit compatible PREBUILT dependencies
are also supported. `build_a800.sh` is now a local compatibility wrapper and does
not deploy over SSH or choose a private bundle/toolchain. GPU architectures are
inherited from `CMAKE_CUDA_ARCHITECTURES`, not forced to SM80. See
[the portable build guide](../BUILDING.md). Historical validation below remains
specific to its original artifacts.

Local CPU/CUDA build and link succeeded (GCC15/nvcc13.4). All three CTests pass;
real N1024/Q6/P2 encoding at scale30 and scale56 has period512, zero compression
fallbacks, and exact RNS reconstruction. The real encrypted CPU scale check
matches actual prime division to 7.11e-15 log2 and decodes within1.66e-7.
`modular_check.py` checks unsigned Barrett and signed-residue MACs (41,160 products,
including near-32-bit primes). `audit_manifest.py` validates API output sequence,
physical spec, and unchanged input SSA ids; the parent's tiny cyclic3-hidden /
12-FFN plan passes all27 calls /six matrix sequences. Exceptional prepare/run
cleanup retains the private stream and H2D source lifetime. GPU execution remains
separately permission-gated; CPU encoder timings below are not GPU timings.

## Standalone probe (requires explicit GPU-window permission)

`compact_linear_probe --help` is safe and performs no CUDA calls. Every actual
probe requires `--run-gpu` plus original `--weights` and `--bias` files; synthetic
weight substitution is refused. The build script never invokes it. The probe
supports all input/output widths divisible by256, physical levels, rescale counts,
direct or binary keys, and full raw float32 weight/bias/input files. It decrypts genuine
results and compares every token/output against CPU dense Linear+bias.

Use `export_fixture.py` to export exact original float32 matrices from the shared
NPZ. Inputs may be float32 or float64; standalone raw input export explicitly
converts the latter to float32 and records that conversion. Raw output metadata
records SHA256 and byte counts. Parent integration should consume its own
`linear-weights.bin` offsets directly instead of using this probe exporter.

## Optional small-ring setup encoder (strict remains default)

Set `MatrixSpec.periodic_encoding=true` explicitly (standalone probe:
`--periodic-encoding`). The online CUDA MAC/BSGS path and all original float32
weights are unchanged. `PeriodicEncoderExperiment` uses the real Poseidon
CKKSEncoder at degree512 with the SAME Q primes and physical scale, then lifts
the integer polynomial g(Y) to g(X^(N/512)). No weights are quantized or replaced.
CKKS uses generator5; the unrelated generic generator3 FFT is not substituted.

For each Q limb, the independently chosen small/full primitive roots are related
by psi_small^t=psi_full^(N/512). The exact compact index map is
`small_index=rev9(((t*(2*k+1) mod1024)-1)/2)`. The map is verified as a permutation
and with modular root equalities. Actual full NTT of the monomial X^(N/512) is
compared at EVERY full-N residue of EVERY Q limb once per preparation contract.
Each diagonal then materializes its genuine full-N NTT residues and passes the
SAME full-N exact compression/reconstruction check and actual upload readback as
strict mode. Only the expensive full complex FFT/full NTT per term are avoided.

Unsupported fast initialization/encoding falls back to native fullEncode and is
counted. Fast mode additionally audits d=-255,0,255 against actual fullEncode per
matrix and reports any term/word differences. Separate FFT rounding paths are
**not universally claimed byte-identical**, even when these audits pass. Stats:
`encoding_mode`, `periodic_encoded_terms`, `strict_encoded_terms`,
`periodic_encoding_fallbacks`, `periodic_fallback_reason`,
`periodic_full_ntt_mapping_verified`, `native_encoder_bitwise_guaranteed` (false
whenever fast is used), `native_encoder_audit_terms`,
`native_encoder_mismatch_terms`, and `native_encoder_mismatch_words`.

CPU-only oracle: enable `COMPACT_BUILD_PERIODIC_EXPERIMENT=ON`, build
`compact_periodic_encoder_check`. `--weights raw.f32 --input-width768
--output-width3072` tests original dense data. Full-N encoding should run on the
remote CPU, not as a local dense pack. `--q-primes comma,separated,actual,primes`
can select the exact context prefix. `--require-native-bitwise` rejects any actual
rounding mismatch; that is a diagnostic guarantee for tested samples only.

Verified remotely with GCC11, the mature CPU library, OMP1, N65536, and actual
Cerium-prefix Q19 primes, using unchanged shared FFN-up weights:

- Scale56:514 terms covering all511 diagonals,640,024,576 full residues checked;
  zero native byte mismatches, decoded max error4.23e-16. Native encoding alone
  took11.7025s; small encoding/remap0.0962s; full materialize+lossless verification
  pipeline1.9372s. These are CPU setup phase timings, NOT end-to-end/GPU speedups.
- Scale55.8125:128 terms,159,383,552 residues, zero native byte mismatches;
  decoded max error4.15e-16; corresponding times2.9045s /0.0237s /0.4624s.
- Scale30:32 genuine terms also reconstructed byte-identically, decode2.67e-8.

Artifacts are under `test-results/bert-a800-20261115/compact-native-build/` with
prefix `cpu-periodic-n65536-`. The strict default stays unchanged, except that its
immutable bit-reversal index map is now cached per CPU thread.
