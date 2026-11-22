# Standalone handwritten Poseidon Llama

New target: **`llama31_poseidon_manual`** (`manual_main.cpp`). This is a direct C++/Poseidon API program, not a DSL interpreter/compiler wrapper. It shares the existing handwritten arithmetic kernels but separates the model, immutable precision plan, resource ownership and qualification from the historical non-gating benchmark.

**Status:** local std-only tests and host code generation only. No new target link/GPU/model accuracy or latency result. Historical b3 Block0/a50 primitive results do not transfer to this new source generation. A800 staging remains stopped after SSH255; no reconnect/retry or new GPU job is implied.

## Model structure

|Layer|Manual implementation|
|---|---|
|Whole decoder model|`manual_main.cpp`: fixed ordered loop over1 or32 blocks|
|Block/program ownership|`source_manual_program.hpp`: `manual::Llama::decoder_block`|
|Explicit Q/scale/Boot schedule|`source_manual_plan.hpp`:15 boundaries, observed counter checks|
|Immutable Boot construction|`source_manual_boot.hpp`, `source_manual_c2s.hpp`|
|Adapted cosine coefficients|`source_manual_coefficients.hpp`|
|RMSNorm|`source_rms.hpp`: raw squares/sum, one relin, rotations, Newton|
|Q/K/V/O/Gate/Up/Down|`source_linear.hpp`: direct Poseidon DFT/BSGS adapter, not Native runtime|
|QK/PV|`source_attention.hpp`: hoisting, masks, raw accumulation/relin|
|Softmax|`source_softmax.hpp`: explicit max/exp/inverse and refresh sites|
|SiLU/SwiGLU|`source_manual_silu.hpp`; optional `source_manual_pair_silu.hpp`|

The model boundary remains encrypted embeddings→decoder blocks→encrypted hidden. No tokenizer/final RMSNorm/LM head/incremental decoding was added. N65536/logN16,32768 complex slots,128token capacity,4096hidden/14336FFN,16/4/56tiles. There is no smaller-N or plaintext fallback.

## Two explicit plans; no silent default

- `manual-highprecision-real-v1`: source-derived252primary+224SiLU=476physical Boot calls/block, paired calls0.
- `manual-highprecision-paired-silu-candidate-v1`:252primary+112SiLU=364physical Boot calls/block,112paired projection calls. Two logical SiLUs share four refresh calls; nonlinear products remain separate per lane. **Candidate only: complete paired-SiLU/model acceptance is outstanding.**

Both use the current Q49/P25 accuracy plan, not the complete Native Cerium prime/digit/scale plan. Application interfaces are scale2^40; the two owning Q27/output2^50 profiles have r8192(RMS/Softmax) andr65536(SiLU), internal all60 arithmetic, actual m/eta/output folds verified. The Q26 arithmetic/phase research is not enabled here. Same N does not imply equal FHE plans or security; sec_level_type::none remains uncertified.

The profiles, APIs and workspaces are separate and immutable after publication. They share the Runtime's actual Context/encoder/application-key objects; that binding is checked by object identity, not guessed from matching ciphertext metadata. The caller must still supply ciphertexts encrypted with that Runtime's key/layout, keep it alive, and serialize construction/submission/environment access. Construct it with `with_boot=false` to avoid an extra legacy profile. This is not a concurrent/multi-GPU API.

Softmax precision is a constructor-frozen `ExpUnitSeedPolicy::IntegerPhysicalScale`, not a live env switch: E=T/S, M=round(E), actual resulting scale S*M. `exp_unit_seed_consumer_version` is required at compile time. The program checks **16 successful selected integer-seed consumers/block and zero standard consumers**, along with actual Boot counters and every boundary Q/id/scale. A SETUP declaration alone is not accepted as proof of selection. Existing legacy Api constructors retain environment semantics.

## API

```cpp
Runtime runtime(literal, false, source_rotations());
manual::Llama llama(runtime, manual::Plan{manual::Mode::HighPrecisionReal});
// state:16 owning Q6/2^40 CTs, mask:Q27/2^40, all from runtime.encrypt.
state = llama.decoder_block(state, layer_weights, encrypted_mask, 0);
// Repeat sequentially with layer1,...; no intermediate decryption.
```

Only the final CLI accuracy measurement decrypts. The API returns encrypted hidden owners. The CLI writes aggregate errors/timings only, no raw hidden tensors, keys or CT/PT serialization.

## Build and CPU-only checks

From an appropriately configured parent project with the matching new Poseidon GPU backend:

```text
cmake --build BUILD --target llama31_poseidon_manual llama31_manual_plan_tests
BUILD/.../llama31_manual_plan_tests
BUILD/.../llama31_poseidon_manual --test-plan-cpu
```

The standalone std test can also be built with g++ using this directory as an include path; it does not link Poseidon/CUDA. It checks plan/order/counter/scale contracts and synthetic manifest declarations, **not actual payload hashes or FHE execution**. The new manual executable requires the real new two-projection backend symbol even when selecting the unpaired plan; the old d93 archive alone is insufficient. Host-only object generation is not target linkage or CUDA evidence.

## Closed input declarations / future admitted run

```text
llama31_poseidon_manual --test-inputs-cpu PLAN FIXTURE WEIGHTS INDEX EMBEDDINGS REFERENCE 1 38 MANIFEST REPORT
```

`--test-inputs-cpu` validates the closed pinned manifest and public fixture literal, then returns before Context/Runtime/keys/GPU/report creation. It does not read model tensor/weight payloads or independently hash them. Supported reference scopes are1 or32blocks/38valid tokens; other scopes need a separately pinned reference workflow, not a loose CLI threshold.

A trusted CPU preflight must hash all frozen payloads BEFORE a separately authorized GPU launch. The C++ reader validates six exact roles and canonical nonsymlink paths:

```text
MANUAL_LLAMA_INPUTS_V1
fixture ABSOLUTE_FIXTURE FIXED_SHA256
weights ABSOLUTE_WEIGHTS FIXED_SHA256
index ABSOLUTE_INDEX FIXED_SHA256
embeddings ABSOLUTE_EMBEDDINGS FIXED_SHA256
reference ABSOLUTE_REFERENCE FIXED_SHA256_FOR_1_OR_32_BLOCKS
report ABSOLUTE_NEW_REPORT -
```

The actual fixed hashes are in `source_manual_io.hpp`; do not create a manifest declaring unverified bytes. Reports use exclusive/no-follow0600 creation, no overwrite. Missing/malformed manifests fail before model payload reads. Only after preflight, frozen build provenance and fresh shared/owned GPU admission may the same argument form be used with `--run` instead of `--test-inputs-cpu`. This documentation is not launch permission or connection-repair permission.

## Qualification and timing

- Final all128rows must be finite; active38 maximum **complex** error≤1e−3. Finite numerical failure emits false/rc2; structure/nonfinite/resource failure rc1. The old `llama31_cerium_source_api` entry remains non-gating and unchanged in purpose.
- The new entry never inherits a historical pass or production/model32/security qualification. Public SiLU Up±8 does not cover historical model32 Up≈26.28. Both plan IDs remain experimental even if a particular future run passes its final reference gate.
- Record context/key/profile/input setup, weight reads, encrypted block execution, final decode and whole process separately. Block execution excludes decryption/RNS diagnostics and weight file reads, but **includes CPU diagonal preparation/H2D and metadata checks**. It is not GPU busy time or proof that weights were prepacked/resident.
- Next optimization controls operate on this manual path: qualify paired SiLU, then qualify r8192 pairing before RMS/Softmax adoption; reuse rotations/PT preparation, optimize constant setup and allocation; independently qualify any Native Q/P/digits/scale/relin-plan migration. Do not combine these into an unmeasured latency claim.
