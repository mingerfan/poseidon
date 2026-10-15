# ResNet18/50 native-aligned DSL work in progress

This checkpoint establishes the user-requested **native no-hoist baseline**.
Neither model has a compiled DSL or full encrypted equivalence result yet.
ResNet20 bootstrap expansion is paused and is not a dependency of this work.

The isolated generator preserves native model arithmetic and changes bootstrap
matrix/key representation to no-hoist, with a common 26 GiB pool. It never edits
the reference repository. `native-source.tar.gz` preserves the reference source.
The original native repository also supplies the pinned ReLU fixture.

## Current evidence

- N65536, Q50/P25; actual input Q32/scale40, 147 image layouts plus one cached
  encrypted constant ONE. Application prefix Q36 is not the input level.
- Native-driven CPU physical traces: ResNet18 851353 operations, 16 logical
  activation positions/60 per-pack bootstrap calls; ResNet50 5109479 operations,
  48 positions/351 per-pack bootstrap calls. Both metadata/payload audits pass.
- ResNet18 ideal-slot replay predicts65 (label65), max error1.27131e-7 against
  the original CHW polynomial oracle. This excludes CKKS/bootstrap noise.
- ResNet50 original plaintext model predicts62 for label65. Preserve this
  behavior; fidelity to the model and label correctness are separate checks.
- Actual native no-hoist bootstrap Q6→Q31/scale40 ran on GPU: measured maximum
  differential error3.76313e-5 with the same key. No full-network claim.

The full traces and lossless zlib payloads remain on the experiment server;
SHA256 and counts are retained in the reports. They are not compiler artifacts.
Old RLE exports and the superseded Q36 input contract are excluded.

## Reproduction

Generate isolated sources with:

```sh
python tools/prepare_native_nohoist.py /path/to/reference/poseidon-gpu /tmp/native-nohoist
```

Configure the adjacent validated `resnet20_no_hoist` CMake project, adding:

```text
-DPOSEIDON_IMAGENET_TARGETS_FILE=/absolute/path/to/tools/native_nohoist_targets.cmake
-DPOSEIDON_IMAGENET_NOHOIST_SOURCE=/tmp/native-nohoist
-DPOSEIDON_IMAGENET_REFERENCE_ROOT=/path/to/reference/poseidon-gpu
```

Build `imagenet_nohoist_bootstrap_probe`, `resnet18_native_nohoist`, and
`resnet50_native_nohoist`. CPU checks: probe `--metadata-only`, model
`--weights-check`, and ResNet18 `--topology-check`.

`tools/run_native_nohoist.py` accepts an explicit bin directory, a new log path,
GPU1/2/3, and model/bootstrap choice. It refuses occupied GPUs, requires28 GiB
free, clears the three forbidden scale overrides, and refuses log overwrite.
Its server model roots are explicit; change those paths for another machine.
CPU guard check: `python tools/test_native_nohoist_guard.py`.

## Remaining gates

Real compiler lowering must address legacy HEVM16-bit index capacity before
compiling these large graphs. Preserve metadata assignments, native grouped
rescale, dynamic encoding, encrypted ONE, and the exact pack bootstrap order.
Then use the same keys and input ciphertexts for native/DSL stage comparisons,
with short encrypted tests before full runs. No arbitrary logits threshold or
single-image dataset-accuracy claim is used.

Compiler capacity checkpoint: an opt-in `poseidon.hevm_wide` function attribute
now emits magic0x48455632 with four uint32 fields per16-byte instruction.
Compiler instruction construction retains int64 values until checked output;
the legacy runtime ABI stays8 bytes. Actual compiler tests preserve constant
65536/uint32 max, register65536 and signed rotation-1, and reject overflow before
creating artifacts. Both the small legacy artifact and full ResNet20 HEVM are
byte-identical (full SHA36b9b6f2e66d85b84cd7ca988a04c50a37635689f7f86a8f6710ba384bf221cd).
The patch applies after the previously checkpointed native primitive definitions;
it is not a standalone patch for upstream Hecate. A wide runtime reader and full
18/50 DSL lowering are still pending. This does not resume bootstrap expansion.

ResNet50 ideal-slot CPU replay has now completed: prediction62 matches the original
oracle, max logits error1.38439e-8, RMS1.74969e-9. ResNet18 native GPU stem also
completed normally, max reported error1.05025e-6. First-block GPU test is pending.

ResNet18 first residual block native no-hoist staged run passed through both
activation points, eight tensor packs each. Measured act1 bootstrap max difference
3.88831e-4; act2 bootstrap 2.18732e-3; act2 ReLU 2.17575e-3, with resulting
output Q9/scale40. These are measured trace values, not newly imposed acceptance
thresholds. Full eight-block native run is underway separately.
