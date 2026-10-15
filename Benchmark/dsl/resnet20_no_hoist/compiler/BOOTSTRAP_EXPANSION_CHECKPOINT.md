# Bootstrap expansion checkpoint — paused at user request, 2026-10-15

This branch is an unfinished opt-in extension. The previously published
ResNet20 benchmark at commit `371761f` remains the validated API-based version.

Implemented and checked with the real Hecate compiler:

- `expand-native-bootstrap` clones an explicit primitive template at the 18
  existing boundaries, retaining group IDs, local indices and template SHA256.
- Seven registered primitive operations cover conjugation, modulus raising,
  exact scale metadata, integer multiplication, raw ciphertext multiplication,
  relinearization and grouped rescale. f64 primitive parameters and group
  membership are bound to emitted HEVM bytes by sidecars.
- A deliberately synthetic template tests structure only. All 18 boundaries
  expand and six negative cases are rejected. This template is never run on GPU
  and must never be presented as a native bootstrap implementation.
- The unchanged original MLIR still compiles to HEVM SHA256
  `36b9b6f2e66d85b84cd7ca988a04c50a37635689f7f86a8f6710ba384bf221cd`.
- The CPU exporter uses an isolated observer copy of native GpuUploader. It
  captures 144 scalar encodings and the native folded EvalMod plan (9 basis
  steps, 8 blocks, 7 combines, 2 double-angle steps), without keys or GPU work.

Not completed: native S2C/C2S diagonal export, full primitive template,
expanded-operator runtime dispatch, placement integration, or same-ciphertext
RNS/GPU comparison against the original bootstrap. Do not run the original
runner on expanded HEVM or use the synthetic tests as numerical acceptance.

The optional `POSEIDON_BUILD_BOOTSTRAP_PLAN_EXPORT` CMake target builds the
CPU exporter; it is disabled by default. Existing native arithmetic was not
modified. Work on this extension stopped when the user returned priority to
ResNet18/50; no GPU task remains running for it.
