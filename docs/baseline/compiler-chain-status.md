# Compiler-chain implementation status (2026-09-05)

**Historical snapshot below, not the latest capability report.** Subsequent
milestones: [48-case live Agent CPU results](deepseek-flash-48-case-results-2026-09-07.md),
[96-case deterministic CPU results](expanded-96-models-cpu.md), and
[real Poseidon GPU Add/drop primitive](poseidon-gpu-modswitch-primitive.md).
The GPU runtime now builds and that primitive executes, but HEVM/Agent GPU
end-to-end execution remains unverified. CUDA installation is no longer pending.

## Goal and gates

Restricted PyTorch -> Hecate Python frontend -> Dacapo Earth/CKKS IR ->
HEVM/CST -> existing Dacapo SEAL CPU runtime -> decrypt and compare with
an independent CPU float64 reference. The user approved SEAL CPU for the
first correctness stage; Poseidon single-GPU integration/performance is a
later, separate gate. Do not block Agent correctness work on CUDA, GPU RNS
profile changes or unrelated desktop/network environment repairs. No new
CPU HEVM backend is being implemented. No Agent implementation before the
handwritten golden sequence and semantic specification gates are satisfied.

### Latest milestone: isolated candidate validation and feedback replay

`run_candidate.py` adds a provider-neutral request/response contract and bounded
feedback loop, currently with a file ReplayProvider only (zero LLM calls).
Candidate Python is NOT exec'd: a checked AST constructs real Hecate expressions.
Tracing, compilation and existing SEAL execution run in fail-closed bubblewrap
namespaces, without mounting reference or the full workspace; only the runtime
gets a read-only key mount. A real capability probe precedes candidate attempts.

The real `candidate-replay-7ommkr7q` scenario rejects malformed JSON, then detects
a numerically wrong but executable Linear reduction (max error 0.6736366599782002),
then passes the saved correct rule answer (max error 8.880283147716383e-9).
The independent `--replay` file entry repeats the expected sequence in
`candidate-replay-rm7lm1v_`. These are scripted fault/recovery tests, not autonomous
Agent synthesis or repair. Security parameters and tolerances are unchanged.

See [candidate feedback runbook](candidate-feedback-loop.md) for the capability
boundary, real evidence, preserved initial loader failure, commands and remaining
provider/model/credential/cost gate. No provider SDK/HTTP client is wired yet.
Regression: 104 tests discovered on the host, 95 passed and nine Torch-only
tests skipped there; all nine were separately rerun and passed in pinned Nix/Torch.
The 19 new candidate tests are included in the 95 host passes. This includes
checks against selected prior 48-case/golden artifacts, not a fresh 48-case run.

### Previous milestone: automatic rule translation passed 48/48 real cases

`run_model_batch.py` now connects a data-only model description to a trusted
PyTorch factory, generic FX checking/lowering, Hecate generation, actual compiler
and SEAL execution with independent numerical comparisons. The Python translator
API accepts trusted nn.Module objects; the CLI deliberately does not import
arbitrary model.py files. It does not select handwritten DSL by family name.

The real `fx-batch-yw9qpdtx` run passes all eight families x six configurations:
48/48 translated/traced/compiled/executed/numerically correct, 192 encrypted
input executions and 640 output values. Max absolute error is 3.972002460272961e-8
with unchanged parameters and tolerances. Fan-out/residual's 12 cases are tagged
as future Agent holdouts, not used as prompt examples. No Agent/LLM calls occurred.
The separate JSON-file CLI example passes in `fx-batch-9fo2xy9m`.

Regression: host discovery ran 85 tests, with 76 passing and nine Torch-only
tests skipped there; those nine separately passed inside the pinned CPU Torch
environment. The old manual base suite (add/mul_plain/Linear) was also freshly
traced, compiled and genuinely executed after the shared executor refactor:
`seal-cpu-golden-98f3gsf8`, all three passed. No runtime backend or safety
parameter changes were needed.

See [rule translator and batch runbook](fx-rule-translator.md) for input limits,
commands, architecture, failures/denominators and per-family results. This is the
deterministic baseline against which Agent work must be compared, not an Agent
contribution. Next gates are safe Agent tracing, structured feedback/repair and
explicit provider/model/credential/cost choices before any external model call.

### Previous milestone: polynomial MLP and semantic probes passed on SEAL CPU

The follow-up run `seal-cpu-golden-l7iu78ak` passed all eight manual cases:
the original add/mul_plain/Linear plus rotate(1), rotate(2), square, quartic
and Linear(4,4)-square-Linear(4,2). Four inputs each yield 112 compared
output values. MLP max absolute error is 5.468147179499283e-8; the largest
error in this run is rotate(2)'s 4.110290798942096e-6. All remain within the
unchanged tolerance. The complete baseline/semantic/static-contract suite
passed 72/72 tests (no skips), including nine new frontend-contract tests.

The MLP artifact contains 8 rotations, 4 MulCC, 12 MulCP and 8 rescales.
A read-only metadata observer confirms actual input/output modulus counts,
scale and two-polynomial ciphertexts. It neither performs arithmetic nor
decrypts intermediate values. The original SEAL_HEVM runtime stays unchanged.

[Hecate semantics v0](hecate-dsl-semantics-v0.md) records the initial verified
operator/layout subset, grammar, layer information loss and remaining gaps.
`scripts/baseline/hecate_contract.py` performs static fragment checks without
executing Python; it is not an Agent, OS sandbox or correctness proof.
The handwritten golden sequence now passes for SEAL CPU. Next is the checked
model/manifest/compile-execute interface and deterministic translator baseline;
external LLM calls still require service/model/cost approval.

### Previous milestone: real compiled Linear passed on SEAL CPU

On 2026-09-05, add, elementwise public-weight multiply and Linear(4,2)
passed genuine encrypt/evaluate/decrypt comparisons for four fixed inputs
each (40 output values total). Maximum absolute error across all cases was
1.9138588935874168e-8, with unchanged atol=1e-5 and rtol=1e-4.
The Linear's maximum error was 1.839301577710728e-8; its cross-element
dot products execute ciphertext rotations and additions, not an elementwise
surrogate. The full local baseline Python suite passed 58/58 tests.

Runtime parameters remain upstream N=32768, 14x60-bit moduli, SEAL tc128
validation; compilation explicitly uses the upstream SEAL CPU profile and
waterline=40. Bootstrap/upscale/unknown opcodes are denied before native
loading. This is not Poseidon GPU success or proof of Agent correctness.

Evidence: `/home/lhy/poseidon-work/results/seal-cpu-golden-7999_hvg`.
See [SEAL CPU golden runbook](seal-cpu-golden.md) for commands, metrics,
key handling, scope and the next MLP/DSL specification gate. Older results
below describe the separate Poseidon adapter bring-up, which remains blocked.

Input cases will provide `build_model()`, fixed array weights/inputs, static
shapes and a declared privacy boundary (public weights, encrypted inputs).
No implicit ReLU/SiLU replacement. The golden sequence is add/multiply,
Linear(4,2), then Linear(4,4)-square-Linear(4,2). Golden acceptance is
`abs(actual-reference) <= 1e-5 + 1e-4*abs(reference)`, checked elementwise.
This is a frozen initial test budget, not a theorem about CKKS accuracy.

## Confirmed current results

Checkout base: `4995e7cadedf2bfb9104658b5638662ecf6a1d0a`, branch
`feat/agent-dsl-correctness`, upstream `origin/gs/feat-application`.
Prior local modifications are retained; no commits or remote writes.

| Gate | Result | Meaning |
|---|---|---|
| CPU CKKS primitives | Passed in previous baseline runs | Direct C++ execution, not compiled DSL |
| CPU adapter/preflight CTest suite | 53 passed, 3 skipped, 0 failed | Mock/captured-format fixtures, not GPU execution |
| HEVM constant contract repair | Repeated/sliced constants and Q-only level guard passed | Real Linear CST encode/decode max error 3.9968e-15; direct encoder unchanged; not encryption |
| Environment-gate unit tests | 8 passed | Reject missing dependencies, wrong versions and uninitialized submodules |
| Dacapo source initialization | Passed after user approval | Fixed commit, clean submodule; parent gitlink unchanged |
| Nix portable bootstrap | Passed after separate user approval | Nix 2.20.6; offline evaluation and sandboxed minimal build, not a compiler build |
| Baseline validation suite | 45/45 passed in the latest run | Includes actual shell entry, real compiler artifacts and isolated constant encode/decode; no encrypted execution |
| SEAL C++ dependency stage | Source build/install checks passed after packaging-path repair | SEAL 4.0.0, GSL 16/16 tests, Zstd 1/1 selected package test; not FHE execution |
| LLVM/MLIR/Clang dependency stage | 18.1.2 build/install checks passed | Single pinned toolchain, not a GPU run |
| Dacapo native C++ build | Passed after shell-entry and compiler-wrapper repairs | hecate-opt, HecateFrontend and SEAL_HEVM; actual CMake linkage/dialects/CDLL checks passed |
| Native C-API compiler diagnostics | Add and mul_plain emitted Earth/CKKS/HEVM/CST | Actual compiler output, not Hecate Python/Torch tracing |
| Poseidon on real tiny artifacts | Rejected ModswitchC (opcode 4) in both cases | Artifacts loaded; no schedule constructed, GPU/communication preflight not run |
| Python dependency stage | Installed; exact package-set check, pip check and actual imports passed | Pinned Nix Python 3.10.14, NumPy 1.25.2, Torch 2.0.1+cpu; no CUDA packages |
| Actual Dacapo prerequisite gate | All checks passed in the approved venv/Nix shell | Prerequisites only; separate real compiler checks provide compilation evidence |
| Initial Dacapo CMake probe | Historical exit 1, superseded by successful native build | Initially missing MLIRConfig.cmake |
| Hecate Python trace/compile | Add, mul_plain and Linear(4,2) passed | Real frontend, Earth/CKKS and HEVM/CST; all three rejected at adapter ModswitchC |
| Compiled-artifact GPU correctness | Not run | No encrypted end-to-end success claim |
| Golden Linear on SEAL CPU | Real encrypted comparison passed for all four inputs | Accepted for the SEAL CPU backend; MLP/Agent and Poseidon GPU remain separate gates |

CPU suite build: `/home/lhy/poseidon-work/build-poseidon/agent-dsl-adapter`.
Logs under `/home/lhy/poseidon-work/results`:
`adapter-20260905T043042-486.log` and `adapter-20260905T043429-485.log`.

The skipped tests are `poseidon_mgpu_resnet20_artifact_path_tests`,
`poseidon_mgpu_resnet20_dump_preflight_tests`, and
`poseidon_mgpu_external_hevm_artifact_tests`. They require external artifacts.
Existing unit fixtures using `sec_level_type::none` remain unit fixtures,
not evidence of approved cryptographic parameters or GPU security.

## Reproduction

From Windows PowerShell:

```powershell
wsl.exe -d Ubuntu-22.04 --cd '/mnt/d/Code Space/Poseidon' -- bash scripts/baseline/run_adapter_checks.sh
wsl.exe -d Ubuntu-22.04 --cd '/mnt/d/Code Space/Poseidon' -- env PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s scripts/baseline -p test_environment_gate.py -v
wsl.exe -d Ubuntu-22.04 --cd '/mnt/d/Code Space/Poseidon' -- env PYTHONDONTWRITEBYTECODE=1 python3 scripts/baseline/check_dacapo_environment.py
```

The adapter runner disables GPU/HPU targets, automatic dependency downloads
and examples, uses the existing GMP 6.3.0 prefix and `-j2`, and applies a
60-second timeout per CTest. It does not need Dacapo source or MLIR. Test names
mentioning clusters/ResNet exercise CPU planning or mock artifacts, not those
applications on hardware.

The environment gate prints JSON and returns nonzero on unmet prerequisites.
Run it with the intended isolated Python interpreter. Optional `--llvm-prefix`,
`--mlir-prefix`, `--clang-prefix`, and `--seal-prefix` select isolated packages.
LLVM/MLIR must report 18.1.2; 18.1.8 is deliberately not treated as the documented
version. Torch 2.0.1 CPU wheels are accepted without CUDA packages for tracing.
This inventory does not replace CMake package discovery, compilation or execution.

## Semantic evidence and unresolved boundaries

### Confirmed: HEVM constant repetition repaired; general encoder unchanged

The original adapter zero-padded short constants. After red-to-green regressions,
the HEVM entry now expands `src[i % src.size()]` across all slots, matching pinned
Dacapo SEAL/HEAAN runtime source. Direct `CKKSEncoder` vector use still zero-pads.
Empty/non-finite constants and QP key contexts presented as HEVM data levels are
rejected. All-or-nothing error handling and scale/parameter metadata are preserved.

The unchanged CPU N=16384/tc128 encoding tests pass at data level 7 and scale
2^48, including actual compiled Linear weights/bias. This is not a GPU-compatible
parameter profile: its 48/50-bit Q/P primes exceed GPU uint32 residues, and it
lacks the artifact's input level 16. See [contract repair and blocker evidence](hevm-compatibility-repair.md).

### Confirmed mappings, unresolved execution compatibility

The adapter maps Encode/RotateC/NegateC/RescaleC/AddCC/AddCP/MulCC/MulCP to
schedule operations. ModswitchC/UpscaleC are unsupported. BootstrapC creates
a fallback marker, not a real bootstrap. HEVM encode scale is interpreted as
a base-2 exponent; level resolves through the context's parameter-id map.

MulCC maps to `Multiply`, while the GPU backend has a separate `Relinearize`
handler requiring keys. Fixed Dacapo's `SEAL_HEVM.cpp:311-316` explicitly
multiplies and relinearizes for MulCC. Poseidon's `GpuEvaluator::multiply`
instead produces `left.size() + right.size() - 1` components without
relinearization. This is a confirmed operation-contract difference relative to
the SEAL HEVM implementation; its effect on real compiled polynomial programs
still needs reproduction. The closed HEAAN library implementation is not inferred
from its `multWithoutRescale` name. Keep this gate before polynomial MLP.

## Next approval boundary

Source initialization was approved and completed on 2026-09-05 after three
successful rounds of bounded network checks. The exact WSL command was:

```bash
timeout -k 3s 180s env GIT_TERMINAL_PROMPT=0 \
  git -c url.https://github.com/.insteadOf=git@github.com: \
  submodule update --init --depth 1 -- third_party/dacapo
```

HEAD is `4616402710f39df3e5f5bd7930a6c036025aaac3`, detached as expected for
a pinned submodule; its working tree is clean. No nested submodules or AGENTS.md
were found within it. Git registered the submodule in the parent's local Git
configuration, but tracked `.gitmodules` was not edited. Its before/after SHA-256
is `EF92574A30A1D180016CEA40155127A5F0BB51D27D732E432672BCBD92C51DD2`.
The parent branch/HEAD, sparse checkout and special-file skip-worktree are intact.

An out-of-source configure probe under
`/home/lhy/poseidon-work/build-dacapo/configure-probe` stopped at missing MLIR.
It installed nothing and is not the eventual Clang/Nix compiler build directory.

Frontend/build inspection confirmed LLVM/MLIR 18.1.2 and SEAL 4.0.0 per README.
Both `HecateFrontend` and `SEAL_HEVM` shared libraries are needed by the stock
Python import path: `__init__.py` imports `runner.py`, which immediately loads
`libSEAL_HEVM.so`. Merely avoiding the SEAL execution backend does not remove
this import dependency. Stock `expr.py` hardcodes `$HECATE/build/lib`; isolated
build-path support must address that before tracing. Stock `config.sh` also has
unquoted paths and must not be sourced blindly from the space-containing checkout.

Installation requests specify versions, isolation, locations and download sizes.
The original LLVM-16 Nix template was not a valid compiler environment. It has
been replaced with pinned recipes; the C++ installation was subsequently approved
and has completed. See [the dependency execution record](dacapo-environment.md)
for successful native builds and diagnosed packaging/shell/ABI failures.
See [the native compiler diagnostic](native-compiler-smoke.md) for actual tiny
HEVM/CST artifacts and their confirmed ModswitchC adapter blocker.

The source-initialization step downloaded only the approved Dacapo checkout
(including its tracked benchmark data). The separately approved Nix bootstrap
is recorded below. SEAL and LLVM/MLIR are installed in the isolated Nix store.
CPU Torch wheels were subsequently approved and installed; CUDA remains uninstalled.
See [the real Python compiler run](python-compiler-smoke.md) for exact locks,
import-path repairs, Linear artifacts and the remaining adapter blocker.
No system, shell-profile or Codex configuration was changed in these steps.

The remaining approved C++ stages have been connected by
`scripts/baseline/continue_dacapo_cpp.py`. It waits for the existing LLVM build
without restarting it, rejects changed recipes/unexpected downloads, and only
then builds/checks Dacapo. After two diagnosed environment failures, the
successful durable status is
`/home/lhy/poseidon-work/results/dacapo-cpp-continuation-20260905T075623-1478.json`.
It is complete; there is no pending compiler build. The separate native C-API
diagnostic produced real compiled artifacts, but it does not advance the Hecate
Python tracing or encrypted GPU correctness gates.

### Historical first installation: Nix bootstrap

After separate user approval on 2026-09-05, downloaded the third-party maintainer's
[nix-portable v012 x86_64 launcher](https://github.com/DavHau/nix-portable/releases/tag/v012).
Its size is 68,062,412 bytes (about 65 MiB), matching release metadata. Downloaded
file SHA-256: `b409c55904c909ac3aeda3fb1253319f86a89ddd1ba31a5dec33d4a06414c72a`.
The archive integrity test passed. This checksum records the HTTPS-downloaded
file and detects subsequent replacement; it is not a publisher signature.

At the bootstrap-only checkpoint, the launcher and extracted runtime/store
occupied approximately 559 MiB under
`/home/lhy/poseidon-work/deps/nix-portable-v012`. Caches and temporary files are
under `/home/lhy/poseidon-work/cache`. No sudo or system package installation
was used. Host `/nix` and `/home/lhy/.nix-portable` remain absent, HOME remains
`/home/lhy`, and `.bashrc`/`.profile` hashes are unchanged.

`scripts/baseline/nix_portable.sh` checks the launcher checksum, selects the
existing `/usr/bin/bwrap` (0.6.1), and sets `NP_GIT=/usr/bin/git` to avoid the
upstream launcher's automatic Git package download. `NP_LOCATION` keeps the
virtual store in the approved dependency directory. The wrapper disables upstream
environment-debug dumping, serializes launcher access and sets Nix `max-jobs=1`,
`cores=2`, with bounded connection/stalled-download settings. Callers must still
use a hard overall timeout and obtain approval before dependency downloads.
The generated Nix configuration has `sandbox = true`; no fallback to a system
installation or proot was used. This dependency environment is not a security
sandbox for untrusted Agent-generated programs.

Confirmed tests:

- `nix --offline --version`: Nix 2.20.6.
- Offline metadata evaluation: x86_64-linux, bwrap, expected HOME/cache/store
  locations, virtual `/nix/store` visible inside the environment.
- Offline minimal derivation: built with the Nix sandbox enabled, without a
  nixpkgs import, fetcher or external inputs; the output contains exactly
  `poseidon-nix-bootstrap-ok` followed by a newline.
- `bash -n scripts/baseline/nix_portable.sh`: passed.

Reproduce from the source checkout in WSL:

```bash
timeout -k 3s 60s bash scripts/baseline/nix_portable.sh \
  nix eval --offline --impure --json --file scripts/baseline/nix_bootstrap_probe.nix
timeout -k 3s 120s bash scripts/baseline/nix_portable.sh \
  nix build --offline --impure --no-link --print-out-paths \
  --file scripts/baseline/nix_bootstrap_smoke.nix
```

Observed output path:
`/nix/store/cqv99103fhfhgrvd6gpzfrg8i3da5zil-poseidon-nix-bootstrap-smoke`.
The corresponding physical file was read and verified below
`/home/lhy/poseidon-work/deps/nix-portable-v012/.nix-portable/nix/store`.
Programs consuming store libraries must run within the same virtual environment;
host processes cannot assume `/nix/store` exists. See the
[v012 README](https://github.com/DavHau/nix-portable/blob/v012/README.md).

That first approval covered only the launcher and its bundled Nix bootstrap
validation. A subsequent approval covers the pinned LLVM/MLIR 18.1.2, SEAL 4.0.0
and supporting C++ dependencies, with a 512 MiB download allowance and a 30 GiB
estimated storage reservation. The [compiler dependency record](dacapo-environment.md)
tracks actual builds; Python wheels remain separate. Do not use the launcher's floating default
nixpkgs channel for the actual compiler build. Native C-API compilation now
passes. The subsequent approved Python stage also passed real Hecate tracing
and compilation; GPU evaluation and golden differential testing have not passed.
