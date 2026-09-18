# Poseidon correctness baseline — 2026-09-05

Follow-up implementation and adapter/preflight results:
[Compiler-chain status](compiler-chain-status.md).

## Scope and verified checkout

- Source: `D:\Code Space\Poseidon`, WSL `/mnt/d/Code Space/Poseidon`.
  Spaces are retained. Windows-native Codex calls Ubuntu-22.04 for builds.
- Branch: `feat/agent-dsl-correctness`; upstream: `origin/gs/feat-application`.
- Base HEAD: `4995e7cadedf2bfb9104658b5638662ecf6a1d0a`.
- Initially clean. Sparse checkout remains enabled; the tracked
  `src/poseidon/Spec/double_hoist_gpu_bootstrap_spec.md:Zone.Identifier`
  retains skip-worktree. No branch changes, commits, or remote writes.
- Read `.agents/AGENTS.md` and `.agents/mgpu_static_schedule_plan.md`.
  The user's no-commit/no-push rule overrides the notes' push instruction.
- Dacapo gitlink: `4616402710f39df3e5f5bd7930a6c036025aaac3`, not initialized.
  `.gitmodules` URL: `git@github.com:corelab-src/dacapo.git`.

## Phase 0 environment evidence

WSL reports mirrored networking, 7.6 GiB RAM and 2 GiB swap. Three consecutive
rounds of HTTPS HEAD plus Poseidon and Dacapo `git ls-remote ... HEAD` passed.
Each curl used a 20-second timeout and each remote Git command a 30-second
timeout, with a 3-second kill grace. This is current reachability evidence,
not a promise of permanent network availability.

| Dependency | Observed state / version evidence | Required stage |
|---|---|---|
| GCC/G++ | GCC 11.4.0; successful C++ build | CPU |
| CMake | 3.22.1 installed; Poseidon minimum 3.12 | CPU/compiler |
| Python | 3.10.12 installed; Dacapo README >=3.10 | Compiler |
| GMP | 6.3.0 isolated prefix present; runtime linkage verified by `ldd` | CPU |
| zlib | Development package 1.2.11 installed and found by CMake | CPU serialization |
| Zstd / Microsoft GSL | Optional integrations explicitly OFF; no auto-download | Not required by this smoke |
| Ninja, Clang, Nix | Not found on WSL PATH | Dacapo isolated environment |
| LLVM/MLIR | Not provisioned; pinned Dacapo README says 18.1.2 | Compiler |
| SEAL | Not provisioned; README says 4.0.0; CMake requires SEAL 4.0 | Dacapo build/runtime dependency |
| CUDA Toolkit | `nvcc` not found; not needed for CPU | GPU only |

Version evidence for the fixed Dacapo revision was read without initializing it:

- [README](https://github.com/corelab-src/dacapo/blob/4616402710f39df3e5f5bd7930a6c036025aaac3/README.md)
- [CMake](https://github.com/corelab-src/dacapo/blob/4616402710f39df3e5f5bd7930a6c036025aaac3/CMakeLists.txt)

Confirmed mismatches requiring resolution before installation:

1. `src/poseidon/tools/dacapo/dacapo-shell.nix` prefers LLVM 16 and falls back
   to a generic package. It imports unpinned nixpkgs. This does not implement
   Dacapo's documented LLVM/MLIR 18.1.2 requirement.
2. That template sets builds under the submodule; use an explicit external
   build directory under `/home/lhy/poseidon-work/build-dacapo` instead, and
   inspect helpers that hardcode `$DACAPO_ROOT/build/bin/hecate-opt`.
3. GPU paths have differing architecture defaults: ResNet20 hardcodes 70;
   optional mgpu GPU objects default to 75. Neither is evidence of the
   appropriate Toolkit configuration for this machine. Do not install CUDA yet.

## Phase 1 CPU test

From Windows PowerShell:

```powershell
wsl.exe -d Ubuntu-22.04 --cd '/mnt/d/Code Space/Poseidon' -- bash scripts/baseline/run_cpu_ckks.sh
```

The script uses the existing isolated GMP prefix, builds only
`test_ckks_deterministic` with `-j2`, and writes a timestamped log beneath
`/home/lhy/poseidon-work/results`. No installation or downloads occur.
The new build directory is
`/home/lhy/poseidon-work/build-poseidon/agent-dsl-cpu`; the old parent build
references the abandoned no-space source path and is deliberately preserved.

Test semantics:

- Existing default CKKS parameter constructor: N=16384, requested `tc128`;
  8192 slots; scale 2^48. Prints actual Q/P primes. No parameter reduction.
- Fixed vectors: `x[i] = ((i % 17) - 8) / 16`,
  `w[i] = ((i % 7) + 1) / 8`, with signed subtraction.
- Roundtrip, ciphertext addition `x+x`, and ciphertext/plaintext multiplication
  followed by rescale `x*w`. Reference values derive from original inputs,
  not from decrypted intermediates.
- All slots checked for finite output, matching count and maximum complex
  absolute error <=1e-6. MAE and relative errors are reported; first eight
  expected/actual slots are printed. Encryption randomness is intentionally
  not fixed. Numeric results vary slightly across runs.
- Relative error with a 1e-12 denominator floor is large at true-zero outputs;
  nonzero-reference relative error is reported separately. Absolute error is
  the acceptance criterion, not a misleading relative score near zero.

First successful run (`cpu-ckks-20260905T034219-1930.log`):

| Operation | MAE | Maximum absolute error |
|---|---:|---:|
| Roundtrip | 1.213e-10 | 7.867e-10 |
| Add | 2.425e-10 | 1.573e-9 |
| Multiply plaintext + rescale | 6.028e-11 | 6.805e-10 |

This is an executed Poseidon CPU primitive baseline, **not** a compiled DSL
test, matrix Linear test, GPU test, bootstrap test, or cryptographic audit.

## Diagnostics encountered

- An instruction search accidentally traversed sibling projects and hit their
  cache permissions; subsequent discovery was scoped to this repository.
- Direct PowerShell forwarding of a CMake argument malformed the path; a
  piped command also carried CRLF into `-j2`. The checked-in Bash runner
  avoids both transport issues and has successfully executed.
- Existing CMake emits a nonfatal GNUInstallDirs warning because it is
  included before `project()`. Build and execution succeeded; no unrelated
  CMake cleanup was performed.

## Remaining phases and next gate

1. Obtain approval to initialize only the fixed Dacapo submodule using
   command-scoped SSH-to-HTTPS rewrite, hard timeout and no `.gitmodules` edit.
   Source stays under `third_party/dacapo`. No recursive dependency downloads.
2. Inspect the fixed frontend, Python requirements and runtime build targets;
   then prepare a pinned Nix dependency solution for LLVM/MLIR 18.1.2 and
   SEAL 4.0.0. Obtain installation/download approval separately. No system-wide
   compiler mixing or blanket package installation.
3. Trace the smallest supported graph, retain Earth/CKKS IR and HEVM/CST, then
   run Poseidon adapter parsing, opcode support, schedule verification and
   preflight before attempting ciphertext execution.
4. Golden-model milestone: a fixed 4-input/2-output affine Linear, followed by
   a small polynomial MLP only after support is confirmed. Save full weights,
   input, plaintext output, handwritten frontend program, artifacts and every
   decrypted output. Compare MAE/max absolute/nonzero relative error; add cosine
   similarity where meaningful. This is a proposed milestone, not yet implemented.
5. Specify actual frontend/IR semantics before Agent synthesis. The README
   confirms Python tracing to Earth IR, but a complete operator/type/packing
   specification still requires frontend source inspection.

The Dacapo README explicitly describes SEAL bootstrap as decrypt-and-encrypt.
Do not use that path to claim real FHE execution. The Poseidon integration plan
also marks lazy rescale/runtime gaps and unsupported ModswitchC/UpscaleC;
successful parsing/preflight does not close those execution gaps.

No Agent, full ResNet, HPU or multi-GPU execution was attempted. Existing single
8-GiB GPU hardware limits remain a planning boundary, not a performance result.
