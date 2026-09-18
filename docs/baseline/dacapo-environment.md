# Pinned Dacapo compiler environment (2026-09-05)

Status: approved C++ dependencies and Dacapo native build **completed** on
2026-09-05. Native C-API add/mul_plain diagnostics emitted real Earth/CKKS and
HEVM/CST; Poseidon then rejected unsupported ModswitchC. See
[the native compiler runbook](native-compiler-smoke.md).
Python DSL tracing, GPU execution and numerical correctness remain unvalidated.

## Version evidence and choices

The initialized Dacapo commit is
`4616402710f39df3e5f5bd7930a6c036025aaac3`. Its README explicitly selects
LLVM/MLIR 18.1.2 and SEAL 4.0.0. Its runtime CMake links `SEAL::seal`, and the
stock Hecate Python import loads both frontend and SEAL_HEVM shared libraries.

The former `src/poseidon/tools/dacapo/dacapo-shell.nix` selected LLVM 16,
used a floating `<nixpkgs>`, silently omitted missing packages, and selected
an in-source build directory. The replacement uses a content-pinned package
snapshot and explicit toolchain recipes; it does not initialize the submodule.

| Component | Planned version | Role / evidence |
|---|---|---|
| nixpkgs source | `cfd6b5fc90b15709b780a5a1619695a88505a176` | Already bundled in approved nix-portable v012; NAR hash verified against its maintainer's lock file |
| LLVM, MLIR, Clang | 18.1.2 | One official monorepo source build; Dacapo README |
| Bootstrap GCC | 13.2.0 | Pinned Nix stdenv, builds Clang itself; Dacapo will use Clang 18.1.2 |
| CMake / Ninja | 3.28.3 / 1.11.1 | Pinned snapshot; satisfies Dacapo requirements |
| SEAL | 4.0.0 | Static PIC library for `SEAL::seal` in shared SEAL_HEVM |
| Microsoft GSL | 3.1.0 | SEAL's ExternalMSGSL.cmake pins this release; preserves GSL support |
| Zlib / Zstd | 1.3.1 / 1.5.5 | Keep SEAL compression support; Zstd built with its required static target |
| Python interpreter | 3.10.14 | Pinned shell interpreter, no tracing wheels yet |
| NumPy / Torch CPU | 1.25.2 / 2.0.1+cpu | Separate venv/wheel-lock stage, not included in this installation |

The snapshot's own LLVM is **18.1.3**, not 18.1.2. Its
[package history](https://github.com/NixOS/nixpkgs/commit/ee4806b35f9646263366bdc238b82c4326529457)
updates 18.1.1 directly to 18.1.3. Thus simply selecting `llvmPackages_18` does
not satisfy this baseline. The custom recipe builds only X86 backend targets,
Clang and MLIR; it does not introduce a second LLVM/MLIR version. MLIR's own
Python bindings are not Hecate's ctypes frontend and are not needed here.

SEAL 4.0.0 calls `find_package(Microsoft.GSL 3 CONFIG)` when dependency fetching
is disabled. The snapshot's GSL 4.0.0 exports SameMajorVersion compatibility,
so it cannot satisfy that call. The recipe explicitly supplies 3.1.0 and retains
its package test patch/checks. SEAL automatic dependency fetching is disabled;
GSL, compression and transparent-ciphertext exceptions remain enabled.
These are dependency/build choices, not changes to CKKS parameters or bootstrap.

## Content pins and provenance

`src/poseidon/tools/dacapo/dependency-lock.json` records exact hashes and sources.

- Bundled nixpkgs NAR hash:
  `sha256-WKm9CvgCldeIVvRz87iOMi8CFVB1apJlkUT4GGvA0iM=`.
  Independently hashed the already extracted tree and compared with
  [nix-portable's v012 lock](https://github.com/DavHau/nix-portable/blob/v012/flake.lock).
  The recipe imports a content-addressed copy, not the floating channel link.
- LLVM archive: official release asset, 132,060,436 bytes. Expected SHA-256
  comes from the [FreeBSD LLVM port's recorded distribution checksum](https://github.com/freebsd/freebsd-ports/commit/935a2fb3225e597d204b459b92a05466a0e8dfab).
  Size agrees with the LLVM GitHub release API. On 2026-09-05 the downloaded
  archive passed both Nix's fixed-output check and an independent `sha256sum`:
  `51073febd91d1f2c3b411d022695744bda322647e76e0b4eb1918229210c48d5`.
- SEAL archive: 612,928 bytes according to the
  [FreeBSD SEAL 4.0.0 port](https://github.com/freebsd/freebsd-ports/blob/2db69f114426f6c90a2ecba5253c38a5248b4d88/security/seal/distinfo).
  Important: `4.0.0` currently resolves to a compatibility **branch**, whose HEAD
  matches tag `v4.0.0` at `a0fc0b732f44fa5242593ab488c8b2b3076a5f76`.
  The archive checksum locks its contents despite the movable URL. Any mismatch
  must stop the build, never trigger automatic checksum replacement.
  The downloaded archive passed both checks on 2026-09-05, with SHA-256
  `616653498ba8f3e0cd23abef1d451c6e161a63bd88922f43de4b3595348b5c7e`.
- GSL source/tree hash and small test-discovery patch are pinned using the
  [Nixpkgs GSL 3.1.0 recipe](https://github.com/NixOS/nixpkgs/blob/2cceb4b94829fbec2573f3f7293066cdc7fa3dd6/pkgs/development/libraries/microsoft_gsl/default.nix).

These are upstream/packager checksum records, not a claim of locally verified
publisher signatures. This historical environment reproduces the research stack;
it is not a currently supported production deployment recommendation.

## Installation request and resource budget

The latest **dry-run only** reports:

- 148 cached store paths: **232.34 MiB** download, **969.38 MiB** unpacked.
- 9 uncached derivations, including the LLVM/SEAL source fetchers, LLVM build,
  Clang wrapper/stdenv, GSL, static-enabled Zstd, SEAL and shell.
- Add LLVM and SEAL archives: about **126.53 MiB**; total payload estimate
  **358.87 MiB**. Source archives are not counted in the binary-cache subtotal.
- Proposed download allowance: **512 MiB**, excluding already installed Nix.
  If cache availability changes the plan above this allowance or adds substantial
  source builds, stop and reassess before realizing the expanded plan.
- Proposed build/storage reservation: **30 GiB**, an estimate, not a measured
  peak. Current WSL has approximately 198 GiB free, 7.6 GiB RAM and 2 GiB swap.
  LLVM compilation may take hours; no completion-time guarantee.

Install/build only within the existing nix-portable virtual store under
`/home/lhy/poseidon-work/deps/nix-portable-v012`. Cache/temp artifacts belong
under `/home/lhy/poseidon-work/cache`, logs under `/home/lhy/poseidon-work/results`.
The successful Dacapo build directory is
`/home/lhy/poseidon-work/build-dacapo/hecate-18.1.2-nix`, not the checkout.
The earlier `hecate-18.1.2` directory is retained for failed-link diagnosis.

Use the existing wrapper: Nix maximum simultaneous builds 1, cores 2; LLVM
compile jobs 2 and link jobs 1. Preserve `sandbox=true`; no sudo, system package
installation, HOME/profile edit or CUDA/display driver installation is included.
The Nix library environment must be used for Dacapo build and execution; host
processes cannot resolve virtual `/nix/store` libraries by themselves.

**User approval was received before realization.** This approval covers only the
C++ toolchain, SEAL and the listed supporting packages/interpreter. PyTorch/NumPy
wheels, CUDA, GPU execution and external LLM calls are separate gates.

## Verification and reproduction

Successful checks so far:

- Three rounds of bounded GitHub HTTPS and both repository `ls-remote` probes.
- Offline Nix recipe evaluation and shell derivation instantiation, with
  `allow-import-from-derivation=false` to prevent evaluation-time package builds.
- 20 Python tests: 8 environment-gate, 9 lock/offline-Nix and 3 mocked native
  library-gate tests; no skips. Missing sources fail; unrelated candidate sources
  cannot replace the expected content-pinned snapshot. Mocked CDLL tests do not
  establish actual native library loading.
- Nix dry-run, which queries cache metadata but does not fetch package payloads.
- Bash syntax check of the planning script.

An initial negative test expected an unrelated candidate path always to fail.
Nix instead reused the already cached object with the expected content hash;
that is safe cache reuse, not acceptance of the unrelated tree. The corrected
test requires either a hash rejection or exactly identical pinned metadata.
Some initial metadata queries/file-location guesses failed; bounded retries
using `curl --get --data...` and actual file enumeration resolved those inspection
issues. They are not compiler build failures or a recurring GitHub outage.

From Windows PowerShell:

```powershell
wsl.exe -d Ubuntu-22.04 --cd '/mnt/d/Code Space/Poseidon' -- bash scripts/baseline/plan_dacapo_environment.sh
wsl.exe -d Ubuntu-22.04 --cd '/mnt/d/Code Space/Poseidon' -- env PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s scripts/baseline -p 'test_*.py' -v
```

Reports: `/home/lhy/poseidon-work/results/dacapo-dependency-plan-20260905T051902-757.json`
and the matching `.log`. This script performs only evaluation and dry-run.
SEAL, LLVM/MLIR/Clang, Dacapo native compilation/linkage and native C-API
Earth/CKKS/HEVM emission have passed. Hecate Python import/tracing, encrypted
HEVM execution and a golden differential test have **not** passed.
Before tracing, the hardcoded `$HECATE/build/lib` import
paths and the Python wheel lock must also be handled without modifying the gitlink.

## Approved installation execution

On 2026-09-05, rechecked the parent/submodule commits, three network rounds and
the dry-run budget. The dependency set and 358.87 MiB estimate are unchanged.
Source fetches now have explicit 10-second connect and 600-second total curl
timeouts. The staged runner also has overall hard timeouts and retains failed
build directories for diagnosis; it never installs Python wheels or CUDA.

Commands from the approved WSL checkout, after reviewing the dry-run:

```bash
bash scripts/baseline/build_dacapo_dependencies.sh --approved sources
bash scripts/baseline/build_dacapo_dependencies.sh --approved seal
bash scripts/baseline/build_dacapo_dependencies.sh --approved toolchain
bash scripts/baseline/build_dacapo_dependencies.sh --approved shell
```

Each stage writes a separate `dacapo-build-*.log` under the results directory.
The first invocation log is
`/home/lhy/poseidon-work/results/dacapo-build-sources-20260905T052804-1123.log`.
Subsequent stages are not successful merely because their commands are listed.

### Observed build results

- Source stage: exit 0; LLVM 132,060,436 bytes and SEAL 612,928 bytes, exact
  hashes verified above. LLVM's archive took approximately 5m52s to download;
  this was a progressing bounded transfer, not a hung Git process.
- Microsoft GSL 3.1.0: source build and **16/16 CTest tests passed**.
- Static-enabled Zstd 1.5.5: source build and its package-selected
  **1/1 `playTests` CTest passed** (12.42 seconds).
- Initial SEAL attempt: compilation succeeded, then Nix's pkg-config path
  validation failed. `seal.pc.in` prepends `${prefix}` to
  `SEAL_INCLUDES_INSTALL_DIR`; Nix's default absolute `CMAKE_INSTALL_INCLUDEDIR`
  produced `${prefix}//nix/store/...`. Failure log:
  `dacapo-build-seal-20260905T053504-465.log` under the results directory.
- Scoped repair: pass `-DCMAKE_INSTALL_INCLUDEDIR=include` in the dependency
  recipe, retaining Nix's validator. Add an installed `pkg-config` includedir
  equality check and a header existence check. No tracked upstream source,
  cryptographic parameter, algorithm or validation guard was changed.
- SEAL retry: **exit 0**, all 40 build steps, version/CMake export/header and
  pkg-config path checks passed. Output inside Nix:
  `/nix/store/i44yjj58c5lb117zrcwwd919kr6j3jpz-poseidon-seal-4.0.0`.
  Log: `dacapo-build-seal-20260905T054433-1329.log`. This does not claim SEAL's
  full cryptographic test suite: upstream `SEAL_BUILD_TESTS` remained OFF.
- LLVM/MLIR/Clang: **exit 0**, completed around 07:38 UTC on 2026-09-05. Log:
  `dacapo-build-toolchain-20260905T054514-2184.log`. Do not start a duplicate.
  Nix used max-jobs=1/cores=2 and LLVM compile/link pools 2/1; installed
  version/CMake-export checks passed.

The failed SEAL build tree is retained under the approved cache for diagnosis.
No source checkout files were removed. No Python wheels or CUDA were installed.

`scripts/baseline/build_hecate_cpp.sh` is the later Dacapo C++ build driver. Run
it inside the realized pinned Nix shell; it verifies the gitlink checkout and
tool versions, uses the native build directory, and builds only `hecate-opt`,
`HecateFrontend` and `SEAL_HEVM` with `--parallel 2`. The companion
`check_hecate_libraries.py` loads both shared libraries and resolves symbols
without calling a VM or importing the Hecate Python package. These tests do not
establish tracing, FHE execution or numerical correctness.

## Continuation while LLVM is building

On 2026-09-05 the existing toolchain build was confirmed alive; at the 06:08 UTC
checkpoint it had reached step 1062/5791. Two `cc1plus` processes were observed,
with approximately 5.7 GiB available RAM and zero swap use. Build-step fraction
is not an elapsed-time estimate. Compiler warnings are present; no fatal build
failure was observed at this checkpoint. Completion is still pending.

`scripts/baseline/continue_dacapo_cpp.py` now connects the remaining **already
approved C++ stages**, without restarting the running LLVM derivation:

1. Wait for the Nix launcher's advisory lock, with a 12-hour bound. A second
   continuation is rejected by its own nonblocking lock.
2. Recheck parent/submodule commits, branch, clean submodule and SHA-256s of the
   recipes/drivers. A user edit while waiting stops continuation, not overwrites it.
3. Evaluate pinned outputs offline and require Nix store validity for LLVM,
   its Clang wrapper and SEAL. A failed LLVM build stops here; no automatic rebuild.
4. Dry-run the remaining shell: reject extra source builds or more than 112 MiB
   of remaining downloads (within the original total 512 MiB approval). Only
   the small stdenv/shell derivations may still need building.
5. Realize the pinned shell, enter it with substitution disabled, then build
   `hecate-opt`, `HecateFrontend` and `SEAL_HEVM`. Dacapo's Ninja compile/link pools
   are also limited to 2/1. Run the existing dialect/library-load checks.

Any failing stage stops the sequence and records a structured error with command,
timeout, exit code and log location. JSON status is atomically replaced. Success
would mean C++ build/loading only; tracing and encrypted-execution fields stay
false. Neither Python wheels, CUDA, an LLM service, nor a VM execution is invoked.
This is one finite build process, not a recurring scheduled automation; it does
not install a service or change shell/system/Codex configuration.

Started once from Windows PowerShell:

```powershell
wsl.exe -d Ubuntu-22.04 --cd '/mnt/d/Code Space/Poseidon' -- timeout -k 10s 17h env PYTHONPYCACHEPREFIX=/home/lhy/poseidon-work/cache/python-bytecode python3 -u scripts/baseline/continue_dacapo_cpp.py --approved
```

Live status:
`/home/lhy/poseidon-work/results/dacapo-cpp-continuation-20260905T060701-13481.json`.
At that checkpoint it said `wait_existing_build`; it subsequently failed at
shell entry, as explained below. This old manifest is **not** a native-build pass.
The matching per-stage `.log` files contain stderr and `.stdout` files contain
stdout. Do not edit the pinned drivers/recipes or start a duplicate while this
sequence is running. If WSL/the process stops, inspect status and store validity
before resuming; the script does not promise to survive shutdown or power loss.

Validation in this continuation turn:

- Three bounded network rounds passed again.
- **22/22 tests passed**: 8 environment-gate, 3 mocked library-gate and 11 new
  orchestration/resource-limit tests. Log:
  `/home/lhy/poseidon-work/results/cpp-continuation-tests-20260905T0607.log`.
- A real duplicate-launch attempt returned exit 1 with
  `A C++ continuation is already running; not starting another` as intended.
- Python byte-compilation and the edited Bash driver's syntax check passed.
- The 9 offline-Nix tests passed in the preceding turn; they were intentionally
  not rerun while the active compiler build holds the launcher lock.

## Completed native build and diagnosed failures

1. **Implicit shell dependency (environment layer).** `nix-shell` implicitly
   requested bashInteractive from its default package lookup even though the
   declared shell derivation had already been realized. Disabling substitution
   alone did NOT prevent source builds/downloads. It planned 49 derivations,
   downloaded Bash/support sources into the isolated store (Bash alone about
   10.4 MiB), then failed at a support patch's mirrors (429; earlier 502/504).
   The stop attempt found the process had already exited. No system installation
   occurred; unexpected fetched source cache files are retained, not silently
   deleted. Exact extra transfer bytes were not separately metered.
   Corrected entry sets `NIX_BUILD_SHELL` to the pinned existing Bash and disables
   substitution, local Nix builds (`--max-jobs 0`) and remote builders. An actual
   shell-entry regression verifies this with no implicit builds/downloads.
   These Nix limits do not disable CMake/Ninja builds inside the ready shell.
2. **Compiler ABI mismatch (environment/link layer).** stdenv's setup hook reset
   CC/CXX to bare `clang`/`clang++`; the raw toolchain on PATH won over its wrapper.
   CMake recorded host GCC 11/system libc paths. At link step 75/77, LLVMSupport
   required `__isoc23_strtol` and `arc4random`, which host libc could not satisfy.
   Version-only checks could not detect this. The shell hook now reasserts
   absolute wrapper paths after setup and puts the wrapper first on PATH.
   A real CMake LLVMSupport consumer probe was added before the Dacapo build.
   A new native build directory avoids reusing wrong-ABI objects; the old one
   remains untouched for diagnosis. No Dacapo source changes were needed.

Successful retry manifest:
`/home/lhy/poseidon-work/results/dacapo-cpp-continuation-20260905T075623-1478.json`.
Its status is `complete`, `hecate_cpp_validated=true`, with tracing and encrypted
execution explicitly false. Logs: `hecate-cpp-20260905T075634-2158.*` in results.
The native probe printed `LLVM=18.1.2 glibc=2.39`; all 77 Dacapo build steps,
dialect enumeration and both library-loading/symbol checks passed. Upstream
unused-variable and other compiler warnings remain; this is not a clean-warning
or full upstream test-suite claim.

Observed footprint after builds: nix-portable dependency tree about 5.1 GiB,
successful Dacapo build about 37 MiB, WSL free space about 194 GiB. No LLVM rebuild,
sudo, CUDA, Python wheel installation, profile edit, branch change or remote write
was needed for these repairs.

## Subsequent approved Python stage

The CPU Python wheels were subsequently approved, hash-locked and installed in
the isolated Nix-Python venv. Actual imports initially exposed missing dynamic
search paths for already-installed libstdc++/zlib; only the Python subprocess
environment was adjusted, with no additional dependency download. See
[the Python compiler execution record](python-compiler-smoke.md) for package
versions, failure/success evidence and real Linear trace/compile results.
Encrypted execution remains blocked by the Poseidon adapter, not Python imports.
