# Real Hecate Python compiler baseline (2026-09-05)

## Outcome and evidence scope

Subsequent continuation: [HEVM constant contract repaired](hevm-compatibility-repair.md).
Latest rerun is `python-compiler-r2ekrtjt`, with 45/45 baseline tests passing and
an additional real-CST encode/decode diagnostic. ModswitchC remains blocked.
The earlier run records below are retained as history.

**Confirmed:** the approved CPU Python environment is installed and imports real
NumPy/Torch successfully. Handwritten add, elementwise multiply and Linear(4,2)
programs trace through the pinned Hecate frontend and compile into Earth IR,
CKKS IR and HEVM/CST. Poseidon loads all three artifact pairs, then rejects
`ModswitchC` (opcode 4) before schedule construction.

**Not verified:** encrypted execution, GPU readiness, decrypted outputs, numerical
agreement, or Agent generation. No SEAL_HEVM execution, bootstrap simulation,
CPU execution-backend substitution or cryptographic parameter change was made.
The golden programs remain **candidates**, not end-to-end validated goldens.

Latest real run:
`/home/lhy/poseidon-work/results/python-compiler-wifstqrm`.
Earlier successful trace/compile run, retained:
`/home/lhy/poseidon-work/results/python-compiler-v9s3abqg`.

## Approved Python installation

The approval covered 300 MiB of downloads and a 2 GiB venv/cache reservation.
Three bounded GitHub network rounds passed before wheel downloads. Nine exact
CPU dependency wheels total **222,195,856 bytes (211.90 MiB)**, each checked
against size and SHA-256 from official PyPI metadata / the official Torch CPU
index. No wheel download was retried. Small release metadata is additional and
remains within the ceiling. No CUDA, NVIDIA, Triton or benchmark-only packages.

- Python 3.10.14 from the existing pinned Nix store.
- NumPy 1.25.2; Torch 2.0.1+cpu (`torch.version.cuda is None`).
- filelock 3.12.2; Jinja2 3.1.2; MarkupSafe 2.1.3; mpmath 1.3.0;
  networkx 3.1; sympy 1.12; typing_extensions 4.7.1.
- pip 23.0.1 and setuptools 65.5.0 from this Python's bundled ensurepip wheels;
  no bootstrap download or system pip upgrade.
- Venv: `/home/lhy/poseidon-work/venvs/hecate-2.0.1-cpu` (observed 884 MiB).
- Cache: `/home/lhy/poseidon-work/cache/hecate-python-cpu` (212 MiB).
- Lock: `src/poseidon/tools/dacapo/python-wheels.lock.json`.

The installer uses offline pip resolution with `--require-hashes`, `--no-index`,
`--only-binary=:all:` and an explicit wheelhouse. Wheel METADATA, pip's install
report and install log are preserved in
`/home/lhy/poseidon-work/results/hecate-python-install-8sk5q99p`.
Installed package versions are checked as an exact closed set, including the
two bootstrap packages; `pip check` passed. These historical pins reproduce the
research stack, not a recommendation for serving untrusted production inputs.

### Diagnosed import failure and scoped repair

The first install completed but its actual import probe failed. NumPy's extension
could not find `libstdc++.so.6`; after exposing that library, the next probe
identified `libz.so.1`. This was an **environment / dynamic-linker** failure,
not a Hecate parser or numerical failure. Package metadata checks alone did not
detect it. The original files and failure logs were retained; no reinstallation
or additional dependency download was needed.

`dacapo-shell.nix` now supplies `HECATE_PYTHON_LIBRARY_PATH` with the already
installed, pinned GCC 13.2 runtime and zlib 1.3.1. Only Python subprocesses use
that value as `LD_LIBRARY_PATH`; the host Poseidon dump tool explicitly does
not inherit it. No system library or profile change. This avoids mixing a Nix
C++ runtime into the host executable's glibc environment.

Successful import/version/numerical CPU tensor probe:
`/home/lhy/poseidon-work/results/hecate-python-verify-o_h1z0aa/report.json`.
It records NumPy/Torch versions, CUDA=null, and `[1,2].sum() == 3.0`.
The prior zlib failure log is in `hecate-python-verify-3181fm1i/import-probe.log`.

## Handwritten model-to-DSL examples

Sources: `scripts/baseline/golden_cases/`. Each case has a trusted `model.py`
providing `build_model()`. `fixtures.json` freezes public weights, inputs/range,
seed and tolerances. `trace_golden.py` contains the manual Hecate programs;
there is **no automatic PyTorch-to-Hecate translator or Agent yet**.

The driver creates numeric `arrays.npz` files (loaded with `allow_pickle=False`),
CPU float64 references and FX graph text. Inputs are zero, fixed signed, seeded
uniform random and range boundary, four vectors per case. FX is diagnostic,
not yet a general supported-operator validator.

| Case | Python trace | Compile | Poseidon adapter | Decrypted comparison |
|---|---|---|---|---|
| Add `x+x` | Passed | Passed | ModswitchC rejected | Not run |
| Elementwise `x*w` | Passed | Passed | ModswitchC rejected | Not run |
| Linear(4,2), including bias | Passed | Passed | ModswitchC rejected | Not run |

The Linear uses an input ciphertext whose slots repeat `[x0,x1,x2,x3]` and
two output ciphertexts, one per row. With periodic weights, each row computes:

1. `p = x * weight[row]`.
2. `pairs = p + p.rotate(1)`.
3. `total = pairs + pairs.rotate(2)`.
4. Add the public scalar bias; read slot zero of that output ciphertext.

For the four-slot periodic layout, the two rotate-and-add stages sum all four
weighted elements. It is not an elementwise substitute for Linear. This is a
hand-derived packing argument; the backend has **not** validated that contract.
For signed input `[0.5,-1,0.25,-0.75]`, independent `torch.nn.Linear` produces
`[0.5,0.75]`; zero input produces `[0.125,-0.25]`. These are plaintext references,
not decrypted measurements.

Linear's actual HEVM contains 23 instructions: 6 Alloc, 4 Encode, 4 RotateC,
1 ModswitchC, 4 AddCC, 2 AddCP, 2 MulCP. CST holds the exact two weight rows and
two scalar biases. No BootstrapC or MulCC is required for this model.

## Compiler/runtime gate

The driver uses the unchanged upstream `config.json`, `--eva`, `--verify-each`
and debug artifact emission. This compiler diagnostic profile has polynomial
degree 131072, input scale exponent 20 and input level 16. The Linear CKKS IR
drops 15 levels, computes at level 1 and declares output scale exponent 40.
These are **compiler artifact facts**, not validated GPU cryptographic settings.
Do not run this large profile on the 8 GiB GPU or reduce parameters to force it.

`ModswitchC` rejection is a **Poseidon adapter** blocker, not network/frontend
failure. The existing checks correctly prevent schedule construction. The driver
returns nonzero (inner 2 for adapter-blocked; nix-shell wrapper surfaces 1).
Each artifact report still records loading=true, schedule-built=false,
readiness=false; GPU and communication preflight stages did not run.

Additional unresolved contracts must remain visible:

- At the original run, Dacapo repeated short CST vectors but Poseidon zero-padded
  them. The subsequent HEVM-specific repair now matches Dacapo and preserves the
  general encoder; real Linear CST encode/decode passed separately. This still
  does not establish encrypted execution or runtime parameter compatibility.
- Existing `GpuEvaluator::drop_modulus` is only a candidate primitive for
  ModswitchC. Q/P, parameter IDs, level numbering, NTT form and scale preservation
  require evidence before mapping it. Never map it to rescale merely by analogy.
- MulCC/relinearization alignment remains a later polynomial-MLP gate.

MAE, max absolute error, nonzero-reference relative error and cosine similarity
are deliberately **null**, comparison status `not_run`. Frozen acceptance stays
`abs(actual-reference) <= 1e-5 + 1e-4*abs(reference)`; no threshold was relaxed.

## Reproduction and tests

From PowerShell (no installation or network needed for these checks):

```powershell
wsl.exe -d Ubuntu-22.04 --cd '/mnt/d/Code Space/Poseidon' -- env PYTHONDONTWRITEBYTECODE=1 python3 scripts/baseline/hecate_python_env.py --verify-existing
wsl.exe -d Ubuntu-22.04 --cd '/mnt/d/Code Space/Poseidon' -- env PYTHONDONTWRITEBYTECODE=1 python3 scripts/baseline/python_compiler_smoke.py
```

The second command creates a new results directory, saves all commands and
diagnostics, and deliberately fails the unsupported adapter gate. The derived
`build-dacapo/hecate-python-root/build` symlink points to the successful native
build; HECATE and PYTHONPATH are command-scoped. Upstream `config.sh` is not
sourced, the Dacapo submodule remains clean, and no files are built in it.

Run the validation suite in WSL:

```bash
timeout -k 3s 5m env PYTHONDONTWRITEBYTECODE=1 \
  POSEIDON_NATIVE_COMPILER_RESULTS=/home/lhy/poseidon-work/results/native-compiler-k73nx6wz \
  POSEIDON_PYTHON_COMPILER_RESULTS=/home/lhy/poseidon-work/results/python-compiler-wifstqrm \
  python3 -m unittest discover -s scripts/baseline -p 'test_*.py' -v
```

The artifact tests check actual hashes, frozen references, exact CST weights,
real rotations and the current adapter failure. Passing those tests does not
mean the adapter or encrypted computation passed. No CUDA install, sudo, system
configuration change, commit, push or PR is authorized by this Python stage.

Latest run: **44 tests passed, 0 failed, 0 skipped**. Log:
`/home/lhy/poseidon-work/results/python-compiler-wifstqrm/validation-tests.log`.
The real prerequisite gate also passed all checks; its JSON intentionally does
not claim compiler execution (a different gate).

## Local changes and Git handoff

Branch remains `feat/agent-dsl-correctness`, HEAD
`4995e7cadedf2bfb9104658b5638662ecf6a1d0a`, tracking
`origin/gs/feat-application`. Dacapo remains clean at the pinned gitlink.
Sparse checkout and the Zone.Identifier skip-worktree bit are unchanged.

This stage adds the wheel lock, Python install/verify driver, real compiler
driver, fixtures, three PyTorch model references, handwritten DSL and regression
tests. It updates the Nix shell's Python library-path variable, Python-lock state
and baseline documentation. No existing C++ runtime behavior was changed here;
the user's earlier CMake/plaintext-test modifications are preserved.

Current tracked diff is three files (102 insertions, 20 deletions), including
earlier work; this does **not** count the untracked docs/scripts/lock files.
`git diff --check` passes, apart from an informational existing CRLF warning.
Everything remains uncommitted as requested. Suggested future local commits,
only with separate approval: (1) dependency locks/isolation; (2) handwritten
model/compiler diagnostics/tests. Do not mix the existing CPU C++ changes into
either without reviewing their own baseline commit boundary.

## Next minimal milestone

The constant-layout contract is now repaired. Resolve the ModswitchC and concrete
runtime parameter contracts, then verify an approved GPU execution profile.
CUDA installation needs its own version/size approval. Only
after real decrypt-and-compare passes does Linear become an accepted golden and
the Agent phase become eligible. The 48-case generation study stays deferred.
