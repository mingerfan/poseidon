# Dacapo local patch bundle

The parent repository keeps Dacapo pinned at
`4616402710f39df3e5f5bd7930a6c036025aaac3`.
Agent validation uses the following local fixes in addition to that upstream
commit. They are distributed as patches in the parent repository; no unpublished
submodule commit or Dacapo remote write is required.

Apply in this order to a **clean checkout of the pinned commit**, checking each
patch before applying it:

1. `dacapo-plaintext-subtraction.patch`
2. `dacapo-concat-layout.patch`
3. `dacapo-bootstrap-containers.patch`
4. `dacapo-native-function-calls.patch`
5. `dacapo-refined-function-input-types.patch`
6. `dacapo-augmented-operand-order.patch`

From the parent repository root, inspect `git submodule status third_party/dacapo`
and `git -C third_party/dacapo status --short` first. Do not overwrite a modified
submodule or reapply patches to an already patched checkout.
For each filename above, use `git -C third_party/dacapo apply --check` with its
absolute patch path, then the same command without `--check`. Stop on any failure.
Rebuild affected frontend/compiler binaries in the isolated environment afterwards.
Applying source patches alone does not rebuild or validate those binaries.

The bundle covers four modified upstream files: `tools/frontend.cpp`,
`python/hecate/hecate/expr.py`, `python/poly/poly/MPCB.py`, and
`lib/Dialect/Earth/Transforms/Common.cpp`. The last patch preserves operand order
for augmented arithmetic, particularly `x -= y`; it records an existing local
fix rather than introducing a new evaluator behavior in this publication.

The bootstrap container patch preserves frontend dataflow only. It does not add
real bootstrap support to the Agent's SEAL CPU execution chain.

## Hecate-only development SDK

The default Nix toolchain now builds a Hecate-specific distribution instead of
every LLVM/Clang/MLIR tool. After the six semantic fixes above, apply
`dacapo-minimal-mlir-link.patch`, first with `git apply --check`. This additional
patch changes only `tools/CMakeLists.txt`; it introduces an optional explicit
MLIR link set. `build_hecate_cpp.sh` selects it with
`-DHECATE_LINK_ALL_MLIR_LIBS=OFF`. Upstream's broad link set remains the CMake
default for other build entry points.

The default distribution preserves LLVM/MLIR 18.1.2 and reuses pinned Nix
GCC 13.2.0 to compile Hecate, avoiding a separate Clang build. The optional
`full` profile retains Clang 18.1.2. The SDK includes `mlir-tblgen`, the
generator Hecate actually uses, instead of the general `mlir-opt` driver.
Exact version checks use `mlir-tblgen`; actual compilation, library loading,
tracing and encrypted numeric validation remain separate required gates.
The component list lives in
`src/poseidon/tools/dacapo/hecate-toolchain-components.json`.
Use `toolchainProfile = "full"` only when a separately reviewed task needs the
broader SDK. Changing profiles changes Nix derivations and requires rebuilding;
do not copy old objects or install outputs into a new store path.

After that build patch, apply `dacapo-unused-mlir-dependencies.patch` with the
same check-before-apply procedure. It removes unused direct LLVM conversion /
EmitC links and unused broad dialect/pass includes from Hecate's two drivers
and CodeSegmentation; the optimizer includes its actual Func/Tensor/transform
declarations explicitly, and CodeSegmentation includes PatternMatch for IRRewriter.
It also declares EarthToCKKS's actual dependency on HecateCKKSCommonConversion,
so GNU ld resolves PolyTypeConverter without relying on incidental archive order.
This does not change a pass implementation or the six semantic patches.
The current in-flight SDK is retained to avoid discarding completed compilation;
its upstream Tensor dependencies still require most of the remaining libraries.

Finally apply `dacapo-seal-runtime-exceptions.patch` with the same check-before-apply
procedure. Native ARM GCC 13.2.0 exposed that the LLVM directory's `-fno-exceptions`
also reaches SEAL's throwing inline headers. This patch enables `-fexceptions`
only for `SEAL_HEVM`, on both supported Linux platforms; it does not change SEAL
algorithms, compiler passes, parameters or numerical tolerances. Reconfigure and
incrementally rebuild Hecate; the pinned LLVM/MLIR SDK does not need rebuilding.

Then apply `dacapo-gcc-qualified-casts.patch`, again checking first. GCC 13.2.0
in C++17 mode rejects four unqualified explicit-template `dyn_cast` calls in
CandidateAnalysis/ScaleManagementUnit. Qualifying the existing LLVM function
preserves the selected cast and analysis behavior; no permissive compiler flag,
language-version upgrade or pass algorithm change is used. These build fixes are
shared by x86 and ARM; the original six semantic patches and gitlink stay fixed.

Apply `dacapo-constant-output-path.patch` last. The ARM packed/native regression
exposed a reused ElideConstant pass mutating its output-prefix option, yielding
`first.cstsecond.cst` instead of `second.cst`. Keep the per-function filename local;
constant values, indexing and binary format are unchanged. The native test
`test_native_constant_export.py` runs the real pass with threading disabled:
it fails on the original binary and requires three independent CST files after
rebuilding. This fix is shared by both platforms; it is not an ARM byte-layout
change. Rebuild the affected Hecate compiler and frontend before rerunning goldens.
