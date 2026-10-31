"""Build an isolated driver for fixed-schedule diagnostics; reuse current pass archives."""

from pathlib import Path
import hashlib
import json
import shlex
import subprocess

ROOT = Path("/home/xuming/poseidon-manual-boot-20261026")
OLD = Path("/home/xuming/poseidon-mixed-q-20261026")
PERF = ROOT / "resnet20-optimized-20261027T055506Z/performance"
EXPERIMENT = PERF.parent / "placement-audit/target-scale-experiment"
OUT = EXPERIMENT / "scale-overhead-20261028"
OUT.mkdir(exist_ok=True)
SOURCE = ROOT / "compiler/dacapo"
BUILD = ROOT / "compiler/compiler-build"
original = (SOURCE / "tools/optimizer.cpp").read_text()
assert (
    hashlib.sha256(original.encode()).hexdigest()
    == "cf131d0f28afadcb3ca8c2d760370ee9ddb72fdaadc63ce1eddc99acf2901b03"
)
assert "native-target" not in original
addition = r"""
#include "hecate/Dialect/Earth/Transforms/Common.h"
#include "nlohmann/json.hpp"
#include <fstream>
namespace {
llvm::cl::opt<std::string> auditSpec("audit-native-spec", llvm::cl::init(""));
llvm::cl::opt<std::string> auditSchedule("audit-schedule-path", llvm::cl::init(""));
struct AuditSchedulePass : mlir::PassWrapper<AuditSchedulePass, mlir::OperationPass<mlir::func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AuditSchedulePass)
  llvm::StringRef getArgument() const final { return "audit-attach-schedule"; }
  llvm::StringRef getDescription() const final { return "Attach an external fixed schedule and diagnostic opids"; }
  void runOnOperation() override {
    auto func = getOperation();
    mlir::OpBuilder builder(func);
    auto values = hecate::earth::attachOpid(func);
    for (size_t i = 1; i < values.size(); ++i)
      if (auto *op = values[i].getDefiningOp())
        op->setAttr("audit.original_opid", builder.getI64IntegerAttr(i));
    std::ifstream input(auditSchedule);
    auto schedule = nlohmann::json::parse(input);
    llvm::SmallVector<mlir::Attribute> entries;
    for (const auto &row : schedule)
      entries.push_back(builder.getDenseI64ArrayAttr(row.get<std::vector<int64_t>>()));
    func->setAttr("btp_schedule", builder.getArrayAttr(entries));
    llvm::errs() << "audit fixed schedule: " << entries.size() << " boundaries, " << values.size() - 1 << " original values\n";
  }
};
} // namespace
"""
original = original.replace(
    "using namespace llvm;", addition + "\nusing namespace llvm;", 1
)
original = original.replace(
    "  earth::registerEarthPasses();",
    "  earth::registerEarthPasses();\n  mlir::PassRegistration<AuditSchedulePass>();",
    1,
)
original = original.replace(
    "  earth::EarthDialect::setCKKSParameters(ckks_config);",
    '  earth::EarthDialect::setCKKSParameters(ckks_config);\n  if (!auditSpec.empty()) earth::EarthDialect::setNativeScaleParameters(auditSpec, "poseidon-resnet20-manual-q50-scale40");',
    1,
)
assert original.count("PassRegistration<AuditSchedulePass>") == 1
(OUT / "audit-optimizer.cpp").write_text(original)
local = (OLD / "source-root.txt").read_text()


def mapped(value):
    return (
        value.replace(local + "/build/nix", str(BUILD))
        .replace(local, str(SOURCE))
        .replace("/nix/store/", str(OLD / "deps/nix/store") + "/")
    )


loader = OLD / "toolchain/lib/ld-linux-x86-64.so.2"
libraries = ":".join(map(str, [OLD / "toolchain/lib", OLD / "tools", PERF / "tools"]))
flags = [
    mapped(v)
    for v in shlex.split(
        json.loads((ROOT / "compiler/commands.json").read_text())[0]["command"]
    )[1:]
]
flags[flags.index("-c") + 1] = str(OUT / "audit-optimizer.cpp")
flags[flags.index("-o") + 1] = str(OUT / "audit-optimizer.o")
command = [
    str(loader),
    "--library-path",
    libraries,
    str(OLD / "tools/clang++"),
    "-nostdinc",
    "-nostdinc++",
]
for value in [
    "/nix/store/qxaq7jz61a6zkr2mq49i0zvqip2m2jj8-gcc-15.2.0/include/c++/15.2.0",
    "/nix/store/qxaq7jz61a6zkr2mq49i0zvqip2m2jj8-gcc-15.2.0/include/c++/15.2.0/x86_64-unknown-linux-gnu",
    "/nix/store/i3v1x1v3hzcmsx45q2i60g0248n40r3h-clang-wrapper-18.1.8/resource-root/include",
    "/nix/store/15h9askp4k1lx44d9871wid23j2a8ijp-glibc-2.42-61-dev/include",
]:
    command += ["-isystem", mapped(value)]
command += [
    "-I" + str(EXPERIMENT / "src/include"),
    "-I" + str(EXPERIMENT / "include"),
    *flags,
]
with (OUT / "build-compile.log").open("w") as log:
    result = subprocess.run(command, cwd=BUILD, stdout=log, stderr=subprocess.STDOUT)
assert result.returncode == 0, (OUT / "build-compile.log").read_text()[-5000:]
link = [mapped(v) for v in json.loads((OLD / "link.json").read_text())[1:]]
for i, value in enumerate(link):
    if value.endswith("tools/CMakeFiles/hecate-opt.dir/optimizer.cpp.o"):
        link[i] = str(OUT / "audit-optimizer.o")
    elif value.startswith("lib/") and (EXPERIMENT / Path(value).name).exists():
        link[i] = str(EXPERIMENT / Path(value).name)
link[link.index("-o") + 1] = str(OUT / "audit-opt")
with (OUT / "build-link.log").open("w") as log:
    result = subprocess.run(
        [
            str(loader),
            "--library-path",
            libraries,
            str(OLD / "tools/ld"),
            "-rpath-link",
            str(OLD / "toolchain/lib"),
            *link,
        ],
        cwd=BUILD,
        stdout=log,
        stderr=subprocess.STDOUT,
    )
assert result.returncode == 0, (OUT / "build-link.log").read_text()[-5000:]
wrapper = OUT / "audit-opt-wrapper"
wrapper.write_text(
    "#!/bin/sh\nexec "
    + str(loader)
    + " --library-path "
    + str(OLD / "toolchain/lib")
    + " "
    + str(OUT / "audit-opt")
    + ' "$@"\n'
)
wrapper.chmod(0o755)
(OUT / "audit-binary.sha256").write_text(
    hashlib.sha256((OUT / "audit-opt").read_bytes()).hexdigest() + "\n"
)
print("Isolated diagnostic driver ready:", wrapper, flush=True)
