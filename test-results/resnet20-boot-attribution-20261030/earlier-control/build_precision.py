"""Earlier compiler control, stopped when the user selected Python-only exploration."""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess

ROOT = Path("/home/xuming/poseidon-manual-boot-20261026")
OLD = Path("/home/xuming/poseidon-mixed-q-20261026")
PERF = ROOT / "resnet20-optimized-20261027T055506Z/performance"
EXPERIMENT = PERF.parent / "placement-audit/target-scale-experiment"
OUT = EXPERIMENT / "boot-attribution-20261030"
OUT.mkdir(exist_ok=True)
BUILD = ROOT / "compiler/compiler-build"
SOURCE = ROOT / "compiler/dacapo"
assert (
    hashlib.sha256((EXPERIMENT / "hecate-opt").read_bytes()).hexdigest()
    == "395e6fb9178bb673a197f51de79c2964da7ddb0fc76a7f0dbea7c2b9d7eb7060"
)
header_path = "include/hecate/Support/BootstrapInputAlignment.h"
header = (EXPERIMENT / "src" / header_path).read_text()
assert header.count("if (plain < 30)") == 1
header = header.replace(
    "namespace hecate {", "namespace hecate {\ndouble auditBootstrapEncodingFloor();", 1
)
header = header.replace("if (plain < 30)", "if (plain < auditBootstrapEncodingFloor())")
destination = OUT / header_path
destination.parent.mkdir(parents=True, exist_ok=True)
destination.write_text(header)
paths = [
    ("lib/Dialect/Earth/Transforms/Common.cpp", "libEarthTransforms.a"),
    (
        "lib/Dialect/CKKS/Transforms/MaterializePhysicalLevels.cpp",
        "libCKKSTransforms.a",
    ),
]
for relative, _ in paths:
    source = (EXPERIMENT / "src" / relative).read_text()
    if relative.endswith("Common.cpp"):
        source = source.replace(
            "#include <cmath>",
            """#include <cmath>
#include "llvm/Support/CommandLine.h"
namespace {
llvm::cl::opt<double> auditBootEncodingFloor(
    "audit-boot-min-plain-scale", llvm::cl::init(30.0));
}
double hecate::auditBootstrapEncodingFloor() {
  if (!std::isfinite(auditBootEncodingFloor) || auditBootEncodingFloor < 0)
    llvm::report_fatal_error("invalid audit Boot encoding floor");
  return auditBootEncodingFloor;
}""",
            1,
        )
    (OUT / Path(relative).name).write_text(source)

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
compiler = [
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
    compiler += ["-isystem", mapped(value)]
compiler += [
    "-I" + str(OUT / "include"),
    "-I" + str(EXPERIMENT / "src/include"),
    "-I" + str(EXPERIMENT / "include"),
]


def compile_one(entry):
    name = Path(entry[0]).name
    args = flags.copy()
    args[args.index("-c") + 1] = str(OUT / name)
    args[args.index("-o") + 1] = str(OUT / (name + ".o"))
    with (OUT / (name + ".build.log")).open("w") as log:
        result = subprocess.run(
            [*compiler, *args], cwd=BUILD, stdout=log, stderr=subprocess.STDOUT
        )
    assert result.returncode == 0, (OUT / (name + ".build.log")).read_text()[-5000:]


with ThreadPoolExecutor(max_workers=2) as pool:
    list(pool.map(compile_one, paths))
for source, archive in paths:
    shutil.copyfile(EXPERIMENT / archive, OUT / archive)
    subprocess.run(
        ["ar", "rcs", str(OUT / archive), str(OUT / (Path(source).name + ".o"))],
        check=True,
    )
link = [mapped(v) for v in json.loads((OLD / "link.json").read_text())[1:]]
for i, value in enumerate(link):
    if value.startswith("lib/") and (EXPERIMENT / Path(value).name).exists():
        link[i] = str(
            (
                OUT
                if Path(value).name in {archive for _, archive in paths}
                else EXPERIMENT
            )
            / Path(value).name
        )
link[link.index("-o") + 1] = str(OUT / "hecate-opt")
with (OUT / "link.log").open("w") as log:
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
assert result.returncode == 0, (OUT / "link.log").read_text()[-5000:]
wrapper = OUT / "hecate-opt-wrapper"
wrapper.write_text(
    "#!/bin/sh\nexec "
    + str(loader)
    + " --library-path "
    + str(OLD / "toolchain/lib")
    + " "
    + str(OUT / "hecate-opt")
    + ' "$@"\n'
)
wrapper.chmod(0o755)
(OUT / "binary.sha256").write_text(
    hashlib.sha256((OUT / "hecate-opt").read_bytes()).hexdigest() + "\n"
)
print("Precision-only ablation ready:", wrapper, flush=True)
