#!/usr/bin/env python3
"""Read-only prerequisite gate; never installs, imports Hecate, or claims E2E success."""
import argparse
import importlib.metadata
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from platform_config import configuration, identity, require_platform

DACAPO_COMMIT = "4616402710f39df3e5f5bd7930a6c036025aaac3"
LLVM_VERSION = "18.1.2"
SEAL_VERSION = "4.0.0"


def version_from_output(output):
    match = re.search(r"\b(\d+\.\d+\.\d+)\b", output)
    return match.group(1) if match else None


def version_matches(actual, expected, minimum=False):
    if not actual:
        return False
    if minimum:
        return tuple(map(int, actual.split('.'))) >= tuple(map(int, expected.split('.')))
    return actual == expected


def probe(args):
    try:
        # timeout also terminates subprocess groups, rather than only their shell.
        result = subprocess.run(
            ["timeout", "-k", "3s", "20s", *map(str, args)],
            capture_output=True, text=True, timeout=25, check=False,
        )
        return result.returncode, (result.stdout + result.stderr).strip()
    except (OSError, subprocess.TimeoutExpired) as error:
        return -1, str(error)


def check_tool(name, expected=None, prefix=None, minimum=False):
    path = str(prefix / "bin" / name) if prefix else shutil.which(name)
    record = {"name": name, "path": path, "expected": expected}
    if not path or not Path(path).is_file():
        return dict(record, ok=False, diagnostic="missing executable")
    code, output = probe([path, "--version"])
    version = version_from_output(output)
    ok = code == 0 and (expected is None or version_matches(version, expected, minimum))
    return dict(record, ok=ok, version=version, returncode=code,
                diagnostic="" if ok else output[:1000])


def check_submodule(repo):
    path = repo / "third_party/dacapo"
    code, output = probe(["git", "-C", repo, "ls-tree", "HEAD", "third_party/dacapo"])
    fields = output.split()
    gitlink_ok = code == 0 and len(fields) == 4 and fields[:3] == ["160000", "commit", DACAPO_COMMIT]
    # git -C on an empty submodule can resolve the parent repository. Do not accept it.
    initialized = (path / ".git").exists()
    head = None
    if initialized:
        code, head = probe(["git", "-C", path, "rev-parse", "HEAD"])
        initialized = code == 0
    return {"name": "dacapo_submodule", "path": str(path), "expected": DACAPO_COMMIT,
            "gitlink_ok": gitlink_ok, "initialized": initialized, "head": head,
            "ok": gitlink_ok and initialized and head == DACAPO_COMMIT}


def check_seal(prefix):
    candidates = [] if prefix is None else [
        prefix / lib / "cmake/SEAL-4.0/SEALConfigVersion.cmake" for lib in ("lib", "lib64")
    ]
    config = next((path for path in candidates if path.is_file()), None)
    version = None
    if config:
        match = re.search(r'set\(PACKAGE_VERSION\s+"([^"]+)"\)', config.read_text())
        version = match.group(1) if match else None
    return {"name": "SEAL", "expected": SEAL_VERSION,
            "path": str(config) if config else None, "version": version,
            "ok": version == SEAL_VERSION,
            "diagnostic": "" if version == SEAL_VERSION else "provide the isolated SEAL 4.0.0 prefix"}


def check_python_package(name, expected):
    try:
        version = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        version = None
    # Distribution version must match this platform exactly.
    ok = version == expected
    return {"name": name, "expected": expected, "version": version, "ok": ok}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--llvm-prefix", type=Path)
    parser.add_argument("--mlir-prefix", type=Path)
    nix_cc = os.environ.get("NIX_CC") if os.environ.get("IN_NIX_SHELL") else None
    compiler_prefix = Path(nix_cc) if nix_cc else None
    parser.add_argument("--clang-prefix", type=Path, default=compiler_prefix)
    parser.add_argument("--compiler", choices=("gcc", "clang"),
                        default=os.environ.get("HECATE_C_COMPILER_NAME", "gcc"))
    parser.add_argument("--gcc-prefix", type=Path, default=compiler_prefix)
    parser.add_argument("--seal-prefix", type=Path)
    args = parser.parse_args()
    if args.compiler not in ("gcc", "clang"):
        parser.error("unsupported compiler selected by the Nix environment")
    if sys.platform != "linux":
        parser.error("run inside the Ubuntu/Linux backend (WSL, VM or remote host), not the host OS")
    require_platform()
    checks = [
        check_submodule(args.repo.resolve()),
        check_tool("cmake", "3.22.1", minimum=True),
        check_tool("ninja"),
        check_tool("gcc", "13.2.0", prefix=args.gcc_prefix) if args.compiler == "gcc"
        else check_tool("clang", LLVM_VERSION, prefix=args.clang_prefix),
        check_tool("g++", "13.2.0", prefix=args.gcc_prefix) if args.compiler == "gcc"
        else check_tool("clang++", LLVM_VERSION, prefix=args.clang_prefix),
        check_tool("llvm-config", LLVM_VERSION, prefix=args.llvm_prefix),
        check_tool("mlir-tblgen", LLVM_VERSION, prefix=args.mlir_prefix or args.llvm_prefix),
        check_seal(args.seal_prefix),
        {"name": "python", "version": sys.version.split()[0], "expected": ">=3.10",
         "ok": sys.version_info >= (3, 10)},
        check_python_package("numpy", "1.25.2"),
        check_python_package("torch", configuration()["torch"]),
    ]
    ok = all(check["ok"] for check in checks)
    print(json.dumps({"platform_identity": identity(), "scope": "prerequisites_only", "checks": checks,
                      "prerequisites_ok": ok, "compiler_build_validated": False,
                      "encrypted_execution_validated": False}, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
