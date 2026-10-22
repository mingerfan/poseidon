"""Pinned native-WSL CUDA toolkit installation, never a display driver.

Prints the plan by default; installation requires --install-approved.
Existing files and failed downloads/staging trees are preserved for diagnosis.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tarfile
import tempfile

from workspace_paths import ROOT, WORK, require_linux_backend
LOCK = ROOT / "scripts/baseline/cuda-12.5.1.lock.json"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def safe_member(member, root):
    parts = PurePosixPath(member.name)
    if parts.is_absolute() or ".." in parts.parts or not parts.parts:
        raise ValueError("Unsafe archive member path")
    destination = root / member.name
    if not destination.resolve().is_relative_to(root.resolve()):
        raise ValueError("Archive member escapes extraction root")
    if not (member.isdir() or member.isfile() or member.issym()):
        raise ValueError("Unsupported archive member kind")
    if member.issym() and (PurePosixPath(member.linkname).is_absolute() or
            not (destination.parent / member.linkname).resolve().is_relative_to(root.resolve())):
        raise ValueError("Unsafe archive symlink")


def unpack(path, root, limit):
    root.mkdir()
    with tarfile.open(path, "r:xz") as archive:
        members = archive.getmembers()
        if len(members) > 20000 or sum(m.size for m in members) > limit:
            raise ValueError("Archive expansion exceeds reservation")
        for member in members:
            safe_member(member, root)
        for member in members:
            # Check again after earlier symlinks have materialized.
            safe_member(member, root)
            member.mode &= 0o777
            archive.extract(member, root, set_attrs=False)
            if member.isfile():
                (root / member.name).chmod(member.mode)
    children = list(root.iterdir())
    if len(children) != 1 or not children[0].is_dir() or children[0].is_symlink():
        raise ValueError("Expected one NVIDIA component directory")
    return children[0]


def merge(source, target):
    for item in source.iterdir():
        dest = target / item.name
        if item.is_symlink():
            if dest.exists() or dest.is_symlink():
                if not dest.is_symlink() or os.readlink(dest) != os.readlink(item):
                    raise ValueError("Conflicting CUDA symlink")
            else:
                dest.symlink_to(os.readlink(item))
        elif item.is_dir():
            if dest.is_symlink():
                raise ValueError("CUDA directory collision")
            dest.mkdir(exist_ok=True)
            merge(item, dest)
        elif dest.exists() or dest.is_symlink():
            if dest.is_symlink() or not dest.is_file() or digest(item) != digest(dest):
                raise ValueError("Conflicting CUDA component file")
        else:
            shutil.copy2(item, dest)


def install(plan):
    require_linux_backend()
    if Path.cwd().resolve() != ROOT:
        raise ValueError("Run in the actual WSL source checkout")
    os.umask(0o077)
    prefix = WORK / "deps/cuda-12.5.1"
    if prefix.exists() or prefix.is_symlink():
        raise ValueError("Preserving existing CUDA prefix")
    cache = WORK / "cache/cuda-12.5.1"
    cache.mkdir(parents=True, exist_ok=True)
    with (cache / "install.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if shutil.disk_usage(WORK).free < plan["reservation_bytes"]:
            raise ValueError("Need approved 1 GiB disk reservation")
        folder = Path(tempfile.mkdtemp(prefix="cuda-install-", dir=WORK / "results"))
        report = dict(status="running", prefix=str(prefix), lock_sha256=digest(LOCK),
                      source_sha256=digest(Path(__file__)), components=[], driver_installed=False)
        print(f"CUDA installation evidence: {folder}", flush=True)
        stage = None
        try:
            total = sum(c["bytes"] for c in plan["components"])
            if total > plan["download_limit_bytes"]:
                raise ValueError("Download plan exceeds approval")
            for c in plan["components"]:
                name = c["url"].rsplit("/", 1)[1]
                target = cache / name
                if target.exists():
                    if target.is_symlink() or target.stat().st_size != c["bytes"] or digest(target) != c["sha256"]:
                        raise ValueError("Existing download mismatch; preserved")
                else:
                    if list(cache.glob(name+".partial-*")):
                        raise ValueError("Partial download exists; review before retrying")
                    fd, temporary = tempfile.mkstemp(prefix=name+".partial-", dir=cache)
                    os.close(fd)
                    partial = Path(temporary)
                    print("Downloading " + c["name"] + " " + c["version"], flush=True)
                    subprocess.run(["timeout", "-k", "3s", "180s", "curl", "--fail", "--silent", "--show-error",
                        "--proto", "=https", "--connect-timeout", "10", "--max-time", "170",
                        "--max-filesize", str(c["bytes"]), "--output", str(partial), c["url"]],
                        check=True, timeout=185)
                    if partial.stat().st_size != c["bytes"] or digest(partial) != c["sha256"]:
                        raise ValueError("CUDA payload size/hash mismatch")
                    partial.rename(target)
                report["components"].append(dict(c, archive=str(target)))
            stage = Path(tempfile.mkdtemp(prefix="cuda-12.5.1-stage-", dir=WORK / "deps"))
            components = stage / "components"
            components.mkdir()
            expanded = 0
            for c in report["components"]:
                component = unpack(Path(c["archive"]), components / c["name"], plan["reservation_bytes"]//2)
                expanded += sum(p.stat().st_size for p in component.rglob("*") if p.is_file() and not p.is_symlink())
                if total + 2*expanded > plan["reservation_bytes"]:
                    raise ValueError("Merged CUDA prefix exceeds reservation")
                for name in ("bin", "include", "lib", "lib64", "nvvm", "targets"):
                    source = component / name
                    if source.is_dir():
                        target = stage / name
                        target.mkdir(exist_ok=True)
                        merge(source, target)
            # NVIDIA redistributables put libraries in lib; nvcc's Linux64
            # profile searches lib64. Keep both names within this private prefix.
            if (stage / "lib").is_dir() and not (stage / "lib64").exists():
                (stage / "lib64").symlink_to("lib", target_is_directory=True)
            result = subprocess.run([str(stage / "bin/nvcc"), "--version"],
                                    capture_output=True, text=True, check=True, timeout=15)
            if "V12.5.82" not in result.stdout:
                raise ValueError("Unexpected installed nvcc version")
            report.update(status="installed", nvcc_version=result.stdout,
                          compressed_bytes=total, extracted_component_bytes=expanded)
            (stage / "installation.json").write_text(json.dumps(report, indent=2)+"\n")
            if prefix.exists() or prefix.is_symlink():
                raise ValueError("CUDA prefix appeared during installation")
            stage.rename(prefix)
            stage = None
        except Exception as error:
            report.update(status="failed", diagnostic=str(error), staging=str(stage) if stage else None)
        (folder / "report.json").write_text(json.dumps(report, indent=2)+"\n")
        print(json.dumps(report, indent=2))
        return 0 if report["status"] == "installed" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install-approved", action="store_true")
    args = parser.parse_args()
    plan = json.loads(LOCK.read_text())
    if not args.install_approved:
        print(json.dumps(plan, indent=2))
        return 0
    return install(plan)


if __name__ == "__main__":
    raise SystemExit(main())
