"""Hash-locked CPU-only Python stage; no system packages or implicit Nix builds.

--plan displays the reviewed platform lock without network access. --install-approved downloads <=300 MiB,
then installs offline into a fresh venv using the already installed Nix Python.
The venv/Nix shell is NOT a security sandbox for generated programs.
"""
from platform_config import nix_platform_options, configuration, identity
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from urllib.parse import urlparse
import zipfile

from continue_dacapo_cpp import DEPS, ROOT, SHELL, WORK, WRAPPER
from workspace_paths import require_linux_backend, nix_environment_options

PLATFORM = configuration()
LOCK = ROOT / "src/poseidon/tools/dacapo" / PLATFORM["wheel_lock"]
VENV = WORK / "venvs/hecate-2.0.1-cpu"
CACHE = WORK / "cache/hecate-python-cpu"
LIMIT = 300 * 1024**2
RESERVATION = 2 * 1024**3
PINS = {"numpy": "1.25.2", "torch": PLATFORM["torch"], "filelock": "3.12.2",
        "Jinja2": "3.1.2", "MarkupSafe": "2.1.3", "mpmath": "1.3.0",
        "networkx": "3.1", "sympy": "1.12", "typing_extensions": "4.7.1"}


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def metadata(package):
    name, version = package
    url = f"https://pypi.org/pypi/{name}/{version}/json"
    raw = subprocess.run(["curl", "--fail", "--silent", "--show-error",
        "--connect-timeout", "10", "--max-time", "30", url],
        capture_output=True, check=True, timeout=35)
    candidates = []
    for item in json.loads(raw.stdout)["urls"]:
        filename = item["filename"]
        if (filename.endswith("-py3-none-any.whl") or
            filename.endswith("-py2.py3-none-any.whl") or
            ("-cp310-cp310-manylinux" in filename and filename.endswith(PLATFORM["machine"] + ".whl"))):
            candidates.append(item)
    if len(candidates) != 1:
        raise RuntimeError(f"Expected exactly one compatible wheel for {name}: {len(candidates)}")
    item = candidates[0]
    return dict(name=name, version=version, filename=item["filename"], url=item["url"],
                bytes=item["size"], sha256=item["digests"]["sha256"], evidence=url)


def validate_lock(lock, *, profile=None):
    profile = PLATFORM if profile is None else profile
    pins = dict(PINS, torch=profile["torch"])
    if lock.get("python") != profile["python"] or lock.get("target") != profile["wheel_target"]:
        raise ValueError("Wheel lock does not match selected platform/Python")
    wheels = lock["wheels"]
    if len(wheels) != len(pins) or {w["name"]: w["version"] for w in wheels} != pins:
        raise ValueError("Wheel set differs from approved CPU-only dependency pins")
    if sum(w["bytes"] for w in wheels) > LIMIT - 1024**2:
        raise ValueError("Wheels plus metadata exceed the 300 MiB approval")
    for wheel in wheels:
        parsed = urlparse(wheel["url"])
        if parsed.scheme != "https" or parsed.hostname not in (
                "files.pythonhosted.org", "download.pytorch.org", "download-r2.pytorch.org"):
            raise ValueError("Unapproved wheel host")
        if Path(wheel["filename"]).name != wheel["filename"] or not wheel["filename"].endswith(".whl"):
            raise ValueError("Unsafe wheel filename")
        if len(wheel["sha256"]) != 64 or any(c not in "0123456789abcdef" for c in wheel["sha256"]):
            raise ValueError("Invalid SHA-256")
        filename = wheel["filename"]
        pure = filename.endswith(("-py3-none-any.whl", "-py2.py3-none-any.whl"))
        tags = filename[:-4].rsplit("-", 3)[-3:]
        native = (len(tags) == 3 and tags[:2] == ["cp310", "cp310"] and
                  all(tag in ("linux_" + profile["machine"],
                              "manylinux2014_" + profile["machine"],
                              "manylinux_2_17_" + profile["machine"])
                      for tag in tags[2].split(".")))
        if not pure and not native:
            raise ValueError("Wheel architecture or ABI mismatch")
        if wheel["name"] in ("torch", "numpy", "MarkupSafe") and not native:
            raise ValueError("Native package requires a platform wheel")
        if wheel["bytes"] <= 0:
            raise ValueError("Invalid wheel size")
    return wheels


def enter_nix(command, seconds=600, keep_env=()):
    require_linux_backend()
    if tuple(keep_env) not in ((), ("DEEPSEEK_API_KEY",)):
        raise ValueError("Only the explicit Agent API key may cross the pure-shell boundary")
    keep = nix_environment_options() + [item for name in keep_env for item in ("--keep", name)]
    raw = subprocess.run(["timeout", "-k", "3s", "90s", *WRAPPER, "nix", "eval",
        "--offline", "--json", *nix_platform_options(), "--file", DEPS, "metadata"],
        capture_output=True, text=True, check=True, timeout=100)
    bash = json.loads(raw.stdout)["shell_bash"]
    return subprocess.run(["timeout", "-k", "10s", str(seconds), "env", f"NIX_BUILD_SHELL={bash}",
        *WRAPPER, "nix-shell", "--pure", *keep, "--option", "substitute", "false", "--max-jobs", "0",
        "--option", "builders", "", "--option", "allow-import-from-derivation", "false", *nix_platform_options(), SHELL,
        "--run", command], timeout=seconds + 15).returncode


def download(wheels):
    if VENV.exists():
        raise RuntimeError(f"Preserving existing venv {VENV}; inspect before retrying")
    if shutil.disk_usage(WORK).free < RESERVATION:
        raise RuntimeError("Less than approved 2 GiB disk reservation available")
    CACHE.mkdir(parents=True, exist_ok=True)
    for wheel in wheels:
        target = CACHE / wheel["filename"]
        if target.exists():
            if target.stat().st_size != wheel["bytes"] or digest(target) != wheel["sha256"]:
                raise RuntimeError(f"Existing cache mismatch; preserved: {target}")
            continue
        # No retries or resumptions: every attempted payload counts against the
        # current approval. Failed partial files remain available for diagnosis.
        if list(CACHE.glob(wheel["filename"] + ".partial-*")):
            raise RuntimeError("Prior partial download found; review remaining budget before retry")
        fd, temporary = tempfile.mkstemp(prefix=wheel["filename"] + ".partial-", dir=CACHE)
        os.close(fd)
        partial = Path(temporary)
        print(f"Downloading {wheel['filename']} ({wheel['bytes']} bytes)", flush=True)
        subprocess.run(["curl", "--fail", "--silent", "--show-error", "--location",
            "--proto", "=https", "--proto-redir", "=https", "--connect-timeout", "10",
            "--max-time", "600", "--max-filesize", str(wheel["bytes"]),
            "--output", str(partial), wheel["url"]], check=True, timeout=610)
        if partial.stat().st_size != wheel["bytes"] or digest(partial) != wheel["sha256"]:
            raise RuntimeError(f"Downloaded wheel failed size/hash verification: {partial}")
        partial.rename(target)


def install_inside(wheels):
    require_linux_backend()
    if not os.environ.get("IN_NIX_SHELL") or sys.version_info[:3] != (3, 10, 14):
        raise RuntimeError("Requires pinned Nix Python 3.10.14")
    if VENV.exists():
        raise RuntimeError("Will not modify an existing venv")
    expanded = 0
    metadata_records = {}
    for wheel in wheels:
        path = CACHE / wheel["filename"]
        if digest(path) != wheel["sha256"] or path.stat().st_size != wheel["bytes"]:
            raise RuntimeError("Wheel changed before offline installation")
        with zipfile.ZipFile(path) as archive:
            expanded += sum(item.file_size for item in archive.infolist())
            names = [name for name in archive.namelist() if name.endswith(".dist-info/METADATA")]
            if len(names) != 1:
                raise RuntimeError("Invalid wheel METADATA")
            metadata_records[wheel["name"]] = archive.read(names[0]).decode()
    if expanded + sum(w["bytes"] for w in wheels) + 100 * 1024**2 > RESERVATION:
        raise RuntimeError("Expanded wheels/cache exceed approved 2 GiB reservation")
    VENV.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, "-m", "venv", str(VENV)], check=True, timeout=120)
    python = str(VENV / "bin/python")
    env = dict(os.environ, PIP_DISABLE_PIP_VERSION_CHECK="1", PIP_CONFIG_FILE=os.devnull,
               PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1")
    requirements = CACHE / "requirements.lock.txt"
    requirements.write_text("".join(f"{w['name']}=={w['version']} --hash=sha256:{w['sha256']}\n"
                                    for w in wheels))
    result = Path(tempfile.mkdtemp(prefix="hecate-python-install-", dir=WORK / "results"))
    # Save wheel dependency declarations before invoking the offline resolver.
    (result / "wheel-metadata.json").write_text(json.dumps(metadata_records, indent=2) + "\n")
    command = [python, "-m", "pip", "--isolated", "install", "--no-index", "--no-cache-dir",
               "--only-binary=:all:", "--require-hashes", "--find-links", str(CACHE),
               "--report", str(result / "pip-report.json"), "-r", str(requirements)]
    with (result / "install.log").open("w") as log:
        subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                       check=True, timeout=240)
    verify_existing(wheels, result, expanded)


def verify_existing(wheels, result=None, expanded=None):
    require_linux_backend()
    if not os.environ.get("IN_NIX_SHELL") or sys.version_info[:3] != (3, 10, 14):
        raise RuntimeError("Requires pinned Nix Python 3.10.14")
    library = os.environ["HECATE_PYTHON_LIBRARY_PATH"]
    if not all(any(Path(p, name).exists() for p in library.split(":"))
               for name in ("libstdc++.so.6", "libz.so.1")):
        raise RuntimeError("Pinned C++/zlib runtime is not already installed")
    if result is None:
        result = Path(tempfile.mkdtemp(prefix="hecate-python-verify-", dir=WORK / "results"))
    python = str(VENV / "bin/python")
    env = dict(os.environ, LD_LIBRARY_PATH=library, PYTHONDONTWRITEBYTECODE="1",
               PYTHONNOUSERSITE="1", PIP_DISABLE_PIP_VERSION_CHECK="1", PIP_CONFIG_FILE=os.devnull)
    for wheel in wheels:
        if digest(CACHE / wheel["filename"]) != wheel["sha256"]:
            raise RuntimeError("Wheel cache hash mismatch")
    checked = subprocess.run([python, "-m", "pip", "--isolated", "check"], env=env,
                             text=True, capture_output=True, timeout=30)
    (result / "pip-check.log").write_text(checked.stdout + checked.stderr)
    checked.check_returncode()
    installed = subprocess.run([python, "-m", "pip", "--isolated", "list", "--format=json"],
        env=env, text=True, capture_output=True, check=True, timeout=30)
    versions = {p["name"].lower().replace("-", "_"): p["version"] for p in json.loads(installed.stdout)}
    expected = {k.lower(): v for k, v in PINS.items()}
    expected.update(pip="23.0.1", setuptools="65.5.0")
    if versions != expected:
        raise RuntimeError(f"Unexpected installed package versions: {versions}")
    (result / "installed-packages.json").write_text(installed.stdout)
    probe = subprocess.run([python, "-c", "import json,sys,numpy,torch; from pathlib import Path; "
        "assert torch.version.cuda is None; assert getattr(torch.version, 'hip', None) is None; "
        f"assert torch.__version__=={PLATFORM['torch']!r}; "
        "assert numpy.__version__=='1.25.2'; "
        "print(json.dumps(dict(python=sys.version,numpy=numpy.__version__,torch=torch.__version__,"
        "cuda=torch.version.cuda,mapped_libraries=sorted({line.split(maxsplit=5)[5] for line in Path('/proc/self/maps').read_text().splitlines() if len(line.split(maxsplit=5))==6 and '.so' in line.split(maxsplit=5)[5]}),sum=torch.tensor([1.,2.],dtype=torch.float64).sum().item())))"],
        env=env, text=True, capture_output=True, timeout=60)
    (result / "import-probe.log").write_text(probe.stdout + probe.stderr)
    probe.check_returncode()
    report = dict(platform_identity=identity(), status="installed", venv=str(VENV), wheel_bytes=sum(w["bytes"] for w in wheels),
                  expanded_wheel_bytes=expanded, lock_sha256=digest(LOCK),
                  versions=json.loads(probe.stdout), library_path=library, python_dsl_tracing_validated=False,
                  encrypted_execution_validated=False)
    (result / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    print(f"Installation evidence: {result}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--plan", action="store_true")
    modes.add_argument("--install-approved", action="store_true")
    modes.add_argument("--inside-install", action="store_true")
    modes.add_argument("--verify-existing", action="store_true")
    modes.add_argument("--inside-verify", action="store_true")
    args = parser.parse_args()
    if Path.cwd().resolve() != ROOT:
        raise SystemExit(f"Requires cwd {ROOT}")
    if args.plan:
        # Display the reviewed lock; refreshing upstream metadata is a separate,
        # read-only review and must not silently replace platform-specific pins.
        lock = json.loads(LOCK.read_text())
        validate_lock(lock)
        print(json.dumps(lock, indent=2))
    else:
        require_linux_backend()
        wheels = validate_lock(json.loads(LOCK.read_text()))
        if args.install_approved:
            download(wheels)
            raise SystemExit(enter_nix("python3 scripts/baseline/hecate_python_env.py --inside-install"))
        elif args.verify_existing:
            raise SystemExit(enter_nix("python3 scripts/baseline/hecate_python_env.py --inside-verify"))
        elif args.inside_verify:
            verify_existing(wheels)
        else:
            install_inside(wheels)


if __name__ == "__main__":
    main()
