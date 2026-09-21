"""Manifest-driven native CPU setup. Planning is offline and does not create files."""
import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time

import platform_config

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = "scripts/baseline/agent-requirements.json"
DEPS = "src/poseidon/tools/dacapo/dacapo-dependencies.nix"
SHELL = "src/poseidon/tools/dacapo/dacapo-shell.nix"
WRAPPER = ["bash", "scripts/baseline/nix_portable.sh"]
MIB = 1024**2
GIB = 1024**3


def read_json(path):
    return json.loads(path.read_text())


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(MIB), b""):
            h.update(block)
    return h.hexdigest()


def select_platform(value, *, system=None, machine=None):
    system = sys.platform if system is None else system
    machine = platform.machine().lower() if machine is None else machine.lower()
    if value == "auto":
        if system != "linux":
            raise ValueError("Automatic setup requires Linux Ubuntu, not the macOS host")
        candidates = [name for name, profile in read_json(platform_config.MANIFEST)["platforms"].items()
                      if machine in profile["machine_aliases"]]
        if len(candidates) != 1:
            raise ValueError("No configured native platform for " + machine)
        value = candidates[0]
    return platform_config.configuration(value)


def make_plan(selected, work_base, root=ROOT):
    manifest = read_json(root / MANIFEST)
    if manifest["schema_version"] != 1 or manifest["toolchain_profile"] != "hecate":
        raise ValueError("This setup entry supports only the reviewed hecate CPU toolchain")
    lock = read_json(root / manifest["dependency_lock"])
    wheel_path = root / "src/poseidon/tools/dacapo" / selected["wheel_lock"]
    from hecate_python_env import validate_lock
    wheels = validate_lock(read_json(wheel_path), profile=selected)
    work = work_base / selected["work_subdirectory"]
    if work_base == root or root in work_base.parents or work_base in root.parents:
        raise ValueError("Use a dedicated work root outside the source checkout")
    launcher = selected["nix_portable"]
    files = [root / MANIFEST, root / manifest["platform_profiles"],
             root / manifest["dependency_lock"], wheel_path]
    patches = []
    for name in manifest["patch_series"]:
        if Path(name).name != name or not name.endswith(".patch"):
            raise ValueError("Invalid patch filename")
        path = root / "scripts/baseline/patches" / name
        patches.append({"file": str(path.relative_to(root)), "sha256": digest(path)})
    missing = [name for name in manifest["system_commands"] if shutil.which(name) is None]
    matches = (sys.platform == "linux" and platform.machine().lower() in selected["machine_aliases"])
    known_bytes = launcher["bytes"] + sum(w["bytes"] for w in wheels) + sum(
        lock[name]["archive_bytes"] for name in ("llvm", "seal"))
    return {
        "schema_version": 1, "scope": manifest["scope"], "mode": "plan_only",
        "platform": selected["id"], "actual_system": sys.platform,
        "actual_machine": platform.machine(), "native_platform_matches": matches,
        "source_root": str(root), "work_root_base": str(work_base), "work_root": str(work),
        "toolchain_profile": "hecate",
        "versions": {"gcc": "13.2.0", "llvm_mlir": lock["llvm"]["version"],
                     "seal": lock["seal"]["version"], "gsl": lock["msgsl"]["version"],
                     "python": selected["python"], "numpy": selected["numpy"], "torch": selected["torch"]},
        "nix_launcher": launcher, "nixpkgs": lock["nixpkgs"],
        "dacapo_commit": lock["dacapo_commit"], "patches": patches,
        "python_wheels": wheels, "known_full_payload_bytes": known_bytes,
        "download_note": "Not a total estimate: Nix binary closure, GSL, Git and protocol/retry bytes are additional; cache hits reduce downloads. Nix dry-run is recorded during apply.",
        "missing_system_commands": missing, "system_prerequisites": manifest["system_note"],
        "limits": manifest["limits"], "stages": manifest["stages"],
        "lock_hashes": {str(p.relative_to(root)): digest(p) for p in files},
        "credentials_read": False, "paid_calls": 0,
        "compiler_execution_validated": False, "encrypted_execution_validated": False,
    }


def plan_summary(plan):
    v = plan["versions"]
    missing = ", ".join(plan["missing_system_commands"]) or "无缺失命令（隔离能力将在执行时检查）"
    return "\n".join([
        "Poseidon Agent CPU 依赖计划（未下载、未安装、未构建）",
        f"平台：{plan['platform']}；当前机器：{plan['actual_system']}/{plan['actual_machine']}；匹配：{plan['native_platform_matches']}",
        "工具链：hecate 精简 SDK；GCC " + v["gcc"] + "；LLVM/MLIR " + v["llvm_mlir"],
        "运行库：SEAL " + v["seal"] + "；GSL " + v["gsl"],
        "Python " + v["python"] + "；NumPy " + v["numpy"] + "；Torch " + v["torch"] + "（CPU）",
        "源码：" + plan["source_root"],
        "工作目录：" + plan["work_root"],
        f"Dacapo：固定 {plan['dacapo_commit']} + {len(plan['patches'])} 个补丁",
        f"Python：{len(plan['python_wheels'])} 个带哈希 wheel；可用 --requirements 导出",
        f"已知完整下载载荷：{plan['known_full_payload_bytes'] / MIB:.1f} MiB（未扣缓存）",
        "该数值不是总下载量；另有 Nix 依赖闭包、GSL、Git 和传输开销。",
        "系统前置命令：" + missing,
        "构建上限：编译 2 jobs，链接 1 job，单阶段 12 小时。",
        "执行需要 --apply --download-budget-mib N --disk-budget-gib N；新子模块另加 --init-submodule。",
        "可选 --self-test 运行 Linear 和 MLP 离线密态验收；不会调用付费 API。",
        "完整来源、哈希及阶段：--plan --json",
    ]) + "\n"


def requirements_text(plan):
    lines = ["# Generated from " + plan["platform"] + " wheel lock; CPython 3.10 only.",
             "# Python packages only; use setup_agent.py for the native compiler/runtime."]
    for w in plan["python_wheels"]:
        lines.append(f"{w['name']} @ {w['url']} --hash=sha256:{w['sha256']}")
    return "\n".join(lines) + "\n"


def checked(command, cwd=ROOT, timeout=30):
    return subprocess.run(command, cwd=cwd, capture_output=True, check=True,
                          text=True, timeout=timeout).stdout.strip()


def patch_paths(patches):
    paths = set()
    for patch in patches:
        text = patch.read_text()
        old = re.findall(r"^--- a/(.+)$", text, re.M)
        new = re.findall(r"^\+\+\+ b/(.+)$", text, re.M)
        if not old or sorted(old) != sorted(new):
            raise RuntimeError("Only existing-file patches are supported")
        for name in old:
            if Path(name).is_absolute() or ".." in Path(name).parts:
                raise RuntimeError("Unsafe patch path")
            paths.add(name)
    return sorted(paths)


def patched_prefix(repo, patches, commit, scratch_parent=None):
    """Reproduce all prefixes; accept only an exact prefix, never guess from reverse-checks."""
    if not (repo / ".git").exists():
        raise RuntimeError("Dacapo is not initialized; use --init-submodule with --apply")
    if checked(["git", "rev-parse", "HEAD"], repo) != commit:
        raise RuntimeError("Dacapo gitlink mismatch; preserving checkout")
    if checked(["git", "diff", "--cached", "--name-only"], repo):
        raise RuntimeError("Dacapo has staged changes; preserving index")
    paths = patch_paths(patches)
    changed = checked(["git", "diff", "HEAD", "--name-only"], repo).splitlines()
    if set(changed) - set(paths):
        raise RuntimeError("Dacapo has changes outside the patch bundle; preserving them")
    actual = {}
    for name in paths:
        path = repo / name
        if not path.is_file() or path.is_symlink():
            raise RuntimeError("Unexpected Dacapo file type: " + name)
        actual[name] = (path.read_bytes(), bool(path.stat().st_mode & 0o111))
    with tempfile.TemporaryDirectory(prefix="poseidon-patch-check-", dir=scratch_parent) as tmp:
        scratch = Path(tmp)
        for name in paths:
            path = scratch / name
            path.parent.mkdir(parents=True, exist_ok=True)
            blob = subprocess.run(["git", "show", f"{commit}:{name}"], cwd=repo,
                                  capture_output=True, check=True, timeout=30).stdout
            mode = checked(["git", "ls-tree", commit, "--", name], repo).split()[0]
            path.write_bytes(blob)
            path.chmod(0o755 if mode == "100755" else 0o644)
        def matches():
            return all(actual[n] == ((scratch / n).read_bytes(),
                                     bool((scratch / n).stat().st_mode & 0o111)) for n in paths)
        prefix = 0 if matches() else None
        for i, patch in enumerate(patches, 1):
            checked(["git", "apply", "--check", str(patch)], scratch)
            checked(["git", "apply", str(patch)], scratch)
            if matches():
                prefix = i
        if prefix is None:
            raise RuntimeError("Dacapo modifications are not an exact patch prefix; no files changed")
        return prefix


def ensure_frontend_link(work):
    target = work / "build-dacapo/hecate-18.1.2-nix"
    parent = work / "build-dacapo/hecate-python-root"
    parent.mkdir(parents=True, exist_ok=True)
    link = parent / "build"
    if link.is_symlink():
        if link.resolve() != target.resolve():
            raise RuntimeError("Preserving unexpected HECATE link")
    elif link.exists():
        raise RuntimeError("Preserving existing HECATE build directory")
    else:
        link.symlink_to(target, target_is_directory=True)


def check_build_caches(work, selected):
    for directory in ("hecate-18.1.2-nix", "seal-golden-keys"):
        build = work / "build-dacapo" / directory
        if not (build / "CMakeCache.txt").exists():
            continue
        records = list(build.glob("CMakeFiles/*/CMakeSystem.cmake"))
        if not records:
            raise RuntimeError("Cannot identify existing CMake cache platform: " + str(build))
        for path in records:
            text = path.read_text()
            match = re.search(r'set\(CMAKE_SYSTEM_PROCESSOR "([^"]+)"\)', text)
            if (not match or match[1].lower() not in selected["machine_aliases"]
                    or 'set(CMAKE_SYSTEM_NAME "Linux")' not in text
                    or 'set(CMAKE_CROSSCOMPILING "FALSE")' not in text):
                raise RuntimeError("Preserving mismatched CMake cache: " + str(build))


def source_fingerprint(root):
    # Only project inputs, never .env, Git object data, keys or result arrays.
    paths = [root / "scripts/setup_agent.py", root / "scripts/agent.py", root / ".gitmodules"]
    for folder in ("scripts/baseline", "src/poseidon/tools/dacapo"):
        paths.extend(p for p in (root / folder).rglob("*")
                     if p.is_file() and p.suffix in (".py", ".sh", ".json", ".nix", ".patch", ".cpp", ".h")
                     and "__pycache__" not in p.parts)
    paths.extend((root / "scripts/baseline").rglob("CMakeLists.txt"))
    return {str(p.relative_to(root)): digest(p) for p in sorted(set(paths))}


def submodule_fingerprint(repo):
    names = checked(["git", "ls-files", "-z"], repo).split(chr(0))
    return {name: digest(repo / name) for name in names if name
            and (Path(name).suffix in (".cpp", ".h", ".hpp", ".inc", ".td", ".py", ".json", ".cmake")
                 or Path(name).name == "CMakeLists.txt")}


def network_bytes():
    # Whole guest, including unrelated traffic: conservative, no payload capture.
    paths = [p for p in Path("/sys/class/net").glob("*/statistics/rx_bytes") if p.parts[-3] != "lo"]
    if not paths:
        raise RuntimeError("Cannot meter guest network receive bytes")
    return {str(p): int(p.read_text()) for p in paths}


class Budget:
    def __init__(self, work, download_mib, disk_gib):
        self.work = work
        self.network_start = network_bytes()
        self.free_start = shutil.disk_usage(work).free
        self.download_limit = download_mib * MIB
        self.disk_limit = disk_gib * GIB
        self.usage = {}

    def check(self):
        now = network_bytes()
        if now.keys() != self.network_start.keys() or any(now[p] < n for p, n in self.network_start.items()):
            raise RuntimeError("Network counters changed/reset; review budget before resuming")
        free = shutil.disk_usage(self.work).free
        self.usage = dict(network_rx_bytes=sum(now[p] - n for p, n in self.network_start.items()),
                          filesystem_used_delta_bytes=max(0, self.free_start - free))
        if self.usage["network_rx_bytes"] >= self.download_limit:
            raise RuntimeError("Download budget reached; partial files and evidence preserved")
        if self.usage["filesystem_used_delta_bytes"] >= self.disk_limit or free < 2 * GIB:
            raise RuntimeError("Disk budget/reservation reached; outputs preserved")


class Runner:
    def __init__(self, root, work, plan, budget):
        self.root, self.work, self.budget = root, work, budget
        self.folder = Path(tempfile.mkdtemp(prefix="agent-setup-", dir=work / "results"))
        self.inputs = source_fingerprint(root)
        self.upstream_inputs = None
        self.state = dict(status="running", plan=plan, inputs_sha256=self.inputs, steps=[],
                          encrypted_execution_validated=False, paid_calls=0)
        self.env = {k: v for k, v in os.environ.items()
                    if k in ("PATH", "HOME", "USER", "LOGNAME", "TMPDIR", "http_proxy", "https_proxy",
                             "HTTP_PROXY", "HTTPS_PROXY", "no_proxy", "NO_PROXY")}
        self.env.update(POSEIDON_PLATFORM=plan["platform"], POSEIDON_WORK_ROOT=plan["work_root_base"],
                        PYTHONDONTWRITEBYTECODE="1", LC_ALL="C", GIT_TERMINAL_PROMPT="0",
                        GIT_ASKPASS="/bin/false", SSH_ASKPASS="/bin/false")
        self.save()

    def save(self):
        self.state["budget_usage"] = self.budget.usage
        path = self.folder / "report.json"
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps(self.state, indent=2) + "\n")
        temp.replace(path)

    def verify(self):
        self.budget.check()
        if source_fingerprint(self.root) != self.inputs:
            raise RuntimeError("Setup/build inputs changed during execution; stop at stage boundary")
        if self.upstream_inputs is not None and submodule_fingerprint(
                self.root / "third_party/dacapo") != self.upstream_inputs:
            raise RuntimeError("Dacapo source changed during execution; stop at stage boundary")

    def run(self, name, command, seconds):
        self.verify()
        seconds = min(seconds, 43200)
        record = dict(stage=name, command=command, timeout_seconds=seconds, status="running")
        self.state["steps"].append(record)
        self.save()
        print(f"{name}: {self.folder / (name + '.log')}", flush=True)
        with (self.folder / (name + ".stdout")).open("w") as out, \
                (self.folder / (name + ".log")).open("w") as err:
            p = subprocess.Popen(["timeout", "-k", "10s", str(seconds), *command], cwd=self.root,
                                 env=self.env, stdout=out, stderr=err, start_new_session=True)
            try:
                start = time.monotonic()
                while p.poll() is None:
                    self.budget.check()
                    if time.monotonic() - start > seconds + 15:
                        raise RuntimeError("Stage deadline exceeded")
                    time.sleep(0.25)
                record.update(exit_code=p.returncode, status="passed" if p.returncode == 0 else "failed")
                if p.returncode:
                    raise RuntimeError(f"{name} failed ({p.returncode}); see {self.folder}")
                self.verify()
            except BaseException:
                record["status"] = "failed"
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(p.pid, signal.SIGTERM)
                try:
                    p.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    with contextlib.suppress(ProcessLookupError):
                        os.killpg(p.pid, signal.SIGKILL)
                    p.wait()
                raise
            finally:
                self.save()
        return (self.folder / (name + ".stdout")).read_text()


def provision_launcher(run, selected):
    target = run.work / "deps/nix-portable-v012/nix-portable"
    pin = selected["nix_portable"]
    if target.is_symlink():
        raise RuntimeError("Preserving unexpected launcher symlink")
    if target.exists():
        if target.stat().st_size != pin["bytes"] or digest(target) != pin["sha256"] or not os.access(target, os.X_OK):
            raise RuntimeError("Existing Nix launcher size/hash/mode mismatch; preserved")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(".partial")
    if partial.exists():
        raise RuntimeError("Prior launcher partial exists; inspect remaining budget before retry")
    run.run("launcher_download", ["curl", "--fail", "--silent", "--show-error", "--location",
        "--proto", "=https", "--proto-redir", "=https", "--connect-timeout", "10",
        "--max-time", "600", "--max-filesize", str(pin["bytes"]),
        "--output", str(partial), pin["url"]], 610)
    if partial.stat().st_size != pin["bytes"] or digest(partial) != pin["sha256"]:
        raise RuntimeError("Nix launcher download size/hash mismatch; partial preserved")
    partial.chmod(0o755)
    partial.replace(target)


def build_sequence(run, selected, init_submodule=False, self_test=False):
    root, work = run.root, run.work
    manifest = read_json(root / MANIFEST)
    commit = run.state["plan"]["dacapo_commit"]
    tree = checked(["git", "ls-tree", "HEAD", "--", "third_party/dacapo"], root).split()
    if tree[:3] != ["160000", "commit", commit]:
        raise RuntimeError("Parent gitlink differs from lock; preserving checkout")
    repo = root / "third_party/dacapo"
    run.run("bubblewrap", ["/usr/bin/bwrap", "--unshare-all", "--die-with-parent",
        "--new-session", "--ro-bind", "/", "/", "--proc", "/proc", "--dev", "/dev",
        "--clearenv", "/usr/bin/true"], 30)
    if not (repo / ".git").exists():
        if not init_submodule:
            raise RuntimeError("Dacapo initialization needs explicit --init-submodule")
        if repo.exists() and any(repo.iterdir()):
            raise RuntimeError("Uninitialized Dacapo directory is nonempty; preserved")
        run.run("submodule_init", ["git", "-c", "submodule.third_party/dacapo.url=" + manifest["dacapo_url"],
            "-c", "http.lowSpeedLimit=1024", "-c", "http.lowSpeedTime=30",
            "submodule", "update", "--init", "--", "third_party/dacapo"], 600)
    patches = [root / "scripts/baseline/patches" / n for n in manifest["patch_series"]]
    prefix = patched_prefix(repo, patches, commit, run.folder)
    run.state["patch_prefix_before"] = prefix
    for i, patch in enumerate(patches[prefix:], prefix + 1):
        run.run(f"patch_{i}_check", ["git", "-C", str(repo), "apply", "--check", str(patch)], 30)
        run.run(f"patch_{i}_apply", ["git", "-C", str(repo), "apply", str(patch)], 30)
    if patched_prefix(repo, patches, commit, run.folder) != len(patches):
        raise RuntimeError("Final Dacapo patch verification failed")
    run.upstream_inputs = submodule_fingerprint(repo)
    run.state["dacapo_inputs_sha256"] = run.upstream_inputs
    provision_launcher(run, selected)
    opts = ["--argstr", "platform", selected["id"]]
    raw = run.run("nix_metadata", WRAPPER + ["nix", "eval", "--offline", "--json",
        "--option", "allow-import-from-derivation", "false", *opts, "--file", DEPS, "metadata"], 180)
    metadata = json.loads(raw)
    if metadata["platform_id"] != selected["id"] or metadata["toolchain_profile"] != "hecate":
        raise RuntimeError("Nix resolved a different platform/toolchain")
    bash = metadata["shell_bash"]
    if not re.fullmatch(r"/nix/store/[a-z0-9]{32}-bash-[^/\s]+/bin/bash", bash):
        raise RuntimeError("Invalid pinned Bash path")
    run.state["nix_metadata"] = metadata
    run.run("nix_dry_run", WRAPPER + ["nix", "build", "--dry-run", "--no-link",
        "--option", "allow-import-from-derivation", "false", *opts, "--file", SHELL], 180)
    for stage in ("sources", "seal", "toolchain", "shell"):
        run.run("build_" + stage, ["bash", "scripts/baseline/build_dacapo_dependencies.sh",
                                 "--approved", stage], 43200)
    check_build_caches(work, selected)
    run.run("build_native", ["env", "NIX_BUILD_SHELL=" + bash, *WRAPPER, "nix-shell", "--pure",
        "--keep", "POSEIDON_WORK_ROOT", "--keep", "POSEIDON_PLATFORM",
        "--option", "substitute", "false", "--max-jobs", "0", "--option", "builders", "",
        "--option", "allow-import-from-derivation", "false", *opts, SHELL,
        "--run", "bash scripts/baseline/build_agent_cpu.sh"], 10800)
    venv = work / "venvs/hecate-2.0.1-cpu"
    mode = "--verify-existing" if venv.exists() else "--install-approved"
    run.run("python_environment", [sys.executable, "-B", "scripts/baseline/hecate_python_env.py", mode], 7200)
    run.run("doctor", [sys.executable, "-B", "scripts/agent.py", "--backend", "local",
                      "--platform", selected["id"], "doctor"], 60)
    if self_test:
        run.run("linear_self_test", [sys.executable, "-B", "scripts/agent.py", "--backend", "local",
            "--platform", selected["id"], "candidate", "--",
            "--case", "scripts/baseline/cases/linear-example.json", "--self-test"], 1200)
        output = run.run("mlp_golden", [sys.executable, "-B", "scripts/baseline/seal_cpu_golden.py",
                                      "--case", "mlp4x4x2"], 1200)
        evidence = re.findall(r"^SEAL CPU golden evidence: (.+)$", output, re.M)
        if len(evidence) != 1:
            raise RuntimeError("Cannot identify this batch's golden evidence; preserve results")
        from result_retention import cleanup_run
        run.state["golden_key_cleanup"] = cleanup_run(Path(evidence[0]), work / "results")
        run.state["encrypted_execution_validated"] = True
        run.state["validation_scope"] = "scripted Linear and mlp4x4x2 only; no real provider"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--platform", choices=("auto", "x86_64-linux", "aarch64-linux"),
                   default=os.environ.get("POSEIDON_PLATFORM", "auto"))
    p.add_argument("--work-root", default=os.environ.get("POSEIDON_WORK_ROOT"))
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--plan", action="store_true", help="Offline JSON plan (default)")
    mode.add_argument("--requirements", action="store_true", help="Print selected hash-locked pip requirements")
    mode.add_argument("--apply", action="store_true", help="Authorize the displayed dependency/build stages")
    p.add_argument("--json", action="store_true", help="Machine-readable plan instead of concise summary")
    p.add_argument("--init-submodule", action="store_true", help="Also authorize initialization of pinned Dacapo")
    p.add_argument("--self-test", action="store_true", help="After apply, run two offline encrypted cases")
    p.add_argument("--download-budget-mib", type=int)
    p.add_argument("--disk-budget-gib", type=int)
    args = p.parse_args(argv)
    if args.json and (args.apply or args.requirements):
        p.error("--json is only valid for the offline plan")
    if args.apply and (not args.download_budget_mib or args.download_budget_mib <= 0
                       or not args.disk_budget_gib or args.disk_budget_gib <= 0):
        p.error("--apply requires positive --download-budget-mib and --disk-budget-gib")
    if not args.apply and (args.init_submodule or args.self_test):
        p.error("--init-submodule and --self-test require --apply")
    return args


def main(argv=None):
    try:
        args = parse_args(argv)
        selected = select_platform(args.platform)
        os.environ["POSEIDON_PLATFORM"] = selected["id"]
        from workspace_paths import work_root
        base = work_root(args.work_root)
        os.environ["POSEIDON_WORK_ROOT"] = str(base)
        plan = make_plan(selected, base)
        if not args.apply:
            print(requirements_text(plan) if args.requirements else
                  json.dumps(plan, indent=2) if args.json else plan_summary(plan))
            return 0
        platform_config.require_platform(selected["id"])
        if sys.version_info < (3, 10):
            raise RuntimeError("Setup launcher requires Python >=3.10")
        if Path.cwd().resolve() != ROOT:
            raise RuntimeError("Run from the source checkout root")
        missing = plan["missing_system_commands"]
        if missing or not all(Path(p).is_file() for p in ("/usr/bin/git", "/usr/bin/bwrap")):
            raise RuntimeError("Install system prerequisites separately: " + ", ".join(missing)
                               + "; /usr/bin/git and /usr/bin/bwrap are required")
        work = Path(plan["work_root"])
        # Fail before provisioning when an existing native cache belongs to another architecture.
        check_build_caches(work, selected)
        work.mkdir(parents=True, exist_ok=True)
        (work / "results").mkdir(exist_ok=True)
        import fcntl
        with (work / "agent-setup.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError("Another setup batch owns this work root")
            budget = Budget(work, args.download_budget_mib, args.disk_budget_gib)
            budget.check()
            run = Runner(ROOT, work, plan, budget)
            run.state["approved_limits"] = dict(download_mib=args.download_budget_mib, disk_gib=args.disk_budget_gib)
            def interrupted(signum, frame):
                raise RuntimeError(f"Setup interrupted by signal {signum}")
            previous = signal.signal(signal.SIGTERM, interrupted)
            try:
                build_sequence(run, selected, args.init_submodule, args.self_test)
                run.state.update(status="complete", compiler_execution_validated=True)
            except BaseException as error:
                run.state.update(status="failed", error=str(error))
                raise
            finally:
                signal.signal(signal.SIGTERM, previous)
                run.save()
            print("Setup complete. Report: " + str(run.folder / "report.json"))
            print("Encrypted self-tests: " + ("passed (two cases)" if args.self_test else "not requested"))
            return 0
    except (ValueError, RuntimeError, OSError, subprocess.SubprocessError) as error:
        print("Setup stopped: " + str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
