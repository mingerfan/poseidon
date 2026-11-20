"""Host-owned coordination across checkouts; no provider or backend imports.

Budgets are checked at admission and checkpoints, not a filesystem quota.
Existing candidate sandboxes, native limits and per-task ledgers remain required.
An interrupted API/execution stage stays uncertain and is never replayed here.
"""
from contextlib import contextmanager, ExitStack
import copy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import time
import uuid

FORMAT = "poseidon-shared-execution-v1"
GIB = 1024**3
LIMITS = dict(evidence_bytes=64*GIB, reserve_disk_bytes=4*GIB,
              batch_bytes=4*GIB, compile_jobs=2, link_jobs=1)
KINDS = ("api", "encrypted", "compile", "link", "build", "qualify")
TERMINAL = {"succeeded", "failed", "cancelled", "interrupted"}
STORE_FORMAT = "poseidon-shared-store-accounting-v1"


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def directory(path, *, private=True):
    p = Path(path)
    require(p.is_absolute() and str(p) == os.path.abspath(p), "Coordination absolute directory")
    require(all(not x.is_symlink() for x in (p, *p.parents)), "Indirect coordination directory")
    s = p.stat()
    require(stat.S_ISDIR(s.st_mode) and s.st_uid == os.getuid() and
            not s.st_mode & (0o022 if private else 0o002),
            "Unsafe coordination directory")
    return p


def artifact(path):
    p = directory(path)
    s = p.stat()
    return dict(path=str(p), device=s.st_dev, inode=s.st_ino)


def artifact_paths(batch):
    result = []
    for record in batch["artifacts"]:
        require(artifact(record["path"]) == record, "Batch artifact directory identity changed")
        result.append(record["path"])
    return result


def growth_paths(batch):
    result = []
    scopes = batch.get("growth_scopes", [])
    require(type(scopes) is list and len(scopes) <= 16, "Growth scope records")
    for record in scopes:
        require(type(record) is dict and set(record) == {"path", "device", "inode", "baseline_bytes"} and
                type(record["baseline_bytes"]) is int and record["baseline_bytes"] >= 0,
                "Growth scope baseline")
        require(artifact(record["path"]) == {k: record[k] for k in ("path", "device", "inode")},
                "Growth directory identity changed")
        result.append((Path(record["path"]), record["baseline_bytes"]))
    return result


def batch_bytes(batch):
    fresh = retained_bytes(artifact_paths(batch), reject_links=True)
    growth = sum(max(0, retained_bytes([path], reject_links=True)-baseline)
                 for path, baseline in growth_paths(batch))
    return fresh+growth


def process(pid=None):
    pid = os.getpid() if pid is None else pid
    try:
        text = Path(f"/proc/{pid}/stat").read_text()
    except FileNotFoundError:
        return None
    fields = text[text.rindex(")")+2:].split()
    if fields[0] == "Z":
        return None
    return dict(pid=pid, start_ticks=int(fields[19]),
                boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip())


def retained_bytes(roots, *, reject_links=False):
    total = entries = 0
    pending = [os.fspath(Path(p)) for p in roots]
    while pending:
        parent = pending.pop()
        with os.scandir(parent) as scan:
            for item in scan:
                entries += 1
                require(entries <= 2000000, "Evidence inventory entry budget")
                try:
                    info = item.stat(follow_symlinks=False)
                except FileNotFoundError:
                    continue
                if stat.S_ISLNK(info.st_mode):
                    require(not reject_links, "Indirect batch artifact")
                elif stat.S_ISDIR(info.st_mode):
                    pending.append(item.path)
                elif stat.S_ISREG(info.st_mode):
                    total += info.st_size
                else:
                    require(not reject_links, "Special batch artifact")
    return total


class Coordinator:
    def __init__(self, storage, evidence_roots, *, limits=None):
        p = Path(storage)
        # Validate the existing parent before creating any state.
        directory(p.parent)
        if not p.exists():
            p.mkdir(mode=0o700, exist_ok=True)
        self.storage = directory(p)
        # Historical inventories are read only. New control/output directories
        # still require private ownership; never chmod historical evidence.
        self.roots = tuple(directory(p, private=False) for p in evidence_roots)
        require(1 <= len(self.roots) <= 8 and len(set(self.roots)) == len(self.roots), "Evidence roots")
        require(not any(a != b and a.is_relative_to(b) for a in self.roots for b in self.roots),
                "Overlapping evidence roots")
        self.limits = dict(LIMITS if limits is None else limits)
        require(set(self.limits) == set(LIMITS) and
                all(type(v) is int and v > 0 for v in self.limits.values()), "Shared limits")
        require(all(self.limits[k] <= LIMITS[k] for k in ("evidence_bytes", "batch_bytes")) and
                self.limits["reserve_disk_bytes"] >= LIMITS["reserve_disk_bytes"] and
                all(self.limits[k] == LIMITS[k] for k in ("compile_jobs", "link_jobs")),
                "Shared limit expansion is not permitted")
        self.policy = dict(format=FORMAT, limits=self.limits, evidence_roots=[
            dict(path=str(p), device=p.stat().st_dev, inode=p.stat().st_ino) for p in self.roots])
        self.journal = self.storage/"ledger.json"
        with self._lock("ledger"):
            if self.journal.exists() or self.journal.is_symlink():
                self._read()
            else:
                # A surviving policy file means initialization was interrupted.
                require(not (self.storage/"policy.json").exists(), "Interrupted coordinator initialization")
                self._write(self.storage/"policy.json", self.policy)
                self._write(self.journal, dict(policy=self.policy, batches={}))

    @contextmanager
    def _lock(self, name, *, deadline=None):
        fd = os.open(self.storage/(name+".lock"), os.O_CREAT|os.O_RDWR|os.O_CLOEXEC|os.O_NOFOLLOW, 0o600)
        try:
            info = os.fstat(fd)
            require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1 and
                    info.st_uid == os.getuid() and not info.st_mode & 0o022, "Unsafe shared lock")
            if deadline is None:
                deadline = time.monotonic()+30
            while True:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX|fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    remaining = deadline-time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("Shared execution resources busy")
                    time.sleep(min(0.02, remaining))
            yield
        finally:
            os.close(fd)

    def _write(self, path, value):
        value = copy.deepcopy(value)
        value["binding"] = digest(value)
        raw = canonical(value)
        require(len(raw) <= 16*1024**2, "Shared ledger byte budget")
        temporary = path.with_name(path.name+"."+uuid.uuid4().hex+".tmp")
        with temporary.open("xb") as stream:
            os.chmod(temporary, 0o600)
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        fd = os.open(path.parent, os.O_RDONLY|os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    def _read(self):
        require(self.journal.is_file() and not self.journal.is_symlink() and
                self.journal.stat().st_size <= 16*1024**2, "Shared ledger file")
        data = json.loads(self.journal.read_bytes())
        binding = data.pop("binding", None)
        require(digest(data) == binding and data.get("policy") == self.policy and
                set(data) in ({"policy", "batches"}, {"policy", "batches", "store_accounting"}) and
                type(data["batches"]) is dict,
                "Shared ledger binding/policy")
        for root in self.policy["evidence_roots"]:
            p = directory(root["path"], private=False)
            require((p.stat().st_dev, p.stat().st_ino) == (root["device"], root["inode"]),
                    "Evidence root identity changed")
        self._inventory_roots(data)
        return data

    def _inventory_roots(self, data):
        extension = data.get("store_accounting", dict(format=STORE_FORMAT, roots=[]))
        require(type(extension) is dict and set(extension) == {"format", "roots"} and
                extension["format"] == STORE_FORMAT and type(extension["roots"]) is list and
                len(extension["roots"]) <= 64, "Store accounting version/records")
        result = list(self.roots)
        for record in extension["roots"]:
            require(type(record) is dict and set(record) == {"path", "device", "inode"} and
                    artifact(record["path"]) == record, "Store inventory identity changed")
            path = Path(record["path"])
            require(path != Path(path.anchor) and path != Path.home() and
                    not self.storage.is_relative_to(path) and not path.is_relative_to(self.storage) and
                    all(not path.is_relative_to(p) and not p.is_relative_to(path) for p in result),
                    "Store inventory overlap/scope")
            require(path.stat().st_dev == self.storage.stat().st_dev, "Shared budgets require one filesystem")
            result.append(path)
        return tuple(result)

    def register_inventory(self, path):
        """Count an existing private task store without adopting or altering files."""
        path = directory(path)
        with self._lock("ledger"):
            data = self._read()
            roots = self._inventory_roots(data)
            if any(path.is_relative_to(root) for root in roots):
                return
            require(path != Path(path.anchor) and path != Path.home() and
                    not self.storage.is_relative_to(path) and not path.is_relative_to(self.storage) and
                    not any(root.is_relative_to(path) for root in roots), "Store inventory overlap/scope")
            require(path.stat().st_dev == self.storage.stat().st_dev, "Shared budgets require one filesystem")
            extension = data.setdefault("store_accounting", dict(format=STORE_FORMAT, roots=[]))
            require(len(extension["roots"]) < 64, "Store inventory count")
            extension["roots"].append(artifact(path))
            self._usage(data)
            self._write(self.journal, data)

    def _usage(self, data):
        observed = retained_bytes(self._inventory_roots(data))
        remaining = 0
        for batch in data["batches"].values():
            if batch["status"] == "running":
                used = batch_bytes(batch)
                require(used <= batch["plan"]["reserve_bytes"], "Batch artifact budget exhausted")
                remaining += batch["plan"]["reserve_bytes"]-used
        require(observed+remaining <= self.limits["evidence_bytes"], "Global evidence budget exhausted")
        # All configured roots are required to share this filesystem.
        require(all(p.stat().st_dev == self.storage.stat().st_dev for p in self.roots),
                "Shared budgets require one filesystem")
        require(shutil.disk_usage(self.storage).free >= self.limits["reserve_disk_bytes"]+remaining,
                "Shared disk reserve unavailable")
        return dict(observed_bytes=observed, remaining_reserved_bytes=remaining,
                    enforcement="checkpoints", hard_filesystem_quota=False)

    def register(self, name, plan, output):
        require(type(name) is str and re.fullmatch(r"[A-Za-z0-9_-]{1,96}", name), "Batch name")
        require(type(plan) is dict and set(plan) == {"branch", "source_binding", "reserve_bytes",
                "wall_seconds", "stages"}, "Frozen batch plan fields")
        require(plan["branch"] in {"lhy-generalization", "lhy-resnet20", "lhy-agent-dsl"} and
                type(plan["source_binding"]) is str and re.fullmatch("[0-9a-f]{64}", plan["source_binding"]),
                "Batch source/branch identity")
        require(type(plan["reserve_bytes"]) is int and 1 <= plan["reserve_bytes"] <= self.limits["batch_bytes"] and
                type(plan["wall_seconds"]) is int and 1 <= plan["wall_seconds"] <= 86400,
                "Finite batch resource budget")
        require(type(plan["stages"]) is dict and set(plan["stages"]) == set(KINDS) and
                all(type(v) is int and 0 <= v <= 4096 for v in plan["stages"].values()), "Finite stage budgets")
        output = directory(output)
        require(any(output != r and output.is_relative_to(r) for r in self.roots) and not any(output.iterdir()),
                "Fresh batch output scope")
        with self._lock("ledger"):
            data = self._read()
            require(name not in data["batches"] and len(data["batches"]) < 4096, "Batch already recorded/ledger full")
            require(not any(output.is_relative_to(Path(p)) or Path(p).is_relative_to(output)
                            for b in data["batches"].values() for p in artifact_paths(b)), "Overlapping batch artifacts")
            data["batches"][name] = dict(plan=copy.deepcopy(plan), plan_binding=digest(plan),
                owner=process(), status="running", started=time.time(),
                started_monotonic=time.monotonic(), artifacts=[artifact(output)],
                used={k: 0 for k in KINDS}, active=None, last_stage=None)
            usage = self._usage(data)
            self._write(self.journal, data)
        return Batch(self, name, digest(plan)), usage

    def snapshot(self):
        with self._lock("ledger"):
            data = self._read()
            return copy.deepcopy(data), self._usage(data)

    def recover_interrupted(self, name):
        """Explicit accounting only: preserve used stages and forbid reuse."""
        with self._lock("ledger"):
            data = self._read()
            batch = data["batches"][name]
            require(batch["status"] == "running" and process(batch["owner"]["pid"]) != batch["owner"],
                    "Batch owner is still alive or already terminal")
            batch.update(status="interrupted", finished=time.time(), uncertain_stage=batch["active"])
            self._write(self.journal, data)
            return copy.deepcopy(batch)


class Batch:
    def __init__(self, coordinator, name, binding):
        self.coordinator, self.name, self.binding = coordinator, name, binding

    def _record(self, data):
        batch = data["batches"][self.name]
        require(batch["status"] == "running" and batch["owner"] == process() and
                batch["plan_binding"] == self.binding == digest(batch["plan"]), "Batch ownership/plan")
        return batch

    def record(self):
        """Read this live worker's counters even when a resource check failed."""
        c = self.coordinator
        with c._lock("ledger"):
            return copy.deepcopy(self._record(c._read()))

    def checkpoint(self):
        c = self.coordinator
        with c._lock("ledger"):
            data = c._read()
            batch = self._record(data)
            require(time.monotonic()-batch["started_monotonic"] <= batch["plan"]["wall_seconds"], "Batch wall budget exhausted")
            return c._usage(data)

    def register_artifact(self, path):
        c = self.coordinator
        path = directory(path)
        require(any(path != r and path.is_relative_to(r) for r in c.roots) and not any(path.iterdir()),
                "Fresh registered artifact scope")
        with c._lock("ledger"):
            data = c._read()
            batch = self._record(data)
            if any(path.is_relative_to(Path(p)) for p in artifact_paths(batch)):
                return
            if any(path.is_relative_to(p) for p, _ in growth_paths(batch)):
                return
            require(len(batch["artifacts"]) < 64 and not any(
                path.is_relative_to(Path(p)) or Path(p).is_relative_to(path)
                for b in data["batches"].values() for p in artifact_paths(b)), "Artifact ownership overlap")
            batch["artifacts"].append(artifact(path))
            c._write(c.journal, data)

    def watch_growth(self, path):
        """Observe writes to an existing task scope; historical bytes stay counted.

        This is accounting, not ownership of the pre-existing files. The original
        Application task monitor still enforces the cumulative per-case budget.
        """
        c = self.coordinator
        path = directory(path)
        with c._lock("ledger"):
            data = c._read()
            batch = self._record(data)
            roots = c._inventory_roots(data)
            require(path not in c.roots and any(path.is_relative_to(root) for root in roots),
                    "Registered task growth scope")
            if any(path.is_relative_to(Path(p)) for p in artifact_paths(batch)):
                return
            require(not any(Path(p).is_relative_to(path) for p in artifact_paths(batch)),
                    "Growth/fresh scope overlap")
            own = growth_paths(batch)
            if any(path.is_relative_to(p) for p, _ in own):
                return
            require(not any(p.is_relative_to(path) for p, _ in own), "Growth scope overlap")
            for name, other in data["batches"].items():
                if name != self.name and other["status"] == "running":
                    require(not any(path.is_relative_to(p) or p.is_relative_to(path)
                                    for p, _ in growth_paths(other)), "Task growth scope already active")
            # The extension tag makes older readers reject accounting they do not
            # understand, rather than silently ignoring growth in existing stores.
            data.setdefault("store_accounting", dict(format=STORE_FORMAT, roots=[]))
            scopes = batch.setdefault("growth_scopes", [])
            require(len(scopes) < 16, "Growth scope count")
            scopes.append(dict(**artifact(path), baseline_bytes=retained_bytes([path], reject_links=True)))
            c._usage(data)
            c._write(c.journal, data)

    @contextmanager
    def stage(self, kind, *, timeout=30):
        require(kind in KINDS and type(timeout) in (int, float) and math.isfinite(timeout) and
                0 < timeout <= 1200, "Finite shared stage wait")
        c = self.coordinator
        self.checkpoint()
        with c._lock("ledger"):
            data = c._read()
            b = self._record(data)
            remaining = b["plan"]["wall_seconds"]-(time.monotonic()-b["started_monotonic"])
        deadline = time.monotonic()+min(timeout, remaining)
        # Build/qualification hold both compile slots and the single link slot.
        # This safely covers an existing -j2 build with its -j1 linker pool.
        resources = []
        if kind in {"api", "encrypted", "qualify"}:
            resources += ["api-encrypted"]
        if kind in {"build", "qualify"}:
            resources += ["compile-0", "compile-1", "link"]
        elif kind == "link":
            resources += ["compile-0", "link"]
        with ExitStack() as stack:
            if kind == "compile":
                # All paths use the same files as build/link locks below.
                while True:
                    selected = None
                    for number in (0, 1):
                        trial = ExitStack()
                        try:
                            trial.enter_context(c._lock(f"compile-{number}", deadline=time.monotonic()))
                        except TimeoutError:
                            trial.close()
                        else:
                            selected = trial
                            break
                    if selected is not None:
                        stack.enter_context(selected)
                        break
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Shared compile resources busy")
                    time.sleep(0.02)
            else:
                for name in resources:
                    stack.enter_context(c._lock(name, deadline=deadline))
            self.checkpoint()
            with c._lock("ledger"):
                data = c._read()
                b = self._record(data)
                require(b["active"] is None and b["used"][kind] < b["plan"]["stages"][kind],
                        "Stage active/budget exhausted")
                b["used"][kind] += 1
                b["active"] = dict(kind=kind, number=b["used"][kind], started=time.time())
                c._write(c.journal, data)
            outcome = "failed"
            try:
                yield
                self.checkpoint()
                outcome = "completed"
            finally:
                with c._lock("ledger"):
                    data = c._read()
                    b = self._record(data)
                    b["last_stage"] = dict(b["active"], outcome=outcome, finished=time.time(),
                        execution_uncertain=outcome != "completed" and kind in {"api", "encrypted", "qualify"})
                    b["active"] = None
                    c._write(c.journal, data)

    def finish(self, status):
        require(status in TERMINAL-{"interrupted"}, "Explicit terminal batch status")
        c = self.coordinator
        with c._lock("ledger"):
            data = c._read()
            b = self._record(data)
            require(b["active"] is None, "Cannot finish an active stage")
            used = batch_bytes(b)
            if status == "succeeded":
                require(time.monotonic()-b["started_monotonic"] <= b["plan"]["wall_seconds"] and
                        used <= b["plan"]["reserve_bytes"], "Finished batch exceeds budget")
                c._usage(data)
            b.update(status=status, finished=time.time(), observed_bytes=used)
            c._write(c.journal, data)
            return copy.deepcopy(b)


class CoordinatedBackend:
    """Wrap the existing trusted adapter; do not replace its validation path."""
    def __init__(self, batch, backend):
        self.batch, self.backend = batch, backend

    def __getattr__(self, name):
        return getattr(self.backend, name)

    def _observer(self, observer):
        def register(path):
            self.batch.register_artifact(path)
            if observer is not None:
                observer(path)
        return register

    def qualify(self, request, candidate, level, *, artifact_observer=None):
        with self.batch.stage("qualify"):
            return self.backend.qualify(request, candidate, level,
                artifact_observer=self._observer(artifact_observer))

    def execute(self, folder, samples, *, artifact_observer=None, sample_format="named"):
        with self.batch.stage("encrypted"):
            options = dict(artifact_observer=self._observer(artifact_observer))
            if sample_format != "named":
                options["sample_format"] = sample_format
            return self.backend.execute(folder, samples, **options)


@contextmanager
def coordinated_provider(batch, provider, *, check=None):
    """Lease each generation on the original, type-checked provider instance.

    Application owns the original HTTP hooks and counters. A proxy object would
    fail its existing DeepSeekProvider check, so scope the method change instead.
    """
    original = provider.generate
    had_instance_method = "generate" in vars(provider)
    instance_method = vars(provider).get("generate")

    def generate(request, feedback_history):
        if check is not None:
            check()
        with batch.stage("api"):
            value = original(request, feedback_history)
        if check is not None:
            check()
        return value

    provider.generate = generate
    try:
        yield provider
    finally:
        intact = provider.generate is generate
        if had_instance_method:
            provider.generate = instance_method
        else:
            del provider.generate
        require(intact, "Coordinated provider method changed")
