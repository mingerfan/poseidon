"""Opt-in request artifact monitor owned by the trusted application controller.

Counts retained regular-file bytes at checkpoints, with a nonrefundable observed
peak across intake and repair phases. This is not a filesystem quota: transient
writes between checkpoints and a final interrupted write can exceed the limit.
No cleanup, network, runtime modification or candidate-supplied path is used.
"""
import copy
import os
from pathlib import Path
import re
import shutil
import stat
import time
from benchmark_graph import digest, require

FORMAT = "poseidon-operational-limits-v1"
MONITOR = "poseidon-artifact-monitor-v1"
JOURNAL = "artifact-usage.json"
MAX_ROOTS = 64
MAX_ENTRIES = 200000
PREFIXES = ("candidate-replay-", "poseidon-program-execution-")

def limits(value):
    require(type(value) is dict and set(value) == {
        "format", "max_new_artifact_bytes", "min_free_disk_bytes"}, "Operational fields")
    require(value["format"] == FORMAT, "Operational format")
    require(type(value["max_new_artifact_bytes"]) is int and
            1 <= value["max_new_artifact_bytes"] <= 2**63-1, "Artifact budget")
    require(type(value["min_free_disk_bytes"]) is int and
            0 <= value["min_free_disk_bytes"] <= 2**63-1, "Free disk reserve")
    return copy.deepcopy(value)

def failure(reason):
    mapping = {
        "artifact_bytes": ("budget_exhausted", "resource", "artifact_budget_exhausted"),
        "disk_reserve": ("service_error", "environment", "disk_reserve_unavailable"),
        "backend_monitor": ("service_error", "environment", "artifact_monitor_unavailable"),
        "integrity": ("service_error", "integrity", "artifact_accounting_rejected"),
        "storage": ("service_error", "environment", "artifact_storage_unavailable"),
    }
    require(reason in mapping, "Operational reason")
    status, layer, code = mapping[reason]
    return status, dict(format=MONITOR, code=code, layer=layer, reason_code=reason,
                        owner="host", retry_policy="never_automatic",
                        location=None, context={})

class OperationalStopped(BaseException):
    def __init__(self, reason):
        self.status, self.public_failure = failure(reason)

def _directory(path):
    path = Path(path)
    require(path.is_absolute() and path == Path(os.path.abspath(path)), "Artifact absolute path")
    # Reject symlink ancestors as well as the directory itself.
    for ancestor in (path, *path.parents):
        require(not ancestor.is_symlink(), "Indirect artifact directory")
    data = path.lstat()
    require(stat.S_ISDIR(data.st_mode), "Artifact directory")
    return dict(path=str(path), device=data.st_dev, inode=data.st_ino)

class ArtifactMonitor:
    def __init__(self, app, folder, value, *, initialize=False):
        self.app = app
        self.folder = Path(folder)
        self.value = limits(value)
        self.active = False
        self.last_check = float("-inf")
        try:
            require(getattr(app.backend, "artifact_monitor_version", None) == MONITOR,
                    "Backend artifact observer contract")
            roots = app.backend.operational_artifact_roots()
            require(type(roots) in (tuple, list) and 1 <= len(roots) <= 4, "Backend roots")
            self.allowed = tuple(Path(r) for r in roots)
            for root in self.allowed:
                _directory(root)
        except Exception:
            # Trusted adapter failures are environment failures; never include
            # their message or route them back to generated-source repair.
            raise OperationalStopped("backend_monitor") from None
        try:
            own = _directory(self.folder)
            self.journal = self.folder / JOURNAL
            if self.journal.exists() or self.journal.is_symlink():
                data = app._load(self.journal)
                require(type(data) is dict and set(data) == {
                    "format", "limits", "roots", "observed_current_bytes",
                    "observed_peak_bytes", "binding"}, "Artifact journal fields")
                require(data["format"] == MONITOR and data["limits"] == self.value and
                        digest({k: v for k,v in data.items() if k != "binding"}) == data["binding"],
                        "Artifact journal binding")
                require(type(data["roots"]) is list and 1 <= len(data["roots"]) <= MAX_ROOTS
                        and data["roots"][0] == own, "Artifact owner")
                require(all(type(data[k]) is int and 0 <= data[k] <= 2**63-1
                            for k in ("observed_current_bytes", "observed_peak_bytes")) and
                        data["observed_current_bytes"] <= data["observed_peak_bytes"], "Artifact journal counts")
                self.data = data
            else:
                require(initialize is True,"Missing artifact journal on continuation")
                self.data = dict(format=MONITOR, limits=self.value, roots=[own],
                                 observed_current_bytes=0, observed_peak_bytes=0)
            self._verify_roots()
            self._persist()
        except (ValueError, TypeError, KeyError, OSError):
            raise OperationalStopped("integrity") from None

    def _verify_roots(self):
        require(len(self.data["roots"]) <= MAX_ROOTS, "Artifact directory limit")
        seen = set()
        for index, record in enumerate(self.data["roots"]):
            require(type(record) is dict and set(record) == {"path", "device", "inode"}, "Artifact root record")
            require(type(record["path"]) is str and all(type(record[k]) is int and record[k] >= 0
                    for k in ("device", "inode")), "Artifact root identity")
            path = Path(record["path"])
            try:
                actual = _directory(path)
            except OSError:
                raise ValueError("Artifact directory identity unavailable") from None
            require(actual == record and str(path) not in seen, "Artifact directory changed")
            seen.add(str(path))
            if index:
                require(path.parent in self.allowed and any(path.name.startswith(p) for p in PREFIXES)
                        and re.fullmatch(r"[A-Za-z0-9_-]+", path.name), "Artifact evidence scope")
                # No overlap with task roots; accounting must never adopt another
                # request, a package cache or arbitrary historical directory.
                require(not path.is_relative_to(self.folder) and not self.folder.is_relative_to(path),
                        "Overlapping artifact roots")

    def _persist(self):
        self.data["binding"] = digest({k: v for k,v in self.data.items() if k != "binding"})
        self.app._write(self.journal, self.data)

    def usage(self):
        return dict(format=MONITOR, enforcement="monitored-v1", hard_filesystem_quota=False,
                    observed_current_bytes=self.data["observed_current_bytes"],
                    observed_peak_bytes=self.data["observed_peak_bytes"],
                    max_new_artifact_bytes=self.value["max_new_artifact_bytes"],
                    min_free_disk_bytes=self.value["min_free_disk_bytes"],
                    evidence_directories=len(self.data["roots"])-1)

    def register(self, directory):
        """Host callback, invoked immediately after a fresh evidence mkdir."""
        # Protect registration/persistence from an asynchronous timer reentry.
        prior = self.active
        self.active = True
        try:
            record = _directory(Path(directory))
            path = Path(record["path"])
            require(path.parent in self.allowed and any(path.name.startswith(p) for p in PREFIXES)
                    and re.fullmatch(r"[A-Za-z0-9_-]+", path.name), "Fresh evidence directory scope")
            require(len(self.data["roots"]) < MAX_ROOTS and not any(path.iterdir()), "Fresh empty evidence directory")
            require(record not in self.data["roots"], "Evidence already registered")
            self.data["roots"].append(record)
            self._verify_roots()
            self._persist()
        except (ValueError, TypeError, OSError):
            raise OperationalStopped("integrity") from None
        finally:
            self.active = prior
        self.check(force=True)

    def _size(self):
        total = 0
        entries = 0
        for record in self.data["roots"]:
            pending = [Path(record["path"])]
            while pending:
                parent = pending.pop()
                with os.scandir(parent) as scan:
                    for item in scan:
                        entries += 1
                        require(entries <= MAX_ENTRIES, "Artifact inventory budget")
                        try:
                            data = item.stat(follow_symlinks=False)
                        except FileNotFoundError:
                            continue
                        require(not stat.S_ISLNK(data.st_mode), "Indirect artifact entry")
                        if stat.S_ISDIR(data.st_mode):
                            pending.append(Path(item.path))
                        else:
                            require(stat.S_ISREG(data.st_mode), "Artifact regular file")
                            total += data.st_size
                            require(total <= 2**63-1, "Artifact byte range")
        return total

    def check(self, *, force=False):
        if self.active or not force and time.monotonic()-self.last_check < .25:
            return
        self.active = True
        try:
            try:
                self._verify_roots()
                current = self._size()
                self.data["observed_current_bytes"] = current
                self.data["observed_peak_bytes"] = max(current, self.data["observed_peak_bytes"])
                self._persist()
            except (ValueError, TypeError, KeyError):
                raise OperationalStopped("integrity") from None
            except OSError:
                raise OperationalStopped("storage") from None
            if self.data["observed_peak_bytes"] > self.value["max_new_artifact_bytes"]:
                raise OperationalStopped("artifact_bytes")
            try:
                locations = (self.folder, *(Path(r["path"]) for r in self.data["roots"][1:]))
                if any(shutil.disk_usage(p).free < self.value["min_free_disk_bytes"] for p in locations):
                    raise OperationalStopped("disk_reserve")
            except OSError:
                raise OperationalStopped("storage") from None
            self.last_check = time.monotonic()
        finally:
            self.active = False
