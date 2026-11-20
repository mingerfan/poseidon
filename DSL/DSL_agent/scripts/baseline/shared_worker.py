"""Default shared admission for trusted component workers.

Existing qualification, sandbox, native limits and numerical audits stay in
component_backend. This module owns host-only leases and fresh output scopes.
It never constructs a Provider, loads credentials or changes SDK configuration.
"""
from contextlib import contextmanager
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from shared_execution import Coordinator, KINDS, digest, require

FORMAT = "poseidon-shared-worker-v1"
ACTIONS = {"candidate-cli", "component-qualify", "application-submit", "application-source-submit",
           "application-import-submit", "application-source-prepare", "application-run", "application-execute"}


def environment():
    from hecate_python_env import ROOT, WORK, VENV
    from workspace_paths import WORK_BASE, require_linux_backend
    require_linux_backend()
    require(os.environ.get("IN_NIX_SHELL") == "pure" and Path(sys.prefix) == VENV,
            "Pinned shared worker required")
    branch = subprocess.check_output(
        ["/usr/bin/git", "--no-optional-locks", "-C", str(ROOT), "branch", "--show-current"],
        text=True, timeout=10, env={"PATH":"/usr/bin:/bin","LANG":"C.UTF-8"}).strip()
    require(branch in {"lhy-generalization", "lhy-resnet20", "lhy-agent-dsl"},
            "Shared worker checkout branch")
    coordinator = Coordinator(WORK_BASE / "coordination/v1", [WORK / "results"])
    return coordinator, WORK / "results", branch


def snapshot():
    """Read current host code and actual SDK bytes; never cache the authority."""
    from hecate_python_env import ROOT
    from application_identity import source_closure
    from poseidon_backend_identity import identity
    baseline = Path(__file__).resolve().parent
    head = subprocess.check_output(
        ["/usr/bin/git", "--no-optional-locks", "-C", str(ROOT), "rev-parse", "HEAD"],
        text=True, timeout=10, env={"PATH":"/usr/bin:/bin","LANG":"C.UTF-8"}).strip()
    branch = subprocess.check_output(
        ["/usr/bin/git", "--no-optional-locks", "-C", str(ROOT), "branch", "--show-current"],
        text=True, timeout=10, env={"PATH":"/usr/bin:/bin","LANG":"C.UTF-8"}).strip()
    roots = ["component_worker.py", "shared_worker.py"]
    return dict(source_root=str(ROOT), git_head=head, branch=branch, backend_identity=identity(),
                source_closure=source_closure(baseline, roots))


def write(path, value):
    raw = json.dumps(value, sort_keys=True, indent=2, allow_nan=False).encode()+b"\n"
    require(len(raw) <= 16*1024**2, "Shared worker receipt size")
    with Path(path).open("wb") as stream:
        os.chmod(path, 0o600)
        stream.write(raw)


class Session:
    def __init__(self, batch, output, frozen, verify_extra=None):
        self.batch, self.output, self.frozen = batch, output, frozen
        self.verify_extra = verify_extra
        self.status = "failed"

    def verify(self):
        require(snapshot() == self.frozen["host"], "Shared worker source/SDK changed")
        if self.verify_extra is not None:
            require(self.verify_extra() == self.frozen["extra_authority"], "Shared caller source changed")
        self.batch.checkpoint()

    def observe(self, path):
        # Qualification invokes this while the native evidence directory is empty.
        self.batch.register_artifact(path)

    def result(self, result):
        self.status = ("succeeded" if result.get("status") == "passed" else
                       "cancelled" if result.get("status") == "cancelled" else "failed")


@contextmanager
def scope(action, binding, *, stages, wall_seconds=900, inventories=(), watch=(), verify_extra=None):
    require(action in ACTIONS, "Known shared worker action")
    coordinator, results, branch = environment()
    for path in inventories:
        coordinator.register_inventory(path)
    frozen = dict(format=FORMAT, action=action, input_sha256=digest(binding), host=snapshot())
    if verify_extra is not None:
        frozen["extra_authority"] = verify_extra()
    require(frozen["host"].get("branch", branch) == branch, "Shared worker branch changed")
    plan = dict(branch=branch, source_binding=digest(frozen),
                reserve_bytes=coordinator.limits["batch_bytes"], wall_seconds=wall_seconds,
                stages={kind: stages.get(kind, 0) for kind in KINDS})
    # mkdtemp creates mode 0700. Admission precedes every candidate/native write.
    output = Path(tempfile.mkdtemp(prefix="shared-component-", dir=results))
    batch, usage = coordinator.register(output.name, plan, output)
    session = Session(batch, output, frozen, verify_extra)
    try:
        for path in watch:
            batch.watch_growth(path)
        write(output / "plan.json", dict(plan=plan, plan_binding=batch.binding,
             authority=frozen, initial_usage=usage))
        session.verify()
        yield session
        session.verify()
    except BaseException as error:
        from component_control import QualificationCancelled
        if isinstance(error, (KeyboardInterrupt, QualificationCancelled)):
            session.status = "cancelled"
        else:
            session.status = "failed"
        raise
    finally:
        used = batch.record()["used"]
        write(output / "report.json", dict(format=FORMAT, status=session.status,
             batch_name=batch.name, plan_binding=batch.binding,
             input_sha256=frozen["input_sha256"], stage_usage=used,
             api_generation_attempts=used["api"]))
        try:
            batch.finish(session.status)
        except BaseException:
            # Failed final admission must never leave a successful receipt.
            write(output / "report.json", dict(format=FORMAT, status="failed",
                 batch_name=batch.name, plan_binding=batch.binding,
                 input_sha256=frozen["input_sha256"], stage_usage=used,
                 api_generation_attempts=used["api"]))
            if session.status == "succeeded":
                batch.finish("failed")
            raise


def qualify(request, candidate):
    """One unchanged component qualification under default global admission."""
    from component_backend import qualify as original
    request, candidate = copy.deepcopy(request), copy.deepcopy(candidate)
    binding = dict(request=request, candidate=candidate, validation_level="numerical")
    before = digest(binding)
    with scope("component-qualify", binding, stages=dict(qualify=1)) as session:
        with session.batch.stage("qualify"):
            result = original(request, candidate, artifact_observer=session.observe)
            require(digest(binding) == before, "Shared worker input changed")
        session.result(result)
    return result
