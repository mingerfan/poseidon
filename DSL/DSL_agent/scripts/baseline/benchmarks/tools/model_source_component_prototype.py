"""Staged model-source intake over the existing Application.

Development prototype only; not imported by the production application.
Model source is snapshotted at submit and captured only by an explicit worker.
The task ID, cancellation and total active execution budget span both phases.
"""
import copy
import hashlib
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import uuid

from application_component import Application, Stopped, TERMINAL
from application_diagnostics import PRECISE, POLICIES
from application_policy import BACKEND, LEVELS, DEFAULT_BUDGET, budget, prepare
from benchmark_graph import canonical, digest, require
from strict_json_files import strict_file
from model_weight_package import regular_bytes
from model_resources import resolve
from public_weight_context import storage_limit

FORMAT = "poseidon-model-source-request-v1"
SNAPSHOT = "poseidon-model-source-snapshot-v1"
IMPLEMENTATION = "source-intake-controller-prototype-v1"
KINDS = ("graph", "restricted_python", "trusted_pytorch")
REQUIREMENTS = {"backend", "validation_level", "options", "limits", "feedback_policy", "resource_profile"}
MANIFEST_LIMIT = 131072


def own_identity():
    return dict(format=IMPLEMENTATION,
                controller_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _regular(path, limit):
    require(not path.is_symlink() and path.is_file(), "Model source regular file")
    return regular_bytes(path, limit)


def _fields(source):
    require(type(source) is dict and source.get("format") == FORMAT, "Model source version")
    kind = source.get("kind")
    expected = {
        "graph": {"format", "kind", "model"},
        "restricted_python": {"format", "kind", "manifest"},
        "trusted_pytorch": {"format", "kind", "manifest", "trust_local_code"},
    }
    require(kind in expected and set(source) == expected[kind], "Model source fields")
    if kind == "trusted_pytorch":
        require(source["trust_local_code"] is True, "Explicit trusted local source required")
    return kind


def _snapshot(source, profile):
    kind = _fields(source)
    limits = resolve(profile)
    if kind == "graph":
        require(type(source["model"]) is dict, "Graph data")
        raw = canonical(source["model"])
        require(len(raw) <= limits.json_bytes, "Graph source byte capacity")
        return kind, {"model.json": raw}
    manifest_path = Path(source["manifest"])
    manifest = strict_file(manifest_path, MANIFEST_LIMIT)
    parent = manifest_path.parent
    if kind == "trusted_pytorch":
        from pytorch_model_intake import prepare as check_manifest
        from pytorch_weight_package import declarations
        checked, _ = check_manifest(manifest_path)
        require(checked == manifest, "Manifest changed during snapshot")
        weight_records = declarations(manifest.get("weights", {}), resource_profile=profile)
    else:
        from restricted_model_python import FORMAT_PY
        require(profile == "legacy-v1", "Restricted source profile not supported")
        require(type(manifest) is dict and manifest.get("format") == FORMAT_PY,
                "Restricted source manifest version")
        weight_records = {}
    listed = manifest.get("files")
    require(type(listed) is dict and 1 <= len(listed) <= 8, "Explicit source file list")
    files = {"manifest.json": canonical(manifest)}
    total = 0
    for name, h in listed.items():
        require(type(name) is str and re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*\.py", name)
                and type(h) is str and re.fullmatch(r"[0-9a-f]{64}", h), "Flat source file/hash")
        raw = _regular(parent / name, 65536)
        raw.decode("utf-8")
        require(_sha(raw) == h, "Source snapshot hash")
        total += len(raw)
        files[name] = raw
    require(total <= 65536, "Source package bytes")
    for item in weight_records.values():
        name = item["file"]
        raw = _regular(parent / name, item["bytes"])
        require(len(raw) == item["bytes"] and _sha(raw) == item["sha256"],
                "Weight snapshot size/hash")
        prior = files.get(name)
        require(prior is None or prior == raw, "Conflicting source/weight file")
        files[name] = raw
    require(len(files) <= limits.constants + 9, "Source snapshot file count")
    return kind, files


class SourceEnvironmentError(RuntimeError):
    """Trusted worker/environment failure; never a generated-program diagnosis."""


class SourceApplication(Application):
    """Staged explicit source API, retaining existing Application and backend."""
    def submit_source(self, source, **requirements):
        task = uuid.uuid4().hex
        path = self._dir(task)
        path.mkdir(mode=0o700)
        level = requirements.get("validation_level", "numerical")
        state = dict(task_id=task, status="queued", phase="intake_queued",
                     validation_level=level, generations=0, http_attempts=0, ledger=[],
                     failure=None, package=None, acceptance=None, generation=None)
        self._write(path / "state.json", state)
        try:
            require(set(requirements) <= REQUIREMENTS, "Source requirement fields")
            level = requirements.get("validation_level", "numerical")
            target = requirements.get("backend", BACKEND)
            policy = requirements.get("feedback_policy", PRECISE)
            require(level in LEVELS and target == BACKEND and policy in POLICIES,
                    "Source backend/validation/feedback policy")
            _fields(source)
            options = requirements.get("options")
            if options is None:
                options = {}
            from component_contract import OPTION_KEYS
            require(type(options) is dict and set(options) <= OPTION_KEYS, "Source option fields")
            profile = requirements.get("resource_profile", "legacy-v1")
            if source.get("kind") == "graph":
                from benchmark_graph import graph_resource_profile
                model = source.get("model")
                if type(model) is dict and model.get("format") in (
                        "poseidon-model-graph-v1", "poseidon-model-graph-v2"):
                    inferred = graph_resource_profile(model)
                    require("resource_profile" not in requirements or profile == inferred,
                            "Graph source/profile mismatch")
                    profile = inferred
            requested_limits = requirements.get("limits")
            bounds = budget(DEFAULT_BUDGET if requested_limits is None else requested_limits)
            kind, files = _snapshot(source, profile)
            # No input execution, Nix entry or Provider is performed by submit.
            snapshot = path / "source"
            snapshot.mkdir(mode=0o700)
            for name, raw in files.items():
                with (snapshot / name).open("xb") as stream:
                    os.chmod(snapshot / name, 0o600)
                    stream.write(raw)
            identity_policy = "staged-v1" if callable(
                getattr(self.backend, "identity_manifest", None)) else "legacy-v1"
            try:
                identity = self._identity(identity_policy)
            except Exception:
                state.update(status="service_error", phase="complete",
                             failure=self._failure("model_source_environment_unavailable", "environment"))
                self._write(path / "state.json", state)
                return task
            frozen = dict(format=SNAPSHOT, kind=kind, resource_profile=profile,
                          files={name: dict(bytes=len(raw), sha256=_sha(raw))
                                 for name, raw in files.items()},
                          requirements=dict(backend=target, validation_level=level,
                                            feedback_policy=policy, options=copy.deepcopy(options)),
                          budget=bounds, identity_policy=identity_policy, identity=identity,
                          implementation=own_identity())
            frozen["binding"] = digest(frozen)
            require(len(canonical(frozen)) <= 4 * 1024**2, "Source metadata capacity")
            self._write(path / "source.json", frozen)
            state["intake"] = dict(kind=kind, status="queued", reference_verified=False,
                                  active_seconds=0., encrypted_execution=False, agent_calls=0)
            self._write(path / "state.json", state)
        except (ValueError, TypeError, KeyError, IndexError, OSError, SyntaxError):
            state.update(status="unsupported", phase="complete",
                         failure=self._failure("model_source_unsupported", "input"))
            self._write(path / "state.json", state)
        return task

    @staticmethod
    def _failure(code, layer):
        return dict(code=code, layer=layer, location=None, context={},
                    owner="host" if layer in ("environment", "integrity") else "model_or_intake",
                    retry_policy="never_automatic")

    def status(self, task):
        result = super().status(task)
        state = self._load(self._dir(task) / "state.json")
        if "intake" in state:
            result["intake"] = copy.deepcopy(state["intake"])
        return result

    def _source(self, path):
        spec = self._load(path / "source.json")
        require(type(spec) is dict and set(spec) == {
            "format", "kind", "resource_profile", "files", "requirements", "budget",
            "identity_policy", "identity", "implementation", "binding"},
            "Source task metadata")
        require(spec["format"] == SNAPSHOT and spec["kind"] in KINDS and
                digest({k: v for k, v in spec.items() if k != "binding"}) == spec["binding"],
                "Source task binding")
        require(spec["implementation"] == own_identity(), "Source controller changed")
        budget(spec["budget"])
        limits = resolve(spec["resource_profile"])
        require(type(spec["files"]) is dict and
                1 <= len(spec["files"]) <= limits.constants + 9, "Source inventory")
        folder = path / "source"
        require(folder.is_dir() and not folder.is_symlink(), "Source snapshot folder")
        require(set(p.name for p in folder.iterdir()) == set(spec["files"]),
                "Source snapshot inventory changed")
        for name, record in spec["files"].items():
            require(type(name) is str and re.fullmatch(r"[A-Za-z][A-Za-z0-9_.-]*", name),
                    "Source snapshot path")
            require(type(record) is dict and set(record) == {"bytes", "sha256"} and
                    type(record["bytes"]) is int and 0 < record["bytes"] <=
                    max(limits.json_bytes, limits.total_constant_elements * 8 + 65536),
                    "Source snapshot size declaration")
            raw = _regular(folder / name, record["bytes"])
            require(len(raw) == record["bytes"] and _sha(raw) == record["sha256"],
                    "Source snapshot changed")
        try:
            actual = self._identity(spec["identity_policy"])
        except Exception as error:
            raise SourceEnvironmentError("Source backend unavailable") from error
        require(actual == spec["identity"], "Source backend identity changed")
        return spec

    def _capture(self, path, spec, check):
        from hecate_python_env import VENV, ROOT
        if os.environ.get("IN_NIX_SHELL") != "pure" or Path(sys.prefix) != VENV:
            raise SourceEnvironmentError("Pinned source worker required")
        output = path / "model-import"
        flag = {"graph": "--graph", "restricted_python": "--python-manifest",
                "trusted_pytorch": "--trusted-pytorch-manifest"}[spec["kind"]]
        source = path / "source" / ("model.json" if spec["kind"] == "graph" else "manifest.json")
        cmd = [str(VENV / "bin/python"), "-B", str(ROOT / "scripts/prepare_model.py"),
               flag, str(source), "--resource-profile", spec["resource_profile"],
               "--write", "--inside", "--output", str(output)]
        if spec["kind"] == "trusted_pytorch":
            cmd.append("--trust-local-code")
        allowed = ("PATH", "IN_NIX_SHELL", "POSEIDON_PLATFORM", "POSEIDON_WORK_ROOT",
                   "HECATE_PYTHON_LIBRARY_PATH", "LANG")
        env = {k: os.environ[k] for k in allowed if k in os.environ}
        env.update(LD_LIBRARY_PATH=os.environ["HECATE_PYTHON_LIBRARY_PATH"],
                   PYTHONPATH=str(ROOT / "scripts/baseline"), PYTHONDONTWRITEBYTECODE="1",
                   PYTHONNOUSERSITE="1", HOME="/nonexistent",
                   OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
        from process_resource_limits import limits
        proc = None
        with (path / "source-import.log").open("xb") as log:
            try:
                proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT,
                                        close_fds=True, start_new_session=True, preexec_fn=limits)
                while proc.poll() is None:
                    check()
                    time.sleep(.025)
                check()
                require(proc.returncode == 0, "Isolated model source import failed")
            finally:
                if proc is not None and proc.poll() is None:
                    # Capture bwrap uses --die-with-parent. Killing its direct
                    # supervisor also ends its independent sandbox session.
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
        return output

    def _admit(self, path, spec, state, model, provenance, deadline):
        options = copy.deepcopy(spec["requirements"]["options"])
        if provenance is not None:
            from pytorch_model_intake import preparation_options
            from operator_arithmetic_graph import preparation_options as graph_options
            options = preparation_options(provenance, model, options)
            options = graph_options(provenance["original"], model, options)
        req = spec["requirements"]
        request, context = prepare(model, req["backend"], req["validation_level"], options)
        if provenance is not None:
            from candidate_bundle import check_provenance
            require(len(canonical(provenance)) <= storage_limit(request), "Model provenance capacity")
            check_provenance(provenance, request["model"])
        remaining = math.floor(deadline - time.monotonic())
        if remaining < 1:
            raise Stopped("timed_out")
        bounds = dict(spec["budget"], wall_seconds=remaining)
        budget(bounds)
        meta = dict(request=request, context=context, budget=bounds, identity=spec["identity"],
                    identity_policy=spec["identity_policy"], validation_level=req["validation_level"],
                    feedback_policy=req["feedback_policy"],
                    model_source=dict(format=FORMAT, snapshot_binding=spec["binding"],
                                      implementation=spec["implementation"]))
        key = dict(request_id=request["request_id"], identity=spec["identity"],
                   validation_level=req["validation_level"], model_source=meta["model_source"])
        if provenance is not None:
            meta["model_provenance"] = copy.deepcopy(provenance)
            key["model_provenance_sha256"] = digest(provenance)
        meta["binding"] = digest(meta)
        require(len(canonical(meta)) <= 4 * 1024**2, "Source task storage capacity")
        self._write(path / "task.json", meta)
        state.update(status="queued", phase="queued", reuse_key=digest(key))
        self._write(path / "state.json", state)

    def prepare_source(self, task):
        """Host worker phase only. Does not generate, compile or execute FHE."""
        path = self._dir(task)
        with self._lock(path):
            state = self._load(path / "state.json")
            if state["status"] in TERMINAL or state.get("phase") == "queued":
                return self.status(task)
            require(state["status"] == "queued" and state["phase"] == "intake_queued",
                    "Recover interrupted intake explicitly")
            try:
                spec = self._source(path)
            except SourceEnvironmentError:
                state.update(status="service_error", phase="complete",
                             failure=self._failure("model_source_environment_unavailable", "environment"))
                self._write(path / "state.json", state)
                return self.status(task)
            except (ValueError, TypeError, KeyError, OSError):
                state.update(status="service_error", phase="complete",
                             failure=self._failure("model_source_integrity_rejected", "integrity"))
                self._write(path / "state.json", state)
                return self.status(task)
            started = time.monotonic()
            deadline = started + spec["budget"]["wall_seconds"]
            def check(*_):
                stop = "cancelled" if (path / "cancel.json").exists() else (
                    "timed_out" if time.monotonic() >= deadline else None)
                if stop is not None:
                    # Cleanup cannot be interrupted by a second alarm while
                    # waiting for the capture supervisor to exit.
                    signal.setitimer(signal.ITIMER_REAL, 0)
                    raise Stopped(stop)
            previous = None
            state.update(status="running", phase="model_import")
            self._write(path / "state.json", state)
            try:
                require(signal.getitimer(signal.ITIMER_REAL) == (0., 0.), "Worker owns an alarm")
                previous = signal.signal(signal.SIGALRM, check)
                signal.setitimer(signal.ITIMER_REAL, .1, .1)
                check()
                kind = spec["kind"]
                source = path / "source"
                model = strict_file(source / "model.json", resolve(spec["resource_profile"]).json_bytes) \
                    if kind == "graph" else None
                provenance = None
                # Already normalized JSON uses the original preparation contract.
                if model is None or model.get("format") not in (
                        "poseidon-model-graph-v1", "poseidon-model-graph-v2"):
                    output = self._capture(path, spec, check)
                    from candidate_bundle import load_model_provenance
                    from public_weight_context import POLICY, TRUSTED_BYTES
                    # Input graph capacity and accepted component provenance
                    # storage are separate, existing policies.
                    limit = TRUSTED_BYTES if spec["requirements"]["options"].get(
                        "public_weight_policy") == POLICY else 131072
                    model = strict_file(output / "model.json", limit)
                    provenance = load_model_provenance(output, model, max_bytes=limit)
                check()
                self._source(path)
                elapsed = time.monotonic() - started
                remaining = math.floor(spec["budget"]["wall_seconds"] - elapsed)
                if remaining < 1:
                    raise Stopped("timed_out")
                state["intake"].update(status="prepared", reference_verified=provenance is not None,
                                       active_seconds=elapsed)
                self._admit(path, spec, state, model, provenance, deadline)
                state["intake"]["active_seconds"] = time.monotonic() - started
                self._write(path / "state.json", state)
                check()
            except Stopped as stop:
                state.update(status=stop.status, phase="complete", failure=None)
                state["intake"].update(status=stop.status, active_seconds=time.monotonic() - started)
                self._write(path / "state.json", state)
            except SourceEnvironmentError:
                state.update(status="service_error", phase="complete",
                             failure=self._failure("model_source_environment_unavailable", "environment"))
                state["intake"].update(status="failed", active_seconds=time.monotonic() - started)
                self._write(path / "state.json", state)
            except (ValueError, TypeError, KeyError, IndexError, OSError):
                state.update(status="validation_failed", phase="complete",
                             failure=self._failure("model_source_intake_rejected", "input"))
                state["intake"].update(status="rejected", active_seconds=time.monotonic() - started)
                self._write(path / "state.json", state)
            finally:
                if previous is not None:
                    signal.setitimer(signal.ITIMER_REAL, 0)
                    signal.signal(signal.SIGALRM, previous)
        return self.status(task)

    def _identity_matches(self, meta, *, delivery=False):
        source = meta.get("model_source")
        if source is not None and source.get("implementation") != own_identity():
            return False
        return super()._identity_matches(meta, delivery=delivery)

    def run(self, task, provider):
        if (self._dir(task) / "source.json").exists():
            self.prepare_source(task)
            if self.status(task)["status"] in TERMINAL:
                return self.result(task)
        return super().run(task, provider)
