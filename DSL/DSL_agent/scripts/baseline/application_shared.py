"""Shared admission around the existing Application; its controller stays authoritative."""
from contextlib import contextmanager, nullcontext
from functools import wraps
import copy
import hashlib
import os
from pathlib import Path
from threading import get_ident

from application_component import Application
from application_policy import budget
from application_artifact_budget import FORMAT as LIMIT_FORMAT
from benchmark_graph import digest, require
from shared_execution import (CoordinatedBackend, coordinated_provider, process, GIB)
from shared_worker import environment, scope

FIXED_LIMITS = dict(format=LIMIT_FORMAT, max_new_artifact_bytes=4*GIB, min_free_disk_bytes=4*GIB)


def input_binding(value):
    """Invalid input still reaches the original admission/diagnostic path."""
    try:
        return dict(sha256=digest(value))
    except (ValueError, TypeError, RecursionError, OverflowError):
        return dict(original_contract_validation_required=True)


def authority():
    from application_identity import source_closure
    return source_closure(Path(__file__).resolve().parent, ("application_shared.py",))


class SharedApplication(Application):
    def __init__(self, store, backend):
        super().__init__(store, backend, global_operational_limits=FIXED_LIMITS)
        self._shared_active = None
        self._shared_action = None
        self._shared_task = None
        self._shared_thread = None
        self._held_task_locks = set()
        self._shared_owner = (os.getpid(), get_ident())
        coordinator, _, _ = environment()
        coordinator.register_inventory(self.store)

    @contextmanager
    def _lock(self, path):
        # Keep the original OS task lock through shared final verification. Its
        # original nested run -> source preparation uses the same trusted owner.
        key = (os.getpid(), get_ident(), str(path))
        if key in self._held_task_locks:
            yield
            return
        with super()._lock(path):
            self._held_task_locks.add(key)
            try:
                yield
            finally:
                self._held_task_locks.remove(key)

    def _task_created(self, path):
        if self._shared_active is not None:
            self._shared_active.batch.watch_growth(path)

    def _options(self, requirements):
        from poseidon_runtime_paths import selected_sdk
        from compiler_configuration import CURRENT_CONFIGURATION
        requirements = copy.deepcopy(requirements)
        options = requirements.get("options")
        if selected_sdk() is not None and (options is None or type(options) is dict):
            options = {} if options is None else copy.deepcopy(options)
            selected=selected_sdk()
            if 'native'in selected:
                from compiler_configuration import NATIVE_CONFIGURATION
                options.setdefault("compiler_configuration",NATIVE_CONFIGURATION)
            else:
                options.setdefault("compiler_configuration", CURRENT_CONFIGURATION)
            requirements["options"] = options
        return requirements

    def _task_input(self, task):
        path = self._dir(task)
        records = {}
        for name in ("task.json", "source.json"):
            file = path / name
            if file.exists() or file.is_symlink():
                from model_weight_package import regular_bytes
                raw = regular_bytes(file, 4*1024**2)
                records[name] = dict(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
        return dict(task_store=str(self.store), task_id=task, metadata=records)

    def _bounds(self, task):
        path = self._dir(task)
        for name in ("task.json", "source.json"):
            if (path / name).is_file():
                try:
                    return budget(self._load(path / name)["budget"])
                except (ValueError, TypeError, KeyError, OSError):
                    break
        # Invalid metadata is rejected by the original controller, with no lease
        # available to generate or execute if it were unexpectedly accepted.
        return dict(generations=0, wall_seconds=300)

    def _operation(self, action, binding, body, *, task=None, stages=None, seconds=300, provider=None):
        require(self._shared_owner == (os.getpid(), get_ident()), "Application requires its original worker")
        if self._shared_active is not None:
            require(self._shared_thread == (os.getpid(), get_ident()) and
                    ((self._shared_action == "application-import-submit" and action == "application-submit") or
                     (self._shared_action == "application-run" and action == "application-source-prepare" and
                      self._shared_task == task)), "Nested application operation is unavailable")
            return body()
        original = self.backend
        result = None
        guard = nullcontext() if task is None else self._lock(self._dir(task))
        with guard:
            try:
                with scope(action, binding, stages={} if stages is None else stages,
                           wall_seconds=seconds, inventories=(self.store,),
                           watch=() if task is None else (self._dir(task),), verify_extra=authority) as session:
                    self._shared_active = session
                    self._shared_action, self._shared_task = action, task
                    self._shared_thread = (os.getpid(), get_ident())
                    self.backend = CoordinatedBackend(session.batch, original)
                    try:
                        if task is not None:
                            record = dict(format="poseidon-application-shared-operation-v1", action=action,
                                task_store=str(self.store), task_id=task, operation_input=binding,
                                batch_name=session.batch.name, plan_binding=session.batch.binding,
                                scope=str(session.output), owner=process())
                            record["binding"] = digest(record)
                            self._write(self._dir(task)/(session.batch.name+".json"), record)
                        lease = (coordinated_provider(session.batch, provider)
                                 if provider is not None and getattr(provider, "kind", None) == "deepseek_api"
                                 else nullcontext())
                        with lease:
                            result = body()
                        state = self.status(result) if type(result) is str else result
                        state_status = state.get("status") if type(state) is dict else None
                        session.status = ("succeeded" if state_status in {"queued", "ready", "succeeded", "executed", "passed"}
                                          else "cancelled" if state_status == "cancelled" else "failed")
                    finally:
                        self.backend = original
                        self._shared_active = None
                        self._shared_action = self._shared_task = self._shared_thread = None
                return result
            except Exception:
                self.backend = original
                self._shared_active = None
                self._shared_action = self._shared_task = self._shared_thread = None
                # A failed final shared check cannot leave a successful package in
                # the task/cache. Keep all files; only the mutable task state changes.
                target = result if type(result) is str else task
                if target is not None:
                    path = self._dir(target)
                    state = self._load(path / "state.json")
                    state.update(status="service_error", phase="complete", package=None, acceptance=None,
                        failure=dict(code="shared_execution_unavailable", layer="environment",
                                     owner="host", retry_policy="never_automatic"))
                    self._write(path / "state.json", state)
                    return target if type(result) is str else self.status(target)
                raise

    @wraps(Application.submit)
    def submit(self, model, **requirements):
        requirements = self._options(requirements)
        binding = dict(task_store=str(self.store), submission=input_binding(dict(model=model, requirements=requirements)))
        return self._operation("application-submit", binding,
            lambda: super(SharedApplication, self).submit(model, **requirements))

    @wraps(Application.submit_source)
    def submit_source(self, source, **requirements):
        requirements = self._options(requirements)
        binding = dict(task_store=str(self.store), submission=input_binding(dict(source=source, requirements=requirements)))
        return self._operation("application-source-submit", binding,
            lambda: super(SharedApplication, self).submit_source(source, **requirements))

    @wraps(Application.submit_import)
    def submit_import(self, model_import, **requirements):
        requirements = self._options(requirements)
        binding = dict(task_store=str(self.store), model_import=str(model_import), requirements=input_binding(requirements))
        return self._operation("application-import-submit", binding,
            lambda: super(SharedApplication, self).submit_import(model_import, **requirements))

    @wraps(Application.prepare_source)
    def prepare_source(self, task):
        return self._operation("application-source-prepare", self._task_input(task),
            lambda: super(SharedApplication, self).prepare_source(task), task=task,
            seconds=self._bounds(task)["wall_seconds"])

    @wraps(Application.run)
    def run(self, task, provider):
        if self.status(task)["status"] != "queued":
            return super().run(task, provider)
        bounds = self._bounds(task)
        kind = getattr(provider, "kind", None)
        binding = dict(**self._task_input(task), provider_kind=kind)
        if kind == "deepseek_api":
            from dataclasses import asdict
            from deepseek_provider import DeepSeekProvider
            if isinstance(provider, DeepSeekProvider):
                binding["provider_configuration"] = asdict(provider.config)
        return self._operation("application-run", binding,
            lambda: super(SharedApplication, self).run(task, provider), task=task, provider=provider,
            seconds=bounds["wall_seconds"], stages=dict(qualify=bounds["generations"],
                api=bounds["generations"] if kind == "deepseek_api" else 0))

    def _execution_input(self, task, samples, sample_format):
        try:
            from candidate_bundle import check_execution_dependencies
            from program_execution import package_execution_metadata, normalize_samples
            current = self.result(task)
            require(current["status"] == "succeeded", "Program not available")
            folder = self._dir(current["package"]["task_id"]) / "program"
            manifest, request = check_execution_dependencies(folder)
            if sample_format == "forward_call":
                from pytorch_input_binding import flatten_samples
                metadata = package_execution_metadata(folder, manifest, request["model"])
                samples = flatten_samples(metadata["input_call_plan"], samples)
            else:
                require(sample_format == "named", "Input sample format")
            normalized = normalize_samples(request, samples)
            return input_binding([{name: value.tolist() for name, value in sample.items()} for sample in normalized])
        except (ValueError, TypeError, KeyError, IndexError, OSError):
            return dict(original_contract_validation_required=True)

    @wraps(Application.execute)
    def execute(self, task, samples, *, operational_limits=None, sample_format="named"):
        binding = dict(**self._task_input(task), sample_format=sample_format,
                       inputs=self._execution_input(task, samples, sample_format),
                       operational_limits=input_binding(operational_limits))
        return self._operation("application-execute", binding,
            lambda: super(SharedApplication, self).execute(task, samples,
                operational_limits=operational_limits, sample_format=sample_format),
            task=task, stages=dict(encrypted=1), seconds=self._bounds(task)["wall_seconds"])

    @wraps(Application.recover)
    def recover(self, task):
        result = super().recover(task)
        coordinator, _, _ = environment()
        records = sorted(self._dir(task).glob("shared-component-*.json"))
        require(len(records) <= 4096, "Bounded operation recovery")
        for path in records:
            record = self._load(path)
            require(record.get("binding") == digest({k:v for k,v in record.items() if k != "binding"}) and
                    record["task_store"] == str(self.store) and record["task_id"] == task,
                    "Shared recovery task binding")
            data, _ = coordinator.snapshot()
            batch = data["batches"][record["batch_name"]]
            require(batch["plan_binding"] == record["plan_binding"] and batch["owner"] == record["owner"] and
                    batch["artifacts"][0]["path"] == record["scope"], "Shared recovery operation binding")
            authority = self._load(Path(record["scope"]) / "plan.json")["authority"]
            require(authority["action"] == record["action"] and
                    authority["input_sha256"] == digest(record["operation_input"]) and
                    record["operation_input"]["task_id"] == task and
                    record["operation_input"]["task_store"] == str(self.store), "Shared recovery input binding")
            if batch["status"] == "running":
                coordinator.recover_interrupted(record["batch_name"])
        return result
