"""One-request Poseidon qualification, shared with the existing CLI validation path.

A session consumes a frozen public request and candidate. Generation, transport,
repair budgets and scheduling remain with the caller's existing controller.
"""
import copy
import json
import os
import tempfile
from pathlib import Path

from candidate_contract import ReplayProvider, request_file_bytes, run_feedback_loop, strict_json
from component_control import QualificationCancelled
from application_artifact_budget import OperationalStopped
from hecate_python_env import ROOT, WORK, VENV, digest
from numerical_results import require, dump
from native_execution_slots import native_slot, NATIVE_CONCURRENCY
from platform_config import identity, require_python_packages
from poseidon_runtime_paths import BUILD, CPU_BUILD, DACAPO, POSEIDON_BUILD, PROFILE
from compiler_configuration import verify_execution_configuration

SOURCE = ROOT / "scripts/baseline/golden_cases"


def read_json(path):
    from candidate_contract import MAX_BYTES
    require(path.is_file() and not path.is_symlink() and path.stat().st_size <= MAX_BYTES,
            "JSON file size limit")
    return strict_json(path.read_text())


def bind_backend(report, request, waterline):
    """Record the same installed tools and immutable execution configuration."""
    from poseidon_backend_identity import identity as backend_identity
    actual = backend_identity()
    require("backend_identity" not in report or report["backend_identity"] == actual,
            "Poseidon backend identity changed during preparation")
    config = request.get("compiler_configuration")
    require(type(config) is dict and config.get("backend") == "poseidon_cpu",
            "Poseidon CPU configuration required")
    report["backend_identity"] = actual
    report.update(request_id=request["request_id"], compiler_profile_sha256=digest(PROFILE),
                  waterline=waterline, compiler_configuration=copy.deepcopy(config),
                  compiler_sha256=digest(BUILD / "bin/hecate-opt"),
                  runtime_sha256=digest(CPU_BUILD / "poseidon_agent_cpu"),
                  poseidon_library_sha256=digest(POSEIDON_BUILD / "libposeidon_shared.so"),
                  operator_spec_sha256=digest(ROOT / "scripts/baseline/poseidon-operator-spec-v1.json"),
                  frontend_sha256=digest(BUILD / "lib/libHecateFrontend.so"),
                  frontend_source_sha256=digest(DACAPO / "tools/frontend.cpp"),
                  frontend_python_sha256=digest(DACAPO / "python/hecate/hecate/expr.py"))
    require(config["source_lock_sha256"] == actual["source_lock_sha256"]
            and config["operator_spec_sha256"] == actual["operator_spec_sha256"],
            "Request dependency lock/spec differs from installed backend")
    verify_execution_configuration(request, digest(PROFILE), waterline)


class QualificationSession:
    """Own the existing probe, slot lock, frozen inputs and validation adapter."""
    def __init__(self, result, request, report, frozen, fixed, waterline, *,
                 helper_environment=None, unified=True, validation_level="numerical"):
        from application_identity import source_closure, source_catalog, _catalog_values, CATALOG
        require(validation_level in ("compiled", "numerical"), "Validation level")
        result = Path(result)
        require(result.is_dir() and not result.is_symlink()
                and result.resolve().is_relative_to((WORK / "results").resolve()),
                "Platform qualification directory")
        self.result, self.request, self.report = result, copy.deepcopy(request), report
        self.frozen, self.fixed, self.waterline = frozen, copy.deepcopy(fixed), waterline
        self.helper_environment = copy.deepcopy(helper_environment)
        self.validation_level, self.unified = validation_level, unified
        bind_backend(report, self.request, waterline)
        require(self.fixed["atol"] == 1e-5 and self.fixed["rtol"] == 1e-4,
                "Frozen numerical tolerance")
        report["tolerance"] = {k: self.fixed[k] for k in ("atol", "rtol")}
        report.setdefault("native_resources", dict(concurrency=NATIVE_CONCURRENCY,
                          acquisitions=0, wait_seconds=0.0))
        baseline = ROOT / "scripts/baseline"
        catalog = source_catalog(baseline)
        self._authorities = _catalog_values(catalog)[2]
        # Existing CLI source observations stay bound when it uses this session.
        # Direct application qualification starts from the shared validation roots.
        declared = copy.deepcopy(report.get("source_hashes", {}))
        require(type(declared) is dict, "Declared validation sources")
        from application_identity import _read
        import hashlib
        require(all(hashlib.sha256(_read(baseline, name)).hexdigest() == h
                    for name, h in declared.items()), "Declared validation source changed")
        self._roots = tuple(sorted(set((Path(__file__).name, "validation_adapter.py",
                          *(name for name in declared if name.endswith(".py"))))))
        self._closure = source_closure(baseline, self._roots,
                                       dynamic_authorities=self._authorities)
        self._declared_assets = {name: h for name, h in declared.items()
                                 if not name.endswith(".py")}
        report["source_hashes"] = dict(self._declared_assets, **self._closure["sources"])
        self._catalog_sha256 = digest(baseline / CATALOG)
        self._fixture_sha256 = digest(SOURCE / "fixtures.json")
        self._bound = {k: copy.deepcopy(report[k]) for k in (
            "backend_identity", "request_id", "compiler_profile_sha256", "waterline",
            "compiler_configuration", "compiler_sha256", "runtime_sha256",
            "poseidon_library_sha256", "operator_spec_sha256", "frontend_sha256",
            "frontend_source_sha256", "frontend_python_sha256", "tolerance", "source_hashes")}
        report["qualification"] = dict(format="poseidon-qualification-session-v1",
            source_catalog_sha256=self._catalog_sha256, validation_level=validation_level,
            generation_performed=False)
        from validation_adapter import CandidateValidationAdapter, ValidationContext
        self.validation = CandidateValidationAdapter(ValidationContext(
            result=result, request=self.request, report=report, frozen=frozen,
            fixed=self.fixed, waterline=waterline, keys=None, unified=unified,
            native_run=self.native_run, verify_frozen=self.verify_frozen,
            validation_level=validation_level))

    def native_run(self, *args, **kwargs):
        import poseidon_candidate_sandbox as sandbox
        if self.helper_environment is not None:
            kwargs["helpers"] = True
        with native_slot(WORK / "cache/agent-native-slots", metrics=self.report["native_resources"]):
            return sandbox.run(*args, **kwargs)

    def preflight(self):
        """Run the existing sandbox sentinel/network/process isolation probe."""
        self.verify_frozen()
        sentinel = self.result / "private-sentinel.txt"
        sentinel.write_text("sandbox must not see this")
        payload = self.result / "probe-payload.json"
        dump(payload, dict(workspace=str(ROOT), sentinel=str(sentinel),
                          net_ns=os.readlink("/proc/self/ns/net"),
                          pid_ns=os.readlink("/proc/self/ns/pid")))
        output = self.result / "probe"
        output.mkdir()
        code = self.native_run(payload, output,
            [str(VENV / "bin/python"), "/app/candidate_worker.py", "probe"],
            self.result / "probe.log")
        require(code == 0, "Sandbox preflight failed; see probe.log")
        self.report["sandbox_probe"] = read_json(output / "probe.json")
        self.verify_frozen()

    def verify_frozen(self):
        from application_identity import source_catalog, source_closure, _catalog_values, CATALOG, _read
        from poseidon_backend_identity import identity as backend_identity
        baseline = ROOT / "scripts/baseline"
        catalog = source_catalog(baseline)
        require(digest(baseline / CATALOG) == self._catalog_sha256,
                "Qualification source catalog changed")
        closure = source_closure(baseline, self._roots,
                                dynamic_authorities=_catalog_values(catalog)[2])
        require(closure == self._closure, "Validation source changed")
        import hashlib
        require(all(hashlib.sha256(_read(baseline, name)).hexdigest() == h
                    for name, h in self._declared_assets.items()), "Declared validation asset changed")
        require(all(self.report.get(k) == v for k, v in self._bound.items()),
                "Qualification binding report changed")
        require(backend_identity() == self._bound["backend_identity"],
                "Poseidon backend source/binary identity changed")
        if self.helper_environment is not None:
            from upstream_candidate_helpers import verify_sources
            from poly_dependencies import verify as verify_dependency
            require(dict(sources=verify_sources(backend="poseidon_cpu"),
                         dependency=verify_dependency()) == self.helper_environment,
                    "Helper source/dependency changed")
        import hashlib
        require(all(hashlib.sha256(_read(self.result, path)).hexdigest() == h
                    for path, h in self.frozen.items()), "Immutable reference/request/weights changed")
        require(digest(SOURCE / "fixtures.json") == self._fixture_sha256,
                "Frozen reference fixture changed")
        for key, path in (
            ("compiler_profile_sha256", PROFILE),
            ("compiler_sha256", BUILD / "bin/hecate-opt"),
            ("runtime_sha256", CPU_BUILD / "poseidon_agent_cpu"),
            ("poseidon_library_sha256", POSEIDON_BUILD / "libposeidon_shared.so"),
            ("operator_spec_sha256", baseline / "poseidon-operator-spec-v1.json"),
            ("frontend_sha256", BUILD / "lib/libHecateFrontend.so"),
            ("frontend_source_sha256", DACAPO / "tools/frontend.cpp"),
            ("frontend_python_sha256", DACAPO / "python/hecate/hecate/expr.py")):
            require(digest(path) == self._bound[key], "Qualification tool/configuration changed")
        require(self.fixed == read_json(SOURCE / "fixtures.json"), "Frozen reference settings changed")
        verify_execution_configuration(self.request, digest(PROFILE), self.waterline)

    def evaluate(self, raw, index):
        return self.validation.evaluate(raw, index)

    @classmethod
    def qualify(cls, request, candidate, *, validation_level="numerical", artifact_observer=None):
        """Single fixed candidate; never prepares a deterministic DSL answer."""
        from component_contract import reconstruct_request
        from unified_graph_contract import validate_candidate
        reconstruct_request(request)
        validate_candidate(candidate, request)
        request, candidate = copy.deepcopy(request), copy.deepcopy(candidate)
        require(request.get("compiler_configuration", {}).get("backend") == "poseidon_cpu",
                "Poseidon CPU configuration required")
        require(validation_level in ("compiled", "numerical"), "Validation level")
        import numpy as np
        import torch
        require_python_packages(torch, np)
        os.umask(0o077)
        torch.set_num_threads(2)
        result = Path(tempfile.mkdtemp(prefix="candidate-replay-", dir=WORK / "results"))
        print(f"Candidate evidence: {result}", flush=True)
        report = dict(validation_level=validation_level, feedback_policy="legacy-v1",
            platform_identity=identity(), status="running", provider="scripted_replay",
            agent_calls=0, backend="poseidon_cpu", poseidon_gpu_validated=False,
            attempts=[], artifact_replay_only=True, llm_generation_validated=False,
            rule_baseline_status="not_requested_by_component")
        frozen, phase = {}, "model_reference"
        try:
            if artifact_observer is not None:
                artifact_observer(result)
            from unified_graph_prepare import prepare_case
            dump(result / "model.json", request["model"])
            layout = request["layout"]
            period = layout["input_slot_period"] if "logical_inputs" in layout else None
            abi = layout["execution_abi"] if "resource_profile" in layout else None
            model = prepare_case(request["model"], result, period, capacity_abi=abi)
            require(read_json(result / "logical-bindings.json") == layout,
                    "Prepared model/layout binding mismatch")
            (result / "model-structure.txt").write_text(str(model))
            (result / "request.json").write_bytes(request_file_bytes(request))
            frozen = {name: digest(result / name) for name in (
                "model.json", "weights.npz", "arrays.npz", "request.json", "logical-bindings.json")}
            fixed = read_json(SOURCE / "fixtures.json")
            helpers = None
            if "upstream_helpers" in request:
                from upstream_candidate_helpers import verify_sources
                from poly_dependencies import verify as verify_dependency
                helpers = dict(sources=verify_sources(backend="poseidon_cpu"),
                               dependency=verify_dependency())
                report["upstream_helper_environment"] = helpers
            phase = "qualification_setup"
            session = cls(result, request, report, frozen, fixed,
                request["compiler_configuration"]["waterline"],
                helper_environment=helpers, validation_level=validation_level)
            from application_capabilities import assess_request
            report["capability_report"] = assess_request(request)
            phase = "sandbox_preflight"
            session.preflight()
            phase = "feedback_loop"
            raw = json.dumps(candidate)
            def record(index, raw, feedback):
                attempt = result / f"attempt-{index:02d}"
                (attempt / "response.txt").write_text(raw)
                dump(attempt / "feedback.json", feedback)
                dump(result / "report.json", report)
                print(f"attempt {index}: {feedback['status']} / {feedback['layer']}", flush=True)
            # Keep the existing evidence/loop schema with a single offline replay.
            # All actual generation and repair decisions remain with Application.
            report["loop"] = run_feedback_loop(request, ReplayProvider([raw]),
                                               session.evaluate, record, max_repairs=0)
            report["status"] = report["loop"]["status"]
            session.verify_frozen()
            code = 0 if report["status"] == "passed" else 1
        except OperationalStopped as stop:
            report.update(status=stop.status, failure_layer=stop.public_failure["layer"],
                          operational_failure=copy.deepcopy(stop.public_failure))
            raise
        except (KeyboardInterrupt, QualificationCancelled):
            report.update(status="cancelled", failure_layer=phase,
                          diagnostic="Host cancelled qualification")
            code = 130
        except Exception as error:
            category = "input" if phase == "model_reference" else "infrastructure"
            report.update(status=category + "_failed", failure_layer=phase, diagnostic=str(error))
            code = 1
        finally:
            report["frozen_hashes"] = frozen
            dump(result / "report.json", report)
            print(f"Candidate status: {report['status']}", flush=True)
        return code, report, result
