"""Pure-standard-library controls; all fabricated numerical data stay in temp dirs."""
import copy
import io
import json
import math
import struct
import tempfile
import unittest
import zipfile
from pathlib import Path

import evidence_audit as audit


def sealed(value, field="binding"):
    value = copy.deepcopy(value)
    value[field] = audit.digest(value)
    return value


def npy_bytes(values, shape, descr="<f8", fortran=False):
    header = repr(dict(descr=descr, fortran_order=fortran, shape=tuple(shape))).encode("latin1")
    header += b" " * ((64 - (10 + len(header) + 1) % 64) % 64) + b"\n"
    return b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header + b"".join(struct.pack("<d", v) for v in values)


def npz_bytes(reference, shape):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as z:
        z.writestr("inputs.npy", npy_bytes((0.0,) * shape[0], (shape[0], 1)))
        z.writestr("reference.npy", npy_bytes(reference, shape))
    return stream.getvalue()


def comparison(values, groups, width):
    rows = [list(values[i*width:(i+1)*width]) for i in range(groups)]
    return dict(passed=True, atol=1e-5, rtol=1e-4, compared_values=len(values), mae=0.0,
        max_absolute_error=0.0, max_nonzero_reference_relative_error=0.0,
        nonzero_reference_count=len(values), cosine_similarity=1.0, actual=rows, reference=rows,
        absolute_error=[[0.0] * width for _ in range(groups)], relative_error=[[0.0] * width for _ in range(groups)],
        elementwise_pass=[[True] * width for _ in range(groups)])


class Fixture:
    """Minimal schema-shaped evidence, intentionally not a real backend run."""
    def __init__(self, root, profile=audit.LEGACY_PROFILE):
        self.root = root
        self.profile = profile
        self.name = profile.cases[0]
        self.directory = root / self.name
        self.directory.mkdir()
        self.task_id = "a" * 32
        self.cache_id = "b" * 32
        self.task = self.directory / "tasks" / self.task_id
        self.package = self.task / "program"
        self.native = root / "candidate-replay-synthetic"
        self.new_native = root / "poseidon-program-execution-synthetic"
        self.output = self.native / "attempt-00/output"
        self.new_output = self.new_native / "output"
        self.backend = dict(format="synthetic-control-only", compiler_sha256="1" * 64, frontend_sha256="2"*64, executor_sha256="3"*64)
        self.candidate = b"# synthetic data-only candidate\n"
        self.request = dict(request_id="c" * 64, compiler_configuration=dict(name="poseidon-cpu-eva-w45-v1", backend="poseidon_cpu"),
            model=dict(resource_profile="general-tensors-v1", inputs=[dict(name="x", shape=[2])], outputs=[dict(name="z", value="x")]),
            public_constants={}, layout=dict(output_selectors=[[0, 0], [0, 1]], outputs=[dict(name="z", shape=[2])]))
        self.source_manifest = dict(format="synthetic-source-only", files={}, weights={})
        self.original = dict(format="synthetic-original-only")
        self.plan = sealed(dict(format=self.profile.format, cases=list(self.profile.cases), phases=list(self.profile.phases),
            candidate_origin=self.profile.origin, candidate_strategy=self.profile.strategy,
            new_agent_generation=False, production_applied=False,
            bindings=dict(inside_sources={"scripts/baseline/helper.py": "d" * 64, "scripts/baseline/application_model_source.py":"6"*64,
                **{"synthetic/%03d.py" % i: "e" * 64 for i in range(704)}},
                backend_identity=self.backend, requests={self.name:dict(request_id=self.request["request_id"],
                    request_canonical_sha256=audit.digest(self.request), model_sha256=audit.digest(self.request["model"]),
                    public_constants_sha256=audit.digest(self.request["public_constants"]))},
                source_reference={self.name:dict(model_sha256=audit.digest(self.request["model"]),
                    original_sha256=audit.digest(self.original), source_package_sha256=audit.digest(self.source_manifest))})))
        if profile.frozen_candidates:
            bindings = self.plan["bindings"]
            bindings["candidate_hashes"] = {name:self.sha_bytes(self.candidate) for name in profile.cases}
            request_binding = copy.deepcopy(bindings["requests"][self.name])
            source_binding = copy.deepcopy(bindings["source_reference"][self.name])
            bindings["requests"] = {name:copy.deepcopy(request_binding) for name in profile.cases}
            bindings["source_reference"] = {name:copy.deepcopy(source_binding) for name in profile.cases}
            self.plan.pop("binding")
            self.plan = sealed(self.plan)
        self.write(root / "plan.json", self.plan)
        self.write(root / "staged-sources.json", self.plan["bindings"]["inside_sources"])
        identity = dict(format="poseidon-application-identities-v1", policy="conservative-local-module-closure-v1", backend=self.backend)
        for stage in audit.STAGES:
            identity[stage] = sealed(dict(format="poseidon-application-stage-identity-v1", stage=stage,
                policy=identity["policy"], roots=["helper.py"], sources={"helper.py": "d" * 64, "application_model_source.py":"6"*64}, configuration_files={},
                import_edges={"helper.py": []}, dynamic_dependencies={}, external_imports=[], backend={}), "sha256")
        self.identity = sealed(identity, "manifest_sha256")
        self.worker = sealed(dict(format="poseidon-worker-context-v1", backend="poseidon_cpu",
            backend_identity=dict(backend="poseidon_cpu", tools=self.backend,
                implementation_digest=audit.digest(audit.implementation_projection(self.plan["bindings"]["inside_sources"]))),
            paths=dict(source_root=str(audit.scope.ROOT), task_store=str(self.directory / "tasks")),
            source_authority=dict(policy="conservative-local-module-closure-v1", sha256="7"*64,
                sources={"helper.py":"d"*64,"application_model_source.py":"6"*64}),
            stage_identity_sha256=self.identity["manifest_sha256"]))
        self.write(self.directory / "tasks/worker-context.json", self.worker)
        task = sealed(dict(request=self.request, validation_level="numerical", identity_policy="staged-v1", identity=self.identity))
        self.write(self.task / "task.json", task)
        source_raw = audit.canonical(self.source_manifest)
        source = sealed(dict(kind="trusted_pytorch", files={"manifest.json":dict(bytes=len(source_raw), sha256=self.sha_bytes(source_raw))},
            implementation=dict(format="poseidon-source-intake-controller-v1", controller_sha256="6"*64),
            identity=self.identity, identity_policy="staged-v1", resource_profile="general-tensors-v1"))
        self.write(self.task / "source.json", source)
        self.source = source
        self.write(self.directory / "task-id.json", dict(task_id=self.task_id))
        self.write(self.directory / "qualification-binding.json", dict(request=self.request, request_sha256=audit.digest(self.request),
            candidate_sha256=self.sha_bytes(self.candidate), identity=self.identity,
            model_source_sha256=audit.digest(self.source_manifest), candidate_origin=self.profile.origin, agent_generated=False))
        self.context = sealed(dict(format="poseidon-request-context-v1", worker_binding=self.worker["binding"],
            request_id=self.request["request_id"], request_sha256=audit.digest(self.request),
            compiler_configuration_sha256=audit.digest(self.request["compiler_configuration"]),
            layout_sha256=audit.digest(self.request["layout"]), validation_level="numerical"))
        self.write(self.native / "request.json", self.request)
        self.write(self.native / "model.json", self.request["model"])
        qualification_values = (0.1, 0.2) * 4
        self.binary(self.native / "arrays.npz", npz_bytes(qualification_values, (4, 2)))
        self.binary(self.native / "attempt-00/candidate.py", self.candidate)
        self.binary(self.output / "candidate_trace.mlir", b"synthetic-control-only")
        self.make_execution(self.output, qualification_values, 4)
        self.gate = dict(synthetic_control_only=True)
        self.entry = dict(format="hecate-private-symbol-dce-v1", function_bodies_unchanged=True)
        self.compile_command = ["/synthetic/hecate-opt", "never-executed"]
        trace = dict(frontend="real_Hecate", candidate_python_executed=False,
            construction="validated_native_AST_to_Hecate_functions", request_id=self.request["request_id"])
        self.write(self.output / "trace-evidence.json", trace)
        attempt = dict(index=0, status="passed", compiled=True, executed=True, numerically_correct=True, trace=trace,
            artifact_hashes={n:self.sha(self.output / n) for n in ("candidate_trace.mlir", "trace-evidence.json")},
            artifact_gate=self.gate, runtime_plan_entry=self.entry, compile_command=self.compile_command,
            execution=self.execution_metadata, comparison=comparison(qualification_values, 4, 2))
        native_report = dict(status="passed", backend="poseidon_cpu", agent_calls=0, llm_generation_validated=False,
            backend_identity=self.backend, request_id=self.request["request_id"], validation_level="numerical",
            source_hashes={"helper.py": "d" * 64}, frozen_hashes={n:self.sha(self.native / n) for n in ("request.json", "model.json", "arrays.npz")},
            attempts=[attempt], compiler_sha256=self.backend["compiler_sha256"], frontend_sha256=self.backend["frontend_sha256"], runtime_sha256=self.backend["executor_sha256"], compiler_configuration=self.request["compiler_configuration"])
        self.write(self.native / "report.json", native_report)
        validation = dict(format="poseidon-agent-component-v1", status="passed", exit_code=0, validation_level="numerical",
            compiled_validated=True, encrypted_execution=True, numerically_validated=True, paid_calls=0, new_agent_generation=False,
            request_id=self.request["request_id"], run_context=self.context, evidence=str(self.native), compared_values=8, max_absolute_error=0.0,
            observed_stages={k:True for k in ("compiled", "encrypted_execution", "evidence_integrity_not_rejected", "numerically_compared", "traced")})
        self.write(self.task / "validation-0.json", validation)
        self.binary(self.package / "candidate.py", self.candidate)
        self.write(self.package / "request.json", self.request)
        self.write(self.package / "model.json", self.request["model"])
        self.write(self.package / "model-provenance.json", dict(original=self.original, trusted_pytorch=dict(manifest=self.source_manifest, files={})))
        self.write(self.package / "artifacts/application-identities.json", self.identity)
        self.write(self.package / "artifacts/validation.json", dict(level="numerical", scope="four fixed inputs", gate=self.gate, runtime_plan_entry=self.entry, compile_command=self.compile_command))
        self.binary(self.package / "artifacts/candidate_trace.mlir", b"synthetic-control-only")
        provenance = json.loads((self.package / "model-provenance.json").read_text())
        self.write(self.package / "artifacts/model-source.json", sealed(dict(format="poseidon-model-source-origin-v1", kind="trusted_pytorch",
            snapshot_binding=self.source["binding"], files=self.source["files"], implementation=self.source["implementation"],
            resource_profile="general-tensors-v1", model_sha256=audit.digest(self.request["model"]), provenance_sha256=audit.digest(provenance))))
        files = {str(p.relative_to(self.package)):self.sha(p) for p in self.package.rglob("*") if p.is_file()}
        self.manifest = sealed(dict(format="poseidon-dsl-bundle-v6", files=files, agent_generated=False,
            compiler_configuration=self.request["compiler_configuration"], validation=dict(level="numerical", encrypted_execution=True, numerically_validated=True),
            model_provenance="included", origin=dict(request_id=self.request["request_id"], frontend_sha256=self.backend["frontend_sha256"],
                runtime_sha256=self.backend["executor_sha256"], report_sha256=self.sha(self.native / "report.json"),
                model_source=dict(format="poseidon-model-source-origin-v1", file="artifacts/model-source.json", sha256=files["artifacts/model-source.json"])),
            dependencies=dict(backend_identity=self.backend, compiler_sha256=self.backend["compiler_sha256"],
                application_identities=dict(file="artifacts/application-identities.json", sha256=files["artifacts/application-identities.json"]))))
        self.write(self.package / "manifest.json", self.manifest)
        result = dict(status="succeeded", validation_level="numerical", generations=1, http_attempts=0,
            generation=dict(agent_generated=False, generations=1), task_id=self.task_id,
            acceptance=dict(level="numerical", artifact_checked=True, compiled=True, encrypted_execution=True,
                numerically_validated=True, input_groups=4, compared_values=8, maximum_absolute_error=0.0),
            package=dict(binding=self.manifest["binding"], task_id=self.task_id, reused=False))
        self.write_phase("qualification", dict(compiled=True, encrypted_execution=True, source_capture_supervisor_executed=True,
            result=result, package=str(self.package), package_binding=self.manifest["binding"], request_id=self.request["request_id"],
            candidate_sha256=self.sha_bytes(self.candidate), input_groups=4, compared_values=8, maximum_absolute_error=0.0))
        new_values = (0.3, 0.4) * 3
        self.make_execution(self.new_output, new_values, 3)
        for filename in files:
            if filename.startswith("artifacts/"):
                self.binary(self.new_output / filename[10:], (self.package / filename).read_bytes())
        self.write(self.new_native / "payload.json", dict(request=self.request, candidate=dict(request_id=self.request["request_id"], hecate_source=self.candidate.decode())))
        self.write(self.new_output / "inputs.json", dict(synthetic_control_only=True))
        execution = dict(format="poseidon-local-execution-v1", status="executed", bundle_binding=self.manifest["binding"], request_id=self.request["request_id"],
            agent_calls=0, compiler_calls=0, samples=3, encrypted_execution=True, executor_secret_key=False,
            package_validation=self.manifest["validation"], numerically_validated_on_supplied_inputs=False,
            execution=self.execution_metadata, evidence=str(self.new_native), outputs=[dict(z=[0.3, 0.4]) for _ in range(3)])
        self.write(self.new_native / "report.json", execution)
        execution = dict(execution, run_context=self.context)
        self.write_phase("new_inputs", dict(compiled=False, encrypted_execution=True, package_reloaded=True, input_groups=3,
            execution=execution, comparison=comparison(new_values, 3, 2)))
        cached = copy.deepcopy(result)
        cached.update(task_id=self.cache_id, generations=0, http_attempts=0)
        cached["package"]["reused"] = True
        self.write(self.directory / "tasks" / self.cache_id / "state.json", cached)
        self.write_phase("cache", dict(compiled=False, encrypted_execution=False, package_reused=True, result=cached))
        self.sync_aggregate()

    @staticmethod
    def sha_bytes(raw):
        import hashlib
        return hashlib.sha256(raw).hexdigest()

    def sha(self, path):
        return self.sha_bytes(path.read_bytes())

    @staticmethod
    def binary(path, raw):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)

    def write(self, path, value):
        self.binary(path, audit.canonical(value))

    def write_phase(self, phase, values):
        self.write(self.directory / (phase + ".json"), dict(case=self.name, phase=phase, status="passed",
            provider_calls=0, new_agent_generation=False, candidate_origin=self.profile.origin, **values))
        self.write(self.directory / (phase + "-started.json"), dict(format="poseidon-offline-phase-start-v1",
            binding=self.plan["binding"], case=self.name, phase=phase, started_wall=1.0))

    def sync_aggregate(self):
        rows = []
        for name in self.profile.cases:
            for phase in self.profile.phases:
                if name == self.name:
                    row = json.loads((self.directory / (phase + ".json")).read_text())
                else:
                    row = dict(case=name, phase=phase, status="passed")
                rows.append(dict(row, exit_code=0))
        self.write(self.root / "report.json", dict(format="poseidon-ordered-common-offline-report-v1",
            binding=self.plan["binding"], status="passed", rows=rows))

    def make_execution(self, output, values, groups):
        self.execution_metadata = dict(backend="poseidon_cpu", encrypted_execution=True, executor_secret_key=False,
            bootstrap=False, degree=32768, data_moduli=13, special_moduli=1, security="tc128",
            q=list(audit.PRIMES[:13]), p=list(audit.PRIMES[13:]))
        self.write(output / "execution-metadata.json", self.execution_metadata)
        self.write(output / "execution.json", dict(self.execution_metadata, outputs=[[list(values[i*2:i*2+2])] for i in range(groups)]))
        self.binary(output / "decrypted.npy", npy_bytes(values, (groups, 2)))

    def mutate(self, relative, fn):
        p = self.root / relative
        value = json.loads(p.read_text())
        fn(value)
        self.write(p, value)
        if p.parent == self.directory and p.name in tuple(x + ".json" for x in self.profile.phases):
            self.sync_aggregate()

    def validate(self):
        return audit.validate_case(self.directory, self.name, self.plan)


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="poseidon-evidence-control-")
        self.addCleanup(self.temp.cleanup)
        self.f = Fixture(Path(self.temp.name))

    def test_complete_control_and_no_array_output(self):
        result = self.f.validate()
        self.assertEqual(result["source_count"], 706)
        self.assertEqual(result["qualification"]["compared_values"], 8)
        self.assertEqual(result["new_inputs"]["compared_values"], 6)
        self.assertEqual(result["cache_generations"], 0)
        self.assertNotIn('"actual"', json.dumps(result))
        self.assertNotIn('"reference"', json.dumps(result))
        self.assertIn(str(self.f.output / "decrypted.npy"), result["evidence_files"])

    def test_old_705_refused(self):
        self.f.plan["bindings"]["inside_sources"].pop("synthetic/000.py")
        self.f.plan.pop("binding")
        self.f.plan = sealed(self.f.plan)
        with self.assertRaisesRegex(ValueError, "joint_706_inventory"):
            self.f.validate()

    def test_changed_staged_full_inventory(self):
        self.f.mutate("staged-sources.json", lambda x:x.update({"synthetic/000.py":"f"*64}))
        with self.assertRaisesRegex(ValueError, "full_staged_706_binding"):
            self.f.validate()

    def test_wrong_phase_case(self):
        self.f.mutate(self.f.name + "/new_inputs.json", lambda x:x.update(case="other"))
        with self.assertRaisesRegex(ValueError, "aggregate_phase_inventory"):
            self.f.validate()

    def test_boolean_zero_rejected(self):
        self.f.mutate(self.f.name + "/cache.json", lambda x:x.update(provider_calls=False))
        with self.assertRaisesRegex(ValueError, "phase_metadata"):
            self.f.validate()

    def test_changed_candidate_bytes(self):
        self.f.binary(self.f.package / "candidate.py", b"changed")
        with self.assertRaisesRegex(ValueError, "evidence_hash_mismatch"):
            self.f.validate()

    def test_execution_decode_tamper(self):
        self.f.mutate("candidate-replay-synthetic/attempt-00/output/execution.json", lambda x:x["outputs"][0][0].__setitem__(0, 9.0))
        with self.assertRaisesRegex(ValueError, "execution_decoding_binding"):
            self.f.validate()

    def test_new_comparison_summary_cannot_hide_failure(self):
        self.f.mutate(self.f.name + "/new_inputs.json", lambda x:x["comparison"]["reference"][0].__setitem__(0, 9.0))
        with self.assertRaisesRegex(ValueError, "comparison_absolute_error"):
            self.f.validate()

    def test_new_named_outputs_bound_to_native(self):
        self.f.mutate(self.f.name + "/new_inputs.json", lambda x:x["execution"]["outputs"][0]["z"].__setitem__(0, 9.0))
        with self.assertRaisesRegex(ValueError, "new_execution_report_binding"):
            self.f.validate()

    def test_cache_current_generation_forbidden(self):
        self.f.mutate(self.f.name + "/cache.json", lambda x:x["result"].update(generations=1))
        with self.assertRaisesRegex(ValueError, "cache_reuse_boundary"):
            self.f.validate()

    def test_historical_nested_generation_does_not_count_again(self):
        self.assertEqual(self.f.validate()["cache_generations"], 0)

    def test_symlink_refused(self):
        path = self.f.output / "decrypted.npy"
        other = self.f.root / "other.npy"
        path.rename(other)
        path.symlink_to(other)
        with self.assertRaisesRegex(ValueError, "regular_evidence_path"):
            self.f.validate()

    def test_duplicate_json_rejected(self):
        self.f.binary(self.f.directory / "cache.json", b'{"case":"a","case":"b"}')
        with self.assertRaisesRegex(ValueError, "duplicate_json_field"):
            self.f.validate()

    def test_npy_object_fortran_shape_and_nonfinite_rejected(self):
        for raw, expected in [(npy_bytes((1.0,), (1,), descr="|O"), (1,)),
                (npy_bytes((1.0,), (1,), fortran=True), (1,)),
                (npy_bytes((1.0,), (1,)), (2,)), (npy_bytes((math.nan,), (1,)), (1,))]:
            with self.subTest():
                with self.assertRaises(ValueError):
                    audit.npy(raw, expected)

    def test_npz_inventory_and_duplicate_rejected(self):
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, "w") as z:
            z.writestr("../reference.npy", npy_bytes((1.0,), (1,)))
            z.writestr("inputs.npy", npy_bytes((1.0,), (1,)))
        with self.assertRaisesRegex(ValueError, "npz_inventory"):
            audit.npz_reference(stream.getvalue(), (1,))

    def test_raw_phase_schema_has_no_exit_code(self):
        for phase in audit.scope.PHASES:
            self.assertNotIn("exit_code", json.loads((self.f.directory / (phase + ".json")).read_text()))
        self.assertEqual(self.f.validate()["status"], "passed")

    def test_partial_failed_aggregate_keeps_successful_case_evidence(self):
        def subset(value):
            value["rows"] = [r for r in value["rows"] if r["case"] == self.f.name]
            value["rows"].append(dict(case=audit.scope.CASES[1], phase="qualification", status="failed", exit_code=1))
            value["status"] = "failed"
        self.f.mutate("report.json", subset)
        self.assertEqual(self.f.validate()["status"], "passed")

    def test_partial_selected_case_must_have_all_three_phases(self):
        self.f.mutate("report.json", lambda value:value["rows"].pop(0))
        with self.assertRaisesRegex(ValueError, "selected_case_complete_phases"):
            self.f.validate()

    def test_duplicate_aggregate_phase_rejected(self):
        self.f.mutate("report.json", lambda value:value["rows"].__setitem__(-1, copy.deepcopy(value["rows"][0])))
        with self.assertRaisesRegex(ValueError, "aggregate_phase_inventory"):
            self.f.validate()

    def test_unknown_aggregate_phase_rejected(self):
        self.f.mutate("report.json", lambda value:value["rows"][-1].update(phase="unknown"))
        with self.assertRaisesRegex(ValueError, "aggregate_phase_inventory"):
            self.f.validate()

    def test_aggregate_owned_child_failure_rejected(self):
        self.f.mutate("report.json", lambda value:value["rows"][0].update(exit_code=124))
        with self.assertRaisesRegex(ValueError, "aggregate_phase_exit"):
            self.f.validate()

    def test_aggregate_raw_phase_disagreement_rejected(self):
        path = self.f.directory / "new_inputs.json"
        value = json.loads(path.read_text())
        value["comparison"]["mae"] = 1.0
        self.f.write(path, value)
        with self.assertRaisesRegex(ValueError, "aggregate_phase_receipt_binding"):
            self.f.validate()

    def test_aggregate_runtime_binding_rejected(self):
        self.f.mutate("report.json", lambda value:value.update(binding="0"*64))
        with self.assertRaisesRegex(ValueError, "aggregate_runtime_binding"):
            self.f.validate()

    def test_changed_backend_identity_rejected(self):
        self.f.plan["bindings"]["backend_identity"]["executor_sha256"] = "9" * 64
        self.f.plan.pop("binding")
        self.f.plan = sealed(self.f.plan)
        self.f.write(self.f.root / "plan.json", self.f.plan)
        self.f.sync_aggregate()
        for phase in audit.scope.PHASES:
            self.f.mutate(self.f.name + "/" + phase + "-started.json", lambda x:x.update(binding=self.f.plan["binding"]))
        with self.assertRaisesRegex(ValueError, "worker_identity_binding"):
            self.f.validate()

    def test_producer_backend_wrapper_shape_is_preserved(self):
        self.assertEqual(set(self.f.worker["backend_identity"]), {"backend", "tools", "implementation_digest"})
        self.assertEqual(self.f.validate()["status"], "passed")

    def test_bare_sdk_identity_is_not_a_worker_context_identity(self):
        worker = copy.deepcopy(self.f.worker)
        worker["backend_identity"] = self.f.backend
        with self.assertRaisesRegex(ValueError, "worker_identity_binding"):
            audit.worker_identity(worker, self.f.directory, self.f.plan, audit.Reader())

    def test_worker_wrapper_rejects_changed_implementation_projection(self):
        worker = copy.deepcopy(self.f.worker)
        worker["backend_identity"]["implementation_digest"] = "0"*64
        with self.assertRaisesRegex(ValueError, "worker_identity_binding"):
            audit.worker_identity(worker, self.f.directory, self.f.plan, audit.Reader())

    def test_worker_wrapper_rejects_different_source_root(self):
        worker = copy.deepcopy(self.f.worker)
        worker["paths"]["source_root"] = "/other/root"
        with self.assertRaisesRegex(ValueError, "worker_identity_binding"):
            audit.worker_identity(worker, self.f.directory, self.f.plan, audit.Reader())

    def test_lossless_producer_binding_table_wire_request(self):
        logical = dict(constant_origins=dict(capacity_tensor_packing=dict(version="capacity-public-tensors-v9",
            bindings=[dict(input=0, output=i) for i in range(24)])), public_constants={})
        stored = copy.deepcopy(logical)
        stored["constant_origins"]["capacity_tensor_packing"]["bindings"] = dict(
            format="capacity-public-bindings-table-v1", columns=["input", "output"], rows=[[0, i] for i in range(24)])
        path = audit.scope.BASE / "unified_request_storage.py"
        plan = dict(bindings=dict(inside_sources={"scripts/baseline/unified_request_storage.py":self.f.sha(path)}))
        self.assertTrue(audit.request_matches(stored, logical, plan, audit.Reader()))
        stored["constant_origins"]["capacity_tensor_packing"]["bindings"]["rows"][0][1] = 99
        self.assertFalse(audit.request_matches(stored, logical, plan, audit.Reader()))

    def test_request_wire_codec_source_must_match_loaded_identity(self):
        logical = dict(constant_origins=dict(capacity_tensor_packing=dict(version="capacity-public-tensors-v9",
            bindings=[dict(input=0, output=i) for i in range(24)])), public_constants={})
        stored = copy.deepcopy(logical)
        stored["constant_origins"]["capacity_tensor_packing"]["bindings"] = {}
        plan = dict(bindings=dict(inside_sources={"scripts/baseline/unified_request_storage.py":"0"*64}))
        with self.assertRaisesRegex(ValueError, "request_codec_source_binding"):
            audit.request_matches(stored, logical, plan, audit.Reader())

    def test_raw_rule_trace_is_ast_constructed_without_candidate_python_exec(self):
        report = json.loads((self.f.native / "report.json").read_text())
        self.assertIs(report["attempts"][0]["trace"]["candidate_python_executed"], False)
        self.assertEqual(self.f.validate()["status"], "passed")

    def test_native_trace_evidence_cannot_be_changed(self):
        self.f.mutate("candidate-replay-synthetic/attempt-00/output/trace-evidence.json",
            lambda value:value.update(candidate_python_executed=True))
        with self.assertRaisesRegex(ValueError, "evidence_hash_mismatch"):
            self.f.validate()

    def test_changed_phase_start_binding_rejected(self):
        self.f.mutate(self.f.name + "/cache-started.json", lambda x:x.update(binding="0"*64))
        with self.assertRaisesRegex(ValueError, "phase_start_binding"):
            self.f.validate()

    def test_changed_native_report_not_rebound_by_phase_summary(self):
        self.f.mutate("candidate-replay-synthetic/report.json", lambda x:x.update(agent_calls=1))
        with self.assertRaisesRegex(ValueError, "native_report_binding"):
            self.f.validate()

    def test_new_current_compiler_call_rejected(self):
        self.f.mutate(self.f.name + "/new_inputs.json", lambda x:x["execution"].update(compiler_calls=1))
        with self.assertRaisesRegex(ValueError, "new_input_execution_boundary"):
            self.f.validate()

    def test_undeclared_private_path_rejected_before_read(self):
        with self.assertRaisesRegex(ValueError, "public_source_path"):
            audit.bound_source("scripts/baseline/.env", "0"*64, self.f.plan, audit.Reader())

    def test_reader_credentials_and_private_keys_refused_before_read(self):
        for name in (".env", ".ssh/id_rsa", "id_ed25519", "private_key.bin", "secret-key.json", "tls.pem"):
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, "credential_or_secret_path"):
                    audit.Reader().bytes(self.f.root / name)

    def test_npz_zip_bomb_declared_size_rejected(self):
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, "w") as z:
            z.writestr("inputs.npy", npy_bytes((1.0,), (1,)))
            z.writestr("reference.npy", npy_bytes((1.0,), (1,)))
        raw = bytearray(stream.getvalue())
        central = raw.find(b"PK\x01\x02")
        raw[central+24:central+28] = (audit.MAX_FILE+1).to_bytes(4, "little")
        with self.assertRaisesRegex(ValueError, "npz_member_budget"):
            audit.npz_reference(bytes(raw), (1,))

    def test_npy_payload_truncation_rejected(self):
        with self.assertRaisesRegex(ValueError, "npy_payload_size"):
            audit.npy(npy_bytes((1.0,), (1,))[:-1], (1,))

    def test_reader_file_budget(self):
        path = self.f.root / "bounded"
        path.write_bytes(b"1234")
        with self.assertRaisesRegex(ValueError, "evidence_file_budget"):
            audit.Reader().bytes(path, 3)


class CapacityAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="poseidon-capacity-evidence-control-")
        self.addCleanup(self.temp.cleanup)
        self.f = Fixture(Path(self.temp.name), audit.CAPACITY_PROFILE)

    def rebind_plan(self, fn):
        fn(self.f.plan)
        self.f.plan.pop("binding")
        self.f.plan = sealed(self.f.plan)
        self.f.write(self.f.root / "plan.json", self.f.plan)
        self.f.sync_aggregate()
        for phase in self.f.profile.phases:
            self.f.mutate(self.f.name + "/" + phase + "-started.json",
                lambda row:row.update(binding=self.f.plan["binding"]))

    def test_capacity_profile_complete_control_preserves_output_schema(self):
        result = self.f.validate()
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["candidate_origin"], audit.CAPACITY_PROFILE.origin)
        self.assertEqual(result["candidate_sha256"], self.f.plan["bindings"]["candidate_hashes"][self.f.name])
        self.assertEqual(result["source_count"], 706)
        self.assertEqual(len(json.loads((self.f.root / "report.json").read_text())["rows"]), 33)
        self.assertEqual(result["provider_calls"], 0)
        self.assertEqual(result["qualification"]["compared_values"], 8)
        self.assertEqual(result["new_inputs"]["compared_values"], 6)
        self.assertNotIn('"actual"', json.dumps(result))
        self.assertNotIn('"reference"', json.dumps(result))

    def test_capacity_origin_in_phase_cannot_claim_legacy(self):
        self.f.mutate(self.f.name + "/cache.json",
            lambda row:row.update(candidate_origin=audit.LEGACY_PROFILE.origin))
        with self.assertRaisesRegex(ValueError, "phase_metadata"):
            self.f.validate()

    def test_capacity_origin_in_binding_cannot_claim_legacy(self):
        self.f.mutate(self.f.name + "/qualification-binding.json",
            lambda row:row.update(candidate_origin=audit.LEGACY_PROFILE.origin))
        with self.assertRaisesRegex(ValueError, "request_source_binding"):
            self.f.validate()

    def test_capacity_frozen_plan_candidate_mismatch_rejected(self):
        self.rebind_plan(lambda plan:plan["bindings"]["candidate_hashes"].update({self.f.name:"0"*64}))
        with self.assertRaisesRegex(ValueError, "capacity_frozen_candidate_binding"):
            self.f.validate()

    def test_capacity_binding_cannot_substitute_candidate(self):
        self.f.mutate(self.f.name + "/qualification-binding.json",
            lambda row:row.update(candidate_sha256="0"*64))
        with self.assertRaisesRegex(ValueError, "capacity_frozen_candidate_binding"):
            self.f.validate()

    def test_capacity_qualification_candidate_mismatch_rejected(self):
        self.f.mutate(self.f.name + "/qualification.json",
            lambda row:row.update(candidate_sha256="0"*64))
        with self.assertRaisesRegex(ValueError, "candidate_byte_binding"):
            self.f.validate()

    def test_capacity_package_candidate_mismatch_rejected(self):
        self.f.binary(self.f.package / "candidate.py", b"# changed synthetic candidate")
        with self.assertRaisesRegex(ValueError, "evidence_hash_mismatch"):
            self.f.validate()

    def test_capacity_native_candidate_mismatch_rejected(self):
        self.f.binary(self.f.native / "attempt-00/candidate.py", b"# changed synthetic candidate")
        with self.assertRaisesRegex(ValueError, "native_candidate_binding"):
            self.f.validate()

    def test_capacity_full_706_staged_binding_still_required(self):
        self.f.mutate("staged-sources.json", lambda row:row.update({"synthetic/000.py":"0"*64}))
        with self.assertRaisesRegex(ValueError, "full_staged_706_binding"):
            self.f.validate()

    def test_capacity_sdk_identity_still_bound(self):
        self.rebind_plan(lambda plan:plan["bindings"]["backend_identity"].update(executor_sha256="0"*64))
        with self.assertRaisesRegex(ValueError, "worker_identity_binding"):
            self.f.validate()

    def test_capacity_partial_selected_case_evidence_kept(self):
        self.f.mutate("report.json",
            lambda row:row.update(status="failed", rows=[item for item in row["rows"] if item["case"] == self.f.name]))
        self.assertEqual(self.f.validate()["status"], "passed")

    def test_capacity_aggregate_cannot_include_boolean(self):
        self.f.mutate("report.json", lambda row:row["rows"].append(
            dict(case="common_boolean", phase="qualification", status="passed", exit_code=0)))
        with self.assertRaisesRegex(ValueError, "aggregate_phase_inventory"):
            self.f.validate()

    def test_capacity_new_input_maths_still_checked(self):
        self.f.mutate(self.f.name + "/new_inputs.json",
            lambda row:row["comparison"]["reference"][0].__setitem__(0, 9.0))
        with self.assertRaisesRegex(ValueError, "comparison_absolute_error"):
            self.f.validate()


class ProfileTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="poseidon-profile-control-")
        self.addCleanup(self.temp.cleanup)
        self.f = Fixture(Path(self.temp.name), audit.CAPACITY_PROFILE)

    def test_only_two_fixed_immutable_profiles(self):
        self.assertEqual(len(audit.AUDIT_PROFILES), 2)
        with self.assertRaises(TypeError):
            audit.AUDIT_PROFILES["other"] = audit.LEGACY_PROFILE
        self.assertEqual(audit.LEGACY_PROFILE.cases, audit.scope.CASES)
        self.assertEqual(audit.LEGACY_PROFILE.phases, audit.scope.PHASES)
        self.assertEqual(audit.LEGACY_PROFILE.origin, audit.scope.CANDIDATE_ORIGIN)
        self.assertEqual(len(audit.CAPACITY_PROFILE.cases), 11)
        self.assertNotIn("common_boolean", audit.CAPACITY_PROFILE.cases)

    def test_unknown_format_is_not_fallback(self):
        plan = copy.deepcopy(self.f.plan)
        plan["format"] = "poseidon-arbitrary-capacity-plan-v1"
        with self.assertRaisesRegex(ValueError, "recognized_audit_profile"):
            audit.audit_profile(plan)

    def test_cross_profile_contracts_refused(self):
        mutations = (
            lambda plan:plan.update(format=audit.LEGACY_PROFILE.format),
            lambda plan:plan.update(candidate_origin=audit.LEGACY_PROFILE.origin),
            lambda plan:plan.update(candidate_strategy=audit.LEGACY_PROFILE.strategy),
            lambda plan:plan["cases"].append("common_boolean"),
            lambda plan:plan["cases"].reverse(),
            lambda plan:plan["phases"].reverse(),
            lambda plan:plan["cases"].append("unreviewed_case"))
        for mutate in mutations:
            with self.subTest():
                plan = copy.deepcopy(self.f.plan)
                mutate(plan)
                with self.assertRaisesRegex(ValueError, "fixed_audit_profile_contract"):
                    audit.audit_profile(plan)

    def test_exact_candidate_hash_inventory_required(self):
        mutations = (
            lambda hashes:hashes.pop(self.f.name),
            lambda hashes:hashes.update(common_boolean="0"*64),
            lambda hashes:hashes.update({self.f.name:"A"*64}),
            lambda hashes:hashes.update({self.f.name:"0"*63}),
            lambda hashes:hashes.update({self.f.name:False}))
        for mutate in mutations:
            with self.subTest():
                plan = copy.deepcopy(self.f.plan)
                mutate(plan["bindings"]["candidate_hashes"])
                with self.assertRaisesRegex(ValueError, "capacity_frozen_candidate_inventory"):
                    audit.audit_profile(plan)

    def test_exact_capacity_request_and_source_inventory_required(self):
        for field in ("requests", "source_reference"):
            for extra in (False, True):
                with self.subTest(field=field, extra=extra):
                    plan = copy.deepcopy(self.f.plan)
                    if extra:
                        plan["bindings"][field]["common_boolean"] = {}
                    else:
                        plan["bindings"][field].pop(self.f.name)
                    with self.assertRaisesRegex(ValueError, "capacity_fixed_request_inventory"):
                        audit.audit_profile(plan)

    def test_capacity_binding_container_must_be_dict(self):
        plan = copy.deepcopy(self.f.plan)
        plan["bindings"] = []
        with self.assertRaisesRegex(ValueError, "capacity_binding_inventory"):
            audit.audit_profile(plan)


if __name__ == "__main__":
    unittest.main()
