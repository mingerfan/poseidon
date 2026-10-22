"""Real Application lifecycle with synthetic package/execution fixtures, not FHE."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

from application_component import Application
from benchmark_graph import digest
from model_output_contract import tensor_leaves
from task_output_adapter import TaskOutputAdapter
from test_application_component import Backend, Provider
from test_model_output_contract import model, tensor


class OutputBackend(Backend):
    """Protocol fixture: never compiles, encrypts or calls a Provider."""
    def __init__(self):
        super().__init__()
        self.execute_calls = 0
        self.override = None
        self.tasks = {}

    def qualify(self, request, candidate, level):
        result = super().qualify(request, candidate, level)
        result["request"] = copy.deepcopy(request)
        return result

    def export(self, result, destination, level, **kwargs):
        super().export(result, destination, level)
        self.tasks[str(destination)] = (copy.deepcopy(result["request"]), level)

    def execute(self, folder, samples):
        self.execute_calls += 1
        request, level = self.tasks[str(folder)]
        outputs = [
            {o["name"]: [float(x) ** 2 for x in sample["x"]]
             for o in request["model"]["outputs"]}
            for sample in samples
        ]
        result = {
            "format": "poseidon-local-execution-v1", "status": "executed",
            "backend": "poseidon_cpu", "request_id": request["request_id"],
            "bundle_binding": self.check_package(folder, level),
            "package_validation": {
                "level": level, "encrypted_execution": level == "numerical",
                "numerically_validated": level == "numerical",
            },
            "encrypted_execution": True, "executor_secret_key": False,
            "numerically_validated_on_supplied_inputs": False,
            "application_accuracy_certified": False,
            "samples": len(samples), "outputs": outputs,
            "agent_calls": 0, "compiler_calls": 0,
        }
        if self.override:
            result.update(copy.deepcopy(self.override))
        return result


class TaskOutputsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.backend = OutputBackend()
        self.app = Application(Path(self.tmp.name) / "tasks", self.backend)
        self.outputs = TaskOutputAdapter(self.app)
        self.model = model()
        self.tree = {"kind": "tuple", "items": [tensor(), {"kind": "none"}]}
        self.samples = [{"x": [.125, -.25]}]

    def tearDown(self):
        self.tmp.cleanup()

    def ready(self, **kwargs):
        task = self.app.submit(self.model, **kwargs)
        provider = Provider()
        outcome = self.app.run(task, provider)
        self.assertEqual(outcome["status"], "succeeded")
        return task, provider, outcome

    def bound(self, **kwargs):
        task, provider, outcome = self.ready(**kwargs)
        frame = self.outputs.bind(task, self.tree)
        self.assertEqual(frame["status"], "succeeded")
        return task, provider, outcome, frame["receipt"]

    def execute(self, task, receipt, **kwargs):
        return self.outputs.execute(
            task, self.samples, receipt,
            expected_binding=kwargs.get("expected_binding", receipt["binding"]),
        )

    def test_binding_preserves_old_request_task_and_budget(self):
        task, provider, outcome = self.ready()
        before = self.app._checked_task_metadata(task)
        original = copy.deepcopy(outcome)
        frame = self.outputs.bind(task, self.tree)
        self.assertEqual(frame["task"], original)
        self.assertEqual(self.app._checked_task_metadata(task), before)
        self.assertEqual(self.app.status(task), original)
        self.assertEqual((provider.calls, self.backend.calls, self.backend.execute_calls), (1, 1, 0))
        self.assertEqual(frame["receipt"]["task_metadata_binding"], before["binding"])

    def test_pending_task_has_no_receipt_or_execute(self):
        task = self.app.submit(self.model)
        result = self.outputs.bind(task, self.tree)
        self.assertEqual(result["status"], "queued")
        self.assertNotIn("receipt", result)
        self.assertEqual(self.backend.execute_calls, 0)

    def test_cancelled_task_stays_cancelled(self):
        task = self.app.submit(self.model)
        self.app.cancel(task)
        result = self.outputs.execute(task, self.samples, {}, expected_binding="0" * 64)
        self.assertEqual(result["status"], "cancelled")
        self.assertEqual(self.backend.execute_calls, 0)

    def test_unsupported_task_stays_unsupported(self):
        task = self.app.submit(self.model, backend="unsupported")
        result = self.outputs.bind(task, self.tree)
        self.assertEqual(result["status"], "unsupported")
        self.assertNotIn("receipt", result)

    def test_original_validation_failure_remains_terminal(self):
        self.backend.failures = ["compiler"]
        task = self.app.submit(self.model)
        original = self.app.run(task, Provider())
        result = self.outputs.bind(task, self.tree)
        self.assertEqual(result["status"], "validation_failed")
        self.assertEqual(result["task"], original)
        self.assertNotIn("receipt", result)

    def test_none_does_not_add_model_or_physical_output(self):
        task, _, _, receipt = self.bound()
        self.assertEqual(len(tensor_leaves(receipt["delivery"]["contract"]["tree"])), 1)
        self.assertEqual(len(self.app._checked_task_metadata(task)["request"]["model"]["outputs"]), 1)

    def test_explicit_projection_rejects_unknown_output(self):
        task, _, before = self.ready()
        frame = self.outputs.bind(task, tensor("unknown"))
        self.assertEqual(frame["status"], "service_error")
        self.assertEqual(frame["failure"]["code"], "output_signature_rejected")
        self.assertEqual(self.app.result(task), before)

    def test_receipt_json_reload_and_result_only(self):
        task, provider, outcome, receipt = self.bound()
        reloaded = json.loads(json.dumps(receipt))
        result = self.outputs.result(task, reloaded, expected_binding=receipt["binding"])
        self.assertEqual(result["status"], "succeeded")
        self.assertEqual(result["receipt"], receipt)
        self.assertEqual((provider.calls, self.backend.calls, self.backend.execute_calls), (1, 1, 0))
        self.assertEqual(result["task"], outcome)

    def test_execute_reuses_existing_package_and_restores_structure(self):
        task, provider, _, receipt = self.bound()
        result = self.execute(task, receipt)
        self.assertEqual(result["status"], "executed")
        self.assertEqual(result["result"]["structured_outputs"], [([.015625, .0625], None)])
        raw = result["result"]["execution"]
        self.assertFalse(raw["numerically_validated_on_supplied_inputs"])
        self.assertFalse(raw["application_accuracy_certified"])
        self.assertEqual((provider.calls, self.backend.calls, self.backend.execute_calls), (1, 1, 1))

    def test_repeated_execution_and_reload_do_not_generate_or_compile(self):
        task, provider, _, receipt = self.bound()
        reloaded = json.loads(json.dumps(receipt))
        first = self.execute(task, receipt)
        self.samples = [{"x": [.0, .5]}]
        second = self.execute(task, reloaded)
        self.assertEqual(first["result"]["structured_outputs"], [([.015625, .0625], None)])
        self.assertEqual(second["result"]["structured_outputs"], [([.0, .25], None)])
        self.assertEqual((provider.calls, self.backend.calls, self.backend.execute_calls), (1, 1, 2))

    def test_compiled_package_grade_stays_compiled_during_execution(self):
        task, _, _, receipt = self.bound(validation_level="compiled")
        result = self.execute(task, receipt)
        grade = result["result"]["execution"]["package_validation"]
        self.assertEqual(grade, dict(level="compiled", encrypted_execution=False, numerically_validated=False))
        self.assertFalse(result["task"]["acceptance"]["numerically_validated"])

    def test_receipt_of_cached_task_uses_original_package(self):
        original, _, _, _ = self.bound()
        again = self.app.submit(self.model)
        unused = Provider()
        result = self.app.run(again, unused)
        receipt = self.outputs.bind(again, self.tree)["receipt"]
        self.assertTrue(result["package"]["reused"])
        self.assertEqual(receipt["task_id"], again)
        self.assertEqual(receipt["package_task_id"], original)
        self.assertEqual(self.execute(again, receipt)["status"], "executed")
        self.assertEqual(unused.calls, 0)

    def test_receipt_cannot_cross_tasks_even_with_same_cached_program(self):
        first, _, _, receipt = self.bound()
        second = self.app.submit(self.model)
        self.app.run(second, Provider())
        result = self.execute(second, receipt)
        self.assertEqual(result["failure"]["code"], "output_binding_rejected")
        self.assertEqual(self.backend.execute_calls, 0)

    def test_external_receipt_binding_is_required(self):
        task, _, _, receipt = self.bound()
        result = self.execute(task, receipt, expected_binding="0" * 64)
        self.assertEqual(result["failure"]["code"], "output_binding_rejected")
        self.assertEqual(self.backend.execute_calls, 0)

    def test_resigned_receipt_cannot_override_trusted_binding(self):
        task, _, _, receipt = self.bound()
        trusted = receipt["binding"]
        receipt["package_task_id"] = "0" * 32
        receipt["binding"] = digest({k: v for k, v in receipt.items() if k != "binding"})
        result = self.execute(task, receipt, expected_binding=trusted)
        self.assertEqual(result["failure"]["code"], "output_binding_rejected")
        self.assertEqual(self.backend.execute_calls, 0)

    def test_task_metadata_changes_invalidate_receipt(self):
        task, _, _, receipt = self.bound()
        path = self.app._dir(task) / "task.json"
        metadata = self.app._load(path)
        metadata["budget"]["wall_seconds"] = 200
        metadata["binding"] = digest({k: v for k, v in metadata.items() if k != "binding"})
        self.app._write(path, metadata)
        result = self.execute(task, receipt)
        self.assertEqual(result["failure"]["code"], "output_binding_rejected")
        self.assertEqual(self.backend.execute_calls, 0)

    def test_checked_package_tampering_stops_before_executor(self):
        task, _, _, receipt = self.bound()
        (self.app._dir(task) / "program/manifest.json").write_text('{"level":"compiled"}')
        result = self.execute(task, receipt)
        self.assertEqual(result["status"], "service_error")
        self.assertEqual(self.backend.execute_calls, 0)

    def test_current_dependency_drift_stops_before_executor(self):
        task, _, _, receipt = self.bound()
        self.backend.version += 1
        result = self.execute(task, receipt)
        self.assertEqual(result["status"], "service_error")
        self.assertEqual(self.backend.execute_calls, 0)

    def test_wrong_execution_request_is_rejected(self):
        task, _, _, receipt = self.bound()
        self.backend.override = {"request_id": "0" * 64}
        result = self.execute(task, receipt)
        self.assertEqual(result["failure"]["code"], "output_execution_identity_rejected")

    def test_non_poseidon_execution_is_rejected(self):
        task, _, _, receipt = self.bound()
        self.backend.override = {"backend": "seal_cpu"}
        result = self.execute(task, receipt)
        self.assertEqual(result["failure"]["code"], "output_execution_identity_rejected")

    def test_original_execution_failure_is_preserved(self):
        task, _, _, receipt = self.bound()
        self.backend.override = {
            "status": "validation_failed",
            "failure": {"code": "approximation_domain_rejected", "layer": "input"},
            "encrypted_execution": False,
        }
        result = self.execute(task, receipt)
        self.assertEqual(result["status"], "validation_failed")
        self.assertEqual(result["result"]["execution"]["failure"]["code"], "approximation_domain_rejected")
        self.assertNotIn("structured_outputs", result["result"])

    def test_private_hooks_are_not_called_for_receipts(self):
        class Unsafe:
            def __deepcopy__(self, memo):
                raise AssertionError("Copy hook")
            def __array__(self, *args, **kwargs):
                raise AssertionError("Array hook")
        task, _, _, _ = self.bound()
        result = self.outputs.execute(task, self.samples, Unsafe(), expected_binding="0" * 64)
        self.assertEqual(result["failure"]["code"], "output_binding_rejected")
        self.assertEqual(self.backend.execute_calls, 0)

    def test_original_application_execute_signature_is_unchanged(self):
        import inspect
        self.assertEqual(list(inspect.signature(Application.execute).parameters), ["self", "task", "samples"])

    def test_adapter_has_no_experiment_provider_or_environment_imports(self):
        import ast
        import task_output_adapter
        tree = ast.parse(Path(task_output_adapter.__file__).read_text())
        names = {
            alias.name for node in ast.walk(tree) if isinstance(node, ast.Import)
            for alias in node.names
        } | {
            node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
        }
        self.assertFalse(any(
            name and any(word in name for word in ("benchmarks", "provider", "hecate_python_env"))
            for name in names
        ))


if __name__ == "__main__":
    unittest.main()
