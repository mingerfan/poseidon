"""Public input diagnostics and terminal application controls; no FHE evidence."""
import contextlib
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from benchmark_graph import digest
from pytorch_approximation import FORMAT as CONFIG_FORMAT, Resolver
from pytorch_intake_diagnostics import (
    ApproximationRequired, EXIT_CODE, FILENAME, MAX_BYTES, approximation_required,
    checked, read, record, write,
)

def configuration(defaults=None, modules=None, nodes=None):
    return dict(format=CONFIG_FORMAT, defaults=defaults or {}, modules=modules or [], nodes=nodes or [])

def diagnostic():
    return approximation_required("normalizer", "layer_norm", "rsqrt", ["block", "block.norm"])

def manifest(configuration_value=None):
    return dict(approximations=configuration_value or configuration())

SOURCE = {"pytorch_approximation.py": "a" * 64}
P = dict(method="newton", domain=[.5, 2.], iterations=1)

class DiagnosticTests(unittest.TestCase):
    def test_missing_configuration_has_stable_public_fields(self):
        for node, op, function, paths in [
            ("normalizer", "layer_norm", "rsqrt", ["block.norm"]),
            ("softmax", "softmax", "exp", []),
            ("attention", "exp", "exp", ["attention"])]:
            with self.subTest(operator=op), self.assertRaises(ApproximationRequired) as caught:
                Resolver(configuration()).select(node, op, function, paths)
            value = caught.exception.public_failure
            self.assertEqual(value["layer"], "input")
            self.assertEqual(value["code"], "explicit_approximation_required")
            self.assertEqual(value["location"], {"kind": "fx_node", "name": node})
            self.assertEqual(value["context"]["function"], function)
            self.assertEqual(value["retry_policy"], "supply_explicit_approximation")
            self.assertIsInstance(caught.exception, ValueError)

    def test_diagnostic_is_a_detached_snapshot(self):
        paths = ["block.norm"]
        value = approximation_required("norm", "layer_norm", "rsqrt", paths)
        paths.append("private")
        rebuilt = checked(value)
        value["context"]["module_paths"].append("private")
        self.assertEqual(rebuilt["context"]["module_paths"], ["block.norm"])

    def test_arbitrary_paths_context_and_codes_rejected(self):
        for mutate in [
            lambda v: v.update(message="PRIVATE /home/user/.env"),
            lambda v: v.update(layer="numerical"),
            lambda v: v.update(code="passed"),
            lambda v: v.update(retry_policy="repair_within_budget"),
            lambda v: v["context"].update(reference=[42.]),
            lambda v: v["context"].update(module_paths=["/private/secret"]),
            lambda v: v["context"].update(module_paths=["a", "a"]),
            lambda v: v["context"].update(function="private_secret"),
            lambda v: v["context"].update(operator="torch.load"),
            lambda v: v["location"].update(name="../../secret"),
            lambda v: v["location"].update(name="n"*257),
            lambda v: v["context"].update(module_paths=["a"]*65),
            lambda v: v["location"].update(kind="source_path"),
        ]:
            value = diagnostic(); mutate(value)
            with self.subTest(value=value), self.assertRaises(ValueError): checked(value)

    def test_envelope_round_trip_and_exact_identity(self):
        spec = manifest()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / FILENAME
            stored = write(path, diagnostic(), spec, SOURCE)
            self.assertEqual(read(path, spec, SOURCE), diagnostic())
            self.assertEqual(stored["manifest_sha256"], digest(spec))
            for changed_spec, changed_source in [(dict(spec, weights={}), SOURCE),
                                                (spec, dict(SOURCE, other="b"*64))]:
                with self.subTest(spec=changed_spec), self.assertRaises(ValueError):
                    read(path, changed_spec, changed_source)

    def test_binding_change_and_duplicate_json_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / FILENAME
            value = record(diagnostic(), manifest(), SOURCE)
            value["binding_sha256"] = "b" * 64
            path.write_text(json.dumps(value))
            with self.assertRaises(ValueError): read(path, manifest(), SOURCE)
            raw = json.dumps(record(diagnostic(), manifest(), SOURCE))
            path.write_text(raw[:-1] + ', "format": "duplicate"}')
            with self.assertRaises(ValueError): read(path, manifest(), SOURCE)

    def test_available_configuration_cannot_claim_missing(self):
        for value in [
            configuration(defaults={"rsqrt": P}),
            configuration(nodes=[dict(name="normalizer", function="rsqrt", profile=P)]),
            configuration(modules=[dict(path="block.norm", function="rsqrt", profile=P)]),
        ]:
            with self.subTest(configuration=value), self.assertRaisesRegex(ValueError, "does not witness"):
                record(diagnostic(), manifest(value), SOURCE)

    def test_selector_conflict_is_not_mislabeled_as_missing(self):
        value = configuration(modules=[
            dict(path="block", function="rsqrt", profile=P),
            dict(path="block.norm", function="rsqrt", profile=P)])
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            record(diagnostic(), manifest(value), SOURCE)

    def test_unrelated_configuration_retains_first_missing_function(self):
        value = configuration(defaults={"exp": dict(method="taylor", domain=[-1., 1.], degree=2)})
        stored = record(diagnostic(), manifest(value), SOURCE)
        self.assertEqual(stored["diagnostic"]["context"]["function"], "rsqrt")

    def test_file_size_symlink_and_exclusive_write(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / FILENAME
            write(path, diagnostic(), manifest(), SOURCE)
            with self.assertRaises(FileExistsError): write(path, diagnostic(), manifest(), SOURCE)
            target = Path(folder) / "link"; target.symlink_to(path)
            with self.assertRaises((OSError, ValueError)): read(target, manifest(), SOURCE)
            path.write_bytes(b"x"*(MAX_BYTES+1))
            with self.assertRaises(ValueError): read(path, manifest(), SOURCE)

    def test_failure_has_no_profiles_arrays_or_environment(self):
        value = record(diagnostic(), manifest(), SOURCE)
        raw = json.dumps(value)
        for private in ("coefficients", "domain", "weights", "reference", "inputs", "environment",
                        ".env", "/home/", "api_key", "tensor", "scale"):
            self.assertNotIn(private, raw)
        self.assertLess(len(raw), MAX_BYTES)

    def test_legacy_missing_profile_requires_no_new_mounted_module(self):
        import builtins
        original_import = builtins.__import__
        def guarded(name, *args, **kwargs):
            if name == "pytorch_intake_diagnostics":
                raise ModuleNotFoundError("Legacy static worker does not mount new diagnostic module")
            return original_import(name, *args, **kwargs)
        with patch.object(builtins, "__import__", side_effect=guarded):
            with self.assertRaises(ApproximationRequired) as caught:
                Resolver(configuration()).select("norm", "layer_norm", "rsqrt", [])
        self.assertEqual(caught.exception.public_failure["context"]["function"], "rsqrt")

    def test_worker_emits_bound_failure_and_defined_exit(self):
        import pytorch_ordered_intake as intake
        value = diagnostic(); spec = manifest()
        with patch.object(intake, "capture_request", side_effect=ApproximationRequired(value)), \
             patch("strict_json_files.strict_file", return_value=dict(manifest=spec, source_hashes=SOURCE)), \
             patch("pytorch_intake_diagnostics.write") as emitted:
            with self.assertRaises(SystemExit) as stop: intake.worker()
        self.assertEqual(stop.exception.code, EXIT_CODE)
        emitted.assert_called_once_with(Path("/out")/FILENAME, value, spec, SOURCE)

    def test_worker_unknown_failures_do_not_gain_typed_claims(self):
        import pytorch_ordered_intake as intake
        with patch.object(intake, "capture_request", side_effect=ValueError("PRIVATE")), \
             patch("pytorch_intake_diagnostics.write") as emitted:
            with self.assertRaisesRegex(ValueError, "PRIVATE"): intake.worker()
        emitted.assert_not_called()

def source_manifest(root):
    content = "import torch\nclass Model(torch.nn.Module):\n def forward(self,x): return x.square()\ndef build(): return Model().double().eval()\n"
    (root/"model.py").write_text(content)
    spec = dict(format="poseidon-trusted-pytorch-v28", entry="model.build",
                files={"model.py":hashlib.sha256(content.encode()).hexdigest()},
                inputs=[dict(name="x",shape=[2,2])], outputs=["out"], weights={},
                approximations=configuration(), dtype_policy="strict-float64-v1",
                factory=dict(args=[],kwargs={}),
                call=dict(format="poseidon-pytorch-call-v1",args=[dict(kind="tensor",name="x")],kwargs={}))
    path=root/"manifest.json";path.write_text(json.dumps(spec))
    return path

class ApplicationTests(unittest.TestCase):
    def setUp(self):
        from application_component import Application
        from test_application_model_source import Backend
        from application_model_source import FORMAT
        self.tmp=tempfile.TemporaryDirectory()
        self.root=Path(self.tmp.name);self.backend=Backend()
        self.app=Application(self.root/"tasks", self.backend)
        self.task=self.app.submit_source(dict(format=FORMAT,kind="trusted_pytorch",
            manifest=str(source_manifest(self.root)),trust_local_code=True),resource_profile="general-tensors-v1")
        self.assertEqual(self.app.status(self.task)["phase"],"intake_queued")
    def tearDown(self): self.tmp.cleanup()

    def test_typed_input_failure_is_terminal_without_generation(self):
        from test_application_component import Provider
        self.app._capture=lambda *args: (_ for _ in ()).throw(ApproximationRequired(diagnostic()))
        value=self.app.prepare_source(self.task)
        self.assertEqual((value["status"],value["phase"]),("unsupported","complete"))
        self.assertEqual(value["failure"],diagnostic())
        self.assertEqual(value["intake"]["status"],"rejected")
        self.assertEqual((value["generations"],value["http_attempts"],self.backend.calls),(0,0,0))
        self.assertIsNone(value["package"]);self.assertIsNone(value["acceptance"])
        provider=Provider()
        self.assertEqual(self.app.run(self.task,provider)["status"],"unsupported")
        self.assertEqual(provider.calls,0)
        self.assertFalse((self.app._dir(self.task)/"task.json").exists())

    def test_unknown_capture_failure_keeps_existing_terminal_result(self):
        self.app._capture=lambda *args: (_ for _ in ()).throw(ValueError("PRIVATE /home/user/key"))
        value=self.app.prepare_source(self.task)
        self.assertEqual(value["status"],"validation_failed")
        self.assertEqual(value["failure"]["code"],"model_source_intake_rejected")
        self.assertNotIn("PRIVATE",json.dumps(value))
        self.assertNotIn(self.tmp.name,json.dumps(value))
        self.assertEqual(self.backend.calls,0)

    def test_cli_reports_typed_input_failure_without_traceback(self):
        import model_import_cli
        from hecate_python_env import VENV
        manifest_path=self.root/"manifest.json"
        output=io.StringIO()
        with patch.object(sys,"argv",["prepare_model.py","--trusted-pytorch-manifest",str(manifest_path),
                    "--trust-local-code","--write","--inside","--output",str(self.root/"capture"),
                    "--resource-profile","general-tensors-v1"]), \
             patch.object(sys,"prefix",str(VENV)), patch.dict(os.environ,{"IN_NIX_SHELL":"pure"}), \
             patch("pytorch_model_intake.run",side_effect=ApproximationRequired(diagnostic())), \
             contextlib.redirect_stdout(output):
            code=model_import_cli.main()
        self.assertEqual(code,EXIT_CODE)
        value=json.loads(output.getvalue())
        self.assertEqual(value["failure"],diagnostic())
        self.assertEqual((value["status"],value["paid_calls"],value["compiler_calls"]),
                         ("unsupported",0,0))
        self.assertFalse(value["encrypted_execution"])

if __name__=="__main__": unittest.main()
