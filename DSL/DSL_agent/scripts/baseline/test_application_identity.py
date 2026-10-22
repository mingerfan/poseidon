import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import application_identity as identity


def sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


def backend():
    value = {"format": "poseidon-backend-identity-v2"}
    for key in set().union(*[set(v) for v in identity.TOOL_FIELDS.values()]):
        value[key] = {"id": "aarch64-linux"} if key == "platform" else ({"schema": 1} if key == "source_lock" else sha(key))
    value["build_configuration"] = {
        "environment_files": {"lock.json": sha("locked")},
        "cmake_cache_sha256": {k: sha(k) for k in ("compiler", "executor", "poseidon")}}
    return value


class IdentityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.base = self.root / "scripts/baseline"
        self.base.mkdir(parents=True)
        (self.base/identity.CATALOG).write_text(json.dumps(identity.source_catalog(Path(identity.__file__).parent)))
        names = set().union(*[set(v) for v in identity.ROOTS.values()])
        names.update(("application_identity.py", "application_backend.py"))
        for name in names:
            (self.base / name).write_text("VALUE = 1\n")
        self.tools = backend()

    def manifest(self):
        return identity.identity_manifest(self.root, self.tools)

    def rewrite(self, path, value):
        p = self.root / path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(value)

    def resign(self, value):
        for stage in identity.STAGES:
            item = value[stage]
            item["sha256"] = identity._digest({k: v for k, v in item.items() if k != "sha256"})
        value["manifest_sha256"] = identity._digest({k: v for k, v in value.items() if k != "manifest_sha256"})

    def test_unrelated_tests_docs_and_research_tools_do_not_invalidate(self):
        before = self.manifest()
        for name in ("scripts/baseline/test_unrelated.py", "docs/baseline/report.json",
                     "scripts/baseline/benchmarks/tools/research.py", "scripts/baseline/unused_probe.py"):
            self.rewrite(name, "UNRELATED = 2")
        self.assertEqual(before, self.manifest())

    def test_transitive_function_import_is_bound_without_execution(self):
        (self.base / "candidate_trace.py").write_text("def trace():\n    from critical import run\n")
        (self.base / "critical.py").write_text("raise RuntimeError('must not execute')\ndef run(): pass\n")
        before = self.manifest()
        (self.base / "critical.py").write_text("raise RuntimeError('still not executed')\ndef run(): pass\n")
        self.assertNotEqual(before["compilation"], self.manifest()["compilation"])

    def test_literal_dynamic_import_is_bound(self):
        (self.base / "candidate_trace.py").write_text("import importlib\ndef f(): return importlib.import_module('critical')\n")
        (self.base / "critical.py").write_text("VALUE=1")
        before = self.manifest()
        (self.base / "critical.py").write_text("VALUE=2")
        self.assertFalse(identity.compatible(before, self.manifest()))

    def test_unknown_dynamic_dependency_fails_closed(self):
        (self.base / "candidate_trace.py").write_text("from importlib import import_module as load\ndef f(name): return load(name)\n")
        with self.assertRaisesRegex(ValueError, "Unbounded dynamic dependency"):
            self.manifest()

    def test_relative_imports_and_package_initializers_are_bound(self):
        (self.base / "candidate_trace.py").write_text("from helpers.child import f\n")
        self.rewrite("scripts/baseline/helpers/__init__.py", "TOKEN=1")
        self.rewrite("scripts/baseline/helpers/child.py", "from .shared import value\ndef f(): return value")
        self.rewrite("scripts/baseline/helpers/shared.py", "value=1")
        before = self.manifest()
        self.rewrite("scripts/baseline/helpers/shared.py", "value=2")
        self.assertFalse(identity.compatible(before, self.manifest()))
        self.assertIn("helpers/__init__.py", before["compilation"]["sources"])

    def test_sandbox_non_imported_worker_and_data_are_bound(self):
        (self.base / "candidate_trace.py").write_text("import sandbox_policy\n")
        self.rewrite("scripts/baseline/sandbox_policy.py", "MODULES=('worker.py','rules.json')")
        self.rewrite("scripts/baseline/worker.py", "VALUE=1")
        self.rewrite("scripts/baseline/rules.json", "{}")
        before = self.manifest()
        self.rewrite("scripts/baseline/worker.py", "VALUE=2")
        self.assertFalse(identity.compatible(before, self.manifest()))

    def test_helper_verifier_inventory_is_preserved_as_data(self):
        (self.base / "candidate_trace.py").write_text("import upstream_candidate_helpers\n")
        self.rewrite("scripts/baseline/upstream_candidate_helpers.py", "VALUE=1")
        self.rewrite("scripts/baseline/upstream_adapters/helper.py", "VALUE=1")
        self.rewrite("scripts/baseline/upstream_adapters/test_helper.py", "exec(unknown)")
        before = self.manifest()  # Data hash must not execute or parse as a runtime loader.
        self.rewrite("scripts/baseline/upstream_adapters/helper.py", "VALUE=2")
        self.assertFalse(identity.compatible(before, self.manifest()))
        self.assertIn("upstream_adapters/test_helper.py", before["compilation"]["sources"])

    def test_explicit_import_of_prehashed_helper_still_checks_dynamic_code(self):
        (self.base / "candidate_trace.py").write_text("import upstream_candidate_helpers\nimport upstream_adapters.helper\n")
        self.rewrite("scripts/baseline/upstream_candidate_helpers.py", "VALUE=1")
        self.rewrite("scripts/baseline/upstream_adapters/helper.py", "exec(unknown)")
        with self.assertRaisesRegex(ValueError, "Unbounded dynamic dependency"):
            self.manifest()

    def test_parameters_locks_patches_invalidate(self):
        for name in ("scripts/baseline/poseidon-compiler-profile-v1.json",
                     "scripts/baseline/poseidon-operator-spec-v1.json",
                     "scripts/baseline/poseidon-upstream-helpers-v1.json",
                     "src/poseidon/tools/dacapo/dependency-lock.json",
                     "src/poseidon/tools/dacapo/poly-python-wheels.lock.json",
                     "scripts/baseline/patches/semantic.patch"):
            self.rewrite(name, "initial")
            before = self.manifest()
            self.rewrite(name, "changed")
            self.assertFalse(identity.compatible(before, self.manifest()), name)

    def test_backend_binary_and_build_changes_invalidate_correct_stage(self):
        before = self.manifest()
        self.tools["executor_sha256"] = sha("new executor")
        after = self.manifest()
        self.assertEqual(before["generation"], after["generation"])
        self.assertEqual(before["compilation"], after["compilation"])
        self.assertNotEqual(before["execution"], after["execution"])
        self.assertFalse(identity.compatible(before, after))
        self.tools = backend()
        self.tools["build_configuration"]["cmake_cache_sha256"]["compiler"] = sha("new compiler flags")
        self.assertNotEqual(before["compilation"], self.manifest()["compilation"])

    def test_unknown_backend_fields_are_bound_conservatively(self):
        before = self.manifest()
        self.tools["new_runtime_constraint"] = {"value": 1}
        after = self.manifest()
        for stage in identity.STAGES:
            self.assertNotEqual(before[stage], after[stage])

    def test_generation_only_change_keeps_delivery_compatible(self):
        before = self.manifest()
        (self.base / "deepseek_provider.py").write_text("VALUE=2")
        after = self.manifest()
        self.assertNotEqual(before["generation"], after["generation"])
        self.assertTrue(identity.compatible(before, after))
        self.assertFalse(identity.compatible(before, after, stages=("generation",)))

    def test_shared_prompt_module_is_not_falsely_excluded(self):
        (self.base / "candidate_trace.py").write_text("import deepseek_provider")
        before = self.manifest()
        (self.base / "deepseek_provider.py").write_text("VALUE=2")
        self.assertFalse(identity.compatible(before, self.manifest()))

    def test_new_program_execution_entry_is_bound(self):
        before = self.manifest()
        (self.base / "program_execution.py").write_text("VALUE=1")
        after = self.manifest()
        self.assertNotEqual(before["execution"], after["execution"])
        self.assertIn("program_execution.py", after["execution"]["roots"])

    def test_identity_tampering_and_omitted_backend_fields_rejected(self):
        good = self.manifest()
        for field in ("manifest_sha256", "format"):
            bad = copy.deepcopy(good)
            bad[field] = "invalid"
            with self.assertRaises(ValueError):
                identity.check_manifest(bad)
        bad = copy.deepcopy(good)
        bad["execution"]["sources"]["candidate.py"] = sha("x")
        with self.assertRaises(ValueError):
            identity.compatible(bad, good)
        bad = copy.deepcopy(good)
        del bad["execution"]["backend"]["executor_sha256"]
        self.resign(bad)
        with self.assertRaisesRegex(ValueError, "projection"):
            identity.check_manifest(bad)

    def test_symlink_source_and_asset_rejected(self):
        outside = self.root / "outside.py"
        outside.write_text("VALUE=1")
        (self.base / "candidate_trace.py").unlink()
        (self.base / "candidate_trace.py").symlink_to(outside)
        with self.assertRaisesRegex(ValueError, "symlink"):
            self.manifest()

    def test_legacy_identity_method_is_unchanged(self):
        # Identity v1's broad inventory stays byte-for-byte equivalent.
        import application_backend
        import hecate_python_env
        import poseidon_backend_identity
        expected_files = list(self.base.glob("*.py")) + list(self.base.glob("*.json")) + list(self.base.glob("patches/*.patch"))
        expected_files += list(self.base.glob("upstream_adapters/*.py"))
        expected = {"backend": application_backend.BACKEND, "tools": self.tools,
                    "implementation_digest": identity._digest({
                        str(p.relative_to(self.base)): hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in expected_files})}
        with patch.object(hecate_python_env, "ROOT", self.root), patch.object(poseidon_backend_identity, "identity", return_value=self.tools):
            self.assertEqual(expected, application_backend.PoseidonBackend().identity())

    def test_backend_explicit_new_api_delegates(self):
        import application_backend
        expected = self.manifest()
        with patch.object(identity, "current_identity_manifest", return_value=expected):
            self.assertIs(expected, application_backend.PoseidonBackend().identity_manifest())


if __name__ == "__main__":
    unittest.main()
