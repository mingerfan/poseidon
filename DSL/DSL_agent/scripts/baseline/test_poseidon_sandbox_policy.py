"""Poseidon sandbox inventory, process isolation and dependency binding tests."""
import copy
import hashlib
import importlib.abc
import json
import os
import resource
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from application_identity import _syntax_facts, source_closure
from poseidon_sandbox_policy import MODULES, limits

class SandboxProfileTests(unittest.TestCase):
    def test_legacy_inventory_and_shared_limits_remain_available(self):
        from sandbox_policy import MODULES as legacy, limits as old_limits
        self.assertIs(limits, old_limits)
        self.assertEqual(set(legacy) - set(MODULES), {"sandbox_policy.py", "upstream_helper_probe.py"})
        self.assertEqual(set(MODULES) - set(legacy), {"poseidon_sandbox_policy.py", "public_math_context.py", "unified_output_binding_guidance.py"})
        self.assertEqual(len(MODULES), len(set(MODULES)))
        base = Path(__file__).parent
        self.assertTrue(all((base/name).is_file() for name in MODULES))
        self.assertIn("seal_artifact_gate.py", MODULES)  # Shared require(), not a SEAL executor.
        self.assertIn("process_resource_limits.py", MODULES)

    def test_math_role_verifier_is_read_only_and_bound(self):
        from application_identity import source_closure
        base=Path(__file__).parent
        self.assertEqual(MODULES.count("public_math_context.py"),1)
        closure=source_closure(base,("poseidon_candidate_sandbox.py",))
        self.assertIn("public_math_context.py",closure["sources"])
        import public_math_context
        self.assertEqual(public_math_context.POLICY,"descriptors-math-roles-v2")

    def test_compact_context_dependency_is_available_and_bound(self):
        from sandbox_policy import MODULES as legacy
        from application_identity import source_closure
        from public_origin_columns import POLICY
        base=Path(__file__).parent
        self.assertEqual(POLICY,"descriptors-compact-origins-v3")
        self.assertEqual(MODULES.count("public_origin_columns.py"),1)
        self.assertEqual(legacy.count("public_origin_columns.py"),1)
        closure=source_closure(base,("poseidon_candidate_sandbox.py",))
        self.assertEqual(closure["sources"]["public_origin_columns.py"],
                         hashlib.sha256((base/"public_origin_columns.py").read_bytes()).hexdigest())

    def test_child_observes_unchanged_resource_limits(self):
        script = """
import json,resource
from process_resource_limits import limits
limits()
names=('RLIMIT_CORE','RLIMIT_AS','RLIMIT_CPU','RLIMIT_FSIZE','RLIMIT_NOFILE')
print(json.dumps({n:resource.getrlimit(getattr(resource,n)) for n in names}))
"""
        script = "import sys;sys.path.insert(0," + repr(str(Path(__file__).parent)) + ")\n" + script
        child = subprocess.run([sys.executable, "-B", "-c", script], capture_output=True,
                               text=True, timeout=15)
        self.assertEqual(child.returncode, 0, child.stderr)
        self.assertEqual(json.loads(child.stdout), {
            "RLIMIT_CORE": [0, 0], "RLIMIT_AS": [4*1024**3, 4*1024**3],
            "RLIMIT_CPU": [150, 155], "RLIMIT_FSIZE": [16*1024**2, 16*1024**2],
            "RLIMIT_NOFILE": [128, 128]})

    def test_poseidon_command_retains_isolation_and_uses_own_inventory(self):
        import poseidon_candidate_sandbox as sandbox
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory); output = work/"results/out"; output.mkdir(parents=True)
            payload = work/"results/payload.json"; payload.write_text("{}")
            with patch.object(sandbox, "WORK", work), patch.dict(os.environ, {"HECATE_PYTHON_LIBRARY_PATH": "/nix/store/pinned/lib"}):
                cmd = sandbox.command(payload, output, ["/frontend/python", "/app/candidate_trace.py"])
                self.assertIs(sandbox.limits, limits)
                with self.assertRaisesRegex(ValueError, "legacy key"):
                    sandbox.command(payload, output, [], keys=work)
                with self.assertRaisesRegex(ValueError, "payload"):
                    sandbox.command(work/"other", output, [])
                with patch.dict(os.environ, {"HECATE_PYTHON_LIBRARY_PATH": "/tmp/library"}):
                    with self.assertRaisesRegex(ValueError, "Non-Nix"):
                        sandbox.command(payload, output, [])
        for flag in ("--unshare-all", "--die-with-parent", "--new-session", "--clearenv"):
            self.assertIn(flag, cmd)
        self.assertEqual(cmd[cmd.index("--cap-drop")+1], "ALL")
        destinations = {cmd[i+2] for i,v in enumerate(cmd) if v == "--ro-bind" and cmd[i+2].startswith("/app/")}
        self.assertEqual(destinations, {"/app/"+name for name in MODULES})
        self.assertNotIn("/app/upstream_helper_probe.py", destinations)
        self.assertNotIn("/app/sandbox_policy.py", destinations)
        environment = {cmd[i+1]:cmd[i+2] for i,v in enumerate(cmd) if v == "--setenv"}
        self.assertEqual(environment["PATH"], "/nonexistent")
        self.assertEqual(environment["PYTHONNOUSERSITE"], "1")
        self.assertNotIn("POSEIDON_PROBE_SECRET", environment)

    def test_fresh_poseidon_import_does_not_load_legacy_probe_or_provider(self):
        script = """
import importlib.abc,sys
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'sandbox_policy','upstream_helper_probe','deepseek_provider','deepseek_http_worker'}:
            raise AssertionError('forbidden import: '+fullname)
sys.meta_path.insert(0, Guard())
import poseidon_candidate_sandbox
assert 'poseidon_sandbox_policy' in sys.modules
assert 'process_resource_limits' in sys.modules
"""
        script = "import sys;sys.path.insert(0," + repr(str(Path(__file__).parent)) + ")\n" + script
        child = subprocess.run([sys.executable, "-B", "-c", script], env=dict(os.environ),
                               capture_output=True, text=True, timeout=30)
        self.assertEqual(child.returncode, 0, child.stderr)

    def test_feedback_dependencies_are_bound_to_the_program_identity(self):
        from candidate_bundle import implementation
        current = implementation()
        closure = source_closure(Path(__file__).parent, ["poseidon_sandbox_policy.py"])
        for name in ("unified_binding_guidance.py", "candidate_check_diagnostic.py"):
            with self.subTest(name=name):
                self.assertEqual(current[name],
                    hashlib.sha256(Path(__file__).parent.joinpath(name).read_bytes()).hexdigest())
                self.assertIn(name, closure["sources"])

    def test_shallow_layout_dependencies_are_mounted_and_identity_bound(self):
        from sandbox_policy import MODULES as legacy
        base=Path(__file__).parent
        bound=source_closure(base,("poseidon_candidate_sandbox.py",))["sources"]
        for name in ("capacity_layout_costs.py", "unified_capacity_tiling.py",
                     "unified_capacity_constants.py"):
            self.assertEqual(MODULES.count(name),1)
            self.assertEqual(legacy.count(name),1)
            self.assertEqual(bound[name],hashlib.sha256((base/name).read_bytes()).hexdigest())

    def test_fresh_inventory_worker_imports_guidance_and_typed_feedback(self):
        import shutil
        with tempfile.TemporaryDirectory() as directory:
            mount = Path(directory)
            for name in MODULES:
                shutil.copy2(Path(__file__).parent/name, mount/name)
            script = """
import ast,sys
sys.path.insert(0,sys.argv[1])
from unified_binding_guidance import VERSION
from candidate_check_diagnostic import CandidateCheckError
from unified_capacity_constants import SHALLOW_VERSION
from unified_capacity_tiling import template
from capacity_layout_costs import template_costs, candidate_costs, choose_candidate
assert SHALLOW_VERSION == "capacity-public-tensors-v10"
cost=template_costs(template([[0.25,-0.125,0.5]],3,16),16)
assert cost["plaintext_depth_upper_bound"] >= 1
assert cost["compiler_feasibility"] == "unknown"
chosen=choose_candidate([candidate_costs("diagonal",1,50,7)])
assert chosen["execution_feasibility"] == "unknown"

assert VERSION == "explicit-v11"
node = ast.parse("unknown(x)", mode="eval").body
error = CandidateCheckError("undefined_value", node)
assert error.reason == "undefined_value"
assert error.location == {"kind":"source","line":1,"column":0}
assert "deepseek_provider" not in sys.modules
assert "sandbox_policy" not in sys.modules
"""
            child = subprocess.run([sys.executable, "-I", "-B", "-c", script, str(mount)],
                cwd=mount, env={"PATH":os.defpath,"PYTHONDONTWRITEBYTECODE":"1"},
                capture_output=True, text=True, timeout=15)
            self.assertEqual(child.returncode, 0, child.stderr)

    def test_program_binding_uses_poseidon_inventory(self):
        from candidate_bundle import implementation
        current = implementation()
        for name in ("poseidon_sandbox_policy.py", "process_resource_limits.py"):
            self.assertEqual(current[name], hashlib.sha256(Path(__file__).parent.joinpath(name).read_bytes()).hexdigest())
        self.assertNotIn("sandbox_policy.py", current)

class SandboxInventoryIdentityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root/"mounted.py").write_text("VALUE=1\n")
    def policy(self, name, source="MODULES=('mounted.py',)\n"):
        (self.root/name).write_text(source)
    def test_both_inventories_bind_mounted_bytes_and_detect_changes(self):
        for name in ("sandbox_policy.py", "poseidon_sandbox_policy.py"):
            self.policy(name)
            first = source_closure(self.root, [name])
            self.assertIn("mounted.py", first["sources"])
            self.assertIn("mounted.py", first["import_edges"][name])
            (self.root/"mounted.py").write_text("VALUE=2\n")
            second = source_closure(self.root, [name])
            self.assertNotEqual(first["sources"]["mounted.py"], second["sources"]["mounted.py"])
            (self.root/"mounted.py").write_text("VALUE=1\n")
    def test_missing_and_symlinked_mounts_fail_closed(self):
        for name in ("sandbox_policy.py", "poseidon_sandbox_policy.py"):
            self.policy(name, "MODULES=('missing.py',)\n")
            with self.assertRaisesRegex(ValueError, "source file"): source_closure(self.root, [name])
            indirect = self.root/"indirect.py"
            if not indirect.exists(): indirect.symlink_to(self.root/"mounted.py")
            self.policy(name, "MODULES=('indirect.py',)\n")
            with self.assertRaisesRegex(ValueError, "symlink"): source_closure(self.root, [name])
    def test_malformed_inventory_cannot_hide_dependencies(self):
        sources = [
            "VALUE=1\n", "MODULES=[]\n", "MODULES=()\n", "MODULES=('mounted.py','mounted.py')\n",
            "MODULES=('../escape.py',)\n", "MODULES=('/absolute.py',)\n", "MODULES=(1,)\n",
            "MODULES=('mounted.py',)\nMODULES=('other.py',)\n", "MODULES=tuple(('mounted.py',))\n"]
        for name in ("sandbox_policy.py", "poseidon_sandbox_policy.py"):
            for source in sources:
                with self.subTest(name=name, source=source):
                    self.policy(name, source)
                    with self.assertRaises(ValueError): source_closure(self.root, [name])
    def test_analyzer_filename_checks_do_not_create_fake_policy_edges(self):
        raw = Path(__file__).parent.joinpath("application_identity.py").read_bytes()
        _, literals, _, mounts = _syntax_facts(raw, "application_identity.py")
        self.assertNotIn("sandbox_policy.py", literals)
        self.assertNotIn("poseidon_sandbox_policy.py", literals)
        self.assertEqual(mounts, ())
        self.policy("poseidon_sandbox_policy.py")
        closure = source_closure(self.root, ["poseidon_sandbox_policy.py"])
        self.assertEqual(set(closure["sources"]), {"poseidon_sandbox_policy.py", "mounted.py"})

if __name__ == "__main__": unittest.main()
