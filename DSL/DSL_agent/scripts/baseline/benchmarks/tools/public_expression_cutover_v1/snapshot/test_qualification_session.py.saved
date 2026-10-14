"""Application qualification boundary and public feedback regressions; no API."""
import copy
import importlib.abc
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from public_feedback_contract import public_feedback, FeedbackError


class PublicFeedbackTests(unittest.TestCase):
    def test_original_public_envelope_and_deep_copy(self):
        value = dict(status="failed", layer="compiler", category="candidate",
            diagnostic="Bounded diagnostic",
            tool_diagnostic=dict(trust="untrusted_tool_output", text="bounded log"),
            numerical_summary=dict(mae=0.01, max_absolute_error=0.02,
                                   compared_values=8, passed=False))
        original = copy.deepcopy(value)
        returned = public_feedback(value)
        self.assertEqual(returned, original)
        returned["numerical_summary"]["mae"] = 9
        self.assertEqual(value, original)

    def test_private_data_and_nonfinite_summary_are_refused(self):
        invalid = [
            dict(status="failed", reference=[1]),
            dict(status="failed", inputs=[1]),
            dict(status="failed", environment={}),
            dict(status="failed", tool_diagnostic=dict(trust="trusted", text="log")),
            dict(status="failed", numerical_summary=dict(actual=[1])),
            dict(status="failed", numerical_summary=dict(mae=float("nan"))),
            dict(status="failed", numerical_summary=dict(mae=float("inf"))),
            dict(status="failed", numerical_summary=dict(compared_values=True)),
            dict(status="failed", diagnostic="x"*8001),
        ]
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(FeedbackError):
                public_feedback(value)

    def test_provider_wrapper_keeps_original_error_codes_and_class(self):
        from deepseek_provider import public_feedback as provider_feedback, ProviderError
        for value in (dict(status="failed", reference=[]),
                      dict(status="failed", numerical_summary=dict(passed=1))):
            with self.assertRaises(FeedbackError) as neutral:
                public_feedback(value)
            with self.assertRaises(ProviderError) as legacy:
                provider_feedback(value)
            self.assertEqual(str(neutral.exception), str(legacy.exception))
        value = dict(status="failed", layer="static_check")
        self.assertEqual(provider_feedback(value), public_feedback(value))


class QualificationBoundaryTests(unittest.TestCase):
    def test_actual_session_and_feedback_import_without_provider_or_cli(self):
        baseline = Path(__file__).parent
        code = """
import importlib.abc,sys,json
sys.path.insert(0,sys.argv[1])
blocked={'deepseek_provider','deepseek_http_worker','run_candidate',
         'unified_graph_lowering','benchmark_runner'}
class Guard(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path,target=None):
  if fullname in blocked:raise AssertionError('Unexpected boundary import: '+fullname)
sys.meta_path.insert(0,Guard())
import qualification_session,component_backend,public_feedback_contract
assert not blocked.intersection(sys.modules)
print(json.dumps({'provider_imported':False,'cli_imported':False,'paid_calls':0}))
"""
        p = subprocess.run([sys.executable, "-B", "-c", code, str(baseline)],
                           text=True, capture_output=True, timeout=20)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertEqual(json.loads(p.stdout)["paid_calls"], 0)

    def test_component_consumes_direct_request_candidate_and_level(self):
        from component_backend import qualify
        request = {"request_id":"f"*64,
                   "compiler_configuration":{"backend":"poseidon_cpu"}}
        candidate = {"schema":1,"request_id":request["request_id"],"hecate_source":"invalid"}
        report = dict(status="cancelled", attempts=[], failure_layer="sandbox_preflight",
                      diagnostic="Host cancelled qualification")
        with patch("qualification_session.QualificationSession.qualify",
                   return_value=(130, report, Path("/private/test-evidence"))) as session:
            outcome = qualify(request, candidate, validation_level="numerical")
        session.assert_called_once_with(request, candidate, validation_level="numerical")
        self.assertEqual(outcome["status"], "cancelled")
        self.assertFalse(outcome["encrypted_execution"])
        self.assertFalse(outcome["numerically_validated"])
        self.assertEqual(outcome["paid_calls"], 0)

    def test_component_preserves_controlled_failure_feedback(self):
        from component_backend import qualify
        public = dict(status="failed",layer="compiler",category="candidate",
                      diagnostic="Compiler rejected candidate")
        report = dict(status="repair_budget_exhausted",
            attempts=[dict(compiled=False, failure_layer="compiler", diagnostic="private host diagnostic")],
            loop=dict(feedback_history=[public]))
        request={"request_id":"f"*64,"compiler_configuration":{"backend":"poseidon_cpu"}}
        with patch("qualification_session.QualificationSession.qualify",
                   return_value=(1,report,Path("/private/test-evidence"))):
            outcome=qualify(request,{},validation_level="compiled")
        self.assertEqual(outcome["public_feedback"],public)
        self.assertFalse(outcome["encrypted_execution"])

    def test_non_poseidon_backend_refused_before_session(self):
        from component_backend import qualify
        with patch("qualification_session.QualificationSession.qualify") as session:
            with self.assertRaises(ValueError):
                qualify({"compiler_configuration":{"backend":"seal_cpu"}},{})
        session.assert_not_called()

    def test_cli_json_assets_remain_bound_and_tampering_is_refused(self):
        import qualification_session as module
        from hecate_python_env import WORK, digest
        baseline=Path(__file__).parent
        fixed=json.loads((baseline/"golden_cases/fixtures.json").read_text())
        def bind(report,request,waterline):
            for key in ("compiler_profile_sha256","compiler_sha256","runtime_sha256",
                        "poseidon_library_sha256","operator_spec_sha256","frontend_sha256",
                        "frontend_source_sha256","frontend_python_sha256"):
                report[key]="0"*64
            report.update(backend_identity={"test_only":True},request_id="a"*64,
                          waterline=waterline,compiler_configuration={"backend":"poseidon_cpu"})
        asset="platform-profiles.json"
        report={"source_hashes":{asset:digest(baseline/asset)}}
        with tempfile.TemporaryDirectory(dir=WORK/"results",prefix="test-qualification-") as folder, \
             patch.object(module,"bind_backend",side_effect=bind):
            session=module.QualificationSession(Path(folder),{},report,{},fixed,45)
            self.assertNotIn(asset,session._roots)
            self.assertEqual(report["source_hashes"][asset],digest(baseline/asset))
            report["source_hashes"][asset]="f"*64
            with self.assertRaisesRegex(ValueError,"binding report changed"):
                session.verify_frozen()

    def test_cli_and_component_share_the_real_validation_adapter(self):
        import ast
        root = Path(__file__).parent
        tree = ast.parse((root/"run_candidate.py").read_text())
        inside = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="inside")
        calls = [ast.unparse(n.func) for n in ast.walk(inside) if isinstance(n,ast.Call)]
        self.assertIn("QualificationSession", calls)
        self.assertIn("session.preflight", calls)
        self.assertNotIn("CandidateValidationAdapter", calls)
        session = ast.parse((root/"qualification_session.py").read_text())
        names = {n.module for n in ast.walk(session) if isinstance(n,ast.ImportFrom)}
        self.assertIn("validation_adapter", names)
        self.assertNotIn("run_candidate", names)
        self.assertNotIn("deepseek_provider", names)
        self.assertNotIn("unified_graph_lowering", names)


if __name__ == "__main__":
    unittest.main()
