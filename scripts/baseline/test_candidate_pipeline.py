"""Local contracts/mocked loop vs opt-in REAL execution evidence, kept separate."""
import copy
import json
import os
from pathlib import Path
from workspace_paths import ROOT, WORK, RESULTS
import unittest
from unittest.mock import patch, mock_open

from candidate_contract import (MAX_REPAIRS, RULES, ReplayProvider, make_request,
                                strict_json, validate_candidate, run_feedback_loop)


def request_fixture():
    return make_request(dict(fx_graph="y = x * w + b", public_constants={"w": [1, 2, 3, 4], "b": .5},
                             constant_origins={"w": "weight", "b": "bias"},
                             layout=dict(input_shape=[4], output_shape=[4], output_ciphertexts=1,
                                         output_selectors=[[0, i] for i in range(4)])),
                        dict(schema=1, family="affine", configuration=0, id="affine-0"), "a" * 64)


class CandidateContractTests(unittest.TestCase):
    def setUp(self):
        self.request = request_fixture()
        self.candidate = dict(schema=1, request_id=self.request["request_id"],
                              hecate_source='@hc.func("c")\ndef golden(x):\n    return x * w + b\n')

    def test_request_does_not_contain_reference_or_rule_answer(self):
        self.assertNotIn("hecate_source", self.request)
        self.assertNotIn("reference", self.request)
        self.assertNotIn("inputs", self.request)
        self.assertIn("Positive rotation is left", self.request["rules"])
        self.assertEqual(self.request, request_fixture())

    def test_request_hash_changes_with_public_model_data(self):
        before = self.request["request_id"]
        other = copy.deepcopy(self.request)
        other["public_constants"]["b"] = 99
        # A response for one case cannot be applied to another authoritative request.
        other["request_id"] = "different"
        with self.assertRaisesRegex(ValueError, "another immutable request"):
            validate_candidate(self.candidate, other)
        self.assertEqual(self.request["request_id"], before)

    def test_valid_schema_does_not_imply_correctness(self):
        check = validate_candidate(self.candidate, self.request)
        self.assertFalse(check["encrypted_correctness_checked"])
        self.assertFalse(check["compilation_checked"])

    def test_no_parameter_weight_layout_reference_override(self):
        for field in ("atol", "rtol", "keys", "public_constants", "layout", "reference", "command", "model"):
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_candidate(dict(self.candidate, **{field: "changed"}), self.request)
        with self.assertRaises(ValueError):
            validate_candidate(dict(self.candidate, schema=True), self.request)

    def test_malformed_duplicate_nonfinite_oversized_response(self):
        for raw in ('{', '{"x": 1, "x": 2}', '{"x": NaN}', '{"x": Infinity}', '{"x": 1e999}', 'x' * 131073):
            with self.subTest(raw=raw[:30]), self.assertRaises(ValueError):
                strict_json(raw)

    def test_python_effects_and_unprovisioned_ops_rejected(self):
        for source in ('import os\n', '@hc.func("c")\ndef golden(x):\n    return open("/keys")\n',
                       '@hc.func("c")\ndef golden(x):\n    return x.__class__\n',
                       '@hc.func("c")\ndef golden(x):\n    return x.rotate(3)\n',
                       '@hc.func("c")\ndef golden(x):\n    return x.bootstrap()\n'):
            with self.subTest(source=source), self.assertRaises(ValueError):
                validate_candidate(dict(self.candidate, hecate_source=source), self.request)

    def test_operation_limit(self):
        source = '@hc.func("c")\ndef golden(x):\n'
        for i in range(257):
            source += f'    v{i} = {"x" if i == 0 else "v" + str(i-1)} + b\n'
        source += '    return v256\n'
        with self.assertRaisesRegex(ValueError, "operation budget"):
            validate_candidate(dict(self.candidate, hecate_source=source), self.request)

    def test_ast_evaluation_dispatch_without_exec(self):
        from candidate_trace import evaluate_tree

        class Symbol:
            def __init__(self, value):
                self.value = value
            def __add__(self, other):
                return Symbol(("add", self.value, other.value if isinstance(other, Symbol) else other))
            def __mul__(self, other):
                return Symbol(("mul", self.value, other))
            def rotate(self, step):
                return Symbol(("rotate", self.value, step))

        source = '@hc.func("c")\ndef golden(x):\n    a = x * w\n    return a + a.rotate(1)\n'
        validate_candidate(dict(self.candidate, hecate_source=source), self.request)
        result = evaluate_tree(source, {"w": "public-weight"}, Symbol("cipher-input"))
        self.assertEqual(result.value, ("add", ("mul", "cipher-input", "public-weight"),
                                       ("rotate", ("mul", "cipher-input", "public-weight"), 1)))


class FeedbackLoopTests(unittest.TestCase):
    def run_loop(self, responses, feedback):
        saved = []
        report = run_feedback_loop(request_fixture(), ReplayProvider(responses),
                                   lambda raw, i: copy.deepcopy(feedback[i]),
                                   lambda i, raw, fb: saved.append((i, raw, fb)))
        return report, saved

    def test_budget_is_initial_plus_three_repairs(self):
        self.assertEqual(MAX_REPAIRS, 3)
        result, saved = self.run_loop(["a"] * 4, [dict(status="failed", category="candidate")] * 4)
        self.assertEqual(result["attempts"], 4)
        self.assertEqual(result["repairs_used"], 3)
        self.assertEqual(result["status"], "repair_budget_exhausted")
        self.assertFalse(result["final_passed"])
        self.assertEqual(len(saved), 4)

    def test_replay_recovery_not_agent_success_rate(self):
        result, _ = self.run_loop(["bad", "good"], [dict(status="failed", category="candidate"), dict(status="passed")])
        self.assertTrue(result["final_passed"])
        self.assertFalse(result["first_attempt_passed"])
        self.assertEqual(result["agent_calls"], 0)
        self.assertIsNone(result["agent_success_rate"])
        self.assertEqual(result["provider"], "scripted_replay")

    def test_infrastructure_and_integrity_do_not_trigger_repairs(self):
        for category in ("infrastructure", "integrity"):
            result, _ = self.run_loop(["bad", "unused"], [dict(status="failed", category=category)])
            self.assertEqual(result["attempts"], 1)
            self.assertEqual(result["status"], category + "_failed")

    def test_provider_exhaustion_is_not_success(self):
        result, _ = self.run_loop(["bad"], [dict(status="failed", category="candidate")])
        self.assertEqual(result["status"], "provider_exhausted")
        self.assertFalse(result["final_passed"])

    def test_provider_cannot_mutate_request_or_prior_feedback(self):
        original = request_fixture()
        frozen = copy.deepcopy(original)
        class Mutator(ReplayProvider):
            def generate(self, request, feedback):
                request["public_constants"]["b"] = 100
                if feedback:
                    feedback[0]["status"] = "passed"
                return super().generate(request, feedback)
        result = run_feedback_loop(original, Mutator(["a", "b"]),
                                   lambda raw, i: dict(status="failed", category="candidate"), lambda *x: None)
        self.assertEqual(original, frozen)
        self.assertFalse(result["first_attempt_passed"])


class SandboxCommandTests(unittest.TestCase):
    def sandbox_command(self, keys=None):
        from candidate_sandbox import command
        from hecate_python_env import WORK
        with patch.dict(os.environ, {"HECATE_PYTHON_LIBRARY_PATH": "/nix/store/example/lib"}), \
             patch.object(Path, "is_dir", return_value=True), patch.object(Path, "is_file", return_value=True):
            return command(WORK / "results/payload.json", WORK / "results/attempt-output",
                           ["/trusted/worker"], keys)

    def test_no_root_home_workspace_or_results_mount(self):
        cmd = self.sandbox_command()
        for flag in ("--unshare-all", "--clearenv", "--die-with-parent", "--new-session", "--cap-drop"):
            self.assertIn(flag, cmd)
        destinations = [cmd[i+2] for i, word in enumerate(cmd) if word in ("--bind", "--ro-bind")]
        for forbidden in ("/", "/home", "/mnt", str(ROOT), str(RESULTS), "/keys"):
            self.assertNotIn(forbidden, destinations)
        self.assertEqual([cmd[i+2] for i, word in enumerate(cmd) if word == "--bind"], ["/out"])

    def test_keys_mount_is_explicit_and_readonly(self):
        from hecate_python_env import WORK
        cmd = self.sandbox_command(WORK / "results/private-keys")
        index = cmd.index("/keys")
        self.assertEqual(cmd[index-2], "--ro-bind")

    def test_resolved_work_root_crosses_clearenv_without_extra_mounts(self):
        cmd = self.sandbox_command()
        index = cmd.index('POSEIDON_WORK_ROOT')
        self.assertEqual(cmd[index-1:index+2], ['--setenv','POSEIDON_WORK_ROOT',str(WORK)])
        self.assertIn('/app/workspace_paths.py', cmd)
        destinations = [cmd[i+2] for i, word in enumerate(cmd) if word in ('--bind','--ro-bind')]
        self.assertNotIn(str(WORK), destinations)

    def test_failure_does_not_fall_back_to_unsandboxed_run(self):
        import candidate_sandbox as sandbox
        from types import SimpleNamespace
        with patch.object(sandbox, "command", return_value=["/usr/bin/bwrap", "--", "/trusted/worker"]), \
             patch.object(Path, "open", mock_open()), \
             patch.object(sandbox.subprocess, "run", return_value=SimpleNamespace(returncode=1)) as spawn:
            code = sandbox.run(Path("payload"), Path("output"), ["worker"], Path("log"))
            self.assertEqual(code, 1)
            spawn.assert_called_once()
            args, kwargs = spawn.call_args
            self.assertEqual(args[0][0], "/usr/bin/timeout")
            self.assertIn("/usr/bin/bwrap", args[0])
            self.assertLessEqual(kwargs["timeout"], 70)
            self.assertNotIn("LD_LIBRARY_PATH", kwargs["env"])


RESULTS = os.environ.get("POSEIDON_CANDIDATE_RESULTS")


@unittest.skipUnless(RESULTS, "requires actual isolated candidate replay run")
class CandidateEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(RESULTS)
        cls.report = json.loads((cls.root / "report.json").read_text())

    def test_real_sandbox_probe_and_no_agent_claim(self):
        self.assertEqual(self.report["status"], "passed")
        self.assertTrue(all(self.report["sandbox_probe"].values()))
        self.assertEqual(self.report["agent_calls"], 0)
        self.assertFalse(self.report["llm_generation_validated"])
        self.assertFalse(self.report["poseidon_gpu_validated"])

    def test_numerical_fault_and_recovery_are_real_encrypted_runs(self):
        attempts = self.report["attempts"]
        self.assertEqual([x.get("failure_layer", "complete") for x in attempts],
                         ["response_parse", "numerical_comparison", "complete"])
        self.assertFalse(attempts[0]["executed"])
        for item in attempts[1:]:
            self.assertTrue(item["execution"]["encrypted_execution"])
            self.assertFalse(item["execution"]["bootstrap_executed"])
            self.assertFalse(item["trace"]["candidate_python_executed"])
            self.assertEqual(item["comparison"]["atol"], 1e-5)
            self.assertEqual(item["comparison"]["rtol"], 1e-4)
        self.assertFalse(attempts[1]["comparison"]["passed"])
        self.assertTrue(attempts[2]["comparison"]["passed"])
        self.assertEqual(self.report["parameters"]["security_check"], "tc128")
        self.assertEqual(self.report["parameters"]["modulus_bits"], [60] * 14)

    def test_private_reference_and_artifact_integrity(self):
        from hecate_python_env import digest
        for name, expected in self.report["frozen_hashes"].items():
            self.assertEqual(digest(self.root / name), expected)
        for item in self.report["attempts"]:
            for name, expected in item["artifact_hashes"].items():
                self.assertEqual(digest(self.root / f"attempt-{item['index']:02d}" / "output" / name), expected)
            fb = json.loads((self.root / f"attempt-{item['index']:02d}" / "feedback.json").read_text())
            self.assertNotIn("reference", fb)
            self.assertNotIn("actual", fb)
            self.assertNotIn("/home/", json.dumps(fb))


if __name__ == "__main__":
    unittest.main()
