"""Precise feedback policy checks with fake adapters; no API/compiler/FHE."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from application_diagnostics import (FORMAT, LEGACY, PRECISE, attach, checked,
                                     from_result, make, render, repeat_key)
from application_component import Application, Controller
from application_policy import DEFAULT_BUDGET
from benchmark_graph import digest
from benchmark_suite import Builder
from test_application_component import Backend, Provider

class DiagnosticsTests(unittest.TestCase):
    def test_coarse_compile_and_numeric_have_no_repeat_key(self):
        for layer in ("compiler", "artifact_gate", "numerical_comparison", "dsl_trace", "poseidon_runtime"):
            result = {"failure": {"layer": layer, "diagnostic": "SECRET /private/reference.npy"},
                      "comparison": {"mae": 0.1}, "location": "/home/private"}
            d = from_result(result)
            self.assertIsNone(repeat_key(d))
            self.assertNotIn("SECRET", str(d))
            self.assertNotIn("/home", str(d))

    def test_exact_compiler_constraint_and_location_required(self):
        context = {"compiler_accumulated_scale": 80, "compiler_budget": 70}
        incomplete = make("compiler_scale_capacity", context=context)
        self.assertIsNone(repeat_key(incomplete))
        d = make("compiler_scale_capacity", location={"kind": "instruction", "index": 4}, context=context)
        self.assertIsNotNone(repeat_key(d))
        self.assertNotEqual(repeat_key(d), repeat_key(make("compiler_scale_capacity",
            location={"kind": "instruction", "index": 5}, context=context)))

    def test_non_violation_cannot_be_repeat_witness(self):
        d = make("compiler_scale_capacity", location={"kind": "instruction", "index": 4},
                 context={"compiler_accumulated_scale": 60, "compiler_budget": 70})
        self.assertIsNone(repeat_key(d))

    def test_invalid_location_and_constraints_are_rejected(self):
        for location in ({"kind": "source", "path": "/private/ref"},
                         {"kind": "instruction", "index": True},
                         {"kind": "entry", "name": "SECRET"}):
            with self.assertRaises(ValueError):
                make("contract_rejected", location=location)
        for context in ({"reference": [1, 2]}, {"actual": float("nan")}, {"actual": True},
                        {"actual": "SECRET"}, {"actual": -1}):
            with self.assertRaises(ValueError):
                make("contract_rejected", context=context)

    def test_repeat_flag_cannot_be_forged(self):
        d = make("compiler_rejected")
        d["repeat_key_available"] = True
        with self.assertRaises(ValueError):
            checked(d)
        result = attach({"status": "failed", "structured_diagnostic": d})
        self.assertEqual(result["failure"]["reason_code"], "environment_failure")
        self.assertEqual(result["public_feedback"]["category"], "infrastructure")

    def test_static_missing_entry_is_specific_and_public(self):
        result = attach({"status": "failed", "failure": {"layer": "static_check",
            "diagnostic": "SECRET /private/ref.npy"}},
            candidate={"hecate_source": "def other(x):\n    return x\n"})
        d = result["failure"]
        self.assertEqual(d["reason_code"], "entry_missing")
        self.assertIsNotNone(repeat_key(d))
        feedback = result["public_feedback"]
        self.assertEqual(set(feedback), {"status", "layer", "category", "diagnostic"})
        self.assertIn("golden", feedback["diagnostic"])
        self.assertNotIn("SECRET", str(feedback))

    def test_syntax_location_but_no_false_exact_repeat(self):
        d = from_result({"failure": {"layer": "source_parse"}},
                        candidate={"hecate_source": "def golden(\n"})
        self.assertEqual(d["location"]["kind"], "source")
        self.assertIsNone(repeat_key(d))

    def test_environment_and_integrity_take_priority(self):
        for layer in ("integrity", "pipeline", "component"):
            d = from_result({"failure": {"layer": layer}},
                            candidate={"hecate_source": "def other(x):\n return x"})
            self.assertEqual(d["owner"], "host")
            self.assertIsNone(repeat_key(d))

    def test_infrastructure_category_overrides_compiler_stage(self):
        result = attach({"status": "failed", "failure": {"layer": "compiler"},
            "public_feedback": {"status": "failed", "layer": "compiler",
                "category": "infrastructure", "diagnostic": "SECRET /host/compiler"}})
        self.assertEqual(result["failure"]["owner"], "host")
        self.assertEqual(result["public_feedback"]["category"], "infrastructure")
        self.assertNotIn("SECRET", str(result["public_feedback"]))

    def test_arbitrary_prior_diagnostic_is_not_forwarded(self):
        prior = {"status": "failed", "layer": "compiler", "category": "candidate",
                 "diagnostic": "SECRET /private/path reference=[1,2,3]"}
        result = render(make("compiler_rejected"), prior)
        for forbidden in ("SECRET", "/private/path", "reference="):
            self.assertNotIn(forbidden, json.dumps(result))
        self.assertEqual(result["diagnostic"],
                         "Compilation rejected the candidate; inspect the approved compiler feedback.")

    def test_controlled_tool_and_aggregate_fields_remain_available(self):
        prior = {"status": "failed", "layer": "compiler", "category": "candidate",
                 "diagnostic": "do not forward",
                 "tool_diagnostic": {"trust": "untrusted_tool_output", "text": "approved adapter tool excerpt"},
                 "numerical_summary": {"compared_values": 8, "max_absolute_error": 0.1}}
        result = render(make("compiler_rejected"), prior)
        self.assertEqual(result["tool_diagnostic"], prior["tool_diagnostic"])
        self.assertEqual(result["numerical_summary"], prior["numerical_summary"])
        self.assertNotIn("do not forward", result["diagnostic"])

    def test_typed_hint_cannot_override_terminal_trust_failure(self):
        typed = make("compiler_scale_capacity", location={"kind": "instruction", "index": 4},
                     context={"compiler_accumulated_scale": 80, "compiler_budget": 70})
        for category, expected in (("integrity", "integrity_rejected"),
                                   ("infrastructure", "environment_failure")):
            d = from_result({"structured_diagnostic": typed,
                "public_feedback": {"status": "failed", "layer": "compiler", "category": category}})
            self.assertEqual(d["reason_code"], expected)
        d = from_result({"failure": {"layer": "integrity"}, "structured_diagnostic": typed})
        self.assertEqual(d["reason_code"], "integrity_rejected")

    def test_legacy_provider_wire_unchanged(self):
        from deepseek_provider import public_feedback
        d = make("artifact_level_exhausted", location={"kind": "instruction", "index": 2},
                 context={"input_level": 0, "required_levels": 1})
        public = render(d)
        self.assertEqual(public_feedback(public), public)
        self.assertNotIn("format", public)
        self.assertNotIn("reason_code", public)

class PreciseApplicationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.backend = Backend()
        self.app = Application(self.tmp.name, self.backend)
        b = Builder([(2,)])
        self.model = b.finish(b.node("square", ["input0"]))

    def tearDown(self):
        self.tmp.cleanup()

    def test_new_task_binds_policy_without_changing_request(self):
        a = self.app.submit(self.model)
        b = self.app.submit(self.model, feedback_policy=LEGACY)
        ma = self.app._load(self.app._dir(a)/"task.json")
        mb = self.app._load(self.app._dir(b)/"task.json")
        self.assertEqual(ma["feedback_policy"], PRECISE)
        self.assertEqual(ma["request"], mb["request"])
        self.assertNotEqual(ma["binding"], mb["binding"])

    def test_distinct_coarse_compiler_failures_can_reach_success(self):
        self.backend.failures = ["compiler", "compiler"]
        p = Provider(["x*x", "x*x+zero_ct", "x*x-zero_ct"])
        s = self.app.run(self.app.submit(self.model), p)
        self.assertEqual(s["status"], "succeeded")
        self.assertEqual(self.backend.calls, 3)

    def test_numeric_aggregates_do_not_trigger_repeat_stop(self):
        self.backend.failures = ["numerical_comparison", "numerical_comparison"]
        s = self.app.run(self.app.submit(self.model), Provider(["x*x", "x*x+zero_ct", "x*x-zero_ct"]))
        self.assertEqual(s["status"], "succeeded")
        self.assertEqual(self.backend.calls, 3)

    def test_exact_repeated_failure_stops_distinct_candidates(self):
        original = self.backend.qualify
        def qualify(*args):
            original(*args)
            return {"status": "failed", "structured_diagnostic":
                make("artifact_level_exhausted", location={"kind": "instruction", "index": 8},
                     context={"input_level": 0, "required_levels": 1})}
        self.backend.qualify = qualify
        s = self.app.run(self.app.submit(self.model), Provider(["x*x", "x*x+zero_ct", "x*x-zero_ct"]))
        self.assertEqual(s["status"], "validation_failed")
        self.assertEqual(s["failure"]["stop_reason"], "repeated_failure")
        self.assertEqual(self.backend.calls, 2)

    def test_coarse_failure_breaks_precise_streak(self):
        task = self.app.submit(self.model)
        path = self.app._dir(task)
        state = self.app._load(path/"state.json")
        c = Controller(self.app, path, state, DEFAULT_BUDGET, policy=PRECISE)
        exact = make("artifact_level_exhausted", location={"kind": "instruction", "index": 1},
                     context={"input_level": 0, "required_levels": 1})
        for i, d in enumerate([exact, make("compiler_rejected"), exact]):
            c.observe_failure({"structured_diagnostic": d})
            self.assertIsNone(c.after_evaluation(render(d), i))
        self.assertEqual(c.repair_decision.repeated, 1)

    def test_different_instruction_is_not_same_failure(self):
        task = self.app.submit(self.model)
        path = self.app._dir(task)
        state = self.app._load(path/"state.json")
        c = Controller(self.app, path, state, DEFAULT_BUDGET, policy=PRECISE)
        for i in (1, 2):
            d = make("artifact_level_exhausted", location={"kind": "instruction", "index": i},
                     context={"input_level": 0, "required_levels": 1})
            c.observe_failure({"structured_diagnostic": d})
            self.assertIsNone(c.after_evaluation(render(d), i))

    def test_task_without_version_retains_legacy_repeat_decision(self):
        task = self.app.submit(self.model, feedback_policy=LEGACY)
        path = self.app._dir(task)/"task.json"
        value = self.app._load(path)
        del value["feedback_policy"]
        value["binding"] = digest({k: v for k, v in value.items() if k != "binding"})
        self.app._write(path, value)
        self.backend.failures = ["compiler"]*4
        s = self.app.run(task, Provider(["x*x", "x*x+zero_ct", "x*x-zero_ct"]))
        self.assertEqual(s["status"], "validation_failed")
        self.assertEqual(self.backend.calls, 2)

    def test_unsupported_policy_rejected_without_generation(self):
        task = self.app.submit(self.model, feedback_policy="unknown")
        p = Provider()
        s = self.app.run(task, p)
        self.assertEqual(s["status"], "unsupported")
        self.assertEqual(p.calls, 0)

    def test_static_feedback_same_renderer_as_backend(self):
        class WrongThenCorrect(Provider):
            def __init__(self):
                super().__init__()
                self.history = []
            def generate(self, q, h):
                self.history.append(copy.deepcopy(h))
                answer = super().generate(q, h)
                if self.calls == 1:
                    answer = answer.replace("golden(", "other(")
                return answer
        p = WrongThenCorrect()
        s = self.app.run(self.app.submit(self.model), p)
        self.assertEqual(s["status"], "succeeded")
        self.assertEqual(p.history[1][0]["layer"], "static_check")
        self.assertIn("Define exactly one top-level function named golden", p.history[1][0]["diagnostic"])

if __name__ == "__main__":
    unittest.main()
