"""CLI feedback adapter tests, scripted only; no tracing or native execution."""
import json
import shlex
import unittest
from application_diagnostics import make
from candidate_contract import ReplayProvider, run_feedback_loop
from component_control import PreciseFeedbackControl

def response(index):
    return json.dumps({"schema": 1, "request_id": "unit-only",
                      "hecate_source": "def golden(x):\n    return x + "+str(index)+"\n"})

class CliFeedbackTests(unittest.TestCase):
    def run_loop(self, failures, responses=None):
        report = {"attempts": []}
        stopped = []
        control = PreciseFeedbackControl(report, stopped_response=lambda i,r: stopped.append((i,r)))
        def evaluator(raw,index):
            if index >= len(failures):
                report["attempts"].append(dict(index=index,status="passed"))
                return dict(status="passed",layer="complete")
            spec = failures[index]
            layer=spec.get("layer","compiler")
            report["attempts"].append(dict(index=index,status="failed",failure_layer=layer,
                diagnostic=spec.get("diagnostic","private compiler diagnostic")))
            return dict(status="failed",layer=layer,category=spec.get("category","pipeline_unclassified"),
                        diagnostic="PRIVATE /host/reference.npy")
        def evaluate(raw,index):
            return control.evaluate(evaluator,raw,index)
        provider=ReplayProvider(responses or [response(i) for i in range(4)])
        loop=run_feedback_loop({},provider,evaluate,lambda *args: None,max_repairs=3,control=control)
        return loop,report,stopped

    def test_coarse_distinct_compiler_errors_reach_third_candidate(self):
        loop,report,_=self.run_loop([{"diagnostic":"first"},{"diagnostic":"different"}])
        self.assertEqual(loop["status"],"passed")
        self.assertEqual(len(report["attempts"]),3)
        self.assertTrue(all(not a["structured_diagnostic"]["repeat_key_available"] for a in report["attempts"][:2]))
        self.assertNotIn("PRIVATE",str(loop["feedback_history"]))

    def test_coarse_errors_end_at_original_budget(self):
        loop,report,_=self.run_loop([{}]*4)
        self.assertEqual(loop["status"],"repair_budget_exhausted")
        self.assertEqual(len(report["attempts"]),4)

    def test_repeated_candidate_not_revalidated(self):
        loop,report,stopped=self.run_loop([{}]*4,[response(0)]*4)
        self.assertEqual(loop["status"],"validation_failed")
        self.assertEqual(len(report["attempts"]),1)
        self.assertEqual(stopped,[(1,response(0))])
        self.assertEqual(report["feedback_stop"]["failure"]["code"],"repeated_candidate")

    def test_environment_failure_never_repairs(self):
        loop,report,_=self.run_loop([{"category":"infrastructure"}])
        self.assertEqual(loop["status"],"service_error")
        self.assertEqual(len(report["attempts"]),1)

    def test_integrity_failure_never_repairs(self):
        loop,report,_=self.run_loop([{"layer":"integrity","category":"integrity"}])
        self.assertEqual(loop["status"],"service_error")
        self.assertEqual(len(report["attempts"]),1)

    def test_missing_entry_uses_exact_shared_decision(self):
        variants=[response(i).replace("golden(","other(") for i in range(4)]
        loop,report,_=self.run_loop([{"layer":"static_check"}]*4,variants)
        self.assertEqual(loop["status"],"validation_failed")
        self.assertEqual(len(report["attempts"]),2)
        self.assertEqual(report["feedback_stop"]["failure"]["stop_reason"],"repeated_failure")
        self.assertIn("golden",loop["feedback_history"][0]["diagnostic"])

    def test_parse_and_forward_explicit_policy_only(self):
        from run_candidate import parse_args,forward_options
        legacy=parse_args(["--case","unit-only.json","--prepare"])
        self.assertEqual(legacy.feedback_policy,"legacy-v1")
        self.assertNotIn("--feedback-policy",forward_options(legacy))
        precise=parse_args(["--case","unit-only.json","--prepare","--feedback-policy","precise-v2"])
        forwarded=shlex.split(forward_options(precise))
        self.assertEqual(forwarded[forwarded.index("--feedback-policy")+1],"precise-v2")

if __name__=="__main__":
    unittest.main()
