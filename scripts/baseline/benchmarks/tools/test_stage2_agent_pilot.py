"""Offline safety tests for proposal identity and paid pilot accounting; no provider calls."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from stage2_agent_pilot_plan import build_plan, check_binding, small, PAID
from stage2_agent_provenance import provider_accounting
from run_stage2_agent_pilot import require_approval, outcome
from benchmark_graph import digest

def metrics(generations=1, retries=0):
    calls=[]
    for g in range(generations):
        for r in range(retries+1):
            calls.append(dict(index=len(calls),generation_index=g,retry_index=r,
                              status="failed" if r<retries else "response_received",
                              retry_scheduled=r<retries))
    return dict(provider="deepseek_api",service_provider="deepseek",model=PAID["model"],
                wire_model=PAID["model"],reasoning_effort="high",max_tokens=384000,max_calls=4,
                timeout_seconds=1200,provider_retries=3,loopback_proxy_port=0,
                calls=calls,generation_attempts=generations,request_attempts=len(calls),
                agent_calls=len(calls),transport_retries=len(calls)-generations)

class PilotSafety(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = build_plan()

    def test_exact_approval_required(self):
        for binding in (None, "", "0"*64):
            with self.assertRaises(ValueError):
                require_approval(self.plan, binding)
        self.assertIs(require_approval(self.plan, self.plan["binding"]), self.plan)

    def test_model_request_and_budget_tampering_rejected(self):
        for kind in ("model", "request", "budget"):
            changed = copy.deepcopy(self.plan)
            if kind == "model":
                changed["cases"][0]["model"]["id"] = "changed"
            elif kind == "request":
                changed["cases"][0]["request"]["rules"] = []
            else:
                changed["limits"]["maximum_generations"] = 49
            with self.assertRaises(ValueError):
                check_binding(changed)
        changed["binding"] = digest({k:v for k,v in changed.items() if k != "binding"})
        with self.assertRaises(ValueError):
            require_approval(changed, self.plan["binding"])

    def test_shard_and_concurrency_rejected(self):
        for cases, concurrency in (([], 1), (self.plan["cases"]*5, 1), (self.plan["cases"], 2)):
            changed = dict(self.plan, cases=cases, limits=dict(self.plan["limits"], concurrency=concurrency))
            changed["binding"] = digest({k:v for k,v in changed.items() if k != "binding"})
            with self.assertRaises(ValueError):
                require_approval(changed, changed["binding"])

    def test_free_selection_not_holdout_and_no_directed_hint(self):
        free = [c for c in self.plan["cases"] if c["track"] == "free"]
        self.assertEqual(len(free), 6)
        for case in free:
            self.assertTrue(small(case["model"]))
            self.assertEqual(case["origin"]["split"], "development")
            self.assertNotIn("construction_exercise", case["request"])
            self.assertNotIn("upstream_exercise", case["request"])

    def test_no_answer_or_test_arrays_exported(self):
        forbidden = {"hecate_source", "source", "golden", "reference", "test_inputs", "arrays", "api_key"}
        def inspect(value):
            if isinstance(value, dict):
                self.assertFalse(set(value) & forbidden)
                for child in value.values(): inspect(child)
            elif isinstance(value, list):
                for child in value: inspect(child)
        # Layout has a trusted auxiliary ciphertext source declaration; it is not
        # a DSL source answer, so inspect exported model and candidate metadata
        # separately and assert the exact allowed request fields.
        allowed = {"schema","task","model","fx_graph","public_constants","constant_origins",
                   "layout","rules","response_schema","semantic_guidance","compiler_profile_sha256",
                   "privacy","compiler_configuration","request_id","construction_profile",
                   "construction_exercise","upstream_helpers","upstream_exercise"}
        for case in self.plan["cases"]:
            inspect(case["model"])
            self.assertLessEqual(set(case["request"]), allowed)
            self.assertEqual(case["request_id"], case["request"]["request_id"])
            self.assertNotIn("source", case)
            self.assertNotIn("hecate_source", case)
            for flag in ("--golden-file", "--self-test", "--replay", "--inside"):
                self.assertNotIn(flag, case["candidate_arguments"])
            self.assertIn("--live", case["candidate_arguments"])

    def test_preliminary_result_cannot_count_as_audit_pass(self):
        report = dict(provider="deepseek_api", agent_calls=1, provider_metrics=metrics(), llm_generation_validated=True,
                      status="passed", attempts=[dict(status="passed", compiled=True, executed=True,
                        numerically_correct=True, trace=dict(frontend="real_Hecate"),
                        execution=dict(encrypted_execution=True), comparison=dict(atol=1e-5,rtol=1e-4))])
        self.assertEqual(outcome(report, 0), "runner_reported_pass_pending_audit")
        for field, value in (("provider","scripted_replay"),("agent_calls",0),("llm_generation_validated",False)):
            with self.assertRaises(ValueError): outcome(dict(report, **{field:value}), 0)
        report["attempts"][0]["comparison"]["atol"] = 1e-3
        with self.assertRaises(ValueError): outcome(report, 0)

    def test_failure_does_not_count_as_pass(self):
        self.assertEqual(outcome(dict(provider="deepseek_api", agent_calls=1, provider_metrics=metrics(),
                                     status="failed", attempts=[]), 1), "failed")

    def test_transport_retries_are_not_extra_generations(self):
        ledger=metrics(4,3)
        account=provider_accounting(dict(provider="deepseek_api",agent_calls=16,provider_metrics=ledger),PAID)
        self.assertEqual(account["generations"],4)
        self.assertEqual(account["http_attempts"],16)
        for field,value in (("generation_attempts",5),("request_attempts",17),("transport_retries",0)):
            bad=copy.deepcopy(ledger);bad[field]=value
            with self.assertRaises(ValueError):
                provider_accounting(dict(provider="deepseek_api",agent_calls=16,provider_metrics=bad),PAID)

    def test_retry_without_failure_or_wrong_configuration_rejected(self):
        ledger=metrics(2,1)
        for kind in ("retry","model","order"):
            bad=copy.deepcopy(ledger)
            if kind=="retry":bad["calls"][0]["retry_scheduled"]=False
            elif kind=="model":bad["model"]="another-model"
            else:bad["calls"][2]["generation_index"]=0
            with self.assertRaises(ValueError):
                provider_accounting(dict(provider="deepseek_api",agent_calls=4,provider_metrics=bad),PAID)

    def test_cli_denies_before_environment_or_credentials(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            plan_path = root/"proposal.json"
            plan_path.write_text(json.dumps(self.plan))
            run = subprocess.run([sys.executable,"-B",str(Path(__file__).with_name("run_stage2_agent_pilot.py")),
                "--plan",str(plan_path),"--approve-live-binding","wrong","--output",str(root/"execution")],
                capture_output=True,text=True,timeout=10)
            self.assertNotEqual(run.returncode,0)
            self.assertIn("Separate paid approval",run.stderr)
            self.assertNotIn("Dacapo root",run.stdout)
            self.assertFalse((root/"execution").exists())

if __name__ == "__main__":
    unittest.main()
