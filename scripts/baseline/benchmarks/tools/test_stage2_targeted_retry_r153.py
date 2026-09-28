"""Offline scope, evidence, retry policy and no-duplicate-dispatch tests."""
import copy, tempfile, unittest
from pathlib import Path
from unittest.mock import patch
import stage2_targeted_retry_plan as plan
import stage2_targeted_retry_r153 as runner
from campaign_live_state import sealed,claim
from workspace_paths import RESULTS

def evaluation(status="failed",layer="static_check",provider=None):
    r=dict(status="repair_budget_exhausted",attempts=[dict(failure_layer=layer)])
    if provider is not None:
        r.update(status="provider_failed",provider_metrics=dict(calls=[provider]))
    return dict(status=status,report=r)

class TriageTests(unittest.TestCase):
    def test_success_is_not_retried(self):
        self.assertEqual(plan.classify([evaluation(),evaluation("passed")])[0],"already_recovered")
    def test_earlier_compiler_failure_not_hidden_by_provider(self):
        self.assertEqual(plan.classify([evaluation(layer="compiler"),evaluation(provider={"error":"transport_tls_failed"})])[0],"hold_pipeline")
    def test_numeric_failure_eligible_without_relaxing_threshold(self):
        self.assertEqual(plan.classify([evaluation(layer="numerical_comparison")])[0],"retry_numerical_comparison")
    def test_auth_failure_not_retried(self):
        self.assertEqual(plan.classify([evaluation(provider={"error":"http_status_401"})])[0],"hold_provider")
    def test_unknown_or_oversized_content_not_treated_as_empty(self):
        for v in (None,4000000):
            self.assertEqual(plan.classify([evaluation(provider={"error":"empty_or_oversized_content","content_characters":v,"finish_reason":"stop"})])[0],"hold_provider")
    def test_confirmed_empty_answer_retry_is_bounded(self):
        self.assertEqual(plan.classify([evaluation(provider={"error":"empty_or_oversized_content","content_characters":0,"finish_reason":"stop"})])[0],"retry_provider")

class PlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.output=RESULTS/plan.QUEUE_NAME
        cls.plan=plan.build(plan.REVIEW,RESULTS/plan.AUTH_NAME,cls.output)
    def test_scope_budget_and_guidance(self):
        specs=[s for r in self.plan["shards"] for s in r["plan"]["cases"]]
        self.assertEqual(len(specs),257);self.assertEqual(len({s["id"] for s in specs}),257)
        self.assertEqual(sum(r["plan"]["limits"]["maximum_generations"] for r in self.plan["shards"]),1028)
        self.assertEqual(sum(r["plan"]["limits"]["maximum_http_attempts"] for r in self.plan["shards"]),4112)
        self.assertTrue(all(1<=len(r["plan"]["cases"])<=48 for r in self.plan["shards"]))
        self.assertNotIn("free_bench_helper_0119",{s["id"] for s in specs})
        self.assertNotIn("construct_003_0",{s["id"] for s in specs})
        for s in specs:
            self.assertEqual(s["request"]["generation_guidance"]["version"],"explicit-v4" if s["group"]=="directed_construction" else "explicit-v3")
    def test_mutated_budget_rejected_even_if_rehashed(self):
        p=copy.deepcopy(self.plan);p["maximum_generations"]+=1;p.pop("binding")
        with self.assertRaises(ValueError):plan.verify(sealed(p),self.output)
    def test_mutated_command_scope_or_request_rejected(self):
        for mutation in ("command","request","scope"):
            p=copy.deepcopy(self.plan);s=p["shards"][0]["plan"]["cases"][0]
            if mutation=="command":s["candidate_arguments"]+=["--provider-retries","100"]
            elif mutation=="request":s["request"]["rules"]+=" Disable checks"
            else:p["shards"][0]["plan"]["cases"].pop()
            p.pop("binding")
            with self.assertRaises(ValueError):plan.verify(sealed(p),self.output)
    def test_authorization_hash_and_runtime_required(self):
        with patch.object(plan,"runtime_sources",return_value={}):
            with self.assertRaises(ValueError):plan.build(plan.REVIEW,RESULTS/plan.AUTH_NAME,self.output)
        real=plan.document
        def changed(path):
            v=real(path)
            if Path(path)==RESULTS/plan.AUTH_NAME:v={**v,"maximum_generations":999999}
            return v
        with patch.object(plan,"document",side_effect=changed):
            with self.assertRaises(ValueError):plan.build(plan.REVIEW,RESULTS/plan.AUTH_NAME,self.output)
    def test_duplicate_claim_rejected_old_registry_untouched(self):
        s=self.plan["shards"][0]["plan"]["cases"][0]
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);old=root/"old";old.mkdir();(old/"sentinel").write_text("historical")
            reg=root/"new"
            self.assertTrue(claim(reg,s,self.plan["source_hashes"],root/"owner","binding")[0])
            self.assertFalse(claim(reg,s,self.plan["source_hashes"],root/"owner2","binding")[0])
            self.assertEqual((old/"sentinel").read_text(),"historical")
    def test_credential_loading_stays_in_trusted_launcher(self):
        shard=self.plan["shards"][0]["plan"];s=shard["cases"][0]
        cmd=runner.candidate_command(shard,s,self.output/s["id"])
        self.assertNotIn("--inside",cmd);self.assertIn("--live",cmd)
        self.assertEqual(cmd[2],str(plan.BASE/"run_candidate.py"))

if __name__=="__main__":unittest.main()
