"""Synthetic provenance tests plus rejection of a real historical manual FHE result."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from benchmark_graph import digest
from fixed_polynomial_cases import cases
from stage2_agent_pilot_plan import build_plan, PAID
from stage2_agent_provenance import verify_answers
from test_stage2_agent_pilot import metrics

class AgentProvenance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec=build_plan()["cases"][-1]
        cls.source=next(c["source"] for c in cases() if c["id"]==cls.spec["id"])

    def fixture(self, root, crlf=False):
        source=self.source.replace("\n","\r\n") if crlf else self.source
        response=dict(schema=1,request_id=self.spec["request_id"],hecate_source=source)
        raw=json.dumps(response,indent=2)
        if crlf:raw=raw.replace("\n","\r\n")
        a=root/"attempt-00";a.mkdir()
        (a/"response.txt").write_bytes(raw.encode())
        (a/"candidate.py").write_bytes(source.encode())
        feedback=dict(status="passed",layer="complete")
        (a/"feedback.json").write_text(json.dumps(feedback))
        ledger=metrics()
        ledger["calls"][0].update(finish_reason="stop",response_model_matches=True,
                                   model=PAID["model"],content_characters=len(raw))
        report=dict(provider="deepseek_api",agent_calls=1,provider_metrics=ledger,status="passed",
                    llm_generation_validated=True,artifact_replay_only=False,
                    attempts=[dict(index=0,status="passed")])
        report["loop"]=dict(status="passed",provider="deepseek_api",agent_calls=1,attempts=1,
                            provider_metrics=copy.deepcopy(ledger),feedback_history=[feedback],
                            first_attempt_passed=True,repairs_used=0)
        return report

    def test_retained_answer_roundtrip_including_crlf(self):
        for crlf in (False,True):
            with tempfile.TemporaryDirectory() as d:
                root=Path(d);report=self.fixture(root,crlf)
                audited=verify_answers(root,report,self.spec["request"],PAID)
                self.assertEqual(audited["generations"],1)
                self.assertEqual(audited["http_attempts"],1)
                self.assertTrue(audited["first_attempt_passed"])

    def test_manual_replacement_of_generated_source_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);report=self.fixture(root)
            (root/"attempt-00/candidate.py").write_text(self.source+"\n")
            with self.assertRaisesRegex(ValueError,"not the retained model answer"):
                verify_answers(root,report,self.spec["request"],PAID)

    def test_other_request_response_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);report=self.fixture(root)
            p=root/"attempt-00/response.txt";r=json.loads(p.read_text())
            r["request_id"]="0"*64;raw=json.dumps(r);p.write_text(raw)
            report["provider_metrics"]["calls"][0]["content_characters"]=len(raw)
            report["loop"]["provider_metrics"]=copy.deepcopy(report["provider_metrics"])
            with self.assertRaises(ValueError):
                verify_answers(root,report,self.spec["request"],PAID)

    def test_changed_feedback_and_transport_diagnostics_rejected(self):
        for kind in ("feedback","diagnostic","loop"):
            with tempfile.TemporaryDirectory() as d:
                root=Path(d);report=self.fixture(root)
                if kind=="feedback":
                    (root/"attempt-00/feedback.json").write_text('{"status":"failed"}')
                elif kind=="diagnostic":
                    report["provider_metrics"]["calls"][0]["response_model_matches"]=False
                    report["loop"]["provider_metrics"]=copy.deepcopy(report["provider_metrics"])
                else:report["loop"]["repairs_used"]=1
                with self.assertRaises(ValueError):
                    verify_answers(root,report,self.spec["request"],PAID)

    def test_historical_manual_fhe_cannot_be_relabelled_by_auditor(self):
        from workspace_paths import RESULTS
        from audit_stage2_live_candidate import verify_candidate
        report=json.loads((RESULTS/"stage2-polynomial-candidates-r89/report.json").read_text())
        folder=Path(report["rows"][0]["evidence"])
        with self.assertRaisesRegex(ValueError,"live provider identity"):
            verify_candidate(folder,PAID)

if __name__=="__main__":
    unittest.main()
