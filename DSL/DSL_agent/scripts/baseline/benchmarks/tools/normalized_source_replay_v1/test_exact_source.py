import copy,json,sys,unittest
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE),str(HERE.parents[2])]
from exact_source import ExactAgentReplay
from benchmark_graph import digest
FIXTURE=json.loads((HERE/"retained-source-fixture.json").read_text())
class ReplayControls(unittest.TestCase):
    def make(self,**changes):
        f=copy.deepcopy(FIXTURE);f.update(changes)
        return ExactAgentReplay(f["request"],f["raw"],f["receipt"],
               generation_identity_sha256=f["generation_identity_sha256"])
    def test_exact_source_without_transport_or_new_generation(self):
        replay=self.make()
        self.assertEqual(replay.kind,"scripted_replay")
        self.assertEqual(replay.agent_calls,0)
        self.assertEqual(replay.generate(FIXTURE["request"]),FIXTURE["raw"])
        self.assertTrue(replay.origin["agent_generated"])
        self.assertEqual(replay.calls,1)
    def test_verified_receipt_binding_is_preserved(self):
        replay=self.make()
        self.assertEqual(replay.origin["receipt_binding"],FIXTURE["receipt"]["binding"])
    def test_repeat_injection_refused(self):
        replay=self.make();replay.generate(FIXTURE["request"])
        with self.assertRaisesRegex(ValueError,"replay_single_immutable_candidate"):
            replay.generate(FIXTURE["request"])
    def test_changed_request_refused(self):
        replay=self.make();q=copy.deepcopy(FIXTURE["request"]);q["request_id"]="f"*64
        with self.assertRaisesRegex(ValueError,"replay_exact_mathematical_request"):replay.generate(q)
    def test_modified_candidate_refused(self):
        answer=json.loads(FIXTURE["raw"]);answer["hecate_source"]+="\n# altered"
        with self.assertRaisesRegex(ValueError,"Generation source identity"):
            self.make(raw=json.dumps(answer))
    def test_missing_response_field_refused(self):
        answer=json.loads(FIXTURE["raw"]);answer.pop("hecate_source")
        with self.assertRaisesRegex(ValueError,"replay_response_fields"):self.make(raw=json.dumps(answer))
    def test_wrong_generation_identity_refused(self):
        with self.assertRaisesRegex(ValueError,"Generation/package stage mismatch"):
            self.make(generation_identity_sha256="f"*64)
    def test_tampered_receipt_refused(self):
        value=copy.deepcopy(FIXTURE["receipt"]);value["candidate_sha256"]="f"*64
        with self.assertRaisesRegex(ValueError,"Generation receipt binding"):self.make(receipt=value)
if __name__=="__main__":unittest.main(verbosity=2)
