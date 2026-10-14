import copy,json,unittest
from pathlib import Path
from planning import aliases,digest,availability
ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5])
class ReplayPlanning(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.p=json.loads((ROOT/"docs/baseline/agent-normalized-source-live-proposal-r1409.json").read_text())
 def reject(self,change):
  p=copy.deepcopy(self.p);change(p);p["binding"]=digest({k:v for k,v in p.items()if k!="binding"})
  with self.assertRaises(ValueError):aliases(p)
 def test_twenty_five_exact_aliases(self):
  a=aliases(self.p);self.assertEqual(len(a),25);self.assertEqual(len({v["id"]for v in a}),25)
 def test_original_binding_required(self):
  p=copy.deepcopy(self.p);p["binding"]="0"*64
  with self.assertRaises(ValueError):aliases(p)
 def test_missing_context(self):self.reject(lambda p:p["source_contexts"].pop())
 def test_duplicate_context(self):self.reject(lambda p:p["source_contexts"].__setitem__(0,p["source_contexts"][1]))
 def test_wrong_group_request(self):self.reject(lambda p:p["request_groups"][0].update(request_id="0"*64))
 def test_wrong_math_signature(self):self.reject(lambda p:p["source_contexts"][0].update(model_sha256="0"*64))
 def test_overlapping_context(self):self.reject(lambda p:p["request_groups"][0]["contexts"].append(p["request_groups"][0]["contexts"][0]))
 def test_pending_is_not_passed(self):
  r=availability(self.p,dict(proposal_binding=self.p["binding"],rows=[]))
  self.assertEqual({v["status"]for v in r},{"waiting_for_representative"})
 def test_human_origin_cannot_be_agent(self):
  a=aliases(self.p)[0]
  r=dict(proposal_binding=self.p["binding"],rows=[dict(case=a["representative_case"],status="passed",
   request_id=a["request_id"],new_agent_generation=False,package_binding="0"*64)])
  with self.assertRaises(ValueError):availability(self.p,r)
if __name__=="__main__":unittest.main()
