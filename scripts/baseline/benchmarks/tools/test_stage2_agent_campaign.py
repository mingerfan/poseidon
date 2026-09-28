"""Pure planner invariants, not API/compilation/encrypted-execution tests."""
import copy,unittest
from stage2_agent_campaign import partition,identity,task_arguments,classify_error,canonical
class CampaignTests(unittest.TestCase):
 def test_2030_bounded_shards_no_loss_or_duplication(self):
  rows=[dict(id="case_"+str(n),model_sha256="m",request_id=str(n)) for n in range(2030)]
  shards=partition(rows)
  self.assertEqual([r for chunk in shards for r in chunk],rows)
  self.assertTrue(all(1<=len(s)<=48 for s in shards))
  self.assertEqual(len(shards),43)
 def test_payload_budget_independent_of_case_count(self):
  row=dict(id="a",request={"public":"x"*80});n=len(canonical(row))
  rows=[dict(row,id=x) for x in ("a","b","c")]
  self.assertEqual([len(s) for s in partition(rows,max_payload_bytes=2*n)],[2,1])
  with self.assertRaisesRegex(ValueError,"Single prepared task"):partition(rows,max_payload_bytes=n-1)
 def test_invalid_or_expanded_budget_rejected(self):
  for n in (0,49,True,1.0):
   with self.subTest(n=n),self.assertRaises(ValueError):partition([],max_cases=n)
  for n in (0,6*1024**2+1,True):
   with self.subTest(n=n),self.assertRaises(ValueError):partition([],max_payload_bytes=n)
 def test_duplicate_task_is_integrity_failure(self):
  with self.assertRaisesRegex(ValueError,"Duplicate"):partition([dict(id="x"),dict(id="x")])
 def test_resume_identity_binds_task_model_request_and_code(self):
  s=dict(id="a",model_sha256="m",request_id="r");src={"a.py":"old"};base=identity(s,src)
  self.assertEqual(base,identity(copy.deepcopy(s),dict(src)))
  for key in s:
   changed=dict(s);changed[key]+="-changed";self.assertNotEqual(base,identity(changed,src))
  self.assertNotEqual(base,identity(s,{"a.py":"new"}))
 def test_blocked_is_not_pass_or_dsl_impossibility(self):
  x=classify_error(ValueError("shape capacity"))
  self.assertEqual(x["status"],"request_preparation_blocked")
  self.assertFalse(x["agent_support_disproved"])
  self.assertNotIn("passed",x)
 def test_cli_uses_frozen_profile_and_paid_boundaries(self):
  s=dict(compiler_configuration="seal-cpu-eva-w45-v1",construction_profile="hecate-unified-public-v1",exercise="public")
  a=task_arguments(s)
  for key,value in (("--compiler-configuration","seal-cpu-eva-w45-v1"),("--unified-guidance","explicit-v2"),
                    ("--max-repairs","3"),("--provider-retries","3"),("--api-timeout","1200"),("--max-tokens","384000")):
   self.assertEqual(a[a.index(key)+1],value)
  self.assertEqual(a.count("--live"),1);self.assertIn("--stream",a)
 def test_helper_requirements_preserved(self):
  s=dict(compiler_configuration="seal-cpu-eva-w45-v1",helper_profile="trusted",required_helpers=["first","second"])
  a=task_arguments(s);self.assertEqual([a[i+1] for i,v in enumerate(a) if v=="--unified-helper-exercise"],["first","second"])
  self.assertNotIn("--unified-profile",a)
 def test_actual_frozen_definitions_preserve_all_tasks(self):
  from stage2_agent_campaign import definitions,GROUPS
  from collections import Counter
  rows,parents,binding=definitions()
  self.assertEqual(Counter(s["group"] for s in rows),Counter(GROUPS))
  self.assertEqual(len({s["id"] for s in rows}),2030)
  self.assertTrue(all("source" not in s and "hecate_source" not in s for s in rows))
  self.assertEqual(len([s for s in rows if s.get("chunk_period")==4]),13)
  self.assertTrue(parents);self.assertEqual(len(binding),64)
if __name__=="__main__":unittest.main()
