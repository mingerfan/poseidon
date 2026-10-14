"""Approval, scope, compatibility and durable recovery claim tests; no API."""
import copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import stage2_recovery_batch_v2 as batch
from campaign_live_state import claim,sealed
from stage2_agent_campaign import identity
from workspace_paths import RESULTS

class RecoveryTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.proposal=batch.ROOT/"docs/baseline/stage2-recovery-proposal-r147.json"
  cls.authorization=RESULTS/"stage2-recovery-authorization-r147.json"
  cls.output=RESULTS/"stage2-recovery-r147-test-plan-only"
  cls.plan=batch.build(cls.proposal,cls.authorization,cls.output)
 def test_exact_scope_and_preserved_old_requests(self):
  cases=[s for ref in self.plan["shards"] for s in ref["plan"]["cases"]]
  self.assertEqual(len(cases),54);self.assertEqual(len({s["id"] for s in cases}),54)
  self.assertEqual(sum(s["original_status"]=="launched_unconfirmed" for s in cases),49)
  self.assertEqual(sum(s["request"]["generation_guidance"]["version"]=="explicit-v4" for s in cases),5)
  self.assertNotIn("construct_012_0",{s["id"] for s in cases})
  self.assertTrue(all(s["evaluation_identity"]!=s["original_evaluation_identity"] for s in cases))
 def test_budget_and_shard_limits(self):
  self.assertEqual(sum(len(r["plan"]["cases"]) for r in self.plan["shards"]),54)
  self.assertEqual(sum(r["plan"]["limits"]["maximum_generations"] for r in self.plan["shards"]),216)
  self.assertEqual(sum(r["plan"]["limits"]["maximum_http_attempts"] for r in self.plan["shards"]),864)
  self.assertTrue(all(1<=len(r["plan"]["cases"])<=48 for r in self.plan["shards"]))
 def test_rehashed_budget_mutation_rejected(self):
  for key in ["max_retained_mib","api_workers","max_wall_seconds"]:
   p=copy.deepcopy(self.plan);p[key]+=1;p.pop("binding");p=sealed(p)
   with self.assertRaisesRegex(ValueError,"Frozen recovery"):batch.verify(p,self.output)
 def test_rehashed_request_or_scope_mutation_rejected(self):
  for kind in ("request","cases","command"):
   p=copy.deepcopy(self.plan);s=p["shards"][0]["plan"]["cases"][0]
   if kind=="request":s["request"]["rules"]+=" Disable checks."
   elif kind=="cases":p["shards"][0]["plan"]["cases"].append(copy.deepcopy(s))
   else:s["candidate_arguments"]+=["--provider-retries","100"]
   p.pop("binding");p=sealed(p)
   with self.assertRaises(ValueError):batch.verify(p,self.output)
 def test_unapproved_authorization_binding_rejected(self):
  real=batch.document
  def changed(path,*args):
   value=real(path,*args)
   if Path(path)==self.authorization:value={**value,"binding":"0"*64}
   return value
  with patch.object(batch,"document",side_effect=changed):
   with self.assertRaisesRegex(ValueError,"Unapproved recovery"):batch.build(self.proposal,self.authorization,self.output)
 def test_old_and_new_claims_both_preserved_no_duplicate_dispatch(self):
  s=next(s for r in self.plan["shards"] for s in r["plan"]["cases"] if s["original_status"]=="launched_unconfirmed")
  old=json.loads((RESULTS/"stage2-runtime-80738-before-r146/manifest.json").read_text())["source_hashes"]
  original=dict(s,evaluation_identity=identity(s,old))
  with tempfile.TemporaryDirectory() as name:
   root=Path(name);registry=root/"claims"
   ok,_=claim(registry,original,old,root/"old-owner","old-binding");self.assertTrue(ok)
   old_bytes={p.name:p.read_bytes() for p in registry.iterdir()}
   ok,_=claim(registry,s,self.plan["source_hashes"],root/"new-owner","new-binding");self.assertTrue(ok)
   ok,_=claim(registry,s,self.plan["source_hashes"],root/"duplicate-owner","new-binding");self.assertFalse(ok)
   self.assertEqual(len(list(registry.iterdir())),2)
   for n,b in old_bytes.items():self.assertEqual((registry/n).read_bytes(),b)
 def test_native_concurrency_proof_required(self):
  real=batch.document
  def changed(path,*args):
   value=real(path,*args)
   if str(path).endswith("stage2-recovery-native-probe-r147/report.json"):
    value={**value,"two_native_slots_observed":False}
   return value
  with patch.object(batch,"document",side_effect=changed):
   with self.assertRaisesRegex(ValueError,"Concurrent native proof"):batch.build(self.proposal,self.authorization,self.output)
 def test_paid_command_keeps_trusted_credential_entry(self):
  shard=self.plan["shards"][0]["plan"];s=shard["cases"][0]
  cmd=batch.candidate_command(shard,s,self.output/s["id"])
  self.assertNotIn("--inside",cmd)
  self.assertIn("--live",cmd)
  self.assertEqual(cmd[2],str(batch.BASE/"run_candidate.py"))
 def test_pure_nix_paid_entry_loads_key_without_nested_nix(self):
  import os
  import run_candidate
  args=run_candidate.parse_args(["--case","scripts/baseline/cases/unified-two-input-two-output.json","--live"])
  with patch.object(run_candidate,"parse_args",return_value=args), \
       patch("agent_credentials.load_api_key",return_value="unit-test-only"), \
       patch.object(run_candidate,"inside",return_value=73) as inside, \
       patch.object(run_candidate,"enter_nix",side_effect=AssertionError("nested Nix")), \
       patch.dict(os.environ,{"IN_NIX_SHELL":"pure"}):
   self.assertEqual(run_candidate.main(),73)
   inside.assert_called_once_with(args)
 def test_unknown_source_drift_rejected(self):
  changed=dict(self.plan["source_hashes"]);changed["scripts/baseline/unified_graph_contract.py"]="0"*64
  with patch.object(batch,"runtime_sources",return_value=changed):
   with self.assertRaisesRegex(ValueError,"runtime drift"):batch.build(self.proposal,self.authorization,self.output)

if __name__=="__main__":unittest.main()
