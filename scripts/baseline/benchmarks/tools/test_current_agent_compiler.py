"""Reject fabricated security/metadata; historical Agent artifacts remain scoped."""
import copy,json,tempfile,unittest
from pathlib import Path
from audit_current_agent_compiler import build,verify_observations,verify_files,ROOT,RESULTS
class ArtifactTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.report=json.loads((RESULTS/"stage2-campaign-summary-r110/current-compiler-partial.json").read_text())
 def sample(self):
  r=self.report["records"][0]
  e=dict(encrypted_execution=True,bootstrap_executed=False,
         rotation_key_check=r["rotation_key_check"],ciphertext_metadata=r["output_metadata"])
  return copy.deepcopy((r["lineage"],e,r["actual_parameters"]))
 def test_four_retained_executions(self):
  for r in self.report["records"]:
   e=dict(encrypted_execution=True,bootstrap_executed=False,
          rotation_key_check=r["rotation_key_check"],ciphertext_metadata=r["output_metadata"])
   self.assertTrue(verify_observations(r["lineage"],e,r["actual_parameters"]))
 def test_no_plaintext_or_bootstrap_substitution(self):
  for key,value in (("encrypted_execution",False),("bootstrap_executed",True)):
   t,e,p=self.sample();e[key]=value
   with self.assertRaises(ValueError):verify_observations(t,e,p)
 def test_security_parameters_unchanged(self):
  for key,value in (("security_check","none"),("modulus_bits",[60]*13),("polynomial_degree",16384)):
   t,e,p=self.sample();p[key]=value
   with self.assertRaises(ValueError):verify_observations(t,e,p)
 def test_actual_rotation_keys_required(self):
  t,e,p=self.sample();e["rotation_key_check"]["actual_key_file_verified"]=False
  with self.assertRaises(ValueError):verify_observations(t,e,p)
 def test_runtime_output_metadata_required(self):
  for key,value in (("polynomials",3),("data_modulus_count",1),("log2_scale",1)):
   t,e,p=self.sample();e["ciphertext_metadata"][0]["outputs"][0][key]=value
   with self.assertRaises(ValueError):verify_observations(t,e,p)
 def test_evidence_path_and_hash_rejected(self):
  with tempfile.TemporaryDirectory() as tmp:
   f=Path(tmp)/"x";f.write_text("changed")
   for manifest in ({"x":"0"*64},{"../x":"0"*64},{str(f):"0"*64}):
    with self.assertRaises(ValueError):verify_files(Path(tmp),manifest)
 def test_reproducible_independent_audit(self):
  self.assertEqual(build(RESULTS/"stage2-composite-agent-r108-campaign-audit/report.json"),self.report)
  self.assertEqual(len(self.report["partitions"]),5)
  self.assertEqual(sum(p["three_contexts_verified"] for p in self.report["partitions"].values()),3)
  self.assertFalse(self.report["compiler_contexts_complete"])
  self.assertFalse(self.report["stage2_complete"])
 def test_historical_agent_results_cannot_be_promoted(self):
  with self.assertRaisesRegex(ValueError,"Historical audit"):
   build(RESULTS/"stage2-agent-pilot-r98-audit/report.json")
if __name__=="__main__":unittest.main()
