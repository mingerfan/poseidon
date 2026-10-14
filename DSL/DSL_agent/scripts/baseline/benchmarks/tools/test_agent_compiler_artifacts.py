"""Reject fabricated security/metadata; historical Agent artifacts remain scoped."""
import copy,json,tempfile,unittest
from pathlib import Path
from audit_agent_compiler_artifacts import build,verify_observations,verify_files,ROOT
class ArtifactTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.report=json.loads((ROOT/"docs/baseline/stage2-agent-compiler-artifacts-r102.json").read_text())
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
  self.assertEqual(build(),self.report)
  self.assertEqual(len(self.report["partitions"]),5)
  self.assertTrue(all(p["distinct_topologies"]>=3 for p in self.report["partitions"].values()))
  self.assertFalse(self.report["current_guidance_agent_acceptance"])
  self.assertFalse(self.report["stage2_complete"])
if __name__=="__main__":unittest.main()
