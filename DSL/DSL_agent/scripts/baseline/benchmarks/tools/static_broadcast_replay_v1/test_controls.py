import sys,unittest,copy
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[2]))
from replay_scope import *
from saved_outputs import declared_outputs
class ReplayControls(unittest.TestCase):
 def test_complete_alias_partition(self):
  value=read(origin_scope.PROPOSAL);cases=aliases(value)
  self.assertEqual(len(cases),3)
  self.assertEqual({c["context"]for c in cases},{"broadcast_to__arithmetic","broadcast_to__transpose_alias","broadcast_to__scalar_linear"})
  self.assertEqual(BOUNDS["paid_generations"],0);self.assertEqual(BOUNDS["http_attempts"],0)
 def test_rebound_model_change(self):
  value=read(origin_scope.PROPOSAL);value["source_contexts"][1]["model_sha256"]="0"*64
  value["binding"]=digest({k:v for k,v in value.items()if k!="binding"})
  with self.assertRaises(ValueError):aliases(value)
 def test_original_proposal_preserved(self):
  original=checked_origin()
  self.assertNotIn("application_stage_identity",original)
  self.assertEqual(origin_scope.read(origin_scope.PROPOSAL),original)
 def test_declared_output_order(self):
  self.assertEqual(list(declared_outputs({"mean":[1],"product":[2]},["product","mean"])),["product","mean"])
  with self.assertRaises(ValueError):declared_outputs({"unknown":[1]},["product"])
if __name__=="__main__":unittest.main()
