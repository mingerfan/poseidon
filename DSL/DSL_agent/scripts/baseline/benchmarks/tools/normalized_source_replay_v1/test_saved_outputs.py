import json,unittest,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parents[2]))
from saved_outputs import declared_outputs
from pytorch_source_contract import restore_outputs
class OutputOrder(unittest.TestCase):
 def test_canonical_json_preserves_declared_tensor_order(self):
  v=json.loads(json.dumps(dict(zeta=[2],alpha=[1]),sort_keys=True))
  self.assertEqual(list(v),["alpha","zeta"])
  self.assertEqual(list(declared_outputs(v,["zeta","alpha"])),["zeta","alpha"])
 def test_named_list_none_tree_is_restored(self):
  tree=dict(kind="dict",items=[dict(key="value",value=dict(kind="tensor",name="zeta")),
   dict(key="extras",value=dict(kind="list",items=[dict(kind="tensor",name="alpha"),dict(kind="none")]))])
  v=json.loads('{"alpha":[1],"zeta":[2]}')
  self.assertEqual(restore_outputs(declared_outputs(v,["zeta","alpha"]),tree,allow_none=True),
   dict(value=[2],extras=[[1],None]))
 def test_wrong_key_is_rejected(self):
  with self.assertRaisesRegex(ValueError,"inventory"):declared_outputs(dict(wrong=[1]),["right"])
 def test_missing_key_is_rejected(self):
  with self.assertRaisesRegex(ValueError,"inventory"):declared_outputs(dict(zeta=[1]),["zeta","alpha"])
 def test_extra_key_is_rejected(self):
  with self.assertRaisesRegex(ValueError,"inventory"):declared_outputs(dict(zeta=[1],extra=[2]),["zeta"])
 def test_duplicate_declaration_is_rejected(self):
  with self.assertRaisesRegex(ValueError,"inventory"):declared_outputs(dict(zeta=[1]),["zeta","zeta"])
if __name__=="__main__":unittest.main()
