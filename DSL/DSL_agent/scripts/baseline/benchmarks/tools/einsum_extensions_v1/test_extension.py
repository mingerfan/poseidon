"""Independent Torch einsum and core-math checks for the unactivated extension."""
import copy,math,pathlib,sys,unittest
ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5]);sys.path.insert(0,str(ROOT/"scripts/baseline"))
import planner as draft
from benchmark_graph import digest,validate,samples
CASES=[
 ("diagonal","ii->i",[[3,3]]),("trace","ii->",[[3,3]]),("implicit_trace","ii",[[3,3]]),
 ("batched_diagonal","bii->bi",[[2,3,3]]),("batched_trace","...ii->...",[[2,2,3,3]]),
 ("hidden_trace_reduction","...ii->",[[2,3,3]]),("triple_diagonal","iii->i",[[2,2,2]]),
 ("two_repeated_axes","ijij->ij",[[2,3,2,3]]),
 ("diagonal_contraction","bii,ij->bj",[[2,3,3],[3,2]]),
 ("diagonal_broadcast","ii,i->i",[[3,3],[1]]),
 ("middle_ellipsis","i...i->...i",[[2,3,2]]),
 ("implicit_diagonal_product","ii,j",[[3,3],[2]]),
 ("scalar_identity","->",[[]]),("implicit_scalar_identity","",[[]]),
 ("two_scalars",",->",[[],[]]),("three_scalars",",,->",[[],[],[]]),
 ("scalar_matrix",",ij->ij",[[],[2,3]]),("scalar_contraction","i,->",[[3],[]]),
 ("scalar_ellipsis","...,ij->ij",[[],[2,3]]),
 ("scalar_batch_matrix",",...ij,...jk->...ik",[[],[2,2,3],[1,3,2]]),
 ("scalar_trace","ii,->",[[3,3],[]])]
def replay(p,values):
 import torch
 refs={"operand_"+str(i):v for i,v in enumerate(values)}
 for i,s in enumerate(p["steps"]):
  values=[refs[r]for r in s["inputs"]];a=s["attrs"]
  if s["op"]=="reshape":value=values[0].reshape(tuple(a["shape"]))
  elif s["op"]=="permute":value=values[0].permute(tuple(a["dims"]))
  elif s["op"]=="public_gather":value=torch.index_select(values[0],a["axis"],torch.tensor(a["indices"],dtype=torch.int64))
  elif s["op"]=="multiply":value=values[0]*values[1]
  elif s["op"]=="sum":value=values[0].sum(dim=tuple(a["axes"]))
  elif s["op"]=="matmul":value=torch.matmul(*values)
  else:raise AssertionError(s["op"])
  assert list(value.shape)==s["shape"],(s,value.shape)
  refs["step_"+str(i)]=value
 return refs[p["output"]]
def check_case(equation,shapes):
 import numpy as np,torch
 from benchmark_math import evaluate
 p=draft.plan(equation,shapes);draft.check(p)
 high,low,provenance=draft.graph(p)
 assert validate(low)["output_shapes"]["out"]==(p["result_shape"]or[1])
 for sample in samples(low,4):
  values=[torch.from_numpy(sample["input_"+str(i)].reshape(shape))for i,shape in enumerate(shapes)]
  expected=torch.einsum(equation,*values)
  actual=replay(p,values)
  torch.testing.assert_close(actual,expected,rtol=1e-10,atol=1e-10)
  core=evaluate(low,sample)["out"]
  np.testing.assert_allclose(core,expected.numpy().reshape(p["result_shape"]or[1]),atol=1e-10,rtol=1e-10)
class ExtensionTests(unittest.TestCase):
 def test_single_diagonal_indices(self):
  p=draft.plan("ii->i",[[3,3]]);self.assertEqual(p["diagonals"][0]["indices"],[0,4,8])
 def test_implicit_repeated_label_not_output(self):
  self.assertEqual(draft.plan("ii",[[3,3]])["result_shape"],[])
 def test_batched_diagonal_indices(self):
  self.assertEqual(draft.plan("bii->bi",[[2,2,2]])["diagonals"][0]["indices"],[0,3,4,7])
 def test_repeated_label_no_within_operand_broadcast(self):
  with self.assertRaisesRegex(ValueError,"must be equal"):draft.plan("ii",[[2,1]])
 def test_repeated_label_unequal_across_ellipsis(self):
  with self.assertRaisesRegex(ValueError,"must be equal"):draft.plan("i...i",[[2,1,3]])
 def test_unknown_output_label(self):
  with self.assertRaises(ValueError):draft.plan("ii->j",[[2,2]])
 def test_repeated_output_label(self):
  with self.assertRaises(ValueError):draft.plan("i->ii",[[2]])
 def test_scalar_output_label(self):
  with self.assertRaises(ValueError):draft.plan("->i",[[]])
 def test_dynamic_equation(self):
  with self.assertRaises(ValueError):draft.plan(lambda:"i",[[2]])
 def test_integer_label_form_not_implicit(self):
  with self.assertRaises(ValueError):draft.plan([0],[[2]])
 def test_shape_type(self):
  for s in[[[True]],[[0]],[[-1]],[[1.5]],[(2,2)]]:
   with self.subTest(shape=s):
    with self.assertRaises(ValueError):draft.plan("i",s)
 def test_rank32(self):
  p=draft.plan("..."+"ii->...",[[1]*30+[2,2]]);self.assertEqual(len(p["result_shape"]),30)
 def test_rank33(self):
  with self.assertRaises(ValueError):draft.plan("...",[[1]*33])
 def test_operand_inventory(self):
  with self.assertRaises(ValueError):draft.plan(",,,,",[[]]*5)
 def test_operand_count(self):
  with self.assertRaises(ValueError):draft.plan("ii,j",[[2,2]])
 def test_source_shape_budget(self):
  with self.assertRaises(ValueError):draft.plan("ii",[[257,257]])
 def test_workspace_budget(self):
  with self.assertRaises(ValueError):draft.plan("i,j,k->ijk",[[64],[64],[64]])
 def test_graph_input_budget(self):
  p=draft.plan("ii->i",[[65,65]])
  with self.assertRaisesRegex(ValueError,"Total input budget"):draft.graph(p)
 def test_lower_resource_profile(self):
  with self.assertRaises(ValueError):draft.plan("...ii",[[1]*4+[2,2]],resource_profile="legacy-v1")
 def test_modified_diagonal_rehashed_rejected(self):
  p=draft.plan("ii->i",[[2,2]]);p["diagonals"][0]["indices"][1]=2
  p["binding"]=digest({k:v for k,v in p.items()if k!="binding"})
  with self.assertRaisesRegex(ValueError,"recomputation"):draft.check(p)
 def test_modified_step_rehashed_rejected(self):
  p=draft.plan("ii->i",[[2,2]]);p["steps"][1]["attrs"]["indices"]=[0,2]
  p["binding"]=digest({k:v for k,v in p.items()if k!="binding"})
  with self.assertRaises(ValueError):draft.check(p)
 def test_modified_output_shape_rehashed_rejected(self):
  p=draft.plan("ii->i",[[2,2]]);p["result_shape"]=[1]
  p["binding"]=digest({k:v for k,v in p.items()if k!="binding"})
  with self.assertRaises(ValueError):draft.check(p)
 def test_dependency_hash_rejected(self):
  p=draft.plan("ii->i",[[2,2]]);p["implementation_sources"]["pytorch_einsum_normalization.py"]="0"*64
  p["binding"]=digest({k:v for k,v in p.items()if k!="binding"})
  with self.assertRaises(ValueError):draft.check(p)
 def test_extra_field_rejected(self):
  p=draft.plan("ii->i",[[2,2]]);p["extra"]=True;p["binding"]=digest({k:v for k,v in p.items()if k!="binding"})
  with self.assertRaises(ValueError):draft.check(p)
 def test_deterministic(self):
  self.assertEqual(draft.plan("ii->i",[[3,3]]),draft.plan("ii->i",[[3,3]]))
 def test_no_production_activation(self):
  p=draft.plan("ii->i",[[3,3]]);self.assertFalse(p["production_source_policy"])
 def test_scalar_carrier_binding(self):
  p=draft.plan(",i->",[[],[2]]);_,low,meta=draft.graph(p)
  self.assertEqual(low["inputs"][0]["shape"],[1]);self.assertEqual(meta["logical_input_shapes"][0],[])
  self.assertEqual(validate(low)["output_shapes"]["out"],[1]);self.assertEqual(meta["logical_output_shape"],[])
 def test_legacy_steps_unchanged(self):
  from pytorch_einsum_normalization import plan
  for eq,shapes in[("ij,jk->ik",[[2,3],[3,2]]),("...i,...i->",[[2,3],[1,3]]),("i,j->ji",[[2],[3]])]:
   old=plan(eq,shapes);new=draft.plan(eq,shapes);self.assertEqual(new["steps"],old["steps"])
for name,equation,shapes in CASES:
 def test(self,eq=equation,s=shapes):check_case(eq,s)
 setattr(ExtensionTests,"test_math_"+name,test)
if __name__=="__main__":unittest.main()
