"""Existing pinned CPU Torch/NumPy only; no Nix entry, API, compiler or FHE."""
import copy,sys,unittest
from pathlib import Path
from cases import CASES,APPROXIMATION,build,output_names,samples
from planner_composition import BASE,legacy,build_graph
from reference_math import composition,independent_mha
if str(BASE) not in sys.path:sys.path.insert(0,str(BASE))
class MathControls(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  import numpy as np,torch
  if (list(sys.version_info[:3]),np.__version__,torch.__version__,torch.version.cuda)!=( [3,10,14],"1.25.2","2.0.1",None):raise RuntimeError("existing pinned ARM CPU Python required")
  cls.prepared=[]
  for case in CASES:
   module,signature,outputs=build(case);graph,lowered,binding,mini=build_graph(case,module,signature,APPROXIMATION)
   cls.prepared.append((case,module,signature,outputs,graph,lowered,binding,mini))
 def native(self,module,batch,signature,outputs):
  import torch,numpy as np
  # Output tree checker is the frozen source draft; None consumes no tensor port.
  path=BASE/"benchmarks/tools/static_mha_source_v16_draft_v1"
  if str(path) not in sys.path:sys.path.insert(0,str(path))
  import output_contract
  with torch.no_grad():raw=module(*[torch.from_numpy(batch[x["name"]].copy()) for x in signature])
  leaves,tree=output_contract.flatten(raw,lambda x:isinstance(x,torch.Tensor),outputs,source_format=output_contract.SOURCE_FORMAT)
  return {name:leaf.detach().numpy().copy() for name,leaf in zip(outputs,leaves)},tree
 def test_exact_original_source_vs_independent_loops(self):
  import numpy as np
  for case,module,signature,outputs,*_ in self.prepared:
   with self.subTest(case=case["id"]):
    for batch in samples(signature):
     native,tree=self.native(module,batch,signature,outputs);ideal=composition(case,module,batch,APPROXIMATION,approximate=False)
     for name in outputs:np.testing.assert_allclose(native[name],ideal[name],atol=1e-12,rtol=1e-12)
 def test_declared_approximation_vs_fixed_decomposition_numpy_and_torch(self):
  import numpy as np
  from benchmark_math import evaluate
  from benchmark_torch import evaluate as torch_graph
  from operator_graph import evaluate_reference
  for case,module,signature,outputs,graph,lowered,*_ in self.prepared:
   with self.subTest(case=case["id"]):
    for batch in samples(signature):
     expected=composition(case,module,batch,APPROXIMATION,approximate=True)
     for actual in (evaluate(lowered,batch,resource_profile="general-tensors-v1"),torch_graph(lowered,batch,resource_profile="general-tensors-v1"),evaluate_reference(graph,batch)):
      for name in outputs:np.testing.assert_allclose(actual[name],expected[name],atol=1e-12,rtol=1e-12)
 def test_independent_old_mha_reference_remains_equivalent(self):
  import numpy as np
  from reference_math import linear
  ref=legacy("mha_source_reference.py")
  for case,module,signature,outputs,graph,lowered,binding,mini in self.prepared:
   request=mini["source_request"]
   for batch in samples(signature):
    raw=dict(batch)
    if module.pre is not None:raw["pre_projected"]=linear(module.pre,raw["q"])
    for approximate in (False,True):
     operands=request["operands"];args=[raw[operands[r]["name"]] for r in ("query","key","value")]
     independent=independent_mha(module.attention,*args,request["options"],APPROXIMATION,approximate=approximate)
     original=ref.evaluate(request,raw,approximate=approximate)
     for name in original:np.testing.assert_allclose(original[name],independent[name],atol=1e-12,rtol=1e-12)
 def test_named_outputs_and_none_tree(self):
  for case,module,signature,outputs,*_ in self.prepared:
   actual,tree=self.native(module,next(samples(signature)),signature,outputs)
   self.assertEqual(list(actual),outputs)
   if case["tree"]!="named":self.assertGreaterEqual(tree["static_none_count"],1)
 def test_stable_graph_and_public_state_immutable(self):
  import torch
  for case,module,signature,outputs,graph,lowered,*_ in self.prepared:
   before={k:v.detach().clone() for k,v in module.state_dict().items()}
   other=build_graph(case,module,signature,APPROXIMATION)
   self.assertEqual(graph,other[0]);self.assertEqual(lowered,other[1])
   for k,v in before.items():self.assertTrue(torch.equal(v,module.state_dict()[k]))
 def test_all_blocked_rows_rejected_independently_and_by_planner(self):
  import numpy as np
  case,module,signature,outputs,graph,lowered,binding,mini=self.prepared[2];request=copy.deepcopy(mini["source_request"]);request["options"]["attn_mask"]["values"]=[[True]*3]*2
  with self.assertRaisesRegex(ValueError,"masked"):legacy("mha_source_plan.py").plan(**request)
  batch=next(samples(signature))
  with self.assertRaisesRegex(ValueError,"all masked"):independent_mha(module.attention,batch["q"],batch["k"],batch["v"],request["options"],APPROXIMATION,approximate=True)
 def test_exp_domain_overflow_rejected_in_reference(self):
  import numpy as np
  case,module,signature,outputs,graph,lowered,binding,mini=self.prepared[3];request=copy.deepcopy(mini["source_request"]);request["options"]["attn_mask"]["values"][0][0][0]=2.
  batch=next(samples(signature))
  with self.assertRaisesRegex(ValueError,"exp approximation domain"):independent_mha(module.attention,batch["q"],batch["k"],batch["v"],request["options"],APPROXIMATION,approximate=True)
 def test_native_only_source_draft_rejects_composition(self):
  path=BASE/"benchmarks/tools/static_mha_source_v16_draft_v1"
  if str(path) not in sys.path:sys.path.insert(0,str(path))
  from source_plan import capture
  for case,module,signature,outputs,*_ in self.prepared:
   if module.pre is not None or module.post is not None:
    with self.assertRaises(ValueError):capture(module,signature,outputs,approximation=APPROXIMATION,trusted=True)
if __name__=="__main__":unittest.main()
