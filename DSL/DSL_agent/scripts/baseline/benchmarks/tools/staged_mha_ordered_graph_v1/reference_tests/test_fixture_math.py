"""Loaded fixture state equals original mathematical controls; pinned Torch only."""
import importlib.util,json,sys,unittest
from pathlib import Path
from cases import CASES,APPROXIMATION,build,samples
from planner_composition import BASE
from reference_math import composition
from test_math import MathControls
ROOT=Path(__file__).resolve().parent/'source_fixtures'
if str(BASE) not in sys.path:sys.path.insert(0,str(BASE))
class FixtureMath(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  import numpy as np,torch
  if (list(sys.version_info[:3]),np.__version__,torch.__version__,torch.version.cuda)!=([3,10,14],'1.25.2','2.0.1',None):raise ValueError('existing pinned CPU Python')
  from pytorch_weight_package import read,load_state
  cls.prepared=[]
  for i,case in enumerate(CASES):
   p=ROOT/case['id'];manifest=json.loads((p/'manifest.json').read_text());spec=importlib.util.spec_from_file_location('_mha_composition_fixture_'+str(i),p/'model.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
   rng=torch.get_rng_state().clone()
   try:source=module.make();arrays,raw=read(p,manifest['weights'],resource_profile='general-tensors-v1');load_state(source,arrays,manifest['weights'])
   finally:torch.set_rng_state(rng)
   original,signature,outputs=build(case);cls.prepared.append((case,source,original,signature,outputs))
 def test_npy_loaded_state_matches_original_math_context_exactly(self):
  import torch
  for case,source,original,signature,outputs in self.prepared:
   self.assertEqual(set(source.state_dict()),set(original.state_dict()))
   for name,value in source.state_dict().items():self.assertTrue(torch.equal(value,original.state_dict()[name]),case['id']+':'+name)
 def test_fixture_source_outputs_match_original_source(self):
  import numpy as np
  for case,source,original,signature,outputs in self.prepared:
   for batch in samples(signature):
    got,tree=MathControls().native(source,batch,signature,outputs);expected,otree=MathControls().native(original,batch,signature,outputs);self.assertEqual(tree,otree)
    for name in outputs:np.testing.assert_array_equal(got[name],expected[name])
 def test_fixture_source_against_independent_exact_original_mha(self):
  import numpy as np
  for case,source,original,signature,outputs in self.prepared:
   for batch in samples(signature):
    got,tree=MathControls().native(source,batch,signature,outputs);expected=composition(case,original,batch,APPROXIMATION,approximate=False)
    for name in outputs:np.testing.assert_allclose(got[name],expected[name],atol=1e-12,rtol=1e-12)
if __name__=='__main__':unittest.main()
