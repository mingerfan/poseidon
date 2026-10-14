"""Independent coordinate tests for an inactive logical normalization draft."""
import copy,itertools,json,math,random,sys,unittest
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parents[2]
sys.path[:0]=[str(BASE),str(HERE)]
from normalize import normalize,permutation,digest
from benchmark_graph import validate
def oracle(shape,axes):
 """Map labeled input coordinates to output coordinates, without stride inference."""
 labels={tuple(coord[i]for i in axes):j for j,coord in enumerate(itertools.product(*(range(d)for d in shape)))}
 return [labels[c]for c in sorted(labels)]
def fixture(shape,axes,*,indices=None,partial=False,changed_root=False):
 indices=oracle(shape,axes)if indices is None else list(indices)
 nodes=[];counter=0
 def emit(op,refs,attrs):
  nonlocal counter
  name="value_"+str(counter);identifier="node_"+str(counter);counter+=1
  nodes.append(dict(id=identifier,op=op,inputs=refs,attrs=attrs,outputs=[name]));return name
 root="x";constants={}
 if changed_root:
  constants["gain"]=2.0;root=emit("add",[root,"gain"],{})
 flat=emit("reshape",[root],dict(shape=[math.prod(shape)]))
 leaves=[emit("slice",[flat],dict(axis=0,start=i,stop=i+1,step=1))for i in indices]
 saved=leaves[0]
 while len(leaves)>1:
  leaves=[emit("concat",leaves[i:i+8],dict(axis=0))for i in range(0,len(leaves),8)]
 final=emit("reshape",[leaves[0]],dict(shape=[shape[i]for i in axes]))
 outputs=[dict(name="result",value=final)]
 if partial:outputs.append(dict(name="partial",value=saved))
 return dict(format="poseidon-model-graph-v2",resource_profile="general-tensors-v1",
  id="view_fixture",inputs=[dict(name="x",shape=list(shape))],constants=constants,nodes=nodes,outputs=outputs)
def evaluate(model,inputs):
 values={n:list(v)for n,v in inputs.items()};shapes=validate(model)["shapes"]
 for n,value in model["constants"].items():values[n]=[float(value)]
 for n in model["nodes"]:
  op=n["op"];refs=n["inputs"];a=n["attrs"];x=values[refs[0]]
  if op=="reshape":value=list(x)
  elif op=="slice":value=x[slice(a["start"],a["stop"],a["step"])]
  elif op=="concat":value=list(itertools.chain.from_iterable(values[r]for r in refs))
  elif op=="permute":value=[x[i]for i in oracle(shapes[refs[0]],a["dims"])]
  elif op=="add":value=[v+values[refs[1]][0]for v in x]
  else:raise AssertionError(op)
  values[n["outputs"][0]]=value
 return {o["name"]:values[o["value"]]for o in model["outputs"]}
class CoordinateTests(unittest.TestCase):
 def assert_equivalent(self,model):
  frozen=copy.deepcopy(model);before=digest(model);new,record=normalize(model)
  inputs={d["name"]:[(-.25 if i%2 else .375)*(i+1)for i in range(math.prod(d["shape"]))]for d in model["inputs"]}
  self.assertEqual(evaluate(model,inputs),evaluate(new,inputs))
  self.assertEqual(model,frozen);self.assertEqual(digest(model),before)
  self.assertEqual(validate(model)["output_shapes"],validate(new)["output_shapes"])
  self.assertEqual(record["original_sha256"],before);self.assertFalse(record["production_active"])
  return new,record
 def test_asymmetric_three_contexts(self):
  for shape,axes in (((2,3),(1,0)),((2,3,2),(2,0,1)),((3,2,4),(2,1,0))):
   with self.subTest(shape=shape):
    original=fixture(shape,axes);new,r=self.assert_equivalent(original)
    self.assertTrue(r["rules"]);self.assertLess(len(new["nodes"]),len(original["nodes"]))
 def test_rank_8_16_32(self):
  for rank in(8,16,32):
   shape=(1,)*(rank-6)+(2,3,2,3,2,3);axes=list(range(rank));axes[-1],axes[-2]=axes[-2],axes[-1]
   new,r=self.assert_equivalent(fixture(shape,axes));self.assertTrue(r["rules"])
   self.assertEqual(new["nodes"][-1]["op"],"permute")
 def test_seeded_permutations(self):
  rng=random.Random(20261013)
  for i in range(36):
   active=rng.randrange(2,6);shape=tuple(rng.choice((2,3))for _ in range(active));axes=list(range(active));rng.shuffle(axes)
   with self.subTest(case=i):self.assert_equivalent(fixture(shape,axes))
 def test_named_partial_preserved(self):
  new,r=self.assert_equivalent(fixture((2,3,4),(2,0,1),partial=True))
  self.assertTrue(r["rules"]);self.assertEqual(len(new["outputs"]),2)
 def test_updated_ssa_root_preserved(self):
  new,r=self.assert_equivalent(fixture((2,3),(1,0),changed_root=True))
  self.assertTrue(r["rules"]);self.assertEqual(new["nodes"][0]["op"],"add")
  self.assertEqual(new["nodes"][-1]["inputs"],new["nodes"][0]["outputs"])
 def test_same_size_not_enough(self):
  model=fixture((2,3),(1,0),indices=(0,1,2,3,5,4));new,r=self.assert_equivalent(model)
  self.assertFalse(r["rules"]);self.assertEqual(new,model)
 def test_duplicate_indices_rejected_as_permutation(self):
  model=fixture((2,3),(1,0),indices=(0,2,4,1,3,3));new,r=self.assert_equivalent(model)
  self.assertFalse(r["rules"]);self.assertEqual(new,model)
 def test_mixed_roots_not_collapsed(self):
  model=fixture((2,3),(1,0));model["inputs"].append(dict(name="y",shape=[6]))
  first_slice=next(n for n in model["nodes"]if n["op"]=="slice");first_slice["inputs"]=["y"]
  new,r=self.assert_equivalent(model);self.assertFalse(r["rules"]);self.assertEqual(new,model)
 def test_budget_preserves_original(self):
  model=fixture((2,3),(1,0));new,r=normalize(model,max_index_cells=1)
  self.assertEqual(new,model);self.assertTrue(r["work_budget_exhausted"])
 def test_idempotence(self):
  model=fixture((2,3,4),(2,0,1));first,_=normalize(model);second,r=normalize(first)
  self.assertEqual(first,second);self.assertFalse(r["rules"])
 def test_invalid_budget_rejected(self):
  for value in(0,-1,True,262145,"262144"):
   with self.subTest(value=value),self.assertRaises(ValueError):normalize(fixture((2,3),(1,0)),max_index_cells=value)
 def test_forward_dependency_rejected(self):
  model=fixture((2,3),(1,0));model["nodes"][0]["inputs"]=[model["nodes"][-1]["outputs"][0]]
  with self.assertRaises(ValueError):normalize(model)
 def test_explicit_axes_verifies_entire_map(self):
  expected=oracle((2,3,4),(2,0,1));self.assertEqual(permutation((2,3,4),(4,2,3),expected),[2,0,1])
  invalid=list(expected);invalid[-1],invalid[-2]=invalid[-2],invalid[-1]
  self.assertIsNone(permutation((2,3,4),(4,2,3),invalid))
if __name__=="__main__":unittest.main()
