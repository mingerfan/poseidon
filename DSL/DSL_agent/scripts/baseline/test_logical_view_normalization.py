"""Coordinate, version-selection and reference-binding rejection tests."""
import copy,itertools,json,math,random,sys,unittest
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from logical_view_normalization import normalize,permutation,digest,apply_binding,check_finite
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
  self.assertEqual(record["original_sha256"],before);self.assertTrue(record["production_active"])
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

class BindingAndScopeTests(unittest.TestCase):
 def test_explicit_version_keeps_v34_and_unknown_closed(self):
  from pytorch_logical_view_source import selected,SOURCE_FORMAT,PACKAGE_FORMAT,REFERENCE_FORMAT
  from pytorch_model_intake import ordered,SOURCE_FORMATS,PACKAGES,REFERENCES
  from pytorch_spatial_policy import supports_spatial,supports_input_call
  from pytorch_static_broadcast import supports_source
  from pytorch_source_package_policy import selected as packaged
  from pytorch_static_tensor_transforms import selected as transforms
  from pytorch_public_expression_capture import capture_format,source_policy
  spec={"format":SOURCE_FORMAT}
  self.assertTrue(selected(spec));self.assertTrue(ordered(spec))
  self.assertTrue(all(f(spec)for f in (supports_spatial,supports_input_call,supports_source,packaged,transforms)))
  self.assertEqual(PACKAGES[SOURCE_FORMAT],PACKAGE_FORMAT)
  self.assertEqual(REFERENCES[SOURCE_FORMAT],REFERENCE_FORMAT)
  self.assertNotEqual(capture_format(spec),capture_format({"format":"poseidon-trusted-pytorch-v34"}))
  self.assertEqual(source_policy(spec),source_policy({"format":"poseidon-trusted-pytorch-v34"}))
  for version in ("poseidon-trusted-pytorch-v34","poseidon-trusted-pytorch-v999"):
   self.assertFalse(selected({"format":version}))
  self.assertIn("poseidon-trusted-pytorch-v34",SOURCE_FORMATS)
  self.assertIn("poseidon-trusted-pytorch-v37",SOURCE_FORMATS)
  self.assertNotIn("poseidon-trusted-pytorch-v999",SOURCE_FORMATS)
 def fixture(self):
  original=fixture((2,3),(1,0));old=dict(model_sha256=digest(original),binding_sha256="old",original_sha256="source",source_hashes={"a":"b"})
  lowered,binding=apply_binding(original,old)
  reference=dict(format="finite",groups=16,status="passed",lowered_sha256=digest(lowered),
      capture_binding_sha256=binding["binding_sha256"],max_decomposition_error=0.,
      max_reference_tolerance_ratio=0.,comparison_tolerance={"atol":1e-12,"rtol":1e-12})
  reference["binding"]=digest(reference)
  original_reference=dict(reference,lowered_sha256=digest(original),capture_binding_sha256="original")
  original_reference.pop("binding");original_reference["binding"]=digest(original_reference)
  return original,lowered,old,binding,original_reference,reference
 def test_binds_original_and_normalized_without_mutating_original(self):
  original,lowered,old,binding,a,b=self.fixture()
  self.assertEqual(old["model_sha256"],digest(original));self.assertEqual(old["binding_sha256"],"old")
  self.assertEqual(binding["base_model_sha256"],digest(original))
  self.assertEqual(binding["model_sha256"],digest(lowered))
  self.assertEqual(binding["binding_sha256"],digest({k:v for k,v in binding.items()if k!="binding_sha256"}))
  self.assertNotEqual(digest(original),digest(lowered))
  self.assertEqual(check_finite(b,a,lowered,binding),b)
 def test_reference_scope_cannot_change(self):
  _,lowered,_,binding,a,b=self.fixture()
  for key,value in (("groups",4),("comparison_tolerance",{"atol":1e-5,"rtol":1e-4}),("status","failed")):
   with self.subTest(key=key):
    altered=copy.deepcopy(b);altered[key]=value;altered.pop("binding");altered["binding"]=digest(altered)
    with self.assertRaises(ValueError):check_finite(altered,a,lowered,binding)
 def test_reference_graph_binding_cannot_change(self):
  _,lowered,_,binding,a,b=self.fixture()
  for key in ("lowered_sha256","capture_binding_sha256"):
   with self.subTest(key=key):
    altered=dict(b);altered[key]="0"*64;altered.pop("binding");altered["binding"]=digest(altered)
    with self.assertRaises(ValueError):check_finite(altered,a,lowered,binding)
 def test_reference_ratio_and_types_reject(self):
  _,lowered,_,binding,a,b=self.fixture()
  for value in (True,-1.,1.01):
   with self.subTest(value=value):
    altered=dict(b);altered["max_reference_tolerance_ratio"]=value;altered.pop("binding");altered["binding"]=digest(altered)
    with self.assertRaises(ValueError):check_finite(altered,a,lowered,binding)
 def test_capabilities_are_explicit_and_registered(self):
  from application_capabilities import input_sources
  c=input_sources()["trusted_pytorch"];self.assertIn("poseidon-trusted-pytorch-v35",c["formats"])
  self.assertEqual(c["logical_axis_views"]["max_proof_elements"],4096)
  self.assertEqual(c["logical_axis_views"]["compiler_status"],"unverified")
  for field in ("spatial_calls","input_call","static_broadcast","static_tensor_transforms","public_mask_selection","source_package"):
   self.assertIn("poseidon-trusted-pytorch-v35",c[field]["versions"])
 def test_new_capture_metadata_not_accepted_by_old_version(self):
  from pytorch_public_expression_capture import capture_format
  self.assertNotEqual(capture_format({"format":"poseidon-trusted-pytorch-v34"}),"poseidon-public-expression-source-capture-v12")

if __name__=="__main__":unittest.main()
