"""Standard-library controls; never import Torch/NumPy or execute source capture."""
import ast, copy, hashlib, importlib.util, json, struct, sys, unittest
from pathlib import Path
import build_overlay as builder

HERE=builder.HERE;C=HERE/"candidate"
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def tree(path):return ast.parse(path.read_text())
def function(path,name):
 return next(n for n in tree(path).body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name==name)
def dump(node):return ast.dump(node,include_attributes=False)
def selected(path,names,namespace):
 nodes=[n for n in tree(path).body if isinstance(n,ast.FunctionDef) and n.name in names]
 exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),"exec"),namespace)
 return namespace
def require(value,message):
 if not value:raise ValueError(message)
def load_outputs():
 spec=importlib.util.spec_from_file_location("_controlled_mha_outputs",C/"pytorch_mha_outputs.py")
 module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
def bindings(record):
 def leaves(v):
  if type(v)is list:
   for x in v:yield from leaves(x)
  else:yield v
 raw=b"".join(struct.pack("<d",float(v)) for v in leaves(record["values"]))
 return dict(format="poseidon-pytorch-state-weights-v1",
             declarations_sha256="placeholder",normalized_float64_sha256={"state":hashlib.sha256(raw).hexdigest()})

class OverlayControls(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.manifest=json.loads((HERE/"overlay.json").read_text())
  cls.outputs=load_outputs()
 def test_candidate_exactly_rebuilds_without_writes(self):
  original,after,frozen=builder.build_sources()
  self.assertEqual(set(after),set(self.manifest["targets"]))
  self.assertEqual(frozen,self.manifest["frozen_inputs"])
  for name,text in after.items():self.assertEqual((C/name).read_text(),text)
 def test_original_four_targets_unchanged(self):
  for name in builder.TARGETS:
   self.assertEqual(sha(builder.BASE/name),self.manifest["targets"][name]["before_sha256"])
   self.assertEqual((builder.BASE/name).read_bytes(),(HERE/"baseline-snapshot"/name).read_bytes())
 def test_all_candidates_compile_in_memory(self):
  for p in C.glob("*.py"):compile(p.read_text(),str(p),"exec")
 def test_unchanged_mathematical_functions(self):
  old=tree(builder.OLD/"mha_source_plan.py")
  new=tree(C/"pytorch_mha_math.py")
  oldf={n.name:dump(n) for n in old.body if isinstance(n,ast.FunctionDef) and n.name!="implementation"}
  newf={n.name:dump(n) for n in new.body if isinstance(n,ast.FunctionDef) and n.name!="implementation"}
  self.assertEqual(oldf,newf)
 def test_independent_reference_byte_identical(self):
  self.assertEqual((C/"pytorch_mha_reference_math.py").read_bytes(),(builder.OLD/"mha_source_reference.py").read_bytes())
 def test_projection_source_and_shape_rules_preserved(self):
  for name in ("_slice_origins","_output_shapes","_state_record","_state_array","_encode","_decode"):
   self.assertEqual(dump(function(C/"pytorch_mha_plan.py",name)),
                    dump(function(builder.DRAFT/"source_plan.py",name)))
 def test_no_normal_runtime_benchmark_path(self):
  for p in C.glob("*.py"):
   source=p.read_text()
   for name in ("benchmarks/tools","r1063","static_mha_source_v16_draft_v1"):
    if name=="r1063":continue # descriptive algebra lineage is allowed
    self.assertNotIn(name,source)
 def test_manifest_not_activation_or_isolated_capture_claim(self):
  for key in ("production_applied","production_capture_validated","unified_ordered_integration_complete"):
   self.assertIs(self.manifest[key],False)
  for key in ("api","nix_entries","compiler","encrypted_execution"):self.assertEqual(self.manifest[key],0)
 def test_old_facade_functions_unchanged_except_dispatch(self):
  a={n.name:dump(n) for n in tree(builder.BASE/"pytorch_model_intake.py").body if isinstance(n,ast.FunctionDef)}
  b={n.name:dump(n) for n in tree(C/"pytorch_model_intake.py").body if isinstance(n,ast.FunctionDef)}
  for name in a:
   if name not in ("ordered","preparation_options"):self.assertEqual(a[name],b[name],name)
 def test_old_v12_v15_package_reference_maps_remain(self):
  path=C/"pytorch_ordered_contract.py";ns={"require":require}
  for node in tree(path).body:
   if isinstance(node,ast.Assign) and all(isinstance(t,ast.Name) for t in node.targets):
    name=node.targets[0].id
    if name in ("FORMAT","PACKAGE_FORMAT","REFERENCE_FORMAT") or name.endswith(("_SOURCE_FORMAT","_PACKAGE_FORMAT","_REFERENCE_FORMAT")) or name=="SOURCE_FORMATS":
     exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),"exec"),ns)
  selected(path,("package_version","reference_version"),ns)
  for number in range(12,17):
   self.assertEqual(ns["package_version"]({"format":f"poseidon-trusted-pytorch-v{number}"}),f"poseidon-trusted-pytorch-provenance-v{number}")
   self.assertEqual(ns["reference_version"]({"format":f"poseidon-trusted-pytorch-v{number}"}),f"poseidon-trusted-pytorch-reference-v{number}")
  with self.assertRaises(ValueError):ns["package_version"]({"format":"poseidon-trusted-pytorch-v17"})
 def test_old_output_validator_default_and_body_unchanged(self):
  a=function(builder.BASE/"pytorch_source_contract.py","validate_output_tree")
  b=copy.deepcopy(function(C/"pytorch_source_contract.py","validate_output_tree"))
  b.body=b.body[1:];b.args.kwonlyargs=b.args.kwonlyargs[:-1];b.args.kw_defaults=b.args.kw_defaults[:-1]
  self.assertEqual(dump(a),dump(b))
 def test_old_restore_body_remains(self):
  a=function(builder.BASE/"pytorch_source_contract.py","restore_outputs")
  b=copy.deepcopy(function(C/"pytorch_source_contract.py","restore_outputs"));b.body=b.body[1:]
  self.assertEqual(dump(a),dump(b))
 def test_old_output_boundary_rejects_versioned_none(self):
  ns=selected(C/"pytorch_source_contract.py",("validate_output_tree",),{"require":require})
  leaf=object();_,record=self.outputs.flatten((leaf,None),lambda v:v is leaf,["out"],source_format=self.outputs.SOURCE_FORMAT)
  with self.assertRaises(ValueError):ns["validate_output_tree"](record,["out"],allow_scalars=True)
  with self.assertRaises(ValueError):ns["validate_output_tree"]({"kind":"none"},[],allow_scalars=True)
 def test_none_contract_no_cipher_slot_and_restore(self):
  leaf=object();leaves,record=self.outputs.flatten((leaf,None),lambda v:v is leaf,["out"],source_format=self.outputs.SOURCE_FORMAT)
  self.assertEqual(len(leaves),1);self.assertEqual(self.outputs.restore({"out":17},record),(17,None))
 def test_none_old_version_cannot_be_resigned(self):
  leaf=object();_,record=self.outputs.flatten((leaf,None),lambda v:v is leaf,["out"],source_format=self.outputs.SOURCE_FORMAT)
  record["source_format"]=self.outputs.OLD_FORMAT
  record["binding"]=self.outputs.digest({k:v for k,v in record.items() if k!="binding"})
  with self.assertRaises(ValueError):self.outputs.validate(record)
 def test_dynamic_output_rejected(self):
  leaf=object()
  with self.assertRaises(ValueError):self.outputs.flatten((leaf,4),lambda v:v is leaf,["out"],source_format=self.outputs.SOURCE_FORMAT)
 def test_three_cipher_roles_are_not_single_input_binding(self):
  source=ast.get_source_segment((C/"pytorch_mha_plan.py").read_text(),function(C/"pytorch_mha_plan.py","capture"))
  self.assertIn('("query","key","value","key_padding_mask","need_weights","attn_mask","average_attn_weights","is_causal")',source)
  self.assertIn('for role in names[:3]',source)
  self.assertNotIn('("input",)',source)
 def test_external_request_anchor_preserved(self):
  self.assertEqual(dump(function(C/"pytorch_mha_plan.py","verify")),
                   dump(function(builder.DRAFT/"source_plan.py","verify")))
 def test_native_exact_class_and_all_blocked_guard(self):
  self.assertIn('type(child) is torch.nn.MultiheadAttention',(C/"pytorch_mha_plan.py").read_text())
  math=(C/"pytorch_mha_math.py").read_text()
  self.assertIn('not np.any(np.all(blocked,axis=-1))',math)
  self.assertIn('mha_all_masked_unsupported',math)
 def test_generic_approximation_selectors_bound_per_original_call(self):
  source=(C/"pytorch_mha_plan.py").read_text()
  self.assertIn('selection.select(call["id"],name,name,call["module_paths"])',source)
  self.assertIn('require(chosen==call["approximation"]',source)
  self.assertGreaterEqual(source.count('selection.finish()'),2)
 def test_state_binding_rejects_changed_bytes_and_package(self):
  if str(builder.BASE) not in sys.path:sys.path.insert(0,str(builder.BASE))
  from benchmark_graph import digest
  ns=selected(C/"pytorch_mha_capture.py",("_scalars","check_state"),
              {"require":require,"hashlib":hashlib,"struct":struct,"math":__import__("math")})
  s=dict(dtype="<f8",shape=[2],values=[.25,-.5],bytes_sha256=hashlib.sha256(struct.pack("<dd",.25,-.5)).hexdigest())
  records={"state":dict(file="state.npy",sha256="0"*64,bytes=144,shape=[2],dtype="<f8")}
  expected=bindings(s);expected["declarations_sha256"]=digest(records)
  ns["check_state"]({"state":s},records,expected)
  bad=copy.deepcopy(s);bad["values"][0]=.5
  with self.assertRaises(ValueError):ns["check_state"]({"state":bad},records,expected)
  bad=copy.deepcopy(expected);bad["normalized_float64_sha256"]["state"]="0"*64
  with self.assertRaises(ValueError):ns["check_state"]({"state":s},records,bad)
 def test_boolean_original_bytes_and_dtype_cannot_be_forged(self):
  if str(builder.BASE) not in sys.path:sys.path.insert(0,str(builder.BASE))
  from benchmark_graph import digest
  ns=selected(C/"pytorch_mha_capture.py",("_scalars","check_state"),
              {"require":require,"hashlib":hashlib,"struct":struct,"math":__import__("math")})
  s=dict(dtype="|b1",shape=[2],values=[False,True],bytes_sha256=hashlib.sha256(bytes((0,1))).hexdigest())
  records={"state":dict(file="state.npy",sha256="0"*64,bytes=130,shape=[2],dtype="|b1")}
  expected=bindings(s);expected["declarations_sha256"]=digest(records)
  ns["check_state"]({"state":s},records,expected)
  bad=copy.deepcopy(s);bad["values"]=[0,1]
  with self.assertRaises(ValueError):ns["check_state"]({"state":bad},records,expected)
 def test_only_manifest_version_dispatches_worker(self):
  p=function(C/"pytorch_ordered_intake.py","capture_request")
  source=ast.get_source_segment((C/"pytorch_ordered_intake.py").read_text(),p)
  self.assertIn('if spec["format"]==MHA_SOURCE_FORMAT:',source)
  self.assertNotIn('case["id"]',source);self.assertNotIn('module_name==',source)
 def test_frozen_inputs_still_current(self):
  for relative,expected in self.manifest["frozen_inputs"].items():
   self.assertEqual(sha(builder.ROOT/relative),expected)
 def test_patch_and_candidate_binding(self):
  self.assertEqual(sha(HERE/"integration.patch"),self.manifest["patch_sha256"])
  for name,record in self.manifest["targets"].items():self.assertEqual(sha(C/name),record["after_sha256"])
  unsigned={k:v for k,v in self.manifest.items() if k!="binding"}
  self.assertEqual(hashlib.sha256(json.dumps(unsigned,sort_keys=True,separators=(",",":")).encode()).hexdigest(),self.manifest["binding"])
if __name__=="__main__":unittest.main()
