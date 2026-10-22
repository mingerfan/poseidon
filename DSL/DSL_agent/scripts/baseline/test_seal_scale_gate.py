"""Pure static metadata tests; no fake execution claims."""
import math,struct,unittest
from seal_artifact_gate import (inspect_artifacts,STOCK_MODULI,modulus_capacity,
    check_capacity,parameter_profile,ScaleState,compatible_scale)
EMPTY=struct.pack("<q",0)
def artifact(ops=(), *, inputs=((13,40),), output=(0,13,40), nc=4,np=4):
 n=len(inputs);dst,level,scale=output
 values=[s for l,s in inputs]+[l for l,s in inputs]+[scale,level,dst]
 return (struct.pack("<IIQQ",0x4845564D,24,n,1)+
         struct.pack("<5Q",40+8*(2*n+3),len(ops),nc,np,13)+
         struct.pack("<"+"Q"*len(values),*values)+
         b"".join(struct.pack("<4H",*op) for op in ops))
def gate(ops=(),**kwargs):
 inputs=kwargs.get("inputs",((13,40),))
 return inspect_artifacts(artifact(ops,**kwargs),EMPTY,expected_inputs=len(inputs))
class ScaleGateTests(unittest.TestCase):
 def test_capacity_is_product_bit_length(self):
  self.assertEqual(modulus_capacity([3,5]),4)
  self.assertNotEqual(modulus_capacity([3,5]),sum(q.bit_length() for q in [3,5]))
  self.assertEqual([modulus_capacity(STOCK_MODULI[:n]) for n in range(1,14)],list(range(60,781,60)))
 def test_exact_profile_only(self):
  p=dict(seal_version="4.0.0",polynomial_degree=32768,security_check="tc128",
    parameters_set=True,data_modulus_count=13,modulus_bits=[60]*14,
    modulus_values=[str(q) for q in STOCK_MODULI])
  parameter_profile(p)
  for key,value in [("seal_version","4.1.0"),("security_check","none"),("data_modulus_count",14),
                    ("modulus_values",[str(q+2) for q in STOCK_MODULI])]:
   with self.subTest(key=key),self.assertRaises(ValueError):parameter_profile(dict(p,**{key:value}))
 def test_above_180_accepted(self):
  r=gate(inputs=((13,200),),output=(0,13,200))
  self.assertEqual(r["propagated_outputs"][0]["scale"],2.**200)
  self.assertFalse(r["execution_validated"])
 def test_input_encode_headroom(self):
  gate(inputs=((13,778),),output=(0,13,778))
  with self.assertRaisesRegex(ValueError,"initial encoding"):gate(inputs=((13,779),),output=(0,13,779))
 def test_level1_scale75_rejected(self):
  with self.assertRaisesRegex(ValueError,"remaining_level=1.*log2_scale=75"):
   gate(inputs=((1,75),),output=(0,1,75))
 def test_operation_specific_boundaries(self):
  check_capacity(1,2.**59,STOCK_MODULI[:-1],"decode","decode")
  with self.assertRaises(ValueError):check_capacity(1,2.**59,STOCK_MODULI[:-1],"encode","encode")
  with self.assertRaises(ValueError):check_capacity(1,2.**60,STOCK_MODULI[:-1],"decode","decode")
 def test_intermediate_multiply_fails_first(self):
  with self.assertRaisesRegex(ValueError,"instruction 0 MulCC"):
   gate(((8,1,0,0),(3,1,1,0)),inputs=((13,390),),output=(1,12,720))
 def test_multiply_and_output_declaration(self):
  r=gate(((8,1,0,0),),output=(1,13,80))
  self.assertEqual(r["propagated_outputs"][0]["scale"],2.**80)
  with self.assertRaisesRegex(ValueError,"Result scale declaration mismatch"):
   gate(((8,1,0,0),),output=(1,13,40))
 def test_real_prime_rescale(self):
  r=gate(((8,1,0,0),(3,1,1,0)),output=(1,12,20))
  self.assertEqual(r["propagated_outputs"][0]["scale"],2.**80/float(STOCK_MODULI[12]))
  self.assertNotEqual(r["propagated_outputs"][0]["scale"],2.**20)
 def test_modswitch_retains_scale(self):
  r=gate(((4,1,0,12),),output=(1,1,40))
  self.assertEqual(r["propagated_outputs"][0]["scale"],2.**40)
  with self.assertRaisesRegex(ValueError,"instruction 0 ModswitchC"):
   gate(((4,1,0,12),),inputs=((13,75),),output=(1,1,75))
 def test_chain_exhaustion(self):
  with self.assertRaisesRegex(ValueError,"instruction 1 RescaleC.*Exhausted"):
   gate(((4,0,0,12),(3,0,0,0)),output=(0,1,40))
 def test_add_scale_mismatch(self):
  with self.assertRaisesRegex(ValueError,"Add scale mismatch"):
   gate(((6,2,0,1),),inputs=((13,40),(13,41)),output=(2,13,41))
 def test_add_drift_and_left_side_effect(self):
  # lhs is rescaled, rhs is input; nominal 20 on both. dst is distinct.
  r=gate(((8,2,0,0),(3,2,2,0),(6,3,2,1)),
         inputs=((13,40),(12,20)),output=(2,12,20))
  self.assertEqual(r["propagated_outputs"][0]["scale"],2.**20)
  self.assertEqual(len(r["scale_alignments"]),1)
 def test_scale_drift_budget_not_unlimited(self):
  with self.assertRaisesRegex(ValueError,"drift exceeds"):
   compatible_scale(ScaleState(13,2.**40*1.001,40),ScaleState(13,2.**40,40),"test")
 def test_aliasing_and_overwrite(self):
  r=gate(((8,1,0,1),(2,0,1,0),(6,0,0,1)),inputs=((13,40),(13,40)),output=(0,13,80))
  self.assertEqual(r["propagated_outputs"][0]["scale"],2.**80)
 def test_plain_add_and_multiply(self):
  cst=struct.pack("<qqd",1,1,0.5)
  for opcode,out in [(7,40),(9,80)]:
   r=inspect_artifacts(artifact(((0,0,0,(13<<10)|40),(opcode,1,0,0)),output=(1,13,out)),cst)
   self.assertEqual(r["res_scale"],[out])
  with self.assertRaisesRegex(ValueError,"Add scale mismatch"):
   inspect_artifacts(artifact(((0,0,0,(13<<10)|41),(7,1,0,0))),cst)
 def test_decode_declaration_capacity(self):
  with self.assertRaisesRegex(ValueError,"output\\[0\\].*remaining_level=1.*log2_scale=75"):
   gate(((4,0,0,12),),output=(0,1,75))
 def test_eager_plain_available_before_encode_instruction(self):
  cst=struct.pack("<qqd",1,1,.5)
  inspect_artifacts(artifact(((7,1,0,0),(0,0,0,(13<<10)|40)),output=(1,13,40)),cst)

class NewGuidanceTests(unittest.TestCase):
 def test_v7_preserved_and_v8_only_public_addition(self):
  import copy,json
  from benchmark_suite import Builder
  from benchmark_graph import digest
  from unified_graph_contract import prepare,validate_request
  from compiler_configuration import PROFILE_SHA256
  from deepseek_provider import public_request
  b=Builder([(2,)]);m=b.finish(b.node("square",["input0"]))
  old=prepare(m,PROFILE_SHA256,generation_guidance="explicit-v7")
  new=prepare(m,PROFILE_SHA256,generation_guidance="explicit-v8")
  validate_request(new);self.assertEqual(public_request(new),new)
  stripped=copy.deepcopy(new);stripped["generation_guidance"].pop("value_and_scale_discipline")
  stripped["generation_guidance"]["version"]="explicit-v7"
  stripped["request_id"]=digest({k:v for k,v in stripped.items() if k!="request_id"})
  self.assertEqual(old,stripped)
  bad=copy.deepcopy(new);bad["generation_guidance"]["value_and_scale_discipline"]["scale_semantics"]="disable limits"
  bad["request_id"]=digest({k:v for k,v in bad.items() if k!="request_id"})
  with self.assertRaises(ValueError):validate_request(bad)

class ExecutionBindingTests(unittest.TestCase):
 def test_actual_parameter_and_output_binding_rejects_tampering(self):
  import tempfile,copy,hashlib
  from pathlib import Path
  from unittest.mock import patch
  from seal_artifact_gate import verify_execution_binding
  p=dict(seal_version="4.0.0",polynomial_degree=32768,security_check="tc128",
   parameters_set=True,data_modulus_count=13,modulus_bits=[60]*14,
   modulus_values=[str(q) for q in STOCK_MODULI])
  g=gate()
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);(root/"scripts/baseline/seal_keys").mkdir(parents=True)
   lib=root/"libseal_artifact_parameters.so";lib.write_bytes(b"unit-only")
   cpp=root/"scripts/baseline/seal_keys/artifact_parameters.cpp";cpp.write_bytes(b"test-source")
   sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
   report=dict(parameters=p,parameter_observer_sha256=sha(lib),parameter_observer_source_sha256=sha(cpp))
   execution=dict(artifact_gate_version=g["gate_version"],public_parameter_binding=dict(
    actual_parameters_verified=True,security_check="tc128",parm_sha256="a"*64,
    observer_sha256=sha(lib),data_modulus_bits=g["level_capacity_bits"],seal_chain_indices=list(range(13))),
    ciphertext_metadata=[dict(outputs=[dict(data_modulus_count=13,log2_scale=40.,polynomials=2)])])
   with patch("seal_cpu_golden.KEY_BUILD",root),patch("hecate_python_env.ROOT",root):
    verify_execution_binding(g,execution,report)
    for key,value in [("actual_parameters_verified",False),("observer_sha256","b"*64),("seal_chain_indices",[13]*13)]:
     changed=copy.deepcopy(execution);changed["public_parameter_binding"][key]=value
     with self.subTest(key=key),self.assertRaises(ValueError):verify_execution_binding(g,changed,report)
    changed=copy.deepcopy(execution);changed["ciphertext_metadata"][0]["outputs"][0]["log2_scale"]=41.
    with self.assertRaises(ValueError):verify_execution_binding(g,changed,report)
if __name__=="__main__":unittest.main()


