"""Independent stdlib role-contract controls; zero execution/API."""
import copy,json,random,unittest
from pathlib import Path
import public_math_context as m
ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5])
CASES=("tanh_module","tanh_functional","tanh_input_keyword")
def fixture(name="tanh_module"):
 return json.loads((ROOT/"docs/baseline/agent-gelu-live-proposal-r1306-cases"/(name+"-request.json")).read_bytes())
def bind(q):
 info=q["public_weight_binding"];info["model_constants"]={k:m.descriptor(v) for k,v in q["model"]["constants"].items()}
 info["public_constants"]={k:m.descriptor(v) for k,v in q["public_constants"].items()}
 c=copy.deepcopy(q);c["model"]["constants"]=copy.deepcopy(info["model_constants"]);c["public_constants"]=copy.deepcopy(info["public_constants"])
 q["request_id"]=m.digest({k:v for k,v in c.items() if k!="request_id"})
 return q
def poly(values,basis="power",plains=None):
 q=fixture();q["model"]["constants"]={"coef":values}
 q["model"]["nodes"]=[dict(id="node",op="polynomial",inputs=["x","coef"],attrs={"basis":basis},outputs=["result"])]
 q["model"]["outputs"]=[dict(name="result",value="result")]
 q["public_constants"]=dict(plains if plains is not None else {f"plain{i}":v for i,v in enumerate(values)})
 return bind(q)
def reb(r):
 r["binding"]=m.digest({k:v for k,v in r.items() if k!="binding"})
class Controls(unittest.TestCase):
 def actual(self,name):
  q=fixture(name);before=m.canonical(q);new=m.upgrade(q);c=m.project(new)
  self.assertEqual(before,m.canonical(q));self.assertNotEqual(new["request_id"],q["request_id"])
  self.assertEqual(new["public_weight_binding"]["policy"],m.POLICY);m.verify_projected(c)
  row=next(x for x in c["public_weight_binding"]["math_roles"]["constants"] if x["kind"]=="polynomial_coefficients")
  self.assertEqual([x["plain_names"] for x in row["elements"]],[["p3"],["p6"],["p5"],["p3"],["p2"]])
  self.assertEqual([x["exact_zero"] for x in row["elements"]],[True,False,False,True,False])
  self.assertEqual(row["polynomial_semantics"][0]["mathematical_degree"],4)
 def test_actual_module(self):self.actual(CASES[0])
 def test_actual_functional(self):self.actual(CASES[1])
 def test_actual_input_keyword(self):self.actual(CASES[2])
 def test_power_basis(self):
  r=m.build(poly([0.,.5,-.125]));self.assertEqual(r["constants"][0]["polynomial_semantics"][0]["basis"],"power")
 def test_chebyshev_basis(self):
  r=m.build(poly([1.,0.,.5],basis="chebyshev"));self.assertEqual(r["constants"][0]["polynomial_semantics"][0]["basis"],"chebyshev")
 def test_tiny_coefficient_retained(self):
  r=m.build(poly([0.,1.,1e-30]));self.assertEqual(r["constants"][0]["polynomial_semantics"][0]["mathematical_degree"],2)
  self.assertFalse(r["constants"][0]["elements"][2]["exact_zero"])
 def test_exact_zero_not_transparent(self):
  r=m.build(poly([0.,0.]));self.assertTrue(r["constants"][0]["polynomial_semantics"][0]["all_zero"])
  self.assertTrue(r["encrypted_zero_available"]);self.assertIsNone(r["constants"][0]["polynomial_semantics"][0]["mathematical_degree"])
 def test_zero_can_omit_plain_binding(self):
  r=m.build(poly([0.,.5],plains={"gain":.5}))
  self.assertEqual(r["constants"][0]["elements"][0]["plain_names"],[])
 def test_missing_zero_capability(self):
  q=poly([0.,0.]);q["layout"]["auxiliary_ciphertexts"]=[]
  with self.assertRaisesRegex(m.RoleContextRejected,"missing_trusted_encrypted_zero"):m.build(q)
 def test_missing_nonzero_binding(self):
  q=poly([0.,.5],plains={"zero":0.})
  with self.assertRaisesRegex(m.RoleContextRejected,"missing_required_public_plain"):m.build(q)
 def test_multiple_exact_aliases(self):
  r=m.build(poly([.5],plains={"second":.5,"first":.5}))
  self.assertEqual(r["constants"][0]["elements"][0]["plain_names"],["first","second"])
 def test_int_to_exact_binary64(self):
  r=m.build(poly([0,1],plains={"zero":0.,"one":1.}))
  self.assertEqual(r["constants"][0]["elements"][1]["plain_names"],["one"])
 def test_signed_zero_not_same_binding(self):
  r=m.build(poly([-0.,1.],plains={"positive":0.,"one":1.}))
  self.assertEqual(r["constants"][0]["elements"][0]["plain_names"],[])
 def test_no_tolerance_for_matching(self):
  q=poly([.5],plains={"near":.5000000000000001})
  with self.assertRaises(m.RoleContextRejected):m.build(q)
 def test_scalar_arithmetic_role(self):
  q=poly([.25]);q["model"]["constants"]={"gain":.25};q["model"]["nodes"]=[dict(id="mul",op="multiply",inputs=["x","gain"],attrs={},outputs=["r"])]
  bind(q);r=m.build(q);self.assertEqual(r["constants"][0]["kind"],"scalar_operand")
 def test_tensor_arithmetic_roles(self):
  q=poly([.25,.5]);q["model"]["nodes"][0].update(op="multiply",attrs={});q["public_constants"]={"whole":[.25,.5]};bind(q)
  r=m.build(q);self.assertEqual(r["constants"],[]);self.assertEqual(r["opaque_constants"][0]["descriptor"]["elements"],2)
 def test_shape_preserved_in_tensor_role(self):
  q=poly([.25,.5]);q["model"]["constants"]["coef"]=[[.25,.5],[.5,.25]]
  q["model"]["nodes"][0].update(op="add",attrs={});bind(q)
  c=m.project(m.upgrade(q));self.assertEqual(c["public_weight_binding"]["math_roles"]["opaque_constants"][0]["descriptor"]["shape"],[2,2])
 def test_large_random_weights_remain_opaque(self):
  q=poly([.5]);rng=random.Random(771);weights=[[rng.uniform(-.9,.9) for j in range(100)] for i in range(100)]
  q["model"]["constants"]={"weights":weights};q["model"]["nodes"][0].update(op="linear",inputs=["x","weights"],attrs={})
  bind(q);c=m.project(m.upgrade(q));r=c["public_weight_binding"]["math_roles"]
  self.assertEqual(r["math_elements"],0);self.assertEqual(r["constants"],[]);self.assertEqual(r["opaque_constants"][0]["descriptor"]["elements"],10000)
  self.assertNotIn("value",r["opaque_constants"][0]);self.assertNotIn("values",r["opaque_constants"][0])
  self.assertLess(len(m.canonical(c["model"]["constants"])),300)
 def test_hash_tamper(self):
  q=poly([.5]);q["public_weight_binding"]["model_constants"]["coef"]["content_sha256"]="0"*64
  with self.assertRaisesRegex(m.RoleContextRejected,"public_descriptor_content_hash"):m.build(q)
 def test_actual_plain_tamper(self):
  q=poly([.5]);q["public_constants"]["plain0"]=.75
  with self.assertRaisesRegex(m.RoleContextRejected,"public_descriptor_content_hash"):m.build(q)
 def test_wrong_coefficient_shape(self):
  q=poly([[.5]]);bind(q)
  with self.assertRaisesRegex(m.RoleContextRejected,"polynomial_coefficient_capacity"):m.build(q)
 def test_wrong_basis(self):
  with self.assertRaisesRegex(m.RoleContextRejected,"explicit_polynomial_basis"):m.build(poly([.5],basis="unknown"))
 def test_boolean_not_scalar(self):
  with self.assertRaises(m.RoleContextRejected):poly([True])
 def test_nonfinite_not_scalar(self):
  with self.assertRaises(m.RoleContextRejected):poly([float("inf")])
 def test_ragged_not_tensor(self):
  with self.assertRaises(m.RoleContextRejected):poly([[1.],[2.,3.]])
 def test_coefficient_capacity(self):
  q=poly([.5]*257,plains={"half":.5})
  with self.assertRaisesRegex(m.RoleContextRejected,"polynomial_coefficient_capacity"):m.build(q)
 def test_total_math_capacity(self):
  q=poly([.5]*129);q["model"]["constants"]["second"]=[.5]*129
  q["model"]["nodes"].append(dict(id="node2",op="polynomial",inputs=["x","second"],attrs={"basis":"power"},outputs=["r2"]));bind(q)
  with self.assertRaisesRegex(m.RoleContextRejected,"public_math_metadata_capacity"):m.build(q)
 def test_private_envelope_rejected(self):
  q=fixture();q["private_reference"]=[2.]
  with self.assertRaisesRegex(m.RoleContextRejected,"public_envelope_fields"):m.build(q)
 def test_private_weights_rejected(self):
  q=fixture();q["privacy"]["weights"]="private"
  with self.assertRaisesRegex(m.RoleContextRejected,"public_model_privacy"):m.build(q)
 def test_backend_unchanged(self):
  q=fixture();q["compiler_configuration"]["backend"]="seal"
  with self.assertRaisesRegex(m.RoleContextRejected,"public_context_backend"):m.build(q)
 def test_legacy_id_tamper(self):
  q=fixture();q["request_id"]="0"*64
  with self.assertRaisesRegex(m.RoleContextRejected,"legacy_request_identity"):m.upgrade(q)
 def test_projected_value_tamper(self):
  c=m.project(m.upgrade(poly([.5])));r=c["public_weight_binding"]["math_roles"];r["constants"][0]["elements"][0]["value"]=.75;reb(r)
  with self.assertRaises(m.RoleContextRejected):m.verify_projected(c)
 def test_projected_degree_tamper(self):
  c=m.project(m.upgrade(poly([.5])));r=c["public_weight_binding"]["math_roles"];r["constants"][0]["polynomial_semantics"][0]["mathematical_degree"]=3;reb(r)
  with self.assertRaises(m.RoleContextRejected):m.verify_projected(c)
 def test_projected_node_role_tamper(self):
  c=m.project(m.upgrade(poly([.5])));r=c["public_weight_binding"]["math_roles"];r["constants"][0]["usages"][0]["input_index"]=0;reb(r)
  with self.assertRaises(m.RoleContextRejected):m.verify_projected(c)
 def test_projected_coefficient_order_tamper(self):
  c=m.project(m.upgrade(poly([.25,.5])));r=c["public_weight_binding"]["math_roles"];r["constants"][0]["elements"].reverse();reb(r)
  with self.assertRaises(m.RoleContextRejected):m.verify_projected(c)
 def test_projected_plain_hash_tamper(self):
  c=m.project(m.upgrade(poly([.5])));r=c["public_weight_binding"]["math_roles"];r["prebound_math_scalars"]["plain0"]["value"]=.75;reb(r)
  with self.assertRaises(m.RoleContextRejected):m.verify_projected(c)
 def test_removed_required_role_tamper(self):
  c=m.project(m.upgrade(poly([.5])));r=c["public_weight_binding"]["math_roles"];r["constants"]=[];r["math_elements"]=0;reb(r)
  with self.assertRaises(m.RoleContextRejected):m.verify_projected(c)
 def test_claimed_execution_tamper(self):
  c=m.project(m.upgrade(poly([.5])));r=c["public_weight_binding"]["math_roles"];r["physical_layout_validated"]=True;reb(r)
  with self.assertRaises(m.RoleContextRejected):m.verify_projected(c)
 def test_id_family_irrelevant(self):
  q=poly([.5]);q["model"]["id"]="unknown_unseen_model";bind(q)
  r=m.build(q);self.assertEqual(r["constants"][0]["elements"][0]["plain_names"],["plain0"])
 def test_no_dsl_answer(self):
  r=m.build(poly([.5]));raw=m.canonical(r)
  self.assertNotIn(b"hecate_source",raw);self.assertNotIn(b"def golden",raw)

 def test_extra_role_private_field_rejected(self):
  c=m.project(m.upgrade(poly([.5])));r=c["public_weight_binding"]["math_roles"];r["private_reference"]=[3.];reb(r)
  with self.assertRaises(m.RoleContextRejected):m.verify_projected(c)
 def test_extra_coefficient_private_field_rejected(self):
  c=m.project(m.upgrade(poly([.5])));r=c["public_weight_binding"]["math_roles"];r["constants"][0]["elements"][0]["private_reference"]=3.;reb(r)
  with self.assertRaises(m.RoleContextRejected):m.verify_projected(c)

if __name__=="__main__":unittest.main()
