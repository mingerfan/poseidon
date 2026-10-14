"""Original-source and derived-state integrity checks over actual isolated captures."""
import copy,json,unittest
from pathlib import Path
import numpy as np
from benchmark_graph import digest
from capture_adapter import check_record,structural
from pytorch_weight_package import read
from sandbox import PROFILE,run
HERE=Path(__file__).resolve().parent;ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5])
OUT=Path("/home/lhohy/poseidon-work/platforms/aarch64-linux/results/agent-public-tensor-source-draft-v2")
class CapturedSourceTests(unittest.TestCase):
 def fixture(self,name="vector_function"):
  manifest=ROOT/next(c["manifest"]for c in json.loads((HERE/"cases.json").read_text())if c["case"]==name)
  spec=json.loads(manifest.read_text())
  record=json.loads((OUT/name/"capture.json").read_text())
  original,_=read(manifest.parent,spec["weights"],resource_profile=PROFILE)
  internal,_=read(OUT/name/"internal-weight-snapshot",record["internal_manifest"]["weights"],resource_profile=PROFILE)
  return record,spec,original,internal
 def checked(self,r,s,a,b,authority=None):
  return check_record(r,s,a,b,expected_record_binding=authority or r["binding"],resource_profile=PROFILE)
 def changed(self,modify,*,name="vector_function",pattern=None):
  r,s,a,b=self.fixture(name)
  # Inject separate altered payloads; authenticated NPY snapshots stay read-only.
  a={k:v.copy()for k,v in a.items()};b={k:v.copy()for k,v in b.items()};modify(r,s,a,b)
  r["binding"]=digest({k:v for k,v in r.items()if k!="binding"})
  # Rebinding the outer record cannot excuse a mismatch with independently loaded source/public state.
  if pattern:
   with self.assertRaisesRegex(ValueError,pattern):self.checked(r,s,a,b)
  else:
   with self.assertRaises(ValueError):self.checked(r,s,a,b)
 def test_six_loaded_records_match_original_sources(self):
  for c in json.loads((HERE/"cases.json").read_text()):
   with self.subTest(case=c["case"]):
    r,s,a,b=self.fixture(c["case"]);self.checked(r,s,a,b)
    self.assertEqual(r["original_reference"]["groups"],16)
    self.assertTrue(r["original_reference"]["public_state_immutable"])
    self.assertTrue(r["original_reference"]["input_storage_isolated"])
 def test_source_weight_value_substitution(self):
  self.changed(lambda r,s,a,b:a["std"].__setitem__(0,.75),pattern="Original source public weights")
 def test_source_weight_shape_substitution(self):
  self.changed(lambda r,s,a,b:a.update(std=a["std"].reshape(1,3)))
 def test_source_manifest_factory_substitution(self):
  self.changed(lambda r,s,a,b:s["factory"]["kwargs"].update(extra=True),pattern="original source contract")
 def test_source_input_signature_substitution(self):
  self.changed(lambda r,s,a,b:s["inputs"][0].update(shape=[3,2]),pattern="original source contract")
 def test_original_state_binding_substitution(self):
  self.changed(lambda r,s,a,b:r["original_public_state"]["std"].update(normalized_float64_sha256="0"*64),pattern="Original source state")
 def test_derived_array_payload_substitution(self):
  def alter(r,s,a,b):
   n=r["division_capture"]["entries"][0]["reciprocal_state_name"];b[n][0]*=2
  self.changed(alter,pattern="Derived public weights")
 def test_derived_array_removed(self):
  def alter(r,s,a,b):del b[r["division_capture"]["entries"][0]["reciprocal_state_name"]]
  self.changed(alter,pattern="Derived public weights")
 def test_internal_source_inputs_cannot_change(self):
  self.changed(lambda r,s,a,b:r["internal_manifest"]["inputs"][0].update(shape=[3,2]),pattern="Internal source changed")
 def test_internal_outputs_cannot_change(self):
  self.changed(lambda r,s,a,b:r["internal_manifest"].update(outputs=["different"]),pattern="Internal source changed")
 def test_original_reference_graph_cannot_change(self):
  self.changed(lambda r,s,a,b:r["original_reference"].update(graph_sha256="0"*64),pattern="Original source numerical reference")
 def test_original_reference_must_preserve_storage(self):
  self.changed(lambda r,s,a,b:r["original_reference"].update(input_storage_isolated=False),pattern="Original source numerical reference")
 def test_original_reference_must_preserve_weights(self):
  self.changed(lambda r,s,a,b:r["original_reference"].update(public_state_immutable=False),pattern="Original source numerical reference")
 def test_original_reference_cannot_claim_less_probes(self):
  self.changed(lambda r,s,a,b:r["original_reference"].update(groups=4),pattern="Original source numerical reference")
 def test_original_reference_out_of_tolerance(self):
  self.changed(lambda r,s,a,b:r["original_reference"].update(max_reference_tolerance_ratio=2),pattern="Original source numerical reference")
 def test_captured_lowered_graph_cannot_change(self):
  def alter(r,s,a,b):r["lowered_graph"]["nodes"][0]["op"]="negate"
  self.changed(alter,pattern="Derived graph and capture identity")
 def test_plain_capture_cannot_claim_api_or_encryption(self):
  for key in("provider_calls","compiler_calls","encrypted_calls"):
   with self.subTest(key=key):self.changed(lambda r,s,a,b:r.update({key:1}),pattern="Capture only scope")
 def test_activation_requires_separate_version(self):
  self.changed(lambda r,s,a,b:r.update(production_intake_integrated=True),pattern="original source contract")
 def test_external_capture_authority_required(self):
  r,s,a,b=self.fixture()
  with self.assertRaisesRegex(ValueError,"External source capture authority"):self.checked(r,s,a,b,"0"*64)
 def test_rebound_outer_record_cannot_replace_original_authority(self):
  r,s,a,b=self.fixture();original=r["binding"];r["source_dtype"]="float32"
  r["binding"]=digest({k:v for k,v in r.items()if k!="binding"})
  with self.assertRaisesRegex(ValueError,"External source capture authority"):self.checked(r,s,a,b,original)
 def test_direct_source_capture_requires_explicit_trust(self):
  with self.assertRaisesRegex(ValueError,"Explicit trusted-local source"):
   run(HERE.parent/"public_tensor_source_v1/fixtures/vector_function/manifest.json",OUT/"not-created",trusted=False)
  self.assertFalse((OUT/"not-created").exists())
 def test_structural_inventory_does_not_accept_duplicate_fx_names(self):
  node=dict(name="same",op="placeholder",target="x",args={},kwargs={})
  with self.assertRaisesRegex(ValueError,"Unique FX inventory"):structural([node,node])

 def test_public_index_raw_dtype_must_not_be_float_normalized(self):
  self.changed(lambda r,s,a,b:a.update(index=a["index"].astype("<f8")),
   name="index_select_function",pattern="Original source state")
 def test_boolean_mask_raw_dtype_must_not_be_float_normalized(self):
  self.changed(lambda r,s,a,b:a.update(mask=a["mask"].astype("<f8")),
   name="bool_multiply",pattern="Original source state")
 def test_batchnorm_counter_raw_value_is_authenticated(self):
  self.changed(lambda r,s,a,b:a["bn.num_batches_tracked"].__setitem__((),9),
   name="batch_norm_module",pattern="Original source public weights")
 def test_unused_original_state_is_still_authenticated(self):
  self.changed(lambda r,s,a,b:a["unused_counter"].__setitem__((),8),
   name="batch_norm_function",pattern="Original source public weights")
 def test_traced_state_cannot_claim_a_different_dtype(self):
  self.changed(lambda r,s,a,b:r["traced_public_state"]["index"].update(dtype="<f8"),
   name="index_select_function",pattern="Trace state subset")
 def test_unreferenced_state_inventory_cannot_disappear(self):
  self.changed(lambda r,s,a,b:r.update(unreferenced_state_names=[]),
   name="batch_norm_function",pattern="Trace state subset")
 def test_nondivisor_state_cannot_be_declared_as_divisor(self):
  def alter(r,s,a,b):
   d=r["division_capture"];d["declared_divisors"]["mask"]=d["public_state"]["mask"]
   d["binding"]=digest({k:v for k,v in d.items()if k!="binding"})
  self.changed(alter,name="bool_multiply",pattern="Exact direct divisor inventory")
 def test_internal_nonfloating_state_dtype_is_preserved(self):
  self.changed(lambda r,s,a,b:b.update(index=b["index"].astype("<f8")),
   name="index_select_function",pattern="Derived public weights")
 def test_mixed_raw_types_preserved_in_both_states(self):
  for name,state,dtype in [("index_select_function","index","<i8"),("bool_multiply","mask","|b1"),
   ("batch_norm_module","bn.num_batches_tracked","<i8")]:
   with self.subTest(case=name):
    r,s,a,b=self.fixture(name);self.checked(r,s,a,b)
    self.assertEqual(r["original_public_state"][state]["dtype"],dtype)
    self.assertEqual(r["division_capture"]["normalized_state"][state]["dtype"],dtype)

 def test_float32_source_and_float64_math_are_distinct(self):
  for name in ("float32_vector","float32_batch_norm","float32_gelu_projection"):
   with self.subTest(case=name):
    r,s,a,b=self.fixture(name);self.checked(r,s,a,b)
    self.assertEqual(s["dtype_policy"],"convert-float64-v1")
    self.assertIn("<f4",{v["dtype"]for v in r["original_public_state"].values()})
    self.assertNotIn("<f4",{v["dtype"]for v in r["normalized_source_state"].values()})
    self.assertFalse(r["float_normalization"]["source_precision_execution_compared"])
 def test_float32_counter_stays_integer_after_normalization(self):
  r,s,a,b=self.fixture("float32_batch_norm");self.checked(r,s,a,b)
  for states in (r["original_public_state"],r["normalized_source_state"],r["division_capture"]["normalized_state"]):
   self.assertEqual(states["bn.num_batches_tracked"]["dtype"],"<i8")
 def test_reciprocal_uses_promoted_float32_value_not_float32_reciprocal(self):
  r,s,a,b=self.fixture("float32_vector");self.checked(r,s,a,b)
  name=r["division_capture"]["entries"][0]["reciprocal_state_name"]
  expected=1.0/a["std"].astype("<f8")
  np.testing.assert_array_equal(b[name],expected)
  self.assertFalse(np.array_equal(expected,np.reciprocal(a["std"]).astype("<f8")))
 def test_conversion_metadata_cannot_claim_original_precision_validation(self):
  self.changed(lambda r,s,a,b:r["float_normalization"].update(source_precision_execution_compared=True),
   name="float32_vector",pattern="float conversion policy binding")
 def test_conversion_metadata_cannot_change_requested_policy(self):
  self.changed(lambda r,s,a,b:r["float_normalization"].update(float_policy="strict-float64-v1"),
   name="float32_vector",pattern="float conversion policy binding")
 def test_normalized_source_payload_cannot_change(self):
  self.changed(lambda r,s,a,b:r["normalized_source_state"]["std"].update(sha256="0"*64),
   name="float32_vector",pattern="Original normalized source state")
 def test_reciprocal_computed_at_float32_precision_is_rejected(self):
  def alter(r,s,a,b):
   name=r["division_capture"]["entries"][0]["reciprocal_state_name"]
   b[name]=np.reciprocal(a["std"]).astype("<f8")
  self.changed(alter,name="float32_vector")
