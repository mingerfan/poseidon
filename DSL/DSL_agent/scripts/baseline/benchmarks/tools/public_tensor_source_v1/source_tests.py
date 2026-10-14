"""Original-source and derived-state integrity checks over actual isolated captures."""
import copy,json,unittest
from pathlib import Path
import numpy as np
from benchmark_graph import digest
from capture_adapter import check_record,structural
from pytorch_weight_package import read
from sandbox import PROFILE,run
HERE=Path(__file__).resolve().parent;ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5])
OUT=Path("/home/lhohy/poseidon-work/platforms/aarch64-linux/results/agent-public-tensor-source-draft-v1")
class CapturedSourceTests(unittest.TestCase):
 def fixture(self,name="vector_function"):
  spec=json.loads((HERE/"fixtures"/name/"manifest.json").read_text())
  record=json.loads((OUT/name/"capture.json").read_text())
  original,_=read(HERE/"fixtures"/name,spec["weights"],resource_profile=PROFILE)
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
   run(HERE/"fixtures/vector_function/manifest.json",OUT/"not-created",trusted=False)
  self.assertFalse((OUT/"not-created").exists())
 def test_structural_inventory_does_not_accept_duplicate_fx_names(self):
  node=dict(name="same",op="placeholder",target="x",args={},kwargs={})
  with self.assertRaisesRegex(ValueError,"Unique FX inventory"):structural([node,node])
