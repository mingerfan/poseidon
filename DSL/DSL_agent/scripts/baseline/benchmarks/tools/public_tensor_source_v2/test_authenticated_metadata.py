"""Revalidate the new data-only policy against six immutable real source captures.

No source import, FX execution, reference evaluation, compiler or Provider call.
"""
import hashlib,json,unittest
from pathlib import Path
from benchmark_graph import digest
from public_state_binding import check_original,check_trace,direct_divisors
HERE=Path(__file__).resolve().parent;ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5])
PRIOR=HERE.parent/"public_tensor_source_v1"
OUT=Path("/home/lhohy/poseidon-work/platforms/aarch64-linux/results/agent-public-tensor-source-draft-v1")
REPORT=ROOT/"docs/baseline/agent-public-tensor-source-draft-r1352.json"
AUTHORITY="a5f434cc36688cfd4215f74ab4121269dfb96854750e19305db7d64daaf35368"
class AuthenticatedSourceMetadataTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  report=json.loads(REPORT.read_text())
  assert report["binding"]==AUTHORITY==digest({k:v for k,v in report.items()if k!="binding"})
  assert report["source_models"]==6 and report["input_groups"]==96
  cls.rows={r["case"]:r for r in report["captures"]}
  cls.cases={r["case"]:r for r in json.loads((PRIOR/"cases.json").read_text())}
 def checked_source_metadata(self,name):
  row=self.rows[name];case=self.cases[name]
  capture=OUT/name/"capture.json";raw=capture.read_bytes()
  self.assertEqual(hashlib.sha256(raw).hexdigest(),row["capture_sha256"])
  record=json.loads(raw);receipt=json.loads((OUT/name/"receipt.json").read_text())
  self.assertEqual(receipt["capture_binding"],record["binding"])
  self.assertEqual(record["binding"],digest({k:v for k,v in record.items()if k!="binding"}))
  manifest=ROOT/case["manifest"];spec=json.loads(manifest.read_text())
  self.assertEqual(hashlib.sha256(manifest.read_bytes()).hexdigest(),case["manifest_sha256"])
  self.assertEqual(record["original_manifest_sha256"],digest(spec))
  state=record["original_public_state"];division=record["division_capture"]
  check_original(state,spec["weights"])
  check_trace(state,division["public_state"],sorted(set(state)-set(division["public_state"])))
  selected=direct_divisors(division["source_inventory"],division["public_state"])
  self.assertEqual(set(selected),{e["state_name"]for e in division["entries"]})
  self.assertEqual(selected,{n:division["declared_divisors"][n]for n in selected})
  self.assertEqual(len(division["entries"]),row["division_entries"])
  self.assertTrue(all(selected[n]["dtype"]=="<f8" for n in selected))
  # Projection weights are public state, not tensor division operands.
  if name=="dense_projection":
   self.assertIn("weight",state);self.assertNotIn("weight",selected)
  self.assertEqual(record["provider_calls"],0);self.assertEqual(record["compiler_calls"],0)
  self.assertEqual(record["encrypted_calls"],0)
for name in ("vector_function","channel_method","inplace_snapshot","mixed_scalar_power","dual_fusion","dense_projection"):
 def test(self,case=name):self.checked_source_metadata(case)
 setattr(AuthenticatedSourceMetadataTests,"test_authenticated_"+name,test)
