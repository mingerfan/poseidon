"""Transport resource and identity controls; no model execution or Provider calls."""
import copy,json,tempfile,unittest
from pathlib import Path
import provenance_json_transport as t
class TransportTests(unittest.TestCase):
 def value(self,n=1):
  leaf={"weights":[float(i)/16 for i in range(128)],"kind":"public"}
  return {"original":{"format":"test"},"decomposition":[leaf]*n,
          "reference_report":{},"trusted_pytorch":{"manifest":{"format":t.SOURCE_FORMAT}}}
 def test_full_large_provenance(self):
  value=self.value(6000);self.assertGreater(len(t.raw(value)),t.WIRE_BYTES)
  encoded=t.serialize(value,purpose="model-provenance")
  self.assertLess(len(encoded),t.WIRE_BYTES)
  restored=t.unpack(json.loads(encoded),purpose="model-provenance")
  self.assertEqual(t.raw(restored),t.raw(value))
 def test_metadata_binding_preserved(self):
  value={"request":{"request_id":"a"*64},"model_provenance":self.value(6000),"budget":{"generations":4}}
  value["binding"]=t.sha(value,sorted_keys=True)
  encoded=t.task_bytes(value);restored=t.unpack(json.loads(encoded),purpose="application-task")
  self.assertEqual(t.raw(restored),t.raw(value))
  self.assertEqual(restored["binding"],t.sha({k:v for k,v in restored.items()if k!="binding"},sorted_keys=True))
 def test_fresh_decoded_records(self):
  v={"a":[{"x":[1]}],"b":[{"x":[1]}]}
  restored=t.decode(t.encode(v,purpose="decomposition"),purpose="decomposition")
  restored["a"][0]["x"][0]=2;self.assertEqual(restored["b"][0]["x"],[1])
 def test_legacy_task_wire_not_expanded(self):
  value={"model_provenance":self.value(6000)}
  value["model_provenance"]["trusted_pytorch"]["manifest"]["format"]="poseidon-trusted-pytorch-v35"
  with self.assertRaises(ValueError):t.task_bytes(value)
 def test_legacy_part_representation(self):
  value={"x":[1,2]}
  self.assertEqual(t.part_bytes(value,"capture-reference",{"format":"poseidon-trusted-pytorch-v35"},4096),
                   json.dumps(value,allow_nan=False).encode())
 def test_unknown_purpose_plain(self):
  with self.assertRaises(ValueError):t.serialize({},purpose="unknown")
 def test_part_contract(self):
  raw=t.raw(t.encode({"reference":"test"},purpose="capture-reference"))
  self.assertEqual(t.parse_part(raw,purpose="capture-reference",spec={"format":t.SOURCE_FORMAT},limit=t.WIRE_BYTES),{"reference":"test"})
  with self.assertRaises(ValueError):t.parse_part(raw,purpose="capture-reference",spec={"format":"poseidon-trusted-pytorch-v35"},limit=t.WIRE_BYTES)
 def test_wrong_purpose(self):
  packet=t.encode(self.value(),purpose="model-provenance")
  with self.assertRaises(ValueError):t.unpack(packet,purpose="application-task")
 def test_packet_non_v36_contract(self):
  value=self.value();value["trusted_pytorch"]["manifest"]["format"]="poseidon-trusted-pytorch-v35"
  with self.assertRaises(ValueError):t.unpack(t.encode(value,purpose="model-provenance"),purpose="model-provenance")
 def test_packet_binding_tamper(self):
  p=t.encode(self.value(),purpose="model-provenance");p["logical_bytes"]+=1
  with self.assertRaises(ValueError):t.unpack(p,purpose="model-provenance")
 def test_legacy_canonical_identity(self):
  value={"z":[1,2],"a":{"scale":-0.0},"t":"中文"}
  raw=t.raw(value);self.assertEqual(t.serialize(value,purpose="decomposition"),raw)
  restored=t.parse_part(raw,purpose="decomposition",spec={"format":t.SOURCE_FORMAT},limit=t.WIRE_BYTES)
  self.assertEqual(t.raw(restored),raw);self.assertEqual(t.sha(value,sorted_keys=True),t.sha(restored,sorted_keys=True))
 def test_duplicate_json(self):
  with self.assertRaises(ValueError):t.parse_part(b'{"x":1,"x":2}',purpose="decomposition",spec={"format":t.SOURCE_FORMAT},limit=t.WIRE_BYTES)
 def test_nonfinite_json(self):
  for raw in (b'{"x":NaN}',b'{"x":1e999}'):
   with self.assertRaises(ValueError):t.parse_part(raw,purpose="decomposition",spec={"format":t.SOURCE_FORMAT},limit=t.WIRE_BYTES)
 def test_scope_budget_not_arbitrary(self):
  for limit in (0,True,t.WIRE_BYTES+1):
   with self.assertRaises(ValueError):t.serialize({},purpose="decomposition",limit=limit)
 def test_wire_capacity(self):
  with self.assertRaises(ValueError):t.serialize({"long":"abcdef"},purpose="decomposition",limit=1)
 def test_logical_capacity(self):
  with self.assertRaises(ValueError):t.serialize({"x":["a"*2048]*9000},purpose="decomposition")
 def test_role_fields_closed(self):
  p=t.encode(self.value(),purpose="model-provenance");p["unused"]=True
  with self.assertRaises(ValueError):t.unpack(p,purpose="model-provenance")
 def test_forward_reference(self):
  p=t.encode({"x":[]},purpose="decomposition");p["nodes"][0][1]=[["ref",1]]
  p["binding"]=t.sha({k:v for k,v in p.items()if k!="binding"},sorted_keys=True)
  with self.assertRaises(ValueError):t.decode(p,purpose="decomposition")
 def test_body_alias_not_shared(self):
  v={"model_provenance":self.value(6000)}
  packed=json.loads(t.task_bytes(v));out=t.unpack(packed,purpose="application-task")
  out["model_provenance"]["decomposition"][0]["weights"][0]=9
  self.assertEqual(out["model_provenance"]["decomposition"][1]["weights"][0],0)
 def test_small_limit_no_large_work_expansion(self):
  with self.assertRaises(ValueError):t.serialize({"x":["a"*100]*20},purpose="decomposition",limit=100)
 def test_nested_envelope_rejected(self):
  inner=t.encode({"x":1},purpose="decomposition")
  outer=t.encode(inner,purpose="decomposition")
  with self.assertRaises(ValueError):t.unpack(outer,purpose="decomposition")
if __name__=="__main__":unittest.main()
