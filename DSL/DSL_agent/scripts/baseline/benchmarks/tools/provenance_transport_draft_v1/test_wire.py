import unittest,json,copy,math
import wire
from codec import encode,raw
class TransportTests(unittest.TestCase):
 def test_legacy_bytes_identity(self):
  v={"z":[1,2],"a":{"scale":-0.0},"t":"中文"}
  b=raw(v);self.assertEqual(wire.serialize(v,"task-metadata"),b)
  out=wire.parse(b,"task-metadata");self.assertEqual(raw(out),b)
  self.assertEqual(wire.logical_digest(out),wire.logical_digest(v))
 def test_large_repeated_records(self):
  leaf={"weights":[float(i)/16 for i in range(128)],"kind":"public"}
  value={"decomposition":[leaf for _ in range(6000)]}
  self.assertGreater(len(raw(value)),wire.WIRE_LIMIT)
  b=wire.serialize(value,"model-provenance")
  self.assertLess(len(b),wire.WIRE_LIMIT)
  self.assertEqual(raw(wire.parse(b,"model-provenance")),raw(value))
 def test_purpose_mismatch(self):
  p=encode(dict(format=wire.FORMAT,purpose="model-provenance",value={"x":1}))
  with self.assertRaisesRegex(ValueError,"scope"):wire.parse(raw(p),"task-metadata")
 def test_packet_tamper(self):
  p=encode(dict(format=wire.FORMAT,purpose="model-provenance",value={"x":1}))
  p["ordered_sha256"]="0"*64
  with self.assertRaises(ValueError):wire.parse(raw(p),"model-provenance")
 def test_logical_metadata_binding_preserved(self):
  v={"request":{"request_id":"a"*64},"budget":{"generations":4},"source":[{"x":1}]*3}
  v["binding"]=wire.logical_digest(v)
  p=encode(dict(format=wire.FORMAT,purpose="task-metadata",value=v))
  restored=wire.parse(raw(p),"task-metadata")
  self.assertEqual(restored["binding"],wire.logical_digest({k:v for k,v in restored.items()if k!="binding"}))
 def test_plain_duplicate_keys(self):
  with self.assertRaisesRegex(ValueError,"duplicate"):wire.parse(b'{"x":1,"x":2}',"task-metadata")
 def test_plain_nonfinite(self):
  for b in (b'{"x":NaN}',b'{"x":1e999}'):
   with self.assertRaises(ValueError):wire.parse(b,"task-metadata")
 def test_fresh_expanded_containers(self):
  v={"a":[{"x":[1]}],"b":[{"x":[1]}]}
  p=encode(dict(format=wire.FORMAT,purpose="model-provenance",value=v))
  restored=wire.parse(raw(p),"model-provenance")
  restored["a"][0]["x"][0]=2
  self.assertEqual(restored["b"][0]["x"],[1])
 def test_unknown_format_is_not_recursively_decoded(self):
  p=encode(dict(format="unknown-v1",purpose="model-provenance",value={"x":1}))
  with self.assertRaises(ValueError):wire.parse(raw(p),"model-provenance")
 def test_invalid_purpose(self):
  for kind in ("model","task","../task-metadata",None):
   with self.assertRaises(ValueError):wire.serialize({},""+kind if type(kind)is str else kind)
 def test_wire_budget(self):
  with self.assertRaises(ValueError):wire.parse(b" "* (wire.WIRE_LIMIT+1),"task-metadata")
  with self.assertRaises(ValueError):wire.serialize({"long":"abcdef"},"task-metadata",limit=1)
 def test_logical_budget(self):
  with self.assertRaises(ValueError):wire.serialize({"x":["a"*2048]*9000},"model-provenance")
if __name__=="__main__":unittest.main()
