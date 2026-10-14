"""Evidence comparator rejection tests; no tracing or FHE claimed."""
import json,struct,tempfile,unittest
from pathlib import Path
from trace_api_diagnostic import compare_artifacts
from upstream_adapters.constants import canonicalize
class ComparisonTests(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
  self.old=Path(self.temp.name)/"old";self.new=Path(self.temp.name)/"new"
  self.old.mkdir();self.new.mkdir();self.name="_hecate_golden.cst"
  for p in (self.old,self.new):(p/"candidate_trace.mlir").write_text("same IR")
  self.raw=struct.pack("<qq",1,16384)+struct.pack("<4d",1,2,3,4)*4096
  self.canonical,self.record=canonicalize(self.raw,4)
  (self.new/self.name).write_bytes(self.raw)
  (self.old/self.name).write_bytes(self.canonical)
  (self.old/(self.name+".upstream-original")).write_bytes(self.raw)
  self.layout=dict(self.record,original_file=self.name+".upstream-original")
  self.save()
 def save(self):
  (self.old/"constant-layout.json").write_text(json.dumps({self.name:self.layout}))
 def test_lossless_lineage(self):
  comparison,lineage=compare_artifacts(self.new,self.old)
  self.assertTrue(all(comparison.values()));self.assertEqual(lineage["period"],4)
 def test_changed_candidate_constant_rejected(self):
  (self.new/self.name).write_bytes(struct.pack("<qq",1,4)+struct.pack("<4d",9,2,3,4))
  comparison,_=compare_artifacts(self.new,self.old)
  self.assertFalse(comparison["raw_cst"]);self.assertFalse(comparison["executed_cst"])
 def test_executed_constant_tamper_rejected(self):
  (self.old/self.name).write_bytes(self.canonical[:-8]+struct.pack("<d",5))
  with self.assertRaisesRegex(ValueError,"Executed CST"):compare_artifacts(self.new,self.old)
 def test_metadata_tamper_rejected(self):
  self.layout["original_sha256"]="0"*64;self.save()
  with self.assertRaisesRegex(ValueError,"metadata"):compare_artifacts(self.new,self.old)
 def test_missing_original_rejected(self):
  (self.old/(self.name+".upstream-original")).unlink()
  with self.assertRaisesRegex(ValueError,"Missing original"):compare_artifacts(self.new,self.old)
 def test_identity(self):
  (self.old/(self.name+".upstream-original")).unlink();(self.old/"constant-layout.json").unlink()
  (self.old/self.name).write_bytes(self.raw)
  comparison,lineage=compare_artifacts(self.new,self.old)
  self.assertTrue(all(comparison.values()));self.assertEqual(lineage["method"],"identity")
 def test_changed_ir_rejected(self):
  (self.new/"candidate_trace.mlir").write_text("different IR")
  comparison,_=compare_artifacts(self.new,self.old);self.assertFalse(comparison["candidate_trace.mlir"])
if __name__=="__main__":unittest.main()
