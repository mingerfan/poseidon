import copy,json,math,tempfile,unittest
from pathlib import Path
from benchmark_graph import canonical,digest
from candidate_bundle import manifest_bytes,dump_manifest
from strict_json_files import dump,strict_file

class BundleManifestStorageTests(unittest.TestCase):
 def test_legacy_small_bytes_are_identical(self):
  value=dict(format="poseidon-dsl-bundle-v6",files={"c.py":"a"*64},binding="b"*64)
  with tempfile.TemporaryDirectory()as d:
   a,b=Path(d)/"a.json",Path(d)/"b.json"
   dump(a,value);dump_manifest(b,value)
   self.assertEqual(a.read_bytes(),b.read_bytes())
 def test_many_artifacts_compact_without_changing_identity_or_content(self):
  value=dict(files={"artifacts/"+str(i)+".json":"a"*64 for i in range(1450)},dependencies={"v":3})
  value["binding"]=digest(value);before=copy.deepcopy(value)
  pretty=json.dumps(value,indent=2,sort_keys=True).encode()+b"\n"
  self.assertGreater(len(pretty),131072)
  raw=manifest_bytes(value);self.assertLessEqual(len(raw),131072)
  decoded=json.loads(raw);self.assertEqual(decoded,before);self.assertEqual(value,before)
  self.assertEqual(decoded["binding"],digest({k:v for k,v in decoded.items()if k!="binding"}))
 def test_exact_compact_boundary_and_one_byte_over(self):
  value={"padding":""};overhead=len(canonical(value))+1
  value["padding"]="x"*(131072-overhead)
  raw=manifest_bytes(value);self.assertEqual(len(raw),131072)
  value["padding"]+="x"
  with self.assertRaisesRegex(ValueError,"manifest size budget"):manifest_bytes(value)
 def test_nonfinite_and_nonmapping_are_rejected(self):
  for v in (float("nan"),float("inf"),-float("inf")):
   with self.assertRaises(ValueError):manifest_bytes({"v":v})
  for v in (None,[],1):
   with self.assertRaises(ValueError):manifest_bytes(v)
 def test_atomic_writer_preserves_strict_reader_and_no_partial_on_oversize(self):
  with tempfile.TemporaryDirectory()as d:
   p=Path(d)/"manifest.json";v={"items":[{"x":"y"*40}for _ in range(2200)]}
   dump_manifest(p,v);self.assertEqual(strict_file(p,131072),v)
   original=p.read_bytes()
   with self.assertRaises(ValueError):dump_manifest(p,{"padding":"x"*131072})
   self.assertEqual(p.read_bytes(),original);self.assertFalse(p.with_name(p.name+".partial").exists())
if __name__=="__main__":unittest.main()
