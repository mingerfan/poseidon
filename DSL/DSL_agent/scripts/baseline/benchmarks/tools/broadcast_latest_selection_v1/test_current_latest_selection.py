"""Latest manifest selection checks. No source import, Provider or FHE."""
import copy,contextlib,importlib.util,io,json,tempfile,unittest
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5])
spec=importlib.util.spec_from_file_location("poseidon_latest_source_selection",ROOT/"scripts/prepare_pytorch_source.py")
cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)
class LatestSelection(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
  self.root=Path(self.temp.name).resolve();self.source=self.root/"model.py"
  self.source.write_text("raise RuntimeError('manifest preparation must not import this file')\n")
 def build(self,**changes):
  opts=dict(entry="model.make",inputs=["x:1,4"],outputs=["output"],source_format="latest")
  opts.update(changes);return cli.build(self.source,**opts)[0]
 def call(self,names):
  return dict(format="poseidon-pytorch-call-v1",args=[dict(kind="tensor",name=n)for n in names],kwargs={})
 def test_latest_is_registered_v28(self):
  self.assertEqual(cli.resolve_source_format("latest"),"poseidon-trusted-pytorch-v28")
  self.assertIn(cli.LATEST_FORMAT,cli.SOURCE_SELECTIONS)
 def test_single_input_flat_call(self):
  self.assertEqual(self.build()["call"],self.call(["x"]))
 def test_multiple_inputs_keep_declared_order(self):
  self.assertEqual(self.build(inputs=["rhs:2","lhs:2"])["call"],self.call(["rhs","lhs"]))
 def test_four_input_boundary(self):
  self.assertEqual(self.build(inputs=["a:1","b:1","c:1","d:1"])["call"],self.call(["a","b","c","d"]))
 def test_latest_and_explicit_v28_are_identical(self):
  self.assertEqual(self.build(),self.build(source_format="poseidon-trusted-pytorch-v28",call=self.call(["x"])))
 def test_latest_preserves_explicit_keyword_binding(self):
  binding=dict(format="poseidon-pytorch-call-v1",args=[],kwargs={"data":dict(kind="tensor",name="x")})
  self.assertEqual(self.build(call=binding)["call"],binding)
 def test_latest_preserves_nested_dictionary_binding(self):
  binding=dict(format="poseidon-pytorch-call-v1",args=[dict(kind="dict",items=[["image",dict(kind="tensor",name="x")]])],kwargs={})
  self.assertEqual(self.build(call=binding)["call"],binding)
 def test_invalid_explicit_call_does_not_fallback(self):
  invalid=self.call(["missing"])
  with self.assertRaisesRegex(ValueError,"tensor_binding"):self.build(call=invalid)
 def test_duplicate_tensor_binding_does_not_fallback(self):
  with self.assertRaisesRegex(ValueError,"missing_or_duplicate"):self.build(call=self.call(["x","x"]))
 def test_explicit_v26_and_v27_still_require_call(self):
  for version in ("poseidon-trusted-pytorch-v26","poseidon-trusted-pytorch-v27","poseidon-trusted-pytorch-v28"):
   with self.subTest(version=version),self.assertRaisesRegex(ValueError,"requires --call"):
    self.build(source_format=version)
 def test_explicit_v21_keeps_flat_manifest(self):
  result=self.build(source_format="poseidon-trusted-pytorch-v21")
  self.assertNotIn("call",result);self.assertEqual(result["format"],"poseidon-trusted-pytorch-v21")
 def test_legacy_default_stays_v18(self):
  self.assertEqual(cli.DEFAULT_FORMAT,"poseidon-trusted-pytorch-v18")
  result=cli.build(self.source,entry="model.make",inputs=["x:1,4"],outputs=["output"])[0]
  self.assertEqual(result["format"],cli.DEFAULT_FORMAT);self.assertNotIn("call",result)
 def test_unknown_version_is_rejected(self):
  with self.assertRaisesRegex(ValueError,"Source version selection"):self.build(source_format="poseidon-trusted-pytorch-v999")
 def test_capacity_still_limits_inputs(self):
  with self.assertRaisesRegex(ValueError,"Input inventory"):self.build(inputs=[str(n)+":1"for n in "abcde"])
 def test_duplicate_input_name_is_rejected(self):
  with self.assertRaisesRegex(ValueError,"Input inventory"):self.build(inputs=["x:1","x:1"])
 def test_latest_cli_does_not_import_model(self):
  out=io.StringIO()
  args=[str(self.source),"--entry","model.make","--input","x:1,4","--output","output","--source-format","latest"]
  with contextlib.redirect_stdout(out):self.assertEqual(cli.main(args),0)
  value=json.loads(out.getvalue());self.assertEqual(value["format"],cli.LATEST_FORMAT)
  self.assertEqual(value["call"],self.call(["x"]))
 def test_saved_manifest_records_explicit_version(self):
  value=self.build();path=cli.save(value,self.root/"source.json",self.root)
  loaded=json.loads(path.read_text());self.assertEqual(loaded,value);self.assertNotEqual(loaded["format"],"latest")
 def test_saved_manifest_not_overwritten(self):
  path=self.root/"source.json";path.write_text("preserved")
  with self.assertRaisesRegex(ValueError,"already exists"):cli.save(self.build(),path,self.root)
  self.assertEqual(path.read_text(),"preserved")
if __name__=="__main__":unittest.main(verbosity=2)
