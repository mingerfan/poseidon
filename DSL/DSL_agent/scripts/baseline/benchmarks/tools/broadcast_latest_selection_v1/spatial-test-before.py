"""Pure file/manifest CLI checks; no model import, Provider, compiler or runtime."""
import contextlib,copy,importlib.util,io,json,tempfile,unittest
from pathlib import Path
from pytorch_spatial_policy import SOURCE_FORMAT,INTERNAL_FORMATS
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("poseidon_test_pytorch_source_cli",ROOT/"prepare_pytorch_source.py")
cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)
class SpatialCliTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name).resolve();self.model=self.root/"model.py"
        self.model.write_text("raise RuntimeError('source import forbidden in manifest preparation')\n")
        self.call=dict(format="poseidon-pytorch-call-v1",args=[dict(kind="tensor",name="x")],kwargs={})
    def build(self,**kw):
        args=dict(entry="model.make",inputs=["x:1,1,3"],outputs=["output"],source_format=SOURCE_FORMAT,call=self.call)
        args.update(kw);return cli.build(self.model,**args)[0]
    def test_v27_flat_binding_and_default_versions(self):
        value=self.build();self.assertEqual(value["call"],self.call);self.assertEqual(value["format"],SOURCE_FORMAT)
        self.assertEqual(cli.DEFAULT_FORMAT,"poseidon-trusted-pytorch-v18")
        self.assertEqual(cli.resolve_source_format("latest"),"poseidon-trusted-pytorch-v27")
    def test_v27_keyword_dictionary_calls(self):
        binding=dict(format=self.call["format"],args=[],kwargs=dict(
            batch=dict(kind="dict",items=[["value",dict(kind="tensor",name="x")]])))
        self.assertEqual(self.build(call=binding)["call"],binding)
    def test_v26_output_is_identical(self):
        value=self.build(source_format="poseidon-trusted-pytorch-v26")
        self.assertEqual(value,dict(self.build(),format="poseidon-trusted-pytorch-v26"))
    def test_v27_and_v26_require_explicit_binding(self):
        for version in(SOURCE_FORMAT,"poseidon-trusted-pytorch-v26"):
            with self.subTest(version=version),self.assertRaisesRegex(ValueError,"requires --call"):
                self.build(source_format=version,call=None)
    def test_older_source_does_not_receive_structured_calls(self):
        with self.assertRaisesRegex(ValueError,"explicit v26"):self.build(source_format="poseidon-trusted-pytorch-v25")
        value=self.build(source_format="poseidon-trusted-pytorch-v25",call=None)
        self.assertNotIn("call",value)
    def test_private_transports_are_not_cli_choices(self):
        for version in INTERNAL_FORMATS:
            self.assertNotIn(version,cli.SOURCE_SELECTIONS)
            with self.assertRaisesRegex(ValueError,"Source version selection"):self.build(source_format=version)
    def test_invalid_binding_fails_before_manifest_save(self):
        invalid=copy.deepcopy(self.call);invalid["args"][0]["name"]="unknown"
        with self.assertRaises(ValueError):self.build(call=invalid)
    def test_main_prints_v27_and_preserves_source_inertness(self):
        call=self.root/"call.json";call.write_text(json.dumps(self.call))
        args=[str(self.model),"--entry","model.make","--input","x:1,1,3","--output","output",
              "--source-format",SOURCE_FORMAT,"--call",str(call)]
        out=io.StringIO()
        with contextlib.redirect_stdout(out):self.assertEqual(cli.main(args),0)
        value=json.loads(out.getvalue());self.assertEqual(value["format"],SOURCE_FORMAT)
        self.assertEqual(value["call"],self.call);self.assertEqual(list(value["files"]),["model.py"])
    def test_existing_manifest_is_never_overwritten(self):
        target=self.root/"source.json";target.write_text("preserved")
        with self.assertRaisesRegex(ValueError,"already exists"):cli.save(self.build(),target,self.root)
        self.assertEqual(target.read_text(),"preserved")
if __name__=="__main__":unittest.main()
