"""Source CLI uses the same explicit-package policy as trusted intake."""
import importlib.util,json,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
class SourceCLI(unittest.TestCase):
 def setUp(self):
  spec=importlib.util.spec_from_file_location("project_prepare_pytorch_source",ROOT/"scripts/prepare_pytorch_source.py")
  self.cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.cli)
  self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
  self.model=self.root/"model.py"
  self.model.write_text('import torch\nclass Model(torch.nn.Module):\n def forward(self,x):\n  return x+1.\ndef make():\n return Model().eval()\n')
  self.call=dict(format="poseidon-pytorch-call-v1",args=[dict(kind="tensor",name="x")],kwargs={})
 def build(self,version,**kwargs):
  return self.cli.build(self.model,entry="model.make",inputs=["x:2,2"],outputs=["output"],
   source_format=version,**kwargs)
 def nested(self):
  d=self.root/"blocks";d.mkdir();init=d/"__init__.py";init.write_text("")
  helper=d/"ops.py";helper.write_text("def affine(x):\n return x+1.\n")
  self.model.write_text('import torch\nfrom blocks.ops import affine\nclass Model(torch.nn.Module):\n def forward(self,x):\n  return affine(x)\ndef make():\n return Model().eval()\n')
  return (init,helper)
 def test_v29_to_v34_nested_contracts(self):
  helpers=self.nested()
  for n in range(29,35):
   with self.subTest(n=n):
    spec,root=self.build("poseidon-trusted-pytorch-v"+str(n),helpers=helpers,call=self.call,source_root=self.root)
    self.assertEqual(root,self.root);self.assertEqual(spec["format"],"poseidon-trusted-pytorch-v"+str(n))
    self.assertEqual(set(spec["files"]),{"model.py","blocks/__init__.py","blocks/ops.py"})
    self.assertEqual(spec["call"],self.call)
 def test_default_legacy_manifest_exact(self):
  spec,root=self.build(self.cli.DEFAULT_FORMAT)
  import hashlib
  expected=dict(format="poseidon-trusted-pytorch-v18",entry="model.make",files={"model.py":hashlib.sha256(self.model.read_bytes()).hexdigest()},
   inputs=[dict(name="x",shape=[2,2])],outputs=["output"],factory=dict(args=[],kwargs={}),
   dtype_policy="strict-float64-v1",weights={},approximations=dict(format="poseidon-pytorch-approximations-v1",defaults={},modules=[],nodes=[]))
  self.assertEqual(self.cli.canonical(spec),self.cli.canonical(expected))
 def test_default_stays_v18(self):
  self.assertEqual(self.cli.DEFAULT_FORMAT,"poseidon-trusted-pytorch-v18")
 def test_explicit_flat_new_contracts(self):
  for n in (32,33,34):
   with self.subTest(n=n):
    spec,root=self.build("poseidon-trusted-pytorch-v"+str(n),call=self.call)
    self.assertEqual(set(spec["files"]),{"model.py"});self.assertEqual(root,self.root)
 def test_explicit_v28_stays_flat(self):
  with self.assertRaisesRegex(ValueError,"Source root requires"):
   self.build("poseidon-trusted-pytorch-v28",call=self.call,source_root=self.root)
 def test_old_contract_cannot_acquire_packages(self):
  helpers=self.nested()
  with self.assertRaises(ValueError):self.build("poseidon-trusted-pytorch-v18",helpers=helpers,source_root=self.root)
 def test_outside_root_rejected(self):
  with tempfile.TemporaryDirectory()as outside:
   helper=Path(outside)/"ops.py";helper.write_text("def affine(x):\n return x\n")
   with self.assertRaisesRegex(ValueError,"outside"):self.build("poseidon-trusted-pytorch-v34",call=self.call,helpers=(helper,),source_root=self.root)
 def test_nested_symlink_rejected(self):
  d=self.root/"blocks";d.mkdir();real=self.root/"real.py";real.write_text("def affine(x):\n return x\n")
  link=d/"ops.py";link.symlink_to(real)
  with self.assertRaises(ValueError):self.build("poseidon-trusted-pytorch-v34",call=self.call,helpers=(link,),source_root=self.root)
 def test_reserved_module_rejected(self):
  helper=self.root/"torch.py";helper.write_text("x=1\n")
  with self.assertRaises(ValueError):self.build("poseidon-trusted-pytorch-v34",call=self.call,helpers=(helper,),source_root=self.root)
 def test_source_bytes_change_binding(self):
  a,_=self.build("poseidon-trusted-pytorch-v34",call=self.call)
  self.model.write_text(self.model.read_text().replace("x+1.","x+2."))
  b,_=self.build("poseidon-trusted-pytorch-v34",call=self.call)
  self.assertNotEqual(a["files"],b["files"])
 def test_saved_manifest_keeps_root_and_exact_files(self):
  helpers=self.nested();spec,root=self.build("poseidon-trusted-pytorch-v34",call=self.call,helpers=helpers,source_root=self.root)
  output=self.root/"source.json";self.cli.save(spec,output,root)
  self.assertEqual(json.loads(output.read_text()),spec)
  with self.assertRaises(ValueError):self.cli.save(spec,output,root)
 def test_latest_creates_ordered_call(self):
  spec,_=self.build("latest")
  self.assertEqual(spec["call"],self.call);self.assertEqual(spec["format"],self.cli.LATEST_FORMAT)
 def test_latest_selects_v34(self):
  self.assertEqual(self.cli.LATEST_FORMAT,"poseidon-trusted-pytorch-v34")
 def test_latest_and_explicit_v34_match_exactly(self):
  a,_=self.build("latest");b,_=self.build("poseidon-trusted-pytorch-v34",call=self.call)
  self.assertEqual(self.cli.canonical(a),self.cli.canonical(b))
 def test_explicit_v28_manifest_exact(self):
  current,_=self.build("poseidon-trusted-pytorch-v28",call=self.call)
  legacy,_=self.build("poseidon-trusted-pytorch-v18")
  legacy["format"]="poseidon-trusted-pytorch-v28";legacy["call"]=self.call
  self.assertEqual(self.cli.canonical(current),self.cli.canonical(legacy))
 def test_latest_nested_creates_ordered_call(self):
  helpers=self.nested();manifest,root=self.build("latest",helpers=helpers,source_root=self.root)
  self.assertEqual(manifest["call"],self.call)
  self.assertEqual(set(manifest["files"]),{"model.py","blocks/__init__.py","blocks/ops.py"})
 def test_nested_entry_preserved(self):
  d=self.root/"models";d.mkdir();init=d/"__init__.py";init.write_text("")
  main=d/"main.py";main.write_bytes(self.model.read_bytes())
  manifest,root=self.cli.build(main,entry="models.main.make",inputs=["x:2,2"],outputs=["output"],
   helpers=[init],source_format="latest",source_root=self.root)
  self.assertEqual(manifest["entry"],"models.main.make")
  self.assertEqual(set(manifest["files"]),{"models/__init__.py","models/main.py"})
if __name__=="__main__":unittest.main(verbosity=2)
