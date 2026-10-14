"""Explicit package task inventory tests; no captured model or backend execution."""
import copy,os,tempfile,unittest
from pathlib import Path
from benchmark_graph import digest
import test_pytorch_source_package_intake as fixtures
from application_model_source import FORMAT
from test_application_model_source import Backend,SourceApplication
class PackageTaskInventory(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
  self.files=fixtures.sources();self.spec=fixtures.manifest(self.files)
  for name,raw in self.files.items():
   p=self.root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
  self.manifest=self.root/"manifest.json";self.manifest.write_bytes(fixtures.canonical(self.spec))
  self.backend=Backend();self.app=SourceApplication(self.root/"tasks",self.backend)
  self.task=self.app.submit_source(dict(format=FORMAT,kind="trusted_pytorch",manifest=str(self.manifest),
   trust_local_code=True),resource_profile="general-tensors-v1")
  self.path=self.app._dir(self.task);self.source=self.path/"source"
 def tearDown(self):self.tmp.cleanup()
 def rejected(self):
  state=self.app.prepare_source(self.task)
  self.assertEqual((state["status"],state["failure"]["layer"]),("service_error","integrity"))
  self.assertEqual((state["generations"],state["http_attempts"],self.backend.calls),(0,0,0))
  self.assertFalse((self.path/"source-import.log").exists())
 def test_nested_and_empty_initializer_valid(self):
  value=self.app._source(self.path)
  self.assertEqual(value["files"]["models/__init__.py"]["bytes"],0)
  self.assertEqual(self.backend.calls,0)
 def test_extra_nested_file_rejected(self):
  (self.source/"models/unlisted.py").write_text("raise RuntimeError('inert')")
  self.rejected()
 def test_extra_empty_directory_rejected(self):
  (self.source/"models/unlisted").mkdir();self.rejected()
 def test_nested_content_tamper_rejected(self):
  (self.source/"models/layers.py").write_text("print('not executed')");self.rejected()
 def test_initializer_content_tamper_rejected(self):
  (self.source/"models/__init__.py").write_text("print('not executed')");self.rejected()
 def test_missing_initializer_rejected(self):
  (self.source/"models/__init__.py").unlink();self.rejected()
 def test_source_symlink_rejected(self):
  p=self.source/"models/layers.py";p.unlink();p.symlink_to(self.root/"models/layers.py");self.rejected()
 def test_source_hardlink_rejected(self):
  p=self.source/"models/layers.py";p.unlink();os.link(self.root/"models/layers.py",p);self.rejected()
 def test_directory_symlink_rejected(self):
  moved=self.source/"moved";(self.source/"models").rename(moved)
  (self.source/"models").symlink_to(moved,target_is_directory=True);self.rejected()
 def test_manifest_symlink_rejected(self):
  p=self.source/"manifest.json";p.unlink();p.symlink_to(self.manifest);self.rejected()
 def test_rebound_path_traversal_metadata_rejected(self):
  p=self.path/"source.json";v=self.app._load(p)
  v["files"]["../outside.py"]=v["files"].pop("models/layers.py")
  v.pop("binding");v["binding"]=digest(v);self.app._write(p,v);self.rejected()
 def test_rebound_incorrect_zero_length_rejected(self):
  p=self.path/"source.json";v=self.app._load(p)
  v["files"]["models/layers.py"]["bytes"]=0
  v.pop("binding");v["binding"]=digest(v);self.app._write(p,v);self.rejected()
 def test_flat_contract_cannot_adopt_nested_paths(self):
  m=copy.deepcopy(self.spec);m["format"]="poseidon-trusted-pytorch-v28"
  (self.source/"manifest.json").write_bytes(fixtures.canonical(m));self.rejected()
