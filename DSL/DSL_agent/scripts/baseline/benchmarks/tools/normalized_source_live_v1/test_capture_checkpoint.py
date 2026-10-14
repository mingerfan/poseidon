import copy,unittest
from pathlib import Path
import capture_checkpoint as checkpoint
import scope
class CaptureCheckpoint(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.here=Path(__file__).resolve().parent
  cls.c=scope.read(cls.here/"capture-checkpoint-first.json")
  cls.inv=scope.read(cls.here.parent/"pending_source_live_inventory_v1/inventory.json")
  cls.sdk=dict(manifest_sha256=cls.c["source_stage_identity_sha256"])
 def valid(self,c):
  return checkpoint.validate(c,self.inv,self.sdk,here=self.here,results=scope.RESULTS)
 def rejected(self,change):
  c=copy.deepcopy(self.c);change(c);c["binding"]=scope.digest({k:v for k,v in c.items()if k!="binding"})
  with self.assertRaises(ValueError):self.valid(c)
 def test_original(self):self.assertEqual(len(self.valid(self.c)),49)
 def test_increased_deadline(self):self.rejected(lambda c:c["continuation"].update(wall_seconds=181))
 def test_wrong_identity(self):self.rejected(lambda c:c.update(source_stage_identity_sha256="0"*64))
 def test_omitted_file(self):self.rejected(lambda c:c["files"].pop(next(iter(c["files"]))))
 def test_changed_file(self):self.rejected(lambda c:c["files"].update({next(iter(c["files"])):"0"*64}))
 def test_wrong_context_order(self):self.rejected(lambda c:c["contexts"].reverse())
 def test_wrong_source_directory(self):self.rejected(lambda c:c["contexts"][0].update(source_folder=str(scope.RESULTS)))
 def test_nonoriginal_log(self):self.rejected(lambda c:c.update(failure_log_sha256="0"*64))
class ExtraCheckpoint(CaptureCheckpoint):
 def valid(self,c):
  return checkpoint.validate_extra(c,self.inv,self.sdk,here=self.here,results=scope.RESULTS)
 @classmethod
 def setUpClass(cls):
  super().setUpClass();cls.c=scope.read(cls.here/"capture-checkpoint-extra.json")
 def test_original(self):self.assertEqual(len(self.valid(self.c)),1)
 def test_increased_deadline(self):self.rejected(lambda c:c.update(context="other"))
 def test_wrong_context_order(self):self.rejected(lambda c:c.update(task_id="0"*32))
 def test_wrong_source_directory(self):self.rejected(lambda c:c.update(source_folder=str(scope.RESULTS)))
if __name__=="__main__":unittest.main()
