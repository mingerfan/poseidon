"""Admission and cumulative-timing regressions, with local process identity simulated only."""
import json,tempfile,unittest,os
from pathlib import Path
from unittest.mock import patch
from campaign_live_state import sealed
from benchmark_runner import dump
from stage2_agent_pilot_plan import sha
import remaining_campaign_queue_v2 as queue
from remaining_campaign_policy_v2 import cumulative_elapsed

class AdmissionTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory(dir=Path(__file__).parent);self.addCleanup(self.tmp.cleanup)
  self.root=Path(self.tmp.name).resolve();self.path=self.root/"plan.json";self.output=self.root/"output"
  self.child=sealed(dict(source_hashes={"runtime":"frozen"},cases=[{"id":"one"}]))
  self.childpath=self.root/"child.json";dump(self.childpath,self.child)
  self.parent=sealed(dict(source_hashes=self.child["source_hashes"],runner_sha256=sha(Path(queue.__file__)),
   shards=[dict(binding=self.child["binding"],output=str(self.output),plan=str(self.childpath),sha256=sha(self.childpath))],
   api_workers=10,native_workers=2,compile_jobs=2,link_jobs=1,max_retained_mib=32768,min_free_mib=4096))
  dump(self.path,self.parent);dump(self.root/"admission-verified.json",queue.admission_proof(self.parent,self.path))
  self.argv=[b"python",b"remaining_campaign_queue_v2.py",b"--output",str(self.root).encode(),
   b"--approve-live-binding",self.parent["binding"].encode(),b"--inside"]
 def verify(self,output=None):
  original=Path.read_bytes
  args=b"\0".join(self.argv)+b"\0"
  def read(path):
   if str(path)=="/proc/123/cmdline":return args
   return original(path)
  with patch.object(queue.os,"getppid",return_value=123),patch.object(Path,"read_bytes",read):
   return queue.verify_child_parent(self.parent,self.child,self.path,output or self.output)
 def test_valid_admission(self):self.verify()
 def test_missing_admission_rejected(self):
  (self.root/"admission-verified.json").unlink()
  with self.assertRaisesRegex(ValueError,"Benchmark file gate"):self.verify()
 def test_wrong_parent_command_rejected(self):
  self.argv[-2]=b"wrong"
  with self.assertRaises(ValueError):self.verify()
 def test_wrong_output_rejected(self):
  with self.assertRaises(ValueError):self.verify(self.root/"other")
 def test_changed_child_file_rejected(self):
  self.childpath.write_text("{}")
  with self.assertRaises(ValueError):self.verify()
 def test_missing_verified_shard_rejected(self):
  proof=queue.admission_proof(self.parent,self.path);proof.pop("binding");proof["verified_shards"]=[]
  dump(self.root/"admission-verified.json",sealed(proof))
  with self.assertRaises(ValueError):self.verify()
 def test_changed_source_rejected(self):
  self.child["source_hashes"]={"runtime":"changed"}
  with self.assertRaises(ValueError):self.verify()

class ClockTests(unittest.TestCase):
 def test_startup_consumes_remaining_time(self):
  self.assertEqual(cumulative_elapsed(3500,100,120),3520)
  self.assertEqual(cumulative_elapsed(3500,100,200),3600)
 def test_prior_approved_extension_clock_preserved(self):
  self.assertEqual(cumulative_elapsed(4167,100,140),4207)
 def test_backwards_clock_rejected(self):
  with self.assertRaises(ValueError):cumulative_elapsed(3500,100,99)

if __name__=="__main__":unittest.main(verbosity=2)
