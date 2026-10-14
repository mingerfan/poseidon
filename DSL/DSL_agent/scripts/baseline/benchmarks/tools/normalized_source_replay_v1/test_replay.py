import copy,time,unittest
from unittest.mock import patch
import replay,replay_scope as scope
class ReplayBoundary(unittest.TestCase):
 def setUp(self):
  self.plan=dict(deadline=time.time()+60,initial_bytes=0)
  self.p=dict(tool_hashes=scope.tool_hashes())
 def run_boundary(self,*,total=0,free=None):
  free=scope.BOUNDS["minimum_free_bytes"]if free is None else free
  with patch.object(replay.supervision,"measure",return_value=total),patch.object(replay.shutil,"disk_usage")as disk:
   disk.return_value.free=free;replay.boundary(self.plan,self.p)
 def test_original_boundary(self):self.run_boundary()
 def test_expired_deadline(self):
  self.plan["deadline"]=time.time()-1
  with self.assertRaisesRegex(ValueError,"time_limit"):self.run_boundary()
 def test_global_storage_ceiling(self):
  self.plan["initial_bytes"]=scope.BOUNDS["global_result_bytes"]
  with self.assertRaisesRegex(ValueError,"storage_limit"):self.run_boundary(total=scope.BOUNDS["global_result_bytes"]+1)
 def test_new_batch_ceiling(self):
  with self.assertRaisesRegex(ValueError,"storage_limit"):self.run_boundary(total=scope.BOUNDS["new_result_bytes"]+1)
 def test_free_space_reserve(self):
  with self.assertRaisesRegex(ValueError,"storage_limit"):self.run_boundary(free=scope.BOUNDS["minimum_free_bytes"]-1)
 def test_changed_controller(self):
  self.p["tool_hashes"]=dict(self.p["tool_hashes"]);self.p["tool_hashes"]["changed"]="0"*64
  with self.assertRaisesRegex(ValueError,"tools_changed"):self.run_boundary()
if __name__=="__main__":unittest.main()
