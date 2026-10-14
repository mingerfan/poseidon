"""Pure cutover gates. No source mutation or execution."""
import copy,unittest
from guarded_apply import eligible
class BoundaryTests(unittest.TestCase):
 def setUp(self):
  self.root=dict(status="complete",planned=32,groups=[
   dict(id="matrix_instance60",status="complete",exit_code=0),dict(id="attention1",status="complete",exit_code=0)])
  self.groups={name:dict(status="complete",planned=n,not_run=0,passed=n-1,failed=1,
     rows=[dict(status="passed",result=dict(phase="complete")) for _ in range(n)]) for name,n in (("matrix_instance60",31),("attention1",1))}
 def test_closed_scope_can_reach_source_checks(self):self.assertTrue(eligible(self.root,self.groups))
 def test_running_batch_never_applies(self):
  self.root["status"]="running";self.assertFalse(eligible(self.root,self.groups))
 def test_provider_stop_with_pending_scope_never_applies(self):
  self.root["status"]="stopped";self.assertFalse(eligible(self.root,self.groups))
 def test_unstarted_attention_never_applies(self):
  del self.groups["attention1"];self.assertFalse(eligible(self.root,self.groups))
 def test_active_task_even_in_closed_report_rejected(self):
  self.groups["matrix_instance60"]["rows"][0]["result"]["phase"]="validating"
  self.assertFalse(eligible(self.root,self.groups))
 def test_duplicate_group_record_rejected(self):
  self.root["groups"].append(copy.deepcopy(self.root["groups"][0]));self.assertFalse(eligible(self.root,self.groups))
 def test_bad_denominator_rejected(self):
  self.groups["matrix_instance60"]["not_run"]=1;self.assertFalse(eligible(self.root,self.groups))
if __name__=="__main__":unittest.main(verbosity=2)
