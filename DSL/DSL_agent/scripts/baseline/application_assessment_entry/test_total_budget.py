"""CLI deadline/cancel controls, including two real owned process trees; no FHE/API."""
import json,os,signal,sys,threading,time,unittest
from pathlib import Path
from unittest.mock import patch
from application_assessment_entry import cli
from application_assessment import read
from application_assessment_entry import test_cli as fixtures
from application_assessment_entry.supervision import same_process
class TotalBudgetCliTests(unittest.TestCase):
 setUp=fixtures.CliTests.setUp
 ready=fixtures.CliTests.ready
 prepared=fixtures.CliTests.prepared
 setup_entry=fixtures.CliTests.setup_entry
 factory=fixtures.CliTests.factory
 call=fixtures.CliTests.call
 submit=fixtures.CliTests.submit
 action=fixtures.CliTests.action
 def test_optional_duration_is_passed_only_when_selected(self):
  self.setup_entry();_,r=self.submit()
  old=read(self.control.store/r["handle"]["evaluation_id"]/"plan.json")
  self.assertEqual(old["format"],"poseidon-application-assessment-plan-v1")
  _,r=self.submit(("--total-wall-seconds","30","--max-new-assessments","1"))
  plan=read(self.control.store/r["handle"]["evaluation_id"]/"plan.json")
  self.assertEqual(plan["format"],"poseidon-application-assessment-plan-v2")
  self.assertEqual(plan["time_budget"]["seconds"],30);self.assertEqual(self.backend.execute_calls,0)
 def test_advance_cannot_change_total_duration_and_invalid_duration_has_no_factory(self):
  self.setup_entry()
  for n in ("0","14401","-1","true","1.0"):
   code,_=self.submit(("--total-wall-seconds",n));self.assertEqual(code,2)
  self.assertEqual(self.factories,0)
  _,r=self.submit(("--total-wall-seconds","30"))
  before=self.factories
  code,_=self.action("advance",r["handle"],("--total-wall-seconds","600"))
  self.assertEqual(code,2);self.assertEqual(self.factories,before)
 def test_new_feedback_codes_have_stable_exit_status_and_no_private_detail(self):
  for code,status,exitcode in (
   ("assessment_entry_total_timed_out","timed_out",4),
   ("assessment_entry_assessment_cancelled","cancelled",4),
   ("assessment_entry_clock_unavailable","service_error",5),
   ("assessment_entry_control_rejected","service_error",5)):
   r=cli.feedback_response("advance",cli.EntryError(code))
   self.assertEqual((r["status"],cli.exit_status(r)),(status,exitcode));self.assertNotIn("PRIVATE",json.dumps(r))
 def test_invalid_watch_handle_is_rejected_without_a_worker(self):
  self.setup_entry()
  args=cli.parse(self.args+["advance","--evaluation-id","0"*32,"--submission-binding","0"*64])
  with patch.object(cli.subprocess,"Popen",side_effect=AssertionError("worker admitted")):
   r=cli.launch(args,[],Path(self.tmp.name)/"invalid-watch-results")
  self.assertEqual(r["failure"]["code"],"assessment_entry_control_rejected")
  self.assertTrue(r["process_cleanup"]["confirmed"]);self.assertEqual(r["process_cleanup"]["status"],"not_started")
 def _worker_command(self,command,handle,marker):
  statefile=self.control.store/handle["evaluation_id"]/"state.json"
  lockfile=self.control.store/handle["evaluation_id"]/"worker.lock"
  child=("import json,os,time;from pathlib import Path;f=Path('/proc').joinpath(str(os.getpid()),'stat').read_text().rsplit(')',1)[1].split();"
   +"Path("+repr(str(marker))+").write_text(json.dumps({'pid':os.getpid(),'start':int(f[19])}));time.sleep(30)")
  worker=("import json,os,hashlib,fcntl,subprocess,sys,time;from pathlib import Path;"
   +"lock=open("+repr(str(lockfile))+",'r+');fcntl.flock(lock,fcntl.LOCK_EX);"
   +"p=Path("+repr(str(statefile))+");s=json.loads(p.read_text());"
   +"s.update(used=1,status='running',ledger=[dict(number=1,part_id='part_0',phase='started',outcome=None)]);"
   +"s.pop('binding');s['binding']=hashlib.sha256(json.dumps(s,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest();"
   +"tmp=p.with_name('.stub-state');tmp.write_text(json.dumps(s));os.chmod(tmp,0o600);tmp.replace(p);"
   +"subprocess.Popen([sys.executable,'-B','-c',"+repr(child)+"],start_new_session=True);time.sleep(30)")
  return command[:5]+[sys.executable,"-B","-c",worker]
 def _cleanup(self,marker):
  if marker.exists():
   v=json.loads(marker.read_text())
   if same_process(v["pid"],v["start"]):os.kill(v["pid"],signal.SIGKILL)
 def _launch_blocked_worker(self,seconds,*,cancel):
  self.setup_entry();_,response=self.submit(("--total-wall-seconds",str(seconds),"--max-new-assessments","1"))
  h=response["handle"];marker=Path(self.tmp.name)/"owned-child.json";self.addCleanup(self._cleanup,marker)
  args=cli.parse(self.args+["--timeout","8","advance","--evaluation-id",h["evaluation_id"],"--submission-binding",h["binding"]])
  original=cli.subprocess.Popen;children=[];cancelled=[]
  def spawn(command,**kwargs):
   child=original(self._worker_command(command,h,marker),**kwargs);children.append(child);return child
  thread=None
  if cancel:
   def request():
    end=time.monotonic()+5
    while not marker.exists() and time.monotonic()<end:time.sleep(.01)
    if marker.exists():cancelled.append(self.control.cancel(h))
   thread=threading.Thread(target=request);thread.start()
  start=time.monotonic()
  with patch.object(cli.subprocess,"Popen",side_effect=spawn):
   r=cli.launch(args,[],Path(self.tmp.name)/"bounded-results")
  if thread is not None:thread.join(timeout=5);self.assertFalse(thread.is_alive())
  self.assertEqual(len(children),1);self.assertIsNotNone(children[0].poll())
  self.assertLess(time.monotonic()-start,7)
  self.assertTrue(marker.exists());v=json.loads(marker.read_text())
  self.assertFalse(same_process(v["pid"],v["start"]))
  self.assertTrue(r["process_cleanup"]["confirmed"]);self.assertEqual(r["process_cleanup"]["status"],"interrupted")
  status=self.control.status(h)
  self.assertEqual(status["budget"]["used"],1);self.assertTrue(status["ledger"][0]["execution_uncertain"])
  self.assertEqual(self.control.advance(h)["binding"],status["binding"])
  self.assertEqual(self.backend.execute_calls,0)
  return r,status,cancelled
 def test_total_deadline_stops_one_real_detached_process_tree_and_preserves_admission(self):
  r,status,_=self._launch_blocked_worker(2,cancel=False)
  self.assertEqual(r["failure"]["code"],"assessment_entry_total_timed_out")
  self.assertEqual((r["status"],status["status"]),("timed_out","timed_out"))
 def test_running_cancel_stops_one_real_detached_process_tree_and_preserves_admission(self):
  r,status,observed=self._launch_blocked_worker(30,cancel=True)
  self.assertEqual(r["failure"]["code"],"assessment_entry_assessment_cancelled")
  self.assertEqual((r["status"],status["status"]),("cancelled","cancelled"))
  self.assertEqual(len(observed),1);self.assertTrue(observed[0]["cancel_requested"])
if __name__=="__main__":unittest.main()
