"""Control-layer tests; no SDK, compiler, FHE or Provider calls."""
import builtins,json,os,subprocess,sys,tempfile,time,types,unittest
from pathlib import Path
from unittest.mock import patch,Mock
from runtime_driver import Driver

def require(value, message):
    if not value: raise ValueError(message)

class DriverControls(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
  self.root=Path(self.tmp.name)
  def write(path,value):
   with Path(path).open("xb") as f:f.write(json.dumps(value).encode())
  self.scope=types.SimpleNamespace(HERE=self.root,ROOT=self.root,BASE=self.root,
   WORK=self.root,RESULTS=self.root,require=require,write=write,
   read=lambda p:json.loads(Path(p).read_text()),UNIQUE_PHASES=True,
   ALLOW_CONTINUATION=False,CASES=("one",),PHASES=("qualification",),
   SCOPE={"phase_seconds":0.02},expected_namespace=Mock(return_value={}),
   resource_boundary=Mock(),before_launch=Mock())
  self.driver=Driver(self.scope,Path(__file__).resolve())
 def test_unapproved_hidden_case_stops_before_import_or_output(self):
  self.scope.before_launch.side_effect=ValueError("approval")
  with patch.object(builtins,"__import__",wraps=builtins.__import__) as imports:
   with self.assertRaisesRegex(ValueError,"approval"):
    self.driver.one(self.root,"one","qualification")
  self.assertFalse(any(c.args[0] in ("numpy","hecate_python_env") for c in imports.call_args_list))
  self.assertEqual(list(self.root.iterdir()),[])
 def test_expired_hidden_case_stops_before_sdk_import_and_phase_reservation(self):
  (self.root/"plan.json").write_text("{}")
  self.scope.resource_boundary.side_effect=ValueError("deadline")
  with patch.object(builtins,"__import__",wraps=builtins.__import__) as imports:
   with self.assertRaisesRegex(ValueError,"deadline"):
    self.driver.one(self.root,"one","qualification")
  self.assertFalse(any(c.args[0] in ("numpy","hecate_python_env") for c in imports.call_args_list))
  self.assertEqual([p.name for p in self.root.iterdir()],["plan.json"])
 def test_no_continuation_rejects_before_plan_or_lock(self):
  self.scope.plan=Mock()
  with self.assertRaisesRegex(ValueError,"continuation"):
   self.driver.outer(self.root/"output",self.root/"previous")
  self.scope.plan.assert_not_called()
 def test_same_phase_cannot_be_claimed_twice(self):
  d=self.root/"one";d.mkdir()
  self.driver.reserve_phase(d,"one","qualification",{"binding":"frozen"})
  first=(d/"qualification-started.json").read_bytes()
  with self.assertRaises(FileExistsError):
   self.driver.reserve_phase(d,"one","qualification",{"binding":"frozen"})
  self.assertEqual(first,(d/"qualification-started.json").read_bytes())
 def test_completed_phase_cannot_be_claimed(self):
  d=self.root/"one";d.mkdir();(d/"qualification.json").write_text("{}")
  with self.assertRaisesRegex(ValueError,"Fresh phase"):
   self.driver.reserve_phase(d,"one","qualification",{"binding":"frozen"})
  self.assertFalse((d/"qualification-started.json").exists())
 def test_running_worker_budget_failure_terminates_and_does_not_wait(self):
  child=Mock();child.poll.return_value=None
  self.scope.resource_boundary.side_effect=ValueError("disk reserve")
  with patch.object(self.driver,"stop_worker") as stop:
   with self.assertRaisesRegex(ValueError,"disk reserve"):
    self.driver.wait_phase(child,self.root,{"deadline_wall":time.time()+30})
  stop.assert_called_once_with(child);child.wait.assert_not_called()
 def test_natural_worker_completion_retains_exit_code(self):
  child=Mock();child.poll.return_value=7;child.returncode=7
  with patch.object(self.driver,"stop_worker") as stop:
   self.assertEqual(self.driver.wait_phase(child,self.root,{"deadline_wall":time.time()+30}),7)
  stop.assert_not_called()
 def test_actual_timeout_stops_separate_session_descendant(self):
  marker=self.root/"descendant.json"
  childcode="import time;time.sleep(30)"
  parentcode=("import subprocess,sys,json,time;from pathlib import Path;"
   "p=subprocess.Popen([sys.executable,'-B','-c',"+repr(childcode)+"],start_new_session=True);"
   "Path("+repr(str(marker))+").write_text(json.dumps({'pid':p.pid}));time.sleep(30)")
  worker=subprocess.Popen([sys.executable,"-B","-c",parentcode],start_new_session=True)
  self.addCleanup(lambda: self.driver.stop_worker(worker) if worker.poll() is None else None)
  limit=time.monotonic()+3
  while not marker.exists() and time.monotonic()<limit:time.sleep(0.01)
  self.assertTrue(marker.exists())
  descendant=json.loads(marker.read_text())["pid"]
  self.assertNotEqual(os.getpgid(worker.pid),os.getpgid(descendant))
  with self.assertRaises(subprocess.TimeoutExpired):
   self.driver.wait_phase(worker,self.root,{"deadline_wall":time.time()+30})
  self.assertIsNotNone(worker.poll())
  stat=Path("/proc")/str(descendant)/"stat"
  if stat.exists():
   self.assertEqual(stat.read_text().rsplit(")",1)[1].split()[0],"Z")

if __name__=="__main__":unittest.main()
