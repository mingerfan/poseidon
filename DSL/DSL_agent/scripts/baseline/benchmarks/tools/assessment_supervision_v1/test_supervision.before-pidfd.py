"""Real local-process controls only; no model, SDK, Provider or FHE execution."""
import json,os,signal,subprocess,sys,tempfile,time,unittest
from pathlib import Path
from unittest.mock import patch
from application_assessment_entry import supervision as s
class SupervisionTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
  self.root=Path(self.tmp.name).resolve();self.root.chmod(0o700)
  self.token="1"*32;self.claim=dict(format="poseidon-assessment-invocation-v1",token=self.token)
  self.claim["binding"]=s.digest(self.claim)
  (self.root/"invocation.json").write_bytes(s.canonical(self.claim))
  self.children=[]
 def spawn(self,worker):
  process=subprocess.Popen([sys.executable,"-B",str(Path(s.__file__).resolve()),
   str(self.root),self.token,sys.executable,"-B","-c",worker],stdin=subprocess.DEVNULL,
   stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
  self.children.append(process)
  self.addCleanup(self.stop,process)
  return process
 def stop(self,process):
  if process.poll()is None:
   process.terminate()
   try:process.wait(timeout=4)
   except subprocess.TimeoutExpired:process.kill();process.wait(timeout=2)
 def read(self):
  value=json.loads((self.root/"process-supervision.json").read_text())
  return s.checked(value,token=self.token,invocation_binding=self.claim["binding"])
 def marker(self,name):
  return self.root/name
 def waiting(self,p):
  limit=time.monotonic()+4
  while not p.exists() and time.monotonic()<limit:time.sleep(.01)
  self.assertTrue(p.exists())
  identity=json.loads(p.read_text())
  self.addCleanup(self.stop_identity,identity)
  return identity
 def stop_identity(self,identity):
  if s.same_process(identity["pid"],identity["start"]):
   os.kill(identity["pid"],signal.SIGKILL)
 def assert_stopped(self,identity):
  limit=time.monotonic()+3
  while s.same_process(identity["pid"],identity["start"])and time.monotonic()<limit:time.sleep(.02)
  self.assertFalse(s.same_process(identity["pid"],identity["start"]))
 def worker(self,marker,ignore_term=False):
  return("import json,os,time,signal;from pathlib import Path;"
   +("signal.signal(signal.SIGTERM,signal.SIG_IGN);"if ignore_term else"")
   +"f=Path('/proc')/str(os.getpid())/'stat';fields=f.read_text().rsplit(')',1)[1].split();"
   +"Path("+repr(str(marker))+").write_text(json.dumps({'pid':os.getpid(),'start':int(fields[19])}));"
   +"time.sleep(30)")
 def test_success_has_no_running_descendants(self):
  process=self.spawn("raise SystemExit(0)");self.assertEqual(process.wait(timeout=5),0)
  value=self.read();self.assertEqual(value["status"],"completed")
  self.assertTrue(value["cleanup_completed"]);self.assertEqual(value["background_processes"],0)
 def test_nonzero_worker_exit_is_preserved(self):
  process=self.spawn("raise SystemExit(3)");self.assertEqual(process.wait(timeout=5),3)
  self.assertEqual(self.read()["worker_exit_code"],3)
 def test_new_session_child_is_stopped_after_parent_exits(self):
  marker=self.marker("escaped.json");child=self.worker(marker)
  parent="import subprocess,sys,time;from pathlib import Path;subprocess.Popen([sys.executable,'-B','-c',"+repr(child)+"],start_new_session=True);p=Path("+repr(str(marker))+");\nwhile not p.exists():time.sleep(.01)\n"
  process=self.spawn(parent);identity=self.waiting(marker)
  self.assertEqual(process.wait(timeout=5),125);value=self.read()
  self.assertEqual(value["status"],"background_exit");self.assertTrue(value["cleanup_completed"])
  self.assertGreaterEqual(value["detached_processes"],1);self.assert_stopped(identity)
 def test_double_fork_orphan_is_adopted_and_stopped(self):
  marker=self.marker("double.json");child=self.worker(marker)
  worker="import os,time;from pathlib import Path\nif os.fork()==0:\n os.setsid()\n if os.fork()==0:\n  "+child.replace(";","\n  ")+"\n  os._exit(0)\n os._exit(0)\np=Path("+repr(str(marker))+")\nwhile not p.exists():time.sleep(.01)\n"
  process=self.spawn(worker);identity=self.waiting(marker)
  self.assertEqual(process.wait(timeout=5),125);value=self.read()
  self.assertEqual(value["status"],"background_exit");self.assertTrue(value["cleanup_completed"])
  self.assert_stopped(identity)
 def test_interruption_stops_term_ignoring_detached_child(self):
  marker=self.marker("ignore.json");child=self.worker(marker,True)
  parent="import subprocess,sys,time;subprocess.Popen([sys.executable,'-B','-c',"+repr(child)+"],start_new_session=True);time.sleep(30)"
  process=self.spawn(parent);identity=self.waiting(marker);started=time.monotonic()
  process.terminate();self.assertEqual(process.wait(timeout=4),125)
  value=self.read();self.assertEqual(value["status"],"interrupted");self.assertTrue(value["cleanup_completed"])
  self.assert_stopped(identity);self.assertLess(time.monotonic()-started,3)
 def test_unrelated_sibling_is_preserved(self):
  outside=subprocess.Popen([sys.executable,"-B","-c","import time;time.sleep(30)"],start_new_session=True)
  self.addCleanup(self.stop,outside)
  process=self.spawn("raise SystemExit(0)");self.assertEqual(process.wait(timeout=5),0)
  self.assertIsNone(outside.poll());self.assertTrue(self.read()["cleanup_completed"])
 def test_pid_reuse_is_never_signalled(self):
  owned=s.OwnedDescendants();owned.owned[(12345,10)]=True
  with patch.object(s,"process_info",return_value=dict(pid=12345,start=11,state="S")),patch.object(s.os,"kill")as kill:
   owned.signal(signal.SIGKILL)
   kill.assert_not_called()
 def test_receipt_tampering_is_rejected(self):
  value=s.receipt(self.token,self.claim["binding"],status="completed",worker_exit_code=0,
   observed=1,detached=0,background=0,cleanup_completed=True)
  for key,v in(("cleanup_completed",False),("token","2"*32),("implementation_sha256","0"*64),("background_processes",1),("observed_processes",True)):
   changed=dict(value);changed[key]=v;changed["binding"]=s.digest({k:v for k,v in changed.items()if k!="binding"})
   with self.subTest(key=key),self.assertRaises(s.SupervisionError):
    s.checked(changed,token=self.token,invocation_binding=self.claim["binding"])
 def test_invalid_invocation_is_rejected_before_spawn(self):
  (self.root/"invocation.json").write_text('{"token":"bad","binding":"0"}')
  with patch.object(s.subprocess,"Popen")as spawn,self.assertRaises(s.SupervisionError):
   s.supervise([sys.executable,"-B","-c","raise SystemExit(0)"],self.root,self.token)
  spawn.assert_not_called()
if __name__=="__main__":unittest.main()
