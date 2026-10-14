import io,json,os,subprocess,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import runtime_acceptance as driver
import runtime_scope as scope

class NamespaceTransportTests(unittest.TestCase):
 def test_large_exact_argument_vector_is_passed_by_fd(self):
  args=["/usr/bin/bwrap","--clearenv"]+["/public/"+str(i)+"/"+"x"*120 for i in range(18000)]
  captured={}
  def launch(actual,**kwargs):
   captured["actual"]=actual;captured["kwargs"]=kwargs
   fd=kwargs["pass_fds"][0]
   with os.fdopen(os.dup(fd),"rb") as stream:captured["data"]=stream.read()
   return object()
  with patch.object(driver.subprocess,"Popen",side_effect=launch):
   driver.spawn_namespace(args,io.BytesIO())
  self.assertEqual(captured["actual"][:2],["/usr/bin/bwrap","--args"])
  self.assertEqual(captured["data"],b"\0".join(os.fsencode(s) for s in args[1:])+b"\0")
  self.assertTrue(captured["kwargs"]["start_new_session"])
  self.assertEqual(captured["kwargs"]["env"],{"PATH":"/usr/bin:/bin"})
  with self.assertRaises(OSError):os.fstat(captured["kwargs"]["pass_fds"][0])
 def test_embedded_nul_is_rejected(self):
  with patch.object(driver.subprocess,"Popen") as launch:
   with self.assertRaisesRegex(ValueError,"NUL"):driver.spawn_namespace(["/usr/bin/bwrap","bad\0value"],io.BytesIO())
   launch.assert_not_called()
 def test_other_executable_is_rejected(self):
  with self.assertRaisesRegex(ValueError,"Pinned"):driver.spawn_namespace(["/bin/true"],io.BytesIO())
 def test_actual_bwrap_fd_option(self):
  with tempfile.TemporaryFile() as stream:
   child=driver.spawn_namespace(["/usr/bin/bwrap","--version"],stream)
   self.assertEqual(child.wait(timeout=5),0)
   stream.seek(0);self.assertTrue(stream.read().startswith(b"bubblewrap "))

 def test_actual_namespace_command_is_not_in_option_fd(self):
  arguments=["/usr/bin/bwrap","--unshare-all","--die-with-parent","--clearenv"]
  for name in ("/usr","/bin","/lib","/lib64"):
   if Path(name).exists():arguments+=["--ro-bind",name,name]
  arguments+=["--proc","/proc","--dev","/dev","--tmpfs","/tmp","--",
              "/usr/bin/python3","-B","-c","print('public-namespace-control')"]
  with tempfile.TemporaryFile() as stream:
   child=driver.spawn_namespace(arguments,stream)
   self.assertEqual(child.wait(timeout=10),0)
   stream.seek(0);self.assertEqual(stream.read(),b"public-namespace-control\n")
 def test_empty_namespace_command_rejected(self):
  with self.assertRaisesRegex(ValueError,"command required"):
   driver.spawn_namespace(["/usr/bin/bwrap","--"],io.BytesIO())

class ContinuationTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
  self.p=Path(self.tmp.name)
  scope.write(self.p/"plan.json",{"immutable":"original"})
  (self.p/"tool-snapshot").mkdir()
  self.failure=dict(bound_plan_sha256=scope.sha(self.p/"plan.json"),code="namespace_argv_e2big",
   nix_entered=True,backend_worker_started=False,compiler_calls=0,encrypted_executions=0,
   provider_calls=0,http_attempts=0,exit_code=1)
  (self.p/"namespace.log").touch()
 def check(self):
  (self.p/"failure-before-sdk.json").write_bytes(scope.canonical(self.failure))
  return driver.unstarted_attempt(self.p)
 def test_explicit_unstarted_overflow_can_continue(self):self.assertEqual(self.check(),self.failure)
 def test_original_missing_platform_can_continue(self):
  self.failure.update(code="explicit_platform_missing",nix_entered=False)
  (self.p/"namespace.log").unlink()
  self.assertEqual(self.check(),self.failure)
 def test_nonempty_namespace_log_rejected(self):
  (self.p/"namespace.log").write_text("started")
  with self.assertRaisesRegex(ValueError,"namespace execution"):self.check()
 def test_completed_result_rejected(self):
  (self.p/"report.json").write_text("{}")
  with self.assertRaisesRegex(ValueError,"pre-worker artifacts"):self.check()
 def test_any_calls_rejected(self):
  self.failure["http_attempts"]=1
  with self.assertRaisesRegex(ValueError,"execution to repeat"):self.check()
 def test_changed_plan_rejected(self):
  (self.p/"plan.json").write_text("{}")
  with self.assertRaisesRegex(ValueError,"plan observation"):self.check()
 def test_observed_zero_case_checkpoint_preserves_original_budget(self):
  original=scope.RESULTS/"agent-ordered-common-runtime-r1260"
  failure=driver.unstarted_attempt(original)
  self.assertEqual(failure["phase_workers_started"],0)
  self.assertEqual(failure["compiler_calls"],0)
 def test_known_bwrap_option_rejection_can_continue(self):
  self.failure["code"]="namespace_argument_count_exceeded"
  log=self.p/"namespace.log"
  log.write_bytes(b"bwrap: Exceeded maximum number of arguments 9000\n")
  self.failure["namespace_log_sha256"]=scope.sha(log)
  self.assertEqual(self.check(),self.failure)
 def test_unknown_failure_rejected(self):
  self.failure["code"]="unknown"
  with self.assertRaisesRegex(ValueError,"Explicit unstarted"):self.check()


class MountPageTests(unittest.TestCase):
 def test_large_inventory_has_exact_readonly_mounts_and_bounded_layers(self):
  entries=["evidence-"+str(i) for i in range(5000)]
  mounted=[];start=0;pages=0
  while True:
   arguments=driver.mount_page(["/usr/bin/bwrap","--bind","/","/"],entries,start,Path("/fresh/output"),"/pinned/python")
   self.assertLessEqual(len(arguments),8500)
   for i,value in enumerate(arguments):
    if value=="--ro-bind":
     self.assertEqual(arguments[i+1],arguments[i+2])
     mounted.append(Path(arguments[i+1]).name)
   pages+=1
   if "--inside" in arguments:break
   self.assertIn("--mount-layer",arguments)
   new_start=int(arguments[-1]);self.assertGreater(new_start,start);start=new_start
  self.assertEqual(mounted,entries);self.assertGreater(pages,1)
 def test_only_native_slot_directory_writable_and_real_pool_acquires(self):
  with tempfile.TemporaryDirectory() as folder:
   cache=Path(folder)/"cache";cache.mkdir(mode=0o700)
   locks=cache/"agent-native-slots";locks.mkdir(mode=0o700)
   frozen=cache/"frozen";frozen.write_text("immutable")
   args=["/usr/bin/bwrap","--unshare-all","--die-with-parent","--clearenv"]
   for name in ("/usr","/bin","/lib","/lib64"):
    if Path(name).exists():args+=["--ro-bind",name,name]
   args+=["--proc","/proc","--dev","/dev","--tmpfs","/tmp","--ro-bind",str(scope.BASE),"/app",
          "--ro-bind",str(cache),"/cache","--bind",str(locks),"/cache/agent-native-slots",
          "--","/usr/bin/python3","-B","-c",
          "import sys,pathlib,errno;sys.path.insert(0,'/app');from native_execution_slots import native_slot;"
          "m={}\nwith native_slot('/cache/agent-native-slots',timeout=2,metrics=m):assert m['acquisitions']==1\n"
          "p=pathlib.Path('/cache/frozen');assert p.read_text()=='immutable'\n"
          "try:p.write_text('changed')\nexcept OSError as e:assert e.errno in (errno.EROFS,errno.EACCES)\n"
          "else:raise AssertionError('other cache was writable')"]
   result=subprocess.run(args,capture_output=True,timeout=10)
   self.assertEqual(result.returncode,0,result.stderr.decode())
   self.assertEqual(frozen.read_text(),"immutable")
   self.assertTrue((locks/"slot-0.lock").is_file())
 def test_native_work_anchor_matches_result_filesystem(self):
  work=str(scope.WORK);results=str(scope.RESULTS)
  args=["/usr/bin/bwrap","--unshare-all","--die-with-parent","--clearenv"]
  for name in ("/usr","/bin","/lib","/lib64"):
   if Path(name).exists():args+=["--ro-bind",name,name]
  args+=["--dev","/dev","--proc","/proc","--ro-bind",work,work,"--ro-bind",results,results,
         "--","/usr/bin/python3","-B","-c",
         "import os; assert os.stat("+repr(work)+").st_dev==os.stat("+repr(results)+").st_dev"]
  result=subprocess.run(args,capture_output=True,timeout=10)
  self.assertEqual(result.returncode,0,result.stderr.decode())
 def test_invalid_readonly_entry_is_rejected(self):
  with self.assertRaisesRegex(ValueError,"inventory"):
   driver.mount_page(["/usr/bin/bwrap"],["../unsafe"],0,Path("/fresh"),"/python")
 def test_actual_nested_bind_keeps_evidence_readonly(self):
  with tempfile.TemporaryDirectory() as folder:
   public=Path(folder)/"frozen";public.write_text("immutable")
   outer=["/usr/bin/bwrap","--unshare-all","--die-with-parent","--clearenv"]
   for name in ("/usr","/bin","/lib","/lib64"):
    if Path(name).exists():outer+=["--ro-bind",name,name]
   outer+=["--proc","/proc","--dev","/dev","--tmpfs","/tmp","--ro-bind",folder,"/evidence","--"]
   inner=["/usr/bin/bwrap","--unshare-all","--die-with-parent","--clearenv","--bind","/","/",
          "--proc","/proc","--dev","/dev","--","/usr/bin/python3","-B","-c",
          "import errno,pathlib; p=pathlib.Path('/evidence/frozen'); assert p.read_text()=='immutable'; "
          "\ntry:p.write_text('changed')\nexcept OSError as e:assert e.errno in (errno.EROFS,errno.EACCES)\n"
          "else:raise AssertionError('readonly submount changed')"]
   result=subprocess.run(outer+inner,capture_output=True,timeout=10)
   self.assertEqual(result.returncode,0,result.stderr.decode())
   self.assertEqual(public.read_text(),"immutable")

if __name__=="__main__":unittest.main(verbosity=2)
