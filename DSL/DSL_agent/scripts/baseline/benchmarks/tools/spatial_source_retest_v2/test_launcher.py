import json,pathlib,subprocess,sys,time,unittest
HERE=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import run
class LauncherControls(unittest.TestCase):
    def args(self):
        entry=[str(HERE/"guard_probe.py"),"--probe"]
        return ["timeout","3s","python3","-B",*entry],entry
    def test_exact_self_timeout_spelling(self):
        args,entry=self.args();self.assertTrue(run.own_timeout_command(args,entry))
    def test_different_target_or_arguments_are_not_self(self):
        args,entry=self.args()
        self.assertFalse(run.own_timeout_command(args,[str(HERE/"guard_probe.py"),"--execute"]))
        self.assertFalse(run.own_timeout_command(["other",*args[1:]],entry))
    def test_external_script_is_not_accepted(self):
        entry=["/tmp/unrelated.py","--probe"]
        self.assertFalse(run.own_timeout_command(["timeout","3s","python3","-B",*entry],entry))
    def test_actual_own_launcher_allowed(self):
        result=subprocess.run(["timeout","3s","python3","-B",str(HERE/"guard_probe.py"),"--probe"],
                              capture_output=True,text=True,timeout=5)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(json.loads(result.stdout)["active_guard"],"passed")
    def test_other_actual_timeout_task_still_blocks(self):
        worker=subprocess.Popen(["timeout","7s","python3","-B",str(HERE/"guard_probe.py"),"--execute"],
                                stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        try:
            self.assertIn("owned_nonancestor_fixture",worker.stdout.readline())
            result=subprocess.run(["timeout","3s","python3","-B",str(HERE/"guard_probe.py"),"--probe"],
                                  capture_output=True,text=True,timeout=5)
            self.assertNotEqual(result.returncode,0)
            self.assertIn("active_project_execution",result.stderr)
        finally:
            if worker.poll()is None:worker.terminate()
            worker.communicate(timeout=5)
if __name__=="__main__":unittest.main(verbosity=2)
