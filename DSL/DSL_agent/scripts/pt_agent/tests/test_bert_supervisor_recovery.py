"""Supervisor quota regression and strict saved-process closure checks."""
import errno
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

from pt_agent.experiments import bert_supervisor_recovery as entry


class SupervisorRecoveryTests(unittest.TestCase):
    def test_one_identified_final_partial_snapshot_is_recorded_without_guessing(self):
        with TemporaryDirectory() as directory:
            root = Path(directory); path = root/'phase-resources.jsonl'
            path.write_bytes(json.dumps(dict(processes=[dict(pid=100,starttime=20)])).encode()+b'\n'+b'{"processes":[')
            births, proof = entry.resource_births(root, truncated_path=path)
            self.assertEqual(births, {(100,20)})
            self.assertEqual(proof['resource_snapshots'],1)
            self.assertEqual(proof['incomplete_final_writes'][0]['bytes'],14)

    def test_unidentified_partial_snapshot_is_rejected(self):
        with TemporaryDirectory() as directory:
            root = Path(directory); (root/'phase-resources.jsonl').write_bytes(b'{"pid":')
            with self.assertRaises(ValueError):entry.resource_births(root)

    def test_malformed_complete_line_is_rejected(self):
        with TemporaryDirectory() as directory:
            root = Path(directory); path=root/'phase-resources.jsonl';path.write_bytes(b'{"pid":\n')
            with self.assertRaises(json.JSONDecodeError):entry.resource_births(root,truncated_path=path)

    def test_same_birth_including_zombie_blocks_and_reused_pid_does_not(self):
        for state in ('R','Z'):
            with self.assertRaises(ValueError):entry.verify_gone({(100,20)},probe=lambda pid:dict(pid=pid,starttime=20,state=state))
        entry.verify_gone({(100,20)},probe=lambda pid:dict(pid=pid,starttime=21,state='R'))

    def test_permission_error_is_not_absence(self):
        def denied(pid):raise PermissionError('fixture')
        with self.assertRaises(PermissionError):entry.verify_gone({(100,20)},probe=denied)

    def test_private_namespace_does_not_change_original_runner(self):
        before=set(entry.original.MODULES);runner=entry.original.run
        values=entry.namespace(limits={'fixture':True})
        self.assertIn('bert_fresh_cold_3h',values['MODULES'])
        self.assertEqual(entry.original.MODULES,before)
        self.assertIs(entry.original.run,runner)
        self.assertIsNot(values['run'].__globals__,vars(entry.original))

    def test_real_parent_resource_write_fails_under_small_limit_and_fixed_parent_completes(self):
        program=r'''
import errno,json,os,resource,sys
from pathlib import Path
from pt_agent.experiments import bert_supervisor_recovery as entry
from pt_agent.experiments import r10_paper_metrics as metrics
root=Path(sys.argv[1]); fixed=sys.argv[2]=='fixed'
resource.setrlimit(resource.RLIMIT_FSIZE,(1024,16*1024**3))
limits=entry.set_supervisor_limits()if fixed else None
before={p for p,r in metrics._scan().items()if r['ppid']==os.getpid()}
try:
 observation=metrics.run_process([sys.executable,'-B','-c','import time;time.sleep(.5)'],cwd=str(Path.cwd()),log=root/'worker.log',resources=root/'worker-resources.jsonl',timeout=5,mem_gib=1,fsize_gib=16)
 result={'ok':True,'observation':observation}
except OSError as error:
 result={'ok':False,'errno':error.errno}
after={p for p,r in metrics._scan().items()if r['ppid']==os.getpid()}
result.update(new_children_remaining=sorted(after-before),bytes=(root/'worker-resources.jsonl').stat().st_size,limits=limits)
print(json.dumps(result))
'''
        for fixed in (False,True):
            with self.subTest(fixed=fixed),TemporaryDirectory() as directory:
                process=subprocess.run([sys.executable,'-B','-c',program,directory,'fixed'if fixed else'original'],
                    capture_output=True,text=True,timeout=20)
                self.assertEqual(process.returncode,0,process.stderr)
                result=json.loads(process.stdout);self.assertEqual(result['new_children_remaining'],[])
                self.assertIs(result['ok'],fixed)
                if fixed:
                    self.assertGreater(result['bytes'],1024)
                    self.assertEqual(result['limits']['after'][0],16*entry.GIB)
                    self.assertEqual(result['observation']['exit_code'],0)
                    self.assertEqual(result['observation']['signals_sent'],[])
                else:
                    self.assertEqual(result['errno'],errno.EFBIG)
                    self.assertEqual(result['bytes'],1024)


if __name__=='__main__':unittest.main()
