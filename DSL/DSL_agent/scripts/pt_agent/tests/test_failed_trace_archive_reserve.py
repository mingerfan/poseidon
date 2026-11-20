"""Focused reserve-policy fixtures; compression and process limits are mocked."""
from contextlib import ExitStack
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pt_agent.experiments import failed_trace_archive as archive
from pt_agent import paths


class ReserveChecks(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory();self.addCleanup(self.temporary.cleanup)
        self.root=Path(self.temporary.name).resolve();self.case=self.root/'source';self.case.mkdir()
        self.source=self.case/'work/trace/traced/VGG16.mlir'
        self.source.parent.mkdir(parents=True);self.source.write_text('synthetic incomplete trace')
        self.result=self.case/'result.json'
        self.result.write_text(json.dumps(dict(model='synthetic-vgg',mode='original-source-native-profile-compile',
            status='error',original_model_selector='vgg16',trace_attempts=1,trace_completed=0,
            compile_attempts=0,native_compile_attempts=0,trace_process=dict(exit_code=1,remaining_pids=[]))))
        (self.case/'completion.json').write_text(json.dumps(dict(status='error',
            registry_persisted=True,required_artifacts_persisted=True)))
        (self.case/'git.log').write_text('synthetic-revision\n')

    def cli(self,extra=()):
        return ['--source-result',str(self.result),'--out',str(self.root/'new')]+list(extra)

    def test_cli_default10_and_explicit4(self):
        for extra,expected in (([],10),(['--minimum-free-gib','4'],4)):
            with self.subTest(extra=extra),patch.object(archive,'run',return_value=dict(status='ok',released_bytes=0,error=None))as run,patch('sys.stdout',io.StringIO()):
                self.assertEqual(archive.main(self.cli(extra)),0)
                self.assertEqual(run.call_args.args[0].minimum_free_gib,expected)

    def test_cli_rejects_nonfinite_and_out_of_range_before_run(self):
        for value in ('3.99','10.01','nan','inf','-inf','abc'):
            with self.subTest(value=value),patch.object(archive,'run')as run,patch('sys.stderr',io.StringIO()),self.assertRaises(SystemExit):
                archive.main(self.cli(['--minimum-free-gib',value]))
            run.assert_not_called()

    def exercise(self,*,reserve=None,free=5):
        out=self.root/('reserve-'+str(reserve)+'-free-'+str(free))
        args=SimpleNamespace(source_result=self.result,out=out)
        if reserve is not None:args.minimum_free_gib=reserve
        watcher=SimpleNamespace(start=lambda:None,is_alive=lambda:False)
        with ExitStack()as stack:
            stack.enter_context(patch.object(paths,'RESULTS_DIR',self.root))
            stack.enter_context(patch.object(archive.shutil,'disk_usage',return_value=SimpleNamespace(free=free*1024**3)))
            stack.enter_context(patch.object(archive.resource,'setrlimit'))
            stack.enter_context(patch.object(archive.signal,'signal'))
            stack.enter_context(patch.object(archive.signal,'setitimer'))
            stack.enter_context(patch.object(archive.threading,'Thread',return_value=watcher))
            compression=stack.enter_context(patch.object(archive,'archive_one',return_value=dict(released_bytes=7)))
            result=archive.run(args)
        self.assertEqual(json.loads((out/'result.json').read_text())['reserve_disk_gib'],10 if reserve is None else reserve)
        self.assertTrue((out/'completion.json').is_file())
        self.assertEqual(self.source.read_text(),'synthetic incomplete trace')
        return result,compression.call_count

    def test_default10_still_refuses5GiB_before_compression(self):
        result,calls=self.exercise(free=5)
        self.assertEqual(result['status'],'error');self.assertEqual(calls,0)
        self.assertIn('Disk reserve unavailable',result['error'])

    def test_explicit4_is_used_by_check_and_persisted(self):
        result,calls=self.exercise(reserve=4,free=5)
        self.assertEqual(result['status'],'ok');self.assertEqual(calls,1)
        self.assertEqual(result['reserve_disk_gib'],4)

    def test_explicit4_boundary_below_is_refused(self):
        result,calls=self.exercise(reserve=4,free=3.99)
        self.assertEqual(result['status'],'error');self.assertEqual(calls,0)

    def test_direct_run_rejects_invalid_reserve_before_creating_case(self):
        args=SimpleNamespace(source_result=self.result,out=self.root/'new',minimum_free_gib=float('nan'))
        with self.assertRaises(ValueError):archive.run(args)
        self.assertFalse(args.out.exists())


if __name__=='__main__':unittest.main()
