"""Only the optional after-trace NPZ archive; synthetic files, no toolchain."""
from argparse import Namespace
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from pt_agent.experiments import r29_local_compile_matrix as matrix


class TraceArchiveTests(unittest.TestCase):
    def test_cli_switch_is_explicit_and_off_by_default(self):
        for extra,expected in [([],False),(['--archive-emit-constants-after-trace'],True)]:
            with patch.object(matrix,'outer',return_value=0) as outer:
                self.assertEqual(matrix.main(['--systems','dacapo-depth-dp',*extra]),0)
            self.assertEqual(outer.call_args.args[0].archive_emit_constants_after_trace,expected)

    def trace_fixture(self, root):
        work=root/'work';emit=work/'emit';emit.mkdir(parents=True)
        (emit/'constants.npz').write_text('synthetic generated constants\n'*2000)
        mlir=emit/'candidate_trace.mlir';mlir.write_text('module { /* dense synthetic constants */ }\n')
        clean=dict(exit_code=0,timed_out=False,signal=None,signals_sent=[],remaining_pids=[])
        result=dict(attempted=dict(emit=1,trace=1),completed=dict(emit=1,trace=1),
                    phases=dict(emit=dict(constants=str(emit/'constants.npz')),
                                trace=dict(status='ok',process=clean,mlir=dict(path=str(mlir)))),
                    processes=[dict(phase='trace',observation=clean)])
        return work,result

    def test_failed_trace_or_started_compile_never_archives(self):
        with TemporaryDirectory() as directory:
            work,original=self.trace_fixture(Path(directory))
            mutations=[lambda r:r['completed'].pop('trace'),
                       lambda r:r['phases']['trace']['process'].update(exit_code=1),
                       lambda r:r['processes'][0]['observation'].update(remaining_pids=[111]),
                       lambda r:r['attempted'].update(compile=1)]
            for mutation in mutations:
                result=deepcopy(original);mutation(result)
                with patch('pt_agent.experiments.r34_archive_text.archive_one') as archive:
                    with self.assertRaises(ValueError):matrix.archive_emit_constants_after_trace(result,work)
                archive.assert_not_called()
            self.assertTrue((work/'emit/constants.npz').is_file())

    def run_case(self, root, *, archive_enabled, fail_archive=False):
        from pt_agent import paths
        source=root/'original';source.mkdir()
        program=source/'program.ptdsl';program.write_text('synthetic saved rule\n')
        info=source/'info.json';info.write_text('{}\n')
        profile=source/'profile.json';profile.write_text('{}\n')
        template=source/'argv.json';template.write_text('[]\n')
        prior=source/'rule.json';prior.write_text(json.dumps(dict(round='R17',status='ok',model=matrix.MODEL,
            worker=dict(heldout_acceptance=True,artifacts=dict(program=str(program))))))
        args=Namespace(rule_result=prior,model_info=info,profile=profile,metadata_template=template,
                       git_rev='synthetic',case_gib=8,memory_gib=6,compile_seconds=7200,trace_seconds=1800,
                       frontend=source/'frontend.py',archive_emit_constants_after_trace=archive_enabled)
        calls=[]
        clean=dict(exit_code=0,timed_out=False,signal=None,signals_sent=[],remaining_pids=[])
        def fake_process(argv,**kwargs):
            phase=argv[argv.index('--worker-phase')+1];work=Path(argv[argv.index('--work-dir')+1])
            record=dict(status='ok')
            if phase=='emit':
                emit=work/'emit';emit.mkdir();(emit/'constants.npz').write_text('synthetic generated constants\n'*2000)
                record['constants']=str(emit/'constants.npz')
            elif phase=='trace':
                mlir=work/'emit/candidate_trace.mlir';mlir.write_text('module { /* dense constant */ }\n')
                record.update(process=clean,mlir=dict(path=str(mlir)))
            else:
                self.assertEqual(phase,'compile')
                self.assertEqual((work/'emit/constants.npz').exists(),not archive_enabled)
                if archive_enabled:
                    self.assertTrue((work/'emit/constants.npz.gz').is_file())
                    recovery=json.loads((work/'emit-constants-recovery.json').read_text())
                    self.assertEqual(recovery['restore_argv'][0:3],['gzip','-dk','--'])
                record.update(compiled_bootstraps=2,predicted_latency_us=None)
            (work/(phase+'-result.json')).write_text(json.dumps(record));calls.append(phase)
            return clean
        original_archive=matrix.archive_emit_constants_after_trace
        def record_archive(result,work):
            calls.append('archive_emit_constants')
            self.assertEqual(result['completed']['trace'],1)
            self.assertNotIn('compile',result['attempted'])
            if fail_archive:
                with patch('pt_agent.experiments.r34_archive_text.os.fsync',side_effect=OSError('synthetic fsync')):
                    return original_archive(result,work)
            return original_archive(result,work)
        with patch.object(paths,'RESULTS_DIR',root),patch.object(matrix,'run_process',side_effect=fake_process),\
                patch.object(matrix,'archive_emit_constants_after_trace',side_effect=record_archive):
            result=matrix.run_case(args,'dacapo-depth-dp',root/'new-case')
        return result,calls

    def test_enabled_stage_is_after_trace_and_before_compile_default_unchanged(self):
        for enabled in [False,True]:
            with self.subTest(enabled=enabled),TemporaryDirectory() as directory:
                result,calls=self.run_case(Path(directory),archive_enabled=enabled)
                self.assertEqual(result['status'],'ok')
                expected=['emit','trace','archive_emit_constants','compile'] if enabled else ['emit','trace','compile']
                self.assertEqual(calls,expected)
                self.assertEqual(list(result['attempted']),expected)
                self.assertEqual(result['completed'],{name:1 for name in expected})
                self.assertEqual(result['timing']['status'],'ok')
                self.assertTrue(result['completion']['registry_persisted'])

    def test_archive_failure_keeps_source_and_never_enters_compile(self):
        with TemporaryDirectory() as directory:
            root=Path(directory);result,calls=self.run_case(root,archive_enabled=True,fail_archive=True)
            self.assertEqual(result['status'],'error')
            self.assertEqual(calls,['emit','trace','archive_emit_constants'])
            self.assertNotIn('compile',result['attempted'])
            self.assertTrue((root/'new-case/work/emit/constants.npz').is_file())
            self.assertFalse((root/'new-case/work/emit/constants.npz.gz').exists())
            recovery=json.loads((root/'new-case/work/emit-constants-recovery.json').read_text())
            self.assertEqual(recovery['restore_argv'][0:3],['gzip','-dk','--'])


if __name__=='__main__':
    unittest.main()
