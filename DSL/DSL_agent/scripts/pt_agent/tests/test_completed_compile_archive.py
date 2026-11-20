"""Small closed-case and failure-order checks; no real artifact compression."""
from contextlib import contextmanager
from copy import deepcopy
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from pt_agent.experiments import completed_compile_archive as archive


class CompletedCompileArchiveTests(unittest.TestCase):
    def fixture(self, root):
        case = root / 'synthetic-vgg' / 'dacapo-depth-dp'
        emit = case / 'work' / 'emit'
        emit.mkdir(parents=True)
        for name in ['candidate_trace.mlir', 'lowered.mlir', 'constants.npz']:
            (emit / name).write_text('synthetic derived content\n' * 2000)
        plan = emit / 'lowered._hecate_golden.runtime-plan.json'
        plan.write_text('{}\n')
        bundle = emit / 'lowered._hecate_golden.bundle'
        (bundle / 'data').mkdir(parents=True)
        (bundle / 'manifest.json').write_text('{"synthetic": true}\n')
        (bundle / 'data' / 'synthetic.bin').write_text('derived fixture\n' * 3000)
        clean = dict(exit_code=0, timed_out=False, signal=None, signals_sent=[], remaining_pids=[])
        result = dict(status='ok', error=None,
                      model='pt_agent.experiments.r13_common_models:vgg16_silu_avgpool',
                      system='dacapo-depth-dp', attempted={'emit':1,'trace':1,'compile':1},
                      completed={'emit':1,'trace':1,'compile':1}, resource_monitor_error=None,
                      run_id='synthetic-vgg-dacapo-depth-dp', metrics={'status':'ok'},
                      processes=[dict(observation=clean) for _ in range(3)],
                      phases=dict(emit=dict(status='ok', constants=str(emit/'constants.npz')),
                                  trace=dict(status='ok', process=clean, mlir=dict(path=str(emit/'candidate_trace.mlir'))),
                                  compile=dict(status='ok', process=clean, runtime_plan=str(plan))))
        (case / 'result.json').write_text(json.dumps(result))
        (case / 'completion.json').write_text(json.dumps(dict(status='ok', required_artifacts_persisted=True,
            registry_persisted=True, timing_status='ok', overlap_and_timing_diagnostics=[])))
        row=dict(run_id=result['run_id'], round='R29', model=result['model'], mode='dacapo-depth-dp',
                 git_rev='synthetic', metrics=result['metrics'], artifacts_dir=str(case),ts='synthetic')
        (root/'registry.jsonl').write_text(json.dumps(row)+'\n')
        return case, emit, bundle, result

    def selection(self, closed, kind):
        source = closed['eligible_bundle'] if kind=='runtime_bundle' else closed['eligible_files'][kind]
        inventory = archive.tree_inventory(source) if kind=='runtime_bundle' else [archive.file_metadata(source,source.name)]
        return dict(kind=kind,path=str(source),inventory=inventory)

    def test_closed_case_and_explicit_four_roles(self):
        with TemporaryDirectory() as directory:
            root=Path(directory).resolve();case,emit,bundle,_=self.fixture(root)
            closed=archive.validate_completed_case(case/'result.json',results_root=root)
            roles=['compiler_ir_input','compiler_ir_final','emit_constants','runtime_bundle']
            selected=archive.validate_manifest(closed,[self.selection(closed,k) for k in roles])
            self.assertEqual([r['kind'] for r in selected],roles)
            self.assertEqual(selected[-1]['restore_argv'][-4:],['--directory',str(emit),'--no-same-owner','--no-same-permissions'])
            self.assertIn(str(emit),selected[-1]['restore_argv'])
            self.assertEqual(selected[-1]['target'],bundle.with_name(bundle.name+'.tar.gz'))

    def test_incomplete_or_unclean_source_case_is_rejected(self):
        with TemporaryDirectory() as directory:
            root=Path(directory).resolve();case,_,_,result=self.fixture(root)
            for mutation in [lambda r:r.update(status='running'),lambda r:r['completed'].pop('compile'),
                             lambda r:r['processes'][0]['observation'].update(remaining_pids=[999])]:
                changed=deepcopy(result);mutation(changed);(case/'result.json').write_text(json.dumps(changed))
                with self.assertRaises(ValueError):archive.validate_completed_case(case/'result.json',results_root=root)

    def test_non_derived_path_and_metadata_change_are_rejected(self):
        with TemporaryDirectory() as directory:
            root=Path(directory).resolve();case,emit,_,_=self.fixture(root)
            closed=archive.validate_completed_case(case/'result.json',results_root=root)
            row=self.selection(closed,'compiler_ir_input')
            for changed in [dict(row,kind='model_weights'),dict(row,path=str(root/'model.pt')),
                            dict(row,inventory=[dict(row['inventory'][0],bytes=1)])]:
                with self.assertRaises(ValueError):archive.validate_manifest(closed,[changed])

    def test_symbolic_and_hard_link_sources_are_rejected(self):
        with TemporaryDirectory() as directory:
            root=Path(directory).resolve();case,emit,bundle,_=self.fixture(root)
            closed=archive.validate_completed_case(case/'result.json',results_root=root)
            target=emit/'candidate_trace.mlir';link=root/'symbolic.mlir';link.symlink_to(target)
            with self.assertRaises(ValueError):archive.file_metadata(link,link.name)
            hard=root/'hard.mlir';os.link(target,hard)
            with self.assertRaises(ValueError):archive.file_metadata(target,target.name)
            hard.unlink()
            (bundle/'data'/'symbolic.bin').symlink_to(target)
            with self.assertRaises(ValueError):archive.tree_inventory(bundle)

    def test_single_file_fsync_failure_preserves_original(self):
        with TemporaryDirectory() as directory:
            root=Path(directory).resolve();case,_,_,_=self.fixture(root)
            closed=archive.validate_completed_case(case/'result.json',results_root=root)
            item=archive.validate_manifest(closed,[self.selection(closed,'compiler_ir_input')])[0]
            with patch('pt_agent.experiments.r34_archive_text.os.fsync',side_effect=OSError('synthetic fsync')):
                with self.assertRaises(OSError):archive.archive_one(item['source'],item['target'],item['temporary'])
            self.assertTrue(item['source'].is_file());self.assertFalse(item['target'].exists())

    def test_bundle_close_failure_preserves_whole_directory(self):
        with TemporaryDirectory() as directory:
            root=Path(directory).resolve();case,_,bundle,_=self.fixture(root)
            closed=archive.validate_completed_case(case/'result.json',results_root=root)
            item=archive.validate_manifest(closed,[self.selection(closed,'runtime_bundle')])[0]
            real=archive.tarfile.open
            @contextmanager
            def failing_close(*args,**kwargs):
                with real(*args,**kwargs) as stream:yield stream
                raise OSError('synthetic close failure before commit')
            with patch.object(archive.tarfile,'open',side_effect=failing_close):
                with self.assertRaises(OSError):archive.archive_tree(bundle,item['target'],item['temporary'],item['inventory'])
            self.assertTrue((bundle/'data/synthetic.bin').is_file());self.assertFalse(item['target'].exists())

    def test_bundle_rename_failure_preserves_whole_directory(self):
        with TemporaryDirectory() as directory:
            root=Path(directory).resolve();case,_,bundle,_=self.fixture(root)
            closed=archive.validate_completed_case(case/'result.json',results_root=root)
            item=archive.validate_manifest(closed,[self.selection(closed,'runtime_bundle')])[0]
            with patch.object(Path,'rename',side_effect=OSError('synthetic rename failure')):
                with self.assertRaises(OSError):archive.archive_tree(bundle,item['target'],item['temporary'],item['inventory'])
            self.assertTrue((bundle/'manifest.json').is_file());self.assertTrue(item['temporary'].is_file())
            self.assertFalse(item['target'].exists())

    def test_partial_directory_delete_retains_archive_and_recovery_record(self):
        from pt_agent import paths
        with TemporaryDirectory() as directory:
            root=Path(directory).resolve();case,emit,bundle,_=self.fixture(root)
            closed=archive.validate_completed_case(case/'result.json',results_root=root)
            manifest=root/'archive-manifest.json';manifest.write_text(json.dumps(dict(case_result=str(case/'result.json'),
                files=[self.selection(closed,'runtime_bundle')])))
            def fail_delete(path):
                (Path(path)/'data/synthetic.bin').unlink()
                raise OSError('synthetic deletion failure after first child')
            out=root/'new-archive'
            with patch.object(paths,'RESULTS_DIR',root),patch.object(archive.shutil,'rmtree',side_effect=fail_delete),\
                    patch('pt_agent.registry.git_rev',return_value='synthetic'):
                self.assertEqual(archive.main(['--manifest',str(manifest),'--out',str(out)]),1)
            result=json.loads((out/'result.json').read_text());state=result['interrupted_file']
            self.assertEqual(result['status'],'error');self.assertTrue(state['archive_exists'])
            self.assertTrue(state['source_exists']);self.assertFalse(state['temporary_exists'])
            self.assertEqual(state['restore_argv'][0],'tar')
            self.assertEqual(state['restore_argv'][state['restore_argv'].index('--directory')+1],str(emit))
            self.assertTrue((out/'manifest.json').is_file());self.assertTrue((out/'progress.json').is_file())

    def test_alex_with_closed_trace_archive_accepts_only_remaining_files(self):
        with TemporaryDirectory() as directory:
            root=Path(directory).resolve();case,emit,_,result=self.fixture(root)
            result['model']='pt_agent.experiments.r13_common_models:alexnet_silu_avgpool'
            result['archive_emit_constants_after_trace']=True
            result['attempted']['archive_emit_constants']=1
            result['completed']['archive_emit_constants']=1
            result['phases']['archive_emit_constants']={'status':'ok'}
            (case/'result.json').write_text(json.dumps(result))
            row=json.loads((root/'registry.jsonl').read_text());row['model']=result['model']
            (root/'registry.jsonl').write_text(json.dumps(row)+'\n')
            (emit/'constants.npz').unlink()
            closed=archive.validate_completed_case(case/'result.json',results_root=root)
            roles=['compiler_ir_input','compiler_ir_final','runtime_bundle']
            self.assertEqual(len(archive.validate_manifest(closed,[self.selection(closed,k) for k in roles])),3)
            with self.assertRaises(FileNotFoundError):self.selection(closed,'emit_constants')
            result['phases']['archive_emit_constants']['status']='error'
            (case/'result.json').write_text(json.dumps(result))
            with self.assertRaises(ValueError):archive.validate_completed_case(case/'result.json',results_root=root)

    def test_unrecognized_model_is_rejected(self):
        with TemporaryDirectory() as directory:
            root=Path(directory).resolve();case,_,_,result=self.fixture(root)
            result['model']='unrelated-model'
            (case/'result.json').write_text(json.dumps(result))
            with self.assertRaises(ValueError):archive.validate_completed_case(case/'result.json',results_root=root)


if __name__=='__main__':
    unittest.main()
