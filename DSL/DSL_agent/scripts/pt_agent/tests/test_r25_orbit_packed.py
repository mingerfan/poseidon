"""R25 stdlib protocol checks and one NumPy-only synthetic-suite wrapper."""

import ast
from contextlib import ExitStack, redirect_stdout
import copy
import io
import json
import os
from pathlib import Path
import re
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from pt_agent import paths
from pt_agent.tests._artifacts import require


NUMPY_PYTHON = Path(os.environ.get('R25_NUMPY_PYTHON',
    str(paths.ORBIT_ROOT.parent / 'orbit-venv/bin/python')))
NUMPY_ARTIFACT = NUMPY_PYTHON.parent if NUMPY_PYTHON.is_symlink() else NUMPY_PYTHON
BOOTSTRAP = r'''import json, os, signal, sys, tempfile, unittest
from pathlib import Path
work = Path(sys.argv[1]).resolve()
os.environ['R25_TEST_WORK'] = str(work)
os.environ['TMPDIR'] = str(work)
tempfile.tempdir = str(work)
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
signal.alarm(300)
suite = unittest.defaultTestLoader.loadTestsFromName('pt_agent.tests._r25_orbit_packed_cases.PackedCases')
result = unittest.TextTestRunner(verbosity=2).run(suite)
record = dict(testsRun=result.testsRun, failures=len(result.failures), errors=len(result.errors),
              skips=len(result.skipped), success=result.wasSuccessful())
(work / 'inner-unit-result.json').write_text(json.dumps(record, allow_nan=False) + '\n')
raise SystemExit(0 if record['success'] and record['testsRun'] > 0 and record['skips'] == 0 else 1)
'''


@require(NUMPY_ARTIFACT)
class ValidationEnvironmentChecks(unittest.TestCase):
    def test_only_current_synthetic_packed_cases_run_once_without_skips(self):
        from pt_agent.experiments.r10_paper_metrics import run_process

        tree = ast.parse(Path(__file__).with_name('_r25_orbit_packed_cases.py').read_text())
        classes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'PackedCases']
        self.assertEqual(len(classes), 1)
        expected = sum(isinstance(node, ast.FunctionDef) and node.name.startswith('test_')
                       for node in classes[0].body)
        self.assertGreater(expected, 0)
        work = Path(os.environ['R25_TEST_WORK']).resolve(); work.mkdir(parents=True, exist_ok=True)
        unit, summary = work / 'inner-unit-result.json', work / 'inner-wrapper-result.json'
        self.assertFalse(unit.exists(), 'The dedicated inner suite runs once')
        self.assertFalse(summary.exists(), 'The dedicated wrapper runs once')
        command = [str(NUMPY_PYTHON), '-B', '-c', BOOTSTRAP, str(work)]
        log = work / 'inner-tests.log'
        observation = run_process(command, cwd=Path(__file__).resolve().parents[2],
            log=log, resources=work / 'inner-resources.jsonl', timeout=360, mem_gib=4, fsize_gib=8,
            env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
        output = log.read_text(); print(output, end='', flush=True)
        self.assertEqual(observation['exit_code'], 0, output)
        self.assertFalse(observation['timed_out'], observation)
        self.assertIsNone(observation['signal'], observation)
        self.assertEqual(observation['remaining_pids'], [], observation)
        counts = re.findall(r'Ran (\d+) tests? in ', output)
        self.assertTrue(counts, output); self.assertEqual(int(counts[-1]), expected, output)
        self.assertNotRegex(output, r'skipped=\d+|\.\.\. skipped ')
        self.assertEqual(json.loads(unit.read_text()),
                         dict(testsRun=expected, failures=0, errors=0, skips=0, success=True))
        summary.write_text(json.dumps(dict(status='ok', wrapper_runs=1, inner_tests=expected,
            inner_skips=0, observation=observation), allow_nan=False) + '\n')
        print('R25_INNER_TESTS=' + str(expected), flush=True)
        print('R25_REAL_CONSTANTS_PROBES_WEIGHTS_AND_GRAPHS_IN_TESTS=0', flush=True)


class OuterProtocolChecks(unittest.TestCase):
    def setUp(self):
        from pt_agent.experiments import r25_orbit_packed_preflight as outer
        self.outer = outer
        temporary = TemporaryDirectory(prefix='r25-outer-', dir=os.environ.get('R25_TEST_WORK'))
        self.addCleanup(temporary.cleanup); self.root = Path(temporary.name)
        self.results = self.root/'results'; self.results.mkdir()
        r13, r22 = self.results/'r13-original', self.results/'r22-original'
        self.locations = dict(results=self.results, repo=self.root/'repo', scripts=self.root/'repo/scripts',
            orbit=self.root/'orbit', numpy_python=self.root/'orbit-venv/bin/python',
            probes=r13/'worker/training_probes.npz', probe_source=r13/'worker/training_probe_source.json',
            r13_result=r13/'result.json', r22_result=r22/'result.json',
            r24_result=self.results/'r24-original/result.json', graph=r22/'work/parse-worker/tdag.json',
            constants=r22/'work/trace-worker/constants/_hecate_ResNetSiLU64k.cst')
        for key in ('numpy_python','probes','probe_source','graph','constants'):
            path = self.locations[key]; path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('small metadata fixture; never interpreted\n')
        self.locations['scripts'].mkdir(parents=True)
        source = dict(split='train', indices=[0,1,2], sample_ids=['train:0','train:1','train:2'])
        self.records = {number:dict(round='R'+str(number), model=outer.MODEL, status='ok') for number in (13,22,24)}
        self.records[13]['worker'] = dict(source=source, artifacts=dict(
            training_probes=str(self.locations['probes']), training_probe_source=str(self.locations['probe_source'])))
        self.records[22].update(mode='orbit-resnet20-trace-tdag-preflight',
            orbit_model_trace_passed=True, tdag_parse_passed=True,
            attempted=dict(trace=1,parse=1), completed=dict(trace=1,parse=1),
            phase_results=dict(trace=dict(status='ok', constant_files=[dict(
                path=str(self.locations['constants']),size_bytes=784347640)],
                mlir=dict(path=str(r22/'work/trace-worker/trace/fixture.mlir'),size_bytes=19)),
                parse=dict(status='ok',tdag=str(self.locations['graph']),parser_metadata=dict(max_slot=65536))))
        self.records[24].update(mode='orbit-native-polynomial-physical-scale-check',
            source_records=dict(records=dict(r13=str(self.locations['r13_result']),r22=str(self.locations['r22_result']))),
            worker=dict(original_reference_passed=True,native_polynomial_math_passed=True,
                physical_scaled_forward_completed=True, original_forward_attempts=3,original_forward_completed=3,
                approx_forward_attempts=3,approx_forward_completed=3,activation_attempts=57,activation_completed=57,
                math_check_attempts=57,math_check_completed=57,
                probe_source=source,approximate_probes=[dict(probe_index=i,sample_id='train:'+str(i),
                    approximate=[[0.0]*10]) for i in range(3)]))
        for number, record in self.records.items(): self._save(self.locations['r'+str(number)+'_result'],record)

    @staticmethod
    def _save(path, record):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, allow_nan=False)+'\n')

    def _args(self): return SimpleNamespace(out=self.results/self.outer.OUT_NAME)

    def _fresh_args(self, label):
        self.results=self.root/label/'results'; self.results.mkdir(parents=True)
        self.locations['results']=self.results
        return self._args()

    def _registry(self):
        path=self.results/'registry.jsonl'
        return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []

    def _source_stats(self):
        original=Path.stat; constant=self.locations['constants']
        def stat(path,*args,**kwargs):
            observed=original(path,*args,**kwargs)
            if path==constant:
                values=list(observed); values[6]=784347640
                return os.stat_result(values)
            return observed
        return patch.object(Path,'stat',stat)

    def _worker(self):
        from pt_agent.experiments import r25_orbit_packed_worker as worker
        record=worker._initial_result()
        record.update(status='ok',source_checked=True,native_packed_equivalence_verified=True,
            native_packed_equivalence_scope=self.outer.SCOPE)
        record.update({name:(30336 if name.startswith('op_') else
                            3 if name.startswith(('packed_forward_','math_check_')) else 1)
                       for name in self.outer.COUNTERS})
        for field in ('op_attempted_by_kind','op_completed_by_kind'):
            record[field]={kind:count*3 for kind,count in self.outer.OP_COUNTS.items()}
        return record

    def _process(self, worker=None, *, missing_worker=False, fail=None, test_count=14, skip=False, **overrides):
        worker=self._worker() if worker is None else worker; calls=[]
        def run(command,**options):
            stage=Path(options['log']).stem; calls.append((stage,command,options))
            observation=dict(exit_code=0,timed_out=False,signal=None,remaining_pids=[],signals_sent=[],
                started_ns=1,ended_ns=2,termination_observed_ns=3,peak_rss_kib_by_pid={},outer_peak_rss_kib=0)
            log=Path(options['log'])
            if stage.startswith('git-'):
                log.write_text('fake-revision\n' if stage=='git-revision' else 'lhy-agent-dsl\n')
            elif stage=='r25-tests':
                log.write_text('Ran '+str(test_count)+' tests in 0.001s\n\nOK'+(' (skipped=1)' if skip else '')+'\n')
                self._save(Path(options['env']['R25_TEST_WORK'])/'inner-wrapper-result.json',
                    dict(status='ok',wrapper_runs=1,inner_tests=14,inner_skips=0,observation=observation))
            else:
                log.write_text('fake packed worker; no official constants, samples or graphs executed\n')
                work=Path(command[command.index('--work-dir')+1]); work.mkdir(parents=True)
                (work/'events.jsonl').touch()
                if not missing_worker: self._save(work/'phase_result.json',worker)
            if stage==fail:
                observation.update(exit_code=1); observation.update(overrides)
            return observation
        run.calls=calls
        return run

    def _execute(self,args,process):
        with ExitStack() as stack:
            stack.enter_context(self._source_stats())
            stack.enter_context(patch.object(self.outer,'_locations',return_value=self.locations))
            stack.enter_context(patch.object(self.outer,'run_process',side_effect=process))
            stack.enter_context(patch.object(self.outer.resource,'setrlimit'))
            stack.enter_context(patch.object(self.outer.signal,'signal'))
            stack.enter_context(patch.object(self.outer.signal,'setitimer'))
            stack.enter_context(patch.object(self.outer.shutil,'disk_usage',return_value=SimpleNamespace(free=11*1024**3)))
            stack.enter_context(patch.object(self.outer.threading,'Thread',
                return_value=SimpleNamespace(start=Mock(),is_alive=lambda:False)))
            stack.enter_context(redirect_stdout(io.StringIO()))
            return self.outer.execute(args)

    def test_source_records_only_original_cached_paths_and_metadata_without_opening_arrays(self):
        with self._source_stats(): source=self.outer._source_preflight(self.locations)
        self.assertEqual(source['saved_sample_ids'],['train:0','train:1','train:2'])
        self.assertFalse(source['sample_contents_read']); self.assertFalse(source['compiled_constants_read'])
        self.assertEqual(source['records'],{key:str(self.locations[key+'_result']) for key in ('r13','r22','r24')})
        self.assertEqual(len(source['inputs']),7)

    def test_source_rejects_foreign_identity_paths_wrong_counts_and_reference_shape(self):
        for variant in ('model','probes','graph','constants','constant_size','slots','r22_count','r24_stage','reference'):
            records=copy.deepcopy(self.records)
            if variant=='model':records[13]['model']='foreign'
            elif variant=='probes':records[13]['worker']['artifacts']['training_probes']=str(self.root/'foreign.npz')
            elif variant=='graph':records[22]['phase_results']['parse']['tdag']=str(self.root/'foreign.json')
            elif variant=='constants':records[22]['phase_results']['trace']['constant_files'][0]['path']=str(self.root/'foreign.cst')
            elif variant=='constant_size':records[22]['phase_results']['trace']['constant_files'][0]['size_bytes']=17
            elif variant=='slots':records[22]['phase_results']['parse']['parser_metadata']['max_slot']=32768
            elif variant=='r22_count':records[22]['completed']['parse']=0
            elif variant=='r24_stage':records[24]['worker']['physical_scaled_forward_completed']=False
            else:records[24]['worker']['approximate_probes'][0]['approximate']=[[0.0]*9]
            for number,record in records.items():self._save(self.locations['r'+str(number)+'_result'],record)
            with self.subTest(variant=variant),self._source_stats(),self.assertRaises(ValueError):
                self.outer._source_preflight(self.locations)
        for number,record in self.records.items():self._save(self.locations['r'+str(number)+'_result'],record)

    def test_command_uses_direct_numpy_python_and_only_explicit_saved_sources(self):
        work=self.root/'case/work'; command=self.outer.phase_command(self.locations,work)
        self.assertEqual(command[:4],[str(self.locations['numpy_python']),'-B','-m',self.outer.MODULE])
        self.assertEqual(command[command.index('--work-dir')+1],str(work/'checks-worker'))
        for key in ('graph','constants','probes','probe_source','r13_result','r22_result','r24_result'):
            option='--'+key.replace('_','-'); self.assertEqual(command.count(option),1)
            self.assertEqual(command[command.index(option)+1],str(self.locations[key]))
        for option in ('--seconds','--python','--checkpoint','--model-source','--poly-dir','--fit','--normalization'):
            self.assertNotIn(option,command)

    def test_environment_preserves_home_and_clears_sdk_paths_and_numeric_threads(self):
        inherited=dict(HOME='/synthetic-home',HECATE='old',HECATE_BUILD='old',LD_LIBRARY_PATH='old',
            NIX_PATH='old',__NIX_TEST='old',IN_NIX_SHELL='old',PYTHONPATH='old',PYTHONHOME='old',VIRTUAL_ENV='old')
        with patch.dict(os.environ,inherited):env=self.outer.child_environment(self.root/'work',self.locations)
        self.assertEqual(env['HOME'],'/synthetic-home')
        for name in inherited.keys()-{'HOME'}:self.assertNotIn(name,env)
        for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
            self.assertEqual(env[name],'1')
        self.assertEqual(env['R25_TEST_WORK'],str(self.root/'work/test-worker'))
        self.assertEqual(env['R25_NUMPY_PYTHON'],str(self.locations['numpy_python']))
        self.assertEqual(env['TMPDIR'],str(self.root/'work/tmp'))
        self.assertEqual((env['PYTHONDONTWRITEBYTECODE'],env['PYTHONNOUSERSITE']),('1','1'))

    def test_success_keeps_finite_scope_counts_null_formal_metrics_and_unique_registry(self):
        args,process=self._args(),self._process(); result=self._execute(args,process)
        self.assertEqual(result['status'],'ok',result.get('error'))
        self.assertEqual([row[0] for row in process.calls],['git-revision','git-branch','r25-tests','model-checks'])
        self.assertEqual(result['attempted'],dict(tests=1,checks=1));self.assertEqual(result['completed'],result['attempted'])
        self.assertEqual((result['git_metadata_attempted'],result['git_metadata_completed']),(2,2))
        self.assertEqual([row[2]['timeout'] for row in process.calls],[30,30,600,960])
        self.assertTrue(all((row[2]['mem_gib'],row[2]['fsize_gib'])==(4,8) for row in process.calls))
        self.assertEqual(result['metrics']['counts'],{key:self._worker()[key] for key in self.outer.COUNTERS})
        self.assertTrue(all(result['metrics'][key] is None for key in self.outer.FORMAL))
        self.assertTrue(result['native_packed_equivalence_verified'])
        self.assertEqual(result['native_packed_equivalence_scope'],self.outer.SCOPE)
        self.assertTrue(all(result[key] is False for key in self.outer.FALSE_FIELDS))
        self.assertEqual(result['case_artifact_bytes'],self.outer.case_size(args.out))
        rows=self._registry();self.assertEqual(len(rows),1)
        self.assertEqual(set(rows[0]),{'run_id','round','model','mode','git_rev','metrics','artifacts_dir','ts'})
        self.assertEqual((rows[0]['round'],rows[0]['mode']),('R25',self.outer.MODE))
        self.assertEqual(rows[0]['metrics'],result['metrics'])
        timing=json.loads((args.out/'timing.json').read_text());self.assertTrue(timing['attribution_ready'])
        self.assertEqual(timing['diagnostics'],[])
        roots=[row for row in timing['spans'] if row['parent_id'] is None];self.assertEqual(len(roots),1)
        self.assertEqual(sum(row['exclusive_ns'] for row in timing['spans']),roots[0]['inclusive_ns'])

    def test_partial_math_failure_keeps_completed_loads_and_stops_without_retry(self):
        worker=self._worker();worker.update(status='math_failed',native_packed_equivalence_verified=False,
            packed_forward_attempts=1,packed_forward_completed=1,math_check_attempts=1,math_check_completed=0,
            op_attempts=10112,op_completed=10112)
        result=self._execute(self._args(),self._process(worker,fail='model-checks'))
        self.assertEqual(result['status'],'stopped_on_error');self.assertFalse(result['native_packed_equivalence_verified'])
        self.assertEqual(result['metrics']['counts']['constant_load_completed'],1)
        self.assertEqual(result['metrics']['counts']['packed_forward_completed'],1)
        self.assertEqual(result['metrics']['counts']['math_check_completed'],0)
        self.assertEqual(result['completed'],dict(tests=1,checks=0));self.assertEqual(result['retry_count'],0)

    def test_missing_worker_keeps_unobserved_counts_null(self):
        result=self._execute(self._args(),self._process(missing_worker=True))
        self.assertEqual(result['status'],'stopped_on_error')
        self.assertTrue(all(value is None for value in result['metrics']['counts'].values()))

    def test_incomplete_counts_scope_identity_and_forbidden_actions_reject_success(self):
        variants=('counts','attempts','by_kind','scope','identity','zero','formal','security')
        for variant in variants:
            args=self._fresh_args(variant);worker=self._worker()
            if variant=='counts':worker['op_completed']=30335
            elif variant=='attempts':worker['constant_load_attempts']=2
            elif variant=='by_kind':worker['op_completed_by_kind']['add']-=1
            elif variant=='scope':worker['native_packed_equivalence_scope']='all graphs'
            elif variant=='identity':worker['round']='R24'
            elif variant=='zero':worker['keygen_runs']=1
            elif variant=='formal':worker['compiled_bootstraps']=19
            else:worker['security_validated']=True
            process=self._process(worker)
            with self.subTest(variant=variant):
                result=self._execute(args,process)
                self.assertEqual(result['status'],'stopped_on_error');self.assertEqual(result['retry_count'],0)
                self.assertEqual(process.calls[-1][0],'model-checks')

    def test_source_failure_stops_before_tests_and_packed_execution(self):
        self.locations['graph'].unlink();process=self._process()
        result=self._execute(self._args(),process)
        self.assertEqual(result['status'],'stopped_on_error');self.assertEqual(process.calls,[])
        self.assertEqual(result['attempted'],dict(tests=0,checks=0))
        self.assertTrue(all(value==0 for value in result['metrics']['counts'].values()))

    def test_empty_or_skipped_tests_stop_before_real_checks(self):
        for label,changes in [('empty',dict(test_count=0)),('skip',dict(skip=True))]:
            process=self._process(**changes)
            with self.subTest(label=label):
                result=self._execute(self._fresh_args(label),process)
                self.assertEqual(result['status'],'stopped_on_error');self.assertEqual(result['attempted']['checks'],0)
                self.assertEqual(process.calls[-1][0],'r25-tests')

    def test_native_signal_timeout_and_remaining_children_are_retained(self):
        for label,observation in [('native',dict(exit_code=-15)),('outer',dict(exit_code=-15,signal=15)),
                                  ('timeout',dict(timed_out=True)),('remaining',dict(remaining_pids=[98765]))]:
            with self.subTest(label=label):
                result=self._execute(self._fresh_args(label),self._process(fail='model-checks',**observation))
                self.assertEqual(result['status'],'stopped_on_error');command=result['processes'][-1]
                self.assertEqual(command['native_termination_signal'],15 if label in ('native','outer') else None)
                self.assertEqual(command['observation']['signal'],observation.get('signal'))
                self.assertEqual(result['outer_received_signals'],[15] if label=='outer' else [])

    def test_terminal_metadata_fixed_point_is_budgeted_and_failure_matches_registry(self):
        args,process=self._args(),self._process();write_json=self.outer.write_json
        def write(path,record):
            write_json(path,record)
            if Path(path).name=='timing.json':(args.out/'late-terminal.log').write_bytes(b'x'*1024**2)
        with patch.object(self.outer,'CASE_BYTES',1024**2),patch.object(self.outer,'write_json',side_effect=write):
            result=self._execute(args,process)
        self.assertEqual(result['status'],'stopped_on_error');self.assertTrue(result['terminal_budget_error'])
        self.assertEqual(result['case_artifact_bytes'],self.outer.case_size(args.out))
        self.assertEqual(json.loads((args.out/'result.json').read_text()),result)
        self.assertEqual(len(self._registry()),1);self.assertEqual(self._registry()[0]['metrics'],result['metrics'])

    def test_existing_or_unapproved_outputs_are_preserved_and_limits_fixed(self):
        existing=self._args().out;existing.mkdir();marker=existing/'saved';marker.write_text('preserved\n')
        for out in (existing,self.results/'other',self.locations['r22_result'].parent):
            with self.subTest(out=out),patch.object(self.outer,'_locations',return_value=self.locations), \
                    patch.object(self.outer,'run_process') as launch,self.assertRaises(ValueError):
                self.outer.execute(SimpleNamespace(out=out))
            launch.assert_not_called()
        self.assertEqual(marker.read_text(),'preserved\n');self.assertEqual(self._registry(),[])
        self.assertEqual((self.outer.OVERALL_SECONDS,self.outer.CASE_BYTES,self.outer.FREE_BYTES),
                         (1800,4*1024**3,10*1024**3))


if __name__ == '__main__':
    unittest.main()
