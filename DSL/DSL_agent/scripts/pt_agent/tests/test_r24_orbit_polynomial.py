"""R24 stdlib protocol checks and one dedicated intake-environment wrapper."""

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


INTAKE_PYTHON = Path(os.environ.get('R24_INTAKE_PYTHON', str(paths.INTAKE_PYTHON)))
INTAKE_ARTIFACT = INTAKE_PYTHON.parent if INTAKE_PYTHON.is_symlink() else INTAKE_PYTHON
BOOTSTRAP = r'''import json, os, signal, sys, tempfile, unittest
from pathlib import Path
work = Path(sys.argv[1]).resolve()
os.environ['R24_TEST_WORK'] = str(work)
os.environ['TMPDIR'] = str(work)
tempfile.tempdir = str(work)
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
signal.alarm(300)
suite = unittest.defaultTestLoader.loadTestsFromName('pt_agent.tests._r24_orbit_polynomial_cases.PolynomialCases')
result = unittest.TextTestRunner(verbosity=2).run(suite)
record = dict(testsRun=result.testsRun, failures=len(result.failures), errors=len(result.errors),
              skips=len(result.skipped), success=result.wasSuccessful())
(work / 'inner-unit-result.json').write_text(json.dumps(record, allow_nan=False) + '\n')
raise SystemExit(0 if record['success'] and record['testsRun'] > 0 and record['skips'] == 0 else 1)
'''


@require(INTAKE_ARTIFACT)
class ValidationEnvironmentChecks(unittest.TestCase):
    def test_only_current_synthetic_polynomial_cases_run_once_without_skips(self):
        from pt_agent.experiments.r10_paper_metrics import run_process

        tree = ast.parse(Path(__file__).with_name('_r24_orbit_polynomial_cases.py').read_text())
        classes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'PolynomialCases']
        self.assertEqual(len(classes), 1)
        expected = sum(isinstance(node, ast.FunctionDef) and node.name.startswith('test_')
                       for node in classes[0].body)
        self.assertGreater(expected, 0)
        work = Path(os.environ['R24_TEST_WORK']).resolve(); work.mkdir(parents=True, exist_ok=True)
        unit, summary = work / 'inner-unit-result.json', work / 'inner-wrapper-result.json'
        self.assertFalse(unit.exists(), 'The dedicated inner suite runs once')
        self.assertFalse(summary.exists(), 'The dedicated wrapper runs once')
        command = [sys.executable, '-B', '-m', 'pt_agent.tools.nixrun', '--seconds', '360',
                   '--python', str(INTAKE_PYTHON), '--', '-B', '-c', BOOTSTRAP, str(work)]
        log = work / 'inner-tests.log'
        observation = run_process(command, cwd=Path(__file__).resolve().parents[2],
            log=log, resources=work / 'inner-resources.jsonl', timeout=480, mem_gib=4, fsize_gib=8,
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
        print('R24_INNER_TESTS=' + str(expected), flush=True)
        print('R24_REAL_WEIGHTS_PROBES_MODELS_AND_TOOLCHAIN_CALLS_IN_TESTS=0', flush=True)


class OuterProtocolChecks(unittest.TestCase):
    def setUp(self):
        from pt_agent.experiments import r24_orbit_polynomial_preflight as outer
        self.outer=outer
        temporary=TemporaryDirectory(prefix='r24-outer-',dir=os.environ.get('R24_TEST_WORK'))
        self.addCleanup(temporary.cleanup);self.root=Path(temporary.name)
        self.results=self.root/'results';self.results.mkdir()
        r13=self.results/'r13-original';r22=self.results/'r22-original'
        self.locations=dict(results=self.results,repo=self.root/'repo',scripts=self.root/'repo/scripts',
            intake_python=self.root/'intake/bin/python',orbit=self.root/'orbit',
            model_source=self.root/'orbit/ResNetSiLU.py',checkpoint=self.root/'original-model.th',
            probes=r13/'worker/training_probes.npz',probe_source=r13/'worker/training_probe_source.json',
            r13_result=r13/'result.json',r16_result=self.results/'r16-original/result.json',
            r22_result=r22/'result.json',r23_result=self.results/'r23-original/result.json',
            poly_dir=r22/'work/local-poly/poly')
        for key in ('intake_python','model_source','checkpoint','probes','probe_source'):
            path=self.locations[key];path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text('synthetic metadata fixture; never loaded\n')
        self.locations['scripts'].mkdir(parents=True)
        for name in outer.POLY_FILES:
            path=self.locations['poly_dir']/name;path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text('synthetic source fixture; never executed\n')
        self.records={}
        for number in (13,16,22,23):
            self.records[number]=dict(round='R'+str(number),model=outer.MODEL,status='ok')
        self.records[13]['worker']=dict(artifacts=dict(training_probes=str(self.locations['probes']),
            training_probe_source=str(self.locations['probe_source'])))
        self.records[16].update(baseline='orbit',worker=dict(compatibility_passed=True,
            source_summary={key:str(self.locations[key]) for key in ('checkpoint','model_source')},
            math_probes=[dict(sample_id='train:'+str(i),passed=True,reference_shape=[1,10],
                reference_dtype='float64',criterion='1e-12+1e-12*abs(reference)') for i in range(3)]))
        self.records[22].update(mode='orbit-resnet20-trace-tdag-preflight',
            orbit_model_trace_passed=True,tdag_parse_passed=True)
        self.records[23].update(mode='orbit-resnet20-native-optimizer-preflight',
            native_optimizer_completed=True,placement_artifact_checked=True,placement_bootstrap_nodes=19)
        for number,record in self.records.items():self._save(self.locations['r'+str(number)+'_result'],record)

    @staticmethod
    def _save(path,record):
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(record,allow_nan=False)+'\n')

    def _args(self):return SimpleNamespace(out=self.results/self.outer.OUT_NAME)

    def _fresh_args(self,label):
        self.results=self.root/label/'results';self.results.mkdir(parents=True)
        self.locations['results']=self.results
        return self._args()

    def _registry(self):
        path=self.results/'registry.jsonl'
        return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []

    def _worker(self):
        from pt_agent.experiments import r24_orbit_polynomial_worker as worker
        result=worker._initial_result()
        result.update(status='ok',**dict.fromkeys(worker.STAGE_FIELDS,True))
        result.update({name:(57 if name.startswith(('activation_','math_check_')) else
                             3 if name.startswith(('original_forward_','approx_forward_')) else 1)
                       for name in self.outer.COUNTERS})
        result.update(original_reference_checks=3,silu_modules_replaced=10)
        return result

    def _process(self,worker=None,*,missing_worker=False,fail=None,test_count=14,skip=False,**overrides):
        worker=self._worker() if worker is None else worker
        calls=[]
        def run(command,**options):
            stage=Path(options['log']).stem;calls.append((stage,command,options))
            observation=dict(exit_code=0,timed_out=False,signal=None,remaining_pids=[],signals_sent=[],
                started_ns=1,ended_ns=2,termination_observed_ns=3,
                peak_rss_kib_by_pid={},outer_peak_rss_kib=0)
            log=Path(options['log'])
            if stage.startswith('git-'):
                log.write_text('fake-revision\n' if stage=='git-revision' else 'lhy-agent-dsl\n')
            elif stage=='r24-tests':
                log.write_text('Ran '+str(test_count)+' tests in 0.001s\n\nOK'+(' (skipped=1)' if skip else '')+'\n')
                self._save(Path(options['env']['R24_TEST_WORK'])/'inner-wrapper-result.json',
                    dict(status='ok',wrapper_runs=1,inner_tests=14,inner_skips=0,observation=observation))
            else:
                log.write_text('fake mathematical worker; no native program or model execution\n')
                work=Path(command[command.index('--work-dir')+1]);work.mkdir(parents=True)
                (work/'events.jsonl').touch()
                if not missing_worker:self._save(work/'phase_result.json',worker)
            if stage==fail:
                observation.update(exit_code=1);observation.update(overrides)
            return observation
        run.calls=calls
        return run

    def _execute(self,args,process):
        with ExitStack() as stack:
            stack.enter_context(patch.object(self.outer,'_locations',return_value=self.locations))
            stack.enter_context(patch.object(self.outer,'run_process',side_effect=process))
            stack.enter_context(patch.object(self.outer,'_base_environment_snapshot',return_value=[]))
            stack.enter_context(patch.object(self.outer.resource,'setrlimit'))
            stack.enter_context(patch.object(self.outer.signal,'signal'))
            stack.enter_context(patch.object(self.outer.signal,'setitimer'))
            stack.enter_context(patch.object(self.outer.shutil,'disk_usage',return_value=SimpleNamespace(free=11*1024**3)))
            stack.enter_context(patch.object(self.outer.threading,'Thread',
                return_value=SimpleNamespace(start=Mock(),is_alive=lambda:False)))
            stack.enter_context(redirect_stdout(io.StringIO()))
            return self.outer.execute(args)

    def test_source_records_keep_original_cached_paths_without_opening_arrays_or_weights(self):
        source=self.outer._source_preflight(self.locations)
        self.assertEqual(source['saved_sample_ids'],['train:0','train:1','train:2'])
        self.assertFalse(source['sample_contents_read']);self.assertFalse(source['weights_read'])
        self.assertFalse(source['native_packed_equivalence_verified'])
        self.assertEqual(len(source['inputs']),13)

    def test_source_rejects_identity_reference_policy_and_foreign_saved_paths(self):
        variants=('model','probes','checkpoint','dtype','criterion','native_stage','poly_dir')
        for variant in variants:
            records=copy.deepcopy(self.records);locations=dict(self.locations)
            if variant=='model':records[13]['model']='foreign'
            elif variant=='probes':records[13]['worker']['artifacts']['training_probes']=str(self.root/'foreign.npz')
            elif variant=='checkpoint':records[16]['worker']['source_summary']['checkpoint']=str(self.root/'foreign.th')
            elif variant=='dtype':records[16]['worker']['math_probes'][0]['reference_dtype']='float32'
            elif variant=='criterion':records[16]['worker']['math_probes'][0]['criterion']='relaxed'
            elif variant=='native_stage':records[23]['placement_artifact_checked']=False
            else:
                locations['poly_dir']=self.root/'foreign-poly'
                for name in self.outer.POLY_FILES:
                    path=locations['poly_dir']/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('fixture\n')
            for number,record in records.items():self._save(locations['r'+str(number)+'_result'],record)
            with self.subTest(variant=variant),self.assertRaises(ValueError):self.outer._source_preflight(locations)
        for number,record in self.records.items():self._save(self.locations['r'+str(number)+'_result'],record)

    def test_command_has_only_explicit_original_sources_and_finite_nix_limits(self):
        work=self.root/'case/work';command=self.outer.phase_command(self.locations,work)
        self.assertEqual(command[command.index('--seconds')+1],'360')
        self.assertEqual(command[command.index('--python')+1],str(self.locations['intake_python']))
        self.assertIn(self.outer.MODULE,command)
        self.assertEqual(command[command.index('--work-dir')+1],str(work/'checks-worker'))
        for key in ('model_source','checkpoint','probes','probe_source','r13_result','r16_result','r22_result','r23_result','poly_dir'):
            option='--'+key.replace('_','-');self.assertEqual(command.count(option),1)
            self.assertEqual(command[command.index(option)+1],str(self.locations[key]))
        for option in ('--normalization','--shape','--provider','--report-samples','--fit'):
            self.assertNotIn(option,command)

    def test_environment_passes_wrapper_paths_and_resets_four_numeric_threads(self):
        with patch.dict(os.environ,HECATE='old',LD_LIBRARY_PATH='old',PYTHONPATH='old'):
            env=self.outer.child_environment(self.root/'work',self.locations)
        for name in ('HECATE','LD_LIBRARY_PATH','PYTHONPATH'):self.assertNotIn(name,env)
        for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
            self.assertEqual(env[name],'1')
        self.assertEqual(env['R24_TEST_WORK'],str(self.root/'work/test-worker'))
        self.assertEqual(env['R24_INTAKE_PYTHON'],str(self.locations['intake_python']))

    def test_success_preserves_exact_counts_null_formal_metrics_and_unique_registry(self):
        args,process=self._args(),self._process();result=self._execute(args,process)
        self.assertEqual(result['status'],'ok',result.get('error'))
        self.assertEqual([row[0] for row in process.calls],['git-revision','git-branch','r24-tests','model-checks'])
        self.assertEqual(result['attempted'],dict(tests=1,checks=1));self.assertEqual(result['completed'],result['attempted'])
        self.assertEqual((result['git_metadata_attempted'],result['git_metadata_completed']),(2,2))
        self.assertEqual([row[2]['timeout'] for row in process.calls],[30,30,600,480])
        self.assertTrue(all((row[2]['mem_gib'],row[2]['fsize_gib'])==(4,8) for row in process.calls))
        self.assertEqual(result['metrics']['counts'],{key:self._worker()[key] for key in self.outer.COUNTERS})
        self.assertTrue(all(result['metrics'][key] is None for key in self.outer.FORMAL))
        self.assertFalse(result['native_packed_equivalence_verified']);self.assertFalse(result['formal_matrix_complete'])
        self.assertEqual(result['case_artifact_bytes'],self.outer.case_size(args.out))
        rows=self._registry();self.assertEqual(len(rows),1)
        self.assertEqual(set(rows[0]),{'run_id','round','model','mode','git_rev','metrics','artifacts_dir','ts'})
        self.assertEqual((rows[0]['round'],rows[0]['mode']),('R24',self.outer.MODE))
        self.assertEqual(rows[0]['metrics'],result['metrics'])
        timing=json.loads((args.out/'timing.json').read_text());self.assertTrue(timing['attribution_ready'])
        self.assertEqual(timing['diagnostics'],[])
        roots=[row for row in timing['spans'] if row['parent_id'] is None];self.assertEqual(len(roots),1)
        self.assertEqual(sum(row['exclusive_ns'] for row in timing['spans']),roots[0]['inclusive_ns'])

    def test_partial_math_failure_keeps_the_three_original_references_passed(self):
        worker=self._worker();worker.update(status='math_failed',original_reference_passed=True,
            native_polynomial_math_passed=False,physical_scaled_forward_completed=False,
            approx_forward_attempts=1,approx_forward_completed=0,activation_attempts=6,activation_completed=5,
            math_check_attempts=6,math_check_completed=5)
        result=self._execute(self._args(),self._process(worker,fail='model-checks'))
        self.assertEqual(result['status'],'stopped_on_error');self.assertTrue(result['original_reference_passed'])
        self.assertFalse(result['native_polynomial_math_passed'])
        self.assertEqual(result['metrics']['counts']['original_forward_completed'],3)
        self.assertEqual(result['metrics']['counts']['activation_completed'],5)
        self.assertEqual(result['attempted'],dict(tests=1,checks=1))
        self.assertEqual(result['completed'],dict(tests=1,checks=0))

    def test_missing_worker_keeps_unobserved_counts_null(self):
        result=self._execute(self._args(),self._process(missing_worker=True))
        self.assertEqual(result['status'],'stopped_on_error')
        self.assertTrue(all(value is None for value in result['metrics']['counts'].values()))

    def test_incomplete_or_extra_attempt_counts_are_rejected_without_retry(self):
        for key in ('math_check_completed','checkpoint_load_attempts'):
            args=self._fresh_args(key);worker=self._worker()
            worker[key]=56 if key=='math_check_completed' else 2
            process=self._process(worker)
            with self.subTest(key=key):
                result=self._execute(args,process);self.assertEqual(result['status'],'stopped_on_error')
                self.assertEqual(result['retry_count'],0);self.assertEqual(process.calls[-1][0],'model-checks')

    def test_source_failure_stops_before_git_tests_and_real_checks(self):
        self.locations['checkpoint'].unlink();process=self._process()
        result=self._execute(self._args(),process)
        self.assertEqual(result['status'],'stopped_on_error');self.assertEqual(process.calls,[])
        self.assertEqual(result['attempted'],dict(tests=0,checks=0))
        self.assertTrue(all(value==0 for value in result['metrics']['counts'].values()))

    def test_empty_or_skipped_tests_stop_before_checks(self):
        for label,changes in [('empty',dict(test_count=0)),('skip',dict(skip=True))]:
            process=self._process(**changes)
            with self.subTest(label=label):
                result=self._execute(self._fresh_args(label),process)
                self.assertEqual(result['status'],'stopped_on_error');self.assertEqual(result['attempted']['checks'],0)
                self.assertEqual(process.calls[-1][0],'r24-tests')

    def test_native_signal_timeout_and_remaining_processes_are_saved_separately(self):
        for label,observation in [('native',dict(exit_code=-15)),('outer',dict(exit_code=-15,signal=15)),
                                  ('timeout',dict(timed_out=True)),('remaining',dict(remaining_pids=[98765]))]:
            with self.subTest(label=label):
                result=self._execute(self._fresh_args(label),self._process(fail='model-checks',**observation))
                self.assertEqual(result['status'],'stopped_on_error');command=result['processes'][-1]
                self.assertEqual(command['native_termination_signal'],15 if label in ('native','outer') else None)
                self.assertEqual(command['observation']['signal'],observation.get('signal'))
                self.assertEqual(result['outer_received_signals'],[15] if label=='outer' else [])

    def test_terminal_metadata_is_in_case_budget_and_registry_matches_failure(self):
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
                         (1500,4*1024**3,10*1024**3))


if __name__ == '__main__':
    unittest.main()
