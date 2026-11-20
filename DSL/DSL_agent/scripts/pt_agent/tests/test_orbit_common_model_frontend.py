"""Small standard-library fakes; no original models, Nix, parser or solver."""
import copy
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pt_agent.experiments import orbit_common_model_frontend as subject
from pt_agent.experiments import orbit_common_model_worker as worker
from pt_agent.experiments.r10_paper_metrics import write_json


class OrbitCommonModelFrontendChecks(unittest.TestCase):
    def args(self, base, model='vgg16'):
        return subject.build_parser().parse_args(['--model',model,'--out',str(base/'new-case'),
            '--registry',str(base/'registry.jsonl')])

    def locations(self, base, model='vgg16'):
        factory=subject.MODEL_RECORDS[model]['factory']
        selected=worker.MODEL_RECORDS[factory]
        root=base/'orbit';poly=base/'local-poly';poly.mkdir()
        for name in ('MPCB.py','Poly.py','Func.py'):
            (poly/name).write_text('# synthetic public source\n')
        (poly/'data').mkdir()
        for name in worker.DATA_FILES:(poly/'data'/name).write_text('synthetic')
        locations=dict(model=factory,scripts=base,repo=base,orbit_root=root,
            frontend=root/'frontend/dacapo',model_source=root/'frontend/dacapo/python/poly/poly/models'/selected['source'],
            benchmark=root/'frontend/dacapo_patch/benchmarks'/(selected['benchmark']+'.py'),checkpoint=base/'checkpoint',
            r13_result=base/'r13.json',r16_result=base/'r16.json',profile=root/worker.PROFILE,
            intake_python=base/'intake-python',orbit_python=base/'orbit-python',poly_dir=poly,
            deps_dir=base/'deps',activations=selected['activations'])
        for key in ('benchmark','model_source','checkpoint','r13_result','r16_result','intake_python','orbit_python'):
            path=locations[key];path.parent.mkdir(parents=True,exist_ok=True);path.write_text('synthetic source only')
        locations['profile'].parent.mkdir(parents=True,exist_ok=True)
        write_json(locations['profile'],dict(polynomialDegree=131072))
        (locations['deps_dir']/'einops').mkdir(parents=True)
        (locations['deps_dir']/'einops/__init__.py').write_text('# fake no import\n')
        return locations

    def _record(self, phase, locations, work):
        worker.configure_model(locations['model'])
        record=worker._initial_result(phase)
        record.update(status='ok',elapsed_ns=100,phase_artifact_bytes=0)
        phase_dir=work/(phase+'-worker');phase_dir.mkdir(parents=True)
        if phase=='trace':
            for key in ('checkpoint_load_attempts','checkpoint_load_completed','model_build_attempts',
                'model_build_completed','expr_load_attempts','expr_load_completed','function_registration_attempts',
                'function_registrations','save_attempts','save_completed','finalize_attempts','finalize_completed','get_model_calls','trace_runs'):
                record[key]=1
            record.update(source_checked=True,environment_restored=True,orbit_model_trace_passed=True,
                activation_calls=locations['activations'],activation_names=['a'+str(i) for i in range(locations['activations'])])
            mlir=phase_dir/'trace/actual_returned_not_a_guessed_name.mlir';mlir.parent.mkdir();mlir.write_text('module { synthetic }')
            record['mlir']=dict(path=str(mlir),size_bytes=mlir.stat().st_size)
        else:
            for key in ('params_calls','params_completed','build_from_mlir_calls','build_from_mlir_completed','parse_runs'):record[key]=1
            record['tdag_parse_passed']=True
            graph=dict(nodes=[dict(id='arg0',op='input',level=None,scale=None),dict(id='0',op='rotate',op_descr=dict(offset=1),level=None,scale=None)],
                edges=[dict(source='arg0',target='0')],inputs=['arg0'],outputs=['0'])
            path=phase_dir/'tdag.json';write_json(path,graph)
            record.update(tdag=str(path),graph=graph,graph_check=worker._check_graph(graph),
                parser_metadata=dict(poly_deg=131072,max_slot=65536,profile=str(locations['profile'])))
        write_json(phase_dir/'phase_result.json',record)
        # Independent synthetic worker PID and same monotonic clock; no source execution.
        tick=time.perf_counter_ns();event=dict(event_id='fake-'+phase,stage_id='fake.'+phase,parent_id=None,start_ns=tick,pid=900001,ppid=1)
        (phase_dir/'events.jsonl').write_text(json.dumps(dict(event,type='begin'))+'\n'+json.dumps(dict(event,type='end',end_ns=tick+1,status='ok'))+'\n')
        return record

    def _execute(self, base, *, failure=None, wrong_coverage=False):
        args=self.args(base);locations=self.locations(base);commands=[]
        def run(argv,**kwargs):
            commands.append(list(argv));log=Path(kwargs['log']);log.write_text('a7b4c291\n' if argv[0]=='git' else 'synthetic\n')
            observation=dict(exit_code=0,timed_out=False,signal=None,signals_sent=[],remaining_pids=[],peak_rss_kib_by_pid={},outer_peak_rss_kib=0)
            if argv[0]!='git':
                phase=argv[argv.index('--phase')+1];work=args.out/'work'
                record=self._record(phase,locations,work)
                if wrong_coverage and phase=='trace':
                    record['activation_calls']-=1;write_json(work/'trace-worker/phase_result.json',record)
                if failure==phase:observation['exit_code']=1
                if phase=='parse':
                    expected=work/'trace-worker/trace/actual_returned_not_a_guessed_name.mlir'
                    self.assertEqual(argv[argv.index('--mlir')+1],str(expected.resolve()))
            return observation
        with patch.object(subject,'selected_paths',return_value=locations),patch.object(subject,'source_preflight',return_value=dict(original_model_modified=False)),\
             patch.object(subject,'run_process',side_effect=run),patch.object(subject.shutil,'disk_usage',return_value=SimpleNamespace(free=1<<40)):
            result=subject.execute(args)
        return args,locations,result,commands

    def test_all_three_explicit_dispatches_use_original_sources_and_same_profile(self):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory)
            for model in subject.MODEL_RECORDS:
                args=self.args(base,model);locations=subject.selected_paths(args)
                self.assertEqual(locations['model'],subject.MODEL_RECORDS[model]['factory'])
                self.assertEqual(locations['profile'],locations['orbit_root']/worker.PROFILE)
                self.assertIn('dacapo_patch/benchmarks',str(locations['benchmark']))
                self.assertEqual(locations['activations'],worker.MODEL_RECORDS[locations['model']]['activations'])

    def test_commands_keep_nix_trace_and_native_parse_limits_and_actual_path(self):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory);args=self.args(base);locations=self.locations(base);work=base/'work'
            trace=subject.phase_command(locations,work,'trace',args)
            self.assertIn('pt_agent.tools.nixrun',trace);self.assertEqual(trace[trace.index('--seconds')+1],'1860')
            self.assertEqual(trace[trace.index('--memory-gib')+1],'12')
            self.assertEqual(trace[trace.index('--phase-seconds')+1],'1800')
            with self.assertRaises(ValueError):subject.phase_command(locations,work,'parse',args)
            parse=subject.phase_command(locations,work,'parse',args,base/'actual.mlir')
            self.assertEqual(parse[:2],[str(locations['orbit_python']),'-B'])
            self.assertNotIn('pt_agent.tools.nixrun',parse)
            self.assertEqual(parse[parse.index('--memory-gib')+1],'4')
            self.assertEqual(parse[parse.index('--mlir')+1],str(base/'actual.mlir'))
            args.trace_seconds=1800.0
            explicit=subject.phase_command(locations,work,'trace',args)
            self.assertEqual(explicit[explicit.index('--seconds')+1],'1860')
            args.trace_seconds=1.25
            fractional=subject.phase_command(locations,work,'trace',args)
            self.assertEqual(fractional[fractional.index('--seconds')+1],'62')

    def test_environment_copy_preserves_home_and_removes_foreign_runtime_keys(self):
        original=dict(HOME='/public/home',KEEP='allowed',LD_LIBRARY_PATH='fake',NIX_PATH='fake',__NIX_TEST='fake',HECATE='fake',PYTHONHOME='fake',PYTHONPATH='fake')
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory);locations=dict(scripts=base/'scripts')
            env,metadata=subject.child_environment(base,locations,original)
            self.assertEqual(original['PYTHONPATH'],'fake');self.assertEqual(env['HOME'],original['HOME'])
            self.assertEqual(env['PYTHONPATH'],str(base/'scripts'));self.assertEqual(env['KEEP'],'allowed')
            self.assertNotIn('LD_LIBRARY_PATH',env);self.assertNotIn('NIX_PATH',env)
            self.assertTrue(all(env[key]=='1' for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')))
            self.assertEqual(metadata['controlled_pythonpath'],[str(base/'scripts')])

    def test_source_missing_dependency_stops_before_any_model_load(self):
        with tempfile.TemporaryDirectory() as directory:
            locations=self.locations(Path(directory));(locations['deps_dir']/'einops/__init__.py').unlink()
            with patch.object(worker,'_check_sources') as check:
                with self.assertRaisesRegex(FileNotFoundError,'einops'):subject.source_preflight(locations)
            check.assert_not_called()

    def test_source_preflight_reuses_small_saved_provenance_without_weight_load(self):
        with tempfile.TemporaryDirectory() as directory:
            locations=self.locations(Path(directory))
            with patch.object(worker,'_check_sources',return_value=dict(original_state_count=102,prior_times_included=False)) as check:
                actual=subject.source_preflight(locations)
            self.assertEqual(actual['logical_slots'],65536);self.assertFalse(actual['weights_modified'])
            self.assertEqual(actual['original_state_count'],102);check.assert_called_once()

    def test_complete_case_uses_actual_mlir_and_eight_field_unique_registry_and_end_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            args,_,result,commands=self._execute(Path(directory))
            self.assertEqual(result['status'],'ok',result.get('error'))
            self.assertEqual(result['attempted'],dict(trace=1,parse=1));self.assertEqual(result['completed'],dict(trace=1,parse=1))
            self.assertEqual(len(commands),3);self.assertTrue(result['completion']['registry_persisted'])
            rows=[json.loads(line) for line in args.registry.read_text().splitlines()]
            self.assertEqual(len(rows),1);self.assertEqual(set(rows[0]),{'run_id','round','model','mode','git_rev','metrics','artifacts_dir','ts'})
            final=json.loads((args.out/'completion.json').read_text());timing=json.loads((args.out/'timing.json').read_text())
            self.assertGreaterEqual(final['ended_ns'],timing['outer_interval']['end_ns'])
            self.assertEqual(final['case_artifact_bytes'],subject.artifact_bytes(args.out))
            self.assertTrue(final['exclusive_attribution_ready'])
            self.assertLessEqual((args.out/'completion.json').stat().st_size,result['completion_reserved_bytes'])
            self.assertTrue(all(result[k] is None for k in worker.UNMEASURED_FIELDS))
            with self.assertRaisesRegex(ValueError,'existing registry'):subject.append_registry(args.registry,result,args.out)

    def test_trace_failure_preserves_partial_and_prevents_parse(self):
        with tempfile.TemporaryDirectory() as directory:
            args,_,result,commands=self._execute(Path(directory),failure='trace')
            self.assertEqual(result['status'],'stopped_on_error');self.assertEqual(result['attempted'],dict(trace=1,parse=0))
            self.assertEqual(result['completed'],dict(trace=0,parse=0));self.assertEqual(len(commands),2)
            self.assertTrue((args.out/'result.json').is_file());self.assertTrue((args.out/'completion.json').is_file())

    def test_wrong_complete_activation_count_stops_before_parse_without_relaxation(self):
        with tempfile.TemporaryDirectory() as directory:
            _,_,result,commands=self._execute(Path(directory),wrong_coverage=True)
            self.assertEqual(result['status'],'stopped_on_error');self.assertIn('count differs',result['error']);self.assertEqual(len(commands),2)

    def test_preflight_error_persists_with_zero_children_and_one_registry(self):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory);args=self.args(base);locations=self.locations(base)
            with patch.object(subject,'selected_paths',return_value=locations),patch.object(subject,'source_preflight',side_effect=ValueError('synthetic source refusal')),\
                 patch.object(subject,'run_process') as run,patch.object(subject.shutil,'disk_usage',return_value=SimpleNamespace(free=1<<40)):
                result=subject.execute(args)
            run.assert_not_called();self.assertEqual(result['attempted'],dict(trace=0,parse=0))
            self.assertEqual(len(args.registry.read_text().splitlines()),1);self.assertTrue((args.out/'completion.json').is_file())

    def test_foreign_tdag_metadata_or_model_identity_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory);locations=self.locations(base);work=base/'work'
            record=self._record('parse',locations,work);record['graph_check']['node_count']+=1
            with self.assertRaisesRegex(ValueError,'Saved TDAG'):subject.validate_phase(record,'parse',locations,work)
            record['model']='custom:model'
            with self.assertRaisesRegex(ValueError,'identity'):subject.validate_phase(record,'parse',locations,work)

    def test_missing_terminal_evidence_keeps_excluded_action_counts_unknown(self):
        record=dict(attempted=dict(trace=1,parse=0),phase_results={})
        self.assertTrue(all(value is None for value in subject.observed_action_counts(record).values()))
        partial=dict(status='running',elapsed_ns=100,**dict.fromkeys(worker.ZERO_FIELDS,0))
        record['phase_results']['trace']=partial
        self.assertTrue(all(value is None for value in subject.observed_action_counts(record).values()))
        partial['status']='error';partial['solver_runs']=1
        actual=subject.observed_action_counts(record)
        self.assertEqual(actual['solver_runs'],1)
        self.assertTrue(all(actual[key]==0 for key in worker.ZERO_FIELDS if key!='solver_runs'))

    def test_disk_reserve_failure_persists_and_never_launches_a_child(self):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory);args=self.args(base);locations=self.locations(base)
            with patch.object(subject,'selected_paths',return_value=locations),patch.object(subject,'source_preflight') as check,\
                 patch.object(subject,'run_process') as run,patch.object(subject.shutil,'disk_usage',return_value=SimpleNamespace(free=9*subject.GIB)):
                result=subject.execute(args)
            run.assert_not_called();check.assert_not_called()
            self.assertEqual(result['status'],'stopped_on_error');self.assertIn('10 GiB',result['terminal_budget_error'])
            self.assertEqual(len(args.registry.read_text().splitlines()),1)
            self.assertTrue((args.out/'completion.json').is_file())

    def test_original_phase_seconds_default_is_preserved_and_out_of_range_refused_early(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(worker.resource,'setrlimit') as limits:
                with self.assertRaises(SystemExit):worker.main(['--phase','parse','--model',subject.MODEL_RECORDS['vgg16']['factory'],
                    '--work-dir',directory,'--phase-seconds','301'])
            limits.assert_not_called()

    def test_completion_allowance_failure_is_recorded_before_registry_success(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(subject,'completion_reserve_bytes',return_value=33*subject.GIB):
                args,_,result,_=self._execute(Path(directory))
            self.assertEqual(result['status'],'stopped_on_error')
            self.assertIn('completion marker',result['terminal_budget_error'])
            row=json.loads(args.registry.read_text())
            self.assertEqual(row['metrics']['status'],'stopped_on_error')
            self.assertEqual(json.loads((args.out/'completion.json').read_text())['status'],'stopped_on_error')


if __name__=='__main__':unittest.main()
