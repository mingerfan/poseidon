"""Resource adapter fixtures never load original models or native Orbit code."""
from contextlib import nullcontext, redirect_stderr
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from pt_agent.experiments import orbit_common_resource_retry as subject
from pt_agent.experiments import orbit_common_model_frontend as frontend
from pt_agent.experiments import orbit_common_model_worker as worker
from pt_agent.tests import test_orbit_common_model_frontend as fixtures


class ResourceRetryChecks(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);self.root=Path(temp.name)

    def safe(self):return dict(stop=False,reasons=[],subtree_rss_bytes=0,
        available_ram_bytes=2*subject.GIB,rss_allowance_bytes=14*subject.GIB,
        minimum_available_bytes=int(1.25*subject.GIB),sample_ts_ns=None)

    def test_outer_default12_and_explicit24_use_private_parser_only(self):
        original=frontend.main;namespace=subject.outer_namespace();seen=[]
        namespace['execute']=lambda args:(seen.append(args.memory_gib)or{'status':'ok'})
        base=['--model','alexnet','--out',str(self.root/'new')]
        self.assertEqual(namespace['main'](base),0)
        self.assertEqual(namespace['main'](base+['--memory-gib','24']),0)
        self.assertEqual(seen,[12,24])
        with redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):namespace['main'](base+['--memory-gib','25'])
        self.assertIs(frontend.main,original)
        with patch.object(frontend,'execute')as execute,redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
            frontend.main(base+['--memory-gib','24'])
        execute.assert_not_called()

    def test_worker24_reaches_real_limit_interface_with_unchanged_trace(self):
        namespace=subject.worker_namespace();calls=[]
        namespace.update(_initialize_environment=lambda:None,_wall_limit=lambda seconds:nullcontext(),
            trace_model=lambda args,recorder,result:calls.append(args.memory_gib))
        argv=['--phase','trace','--model',frontend.MODEL_RECORDS['alexnet']['factory'],
            '--memory-gib','24','--work-dir',str(self.root/'inner')]
        for key in ('frontend','model-source','benchmark','checkpoint','poly-dir','deps-dir','orbit-root','r13-result','r16-result'):
            argv+=['--'+key,str(self.root/key)]
        with patch.object(worker.resource,'setrlimit')as limit:
            self.assertEqual(namespace['main'](argv),0)
        self.assertEqual(calls,[24.])
        self.assertIn((worker.resource.RLIMIT_AS,(24*subject.GIB,24*subject.GIB)),[x.args for x in limit.call_args_list])
        record=json.loads((self.root/'inner/phase_result.json').read_text())
        self.assertEqual(record['address_space_limit_gib'],24)
        self.assertEqual(record['physical_RSS_guard_gib'],14)
        self.assertEqual(record['host_available_reserve_gib'],1.25)

    def test_parse_remains4_and_larger_inner_request_rejected_before_limits(self):
        namespace=subject.worker_namespace()
        with patch.object(worker.resource,'setrlimit')as limit,redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
            namespace['main'](['--phase','parse','--model',frontend.MODEL_RECORDS['alexnet']['factory'],
                '--memory-gib','5','--work-dir',str(self.root/'inner')])
        limit.assert_not_called()
        with patch.object(worker.resource,'setrlimit')as limit,redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
            namespace['main'](['--phase','trace','--model',frontend.MODEL_RECORDS['alexnet']['factory'],
                '--memory-gib','25','--work-dir',str(self.root/'inner')])
        limit.assert_not_called()

    def test_phase_argv_reaches_adapter24_and_actual_parse_path4(self):
        fake=fixtures.OrbitCommonModelFrontendChecks();args=fake.args(self.root,'alexnet');args.memory_gib=24
        locations=fake.locations(self.root,'alexnet');namespace=subject.outer_namespace()
        trace=namespace['phase_command'](locations,self.root/'work','trace',args)
        parsed=namespace['phase_command'](locations,self.root/'work','parse',args,self.root/'actual.mlir')
        self.assertIn(subject.MODULE,trace);self.assertIn(subject.MODULE,parsed)
        self.assertEqual(trace[trace.index('--memory-gib')+1],'24')
        self.assertEqual(parsed[parsed.index('--memory-gib')+1],'4')
        self.assertEqual(trace[trace.index('--phase-seconds')+1],'1800')
        self.assertEqual(parsed[parsed.index('--phase-seconds')+1],'300')
        self.assertEqual(parsed[parsed.index('--mlir')+1],str(self.root/'actual.mlir'))

    def test_all_original_source_graph_and_model_operations_delegate_same_code(self):
        outer=subject.outer_namespace();inner=subject.worker_namespace()
        for name in ('source_preflight','selected_paths','validate_phase','child_environment'):
            self.assertIs(outer[name].__code__,getattr(frontend,name).__code__)
        for name in ('_check_sources','trace_model','parse_model','_load_model','_check_graph','_profile_params'):
            self.assertIs(inner[name].__code__,getattr(worker,name).__code__)
        self.assertEqual(outer['ROUND'],frontend.ROUND);self.assertEqual(outer['MODE'],frontend.MODE)

    def options(self):return dict(cwd=self.root,log=self.root/'trace.log',resources=self.root/'trace-resources.jsonl',
        timeout=2040,mem_gib=24,fsize_gib=8)

    def test_git_probe_has_no_physical_guard_or_changed_allowance(self):
        run=Mock(return_value={'exit_code':0})
        with patch.object(subject,'_guard_observation')as observation:
            subject.guarded_run(['git','rev-parse','HEAD'],state={'phases':[]},run=run,**self.options())
        observation.assert_not_called();self.assertEqual(run.call_args.kwargs['mem_gib'],24)

    def test_admission_memory_failure_starts_no_child_and_saves_reason(self):
        failed=dict(self.safe(),stop=True,reasons=['host available RAM reserve exhausted'])
        run=Mock();state={'phases':[]}
        with patch.object(subject,'_guard_observation',return_value=failed),self.assertRaises(RuntimeError):
            subject.guarded_run(['python','--phase','trace'],state=state,run=run,**self.options())
        run.assert_not_called();self.assertTrue(state['phases'][0]['stop_observed'])
        self.assertTrue((self.root/'physical-memory-guard.json').is_file())

    def test_guard_checks_RSS_and_host_reserve_independent_of_virtual_space(self):
        with patch.object(subject,'_latest_rss_sample',return_value=dict(subtree_rss_bytes=15*subject.GIB)),\
                patch.object(subject,'_available_ram_bytes',return_value=3*subject.GIB):
            self.assertTrue(subject._guard_observation(self.root/'x')['stop'])
        with patch.object(subject,'_latest_rss_sample',return_value=None),\
                patch.object(subject,'_available_ram_bytes',return_value=subject.GIB):
            self.assertTrue(subject._guard_observation(self.root/'x')['stop'])

    def test_normal_run_and_runner_exception_keep_summary_and_no_retry(self):
        for error in (False,True):
            state={'phases':[]};run=Mock(side_effect=OSError('supervisor failed'))if error else Mock(return_value={'exit_code':0})
            with patch.object(subject,'_guard_observation',return_value=self.safe()):
                if error:
                    with self.assertRaises(OSError):subject.guarded_run(['python','--phase','trace'],state=state,run=run,**self.options())
                else:subject.guarded_run(['python','--phase','trace'],state=state,run=run,**self.options())
            self.assertEqual(run.call_count,1);self.assertEqual(state['phases'][0]['AS_gib'],24)
            self.assertTrue((self.root/'physical-memory-summary.json').is_file())

    def fake_execute(self,failure=None):
        fixtures_case=fixtures.OrbitCommonModelFrontendChecks()
        args=fixtures_case.args(self.root,'alexnet');args.memory_gib=24
        locations=fixtures_case.locations(self.root,'alexnet');seen=[]
        def runner(argv,**options):
            Path(options['log']).write_text('synthetic-revision\n')
            observation=dict(exit_code=0,timed_out=False,signal=None,signals_sent=[],remaining_pids=[])
            if argv[0]!='git':
                phase=argv[argv.index('--phase')+1];seen.append((phase,options['mem_gib']))
                fixtures_case._record(phase,locations,args.out/'work')
                if failure==phase:observation['exit_code']=1
            return observation
        with patch.object(frontend,'run_process',side_effect=runner),\
                patch.object(subject,'_guard_observation',return_value=self.safe()),\
                patch.object(frontend.shutil,'disk_usage',return_value=SimpleNamespace(free=1<<40)):
            namespace=subject.outer_namespace()
            namespace.update(selected_paths=lambda args:locations,source_preflight=lambda locations:{'synthetic':True})
            result=namespace['execute'](args)
        return args,result,seen

    def test_success_persists8field_registry_completion_and_separate_memory_semantics(self):
        args,result,seen=self.fake_execute()
        self.assertEqual(result['status'],'ok',result.get('error'))
        self.assertEqual(seen,[('trace',24),('parse',4)])
        self.assertEqual(result['attempted'],{'trace':1,'parse':1});self.assertEqual(result['completed'],{'trace':1,'parse':1})
        self.assertEqual(result['limits']['trace_memory_gib'],24);self.assertEqual(result['limits']['parse_memory_gib'],4)
        self.assertEqual(result['limits']['physical_subtree_RSS_gib'],14)
        row=json.loads(args.registry.read_text())
        self.assertEqual(set(row),{'run_id','round','model','mode','git_rev','metrics','artifacts_dir','ts'})
        self.assertEqual(row['metrics']['resource_adaptation']['automatic_retries'],0)
        self.assertTrue((args.out/'completion.json').is_file());self.assertTrue((args.out/'source/resource-adapter.py').is_file())

    def test_one_trace_failure_preserves_result_and_never_parses(self):
        args,result,seen=self.fake_execute(failure='trace')
        self.assertEqual(result['status'],'stopped_on_error');self.assertEqual(seen,[('trace',24)])
        self.assertEqual(result['attempted'],{'trace':1,'parse':0})
        self.assertTrue((args.out/'result.json').is_file());self.assertTrue((args.out/'completion.json').is_file())

    def test_successful_metadata_passes_original_optimizer_gate_without_optimizer(self):
        from pt_agent.experiments import orbit_common_model_optimizer_worker as optimizer
        args,result,_=self.fake_execute();parsed=result['phase_results']['parse']
        work=self.root/'optimizer-source-check';work.mkdir()
        proof=optimizer._check_frontend_source(SimpleNamespace(model=result['model'],
            source_record=args.out/'result.json',profile=Path(parsed['parser_metadata']['profile']),
            input_mlir=Path(result['phase_results']['trace']['mlir']['path']),work_dir=work))
        self.assertEqual(proof['source_round'],frontend.ROUND)
        self.assertEqual(proof['source_status'],'ok')
        self.assertEqual(proof['activation_count'],7)

    def test_entry_dispatches_inner_phase_without_outer_results_lookup(self):
        with patch.object(subject,'worker_namespace',return_value={'main':lambda argv:7})as inner,\
                patch.object(subject,'outer_namespace')as outer:
            self.assertEqual(subject.main(['--phase','trace']),7)
        inner.assert_called_once();outer.assert_not_called()


if __name__=='__main__':unittest.main()
