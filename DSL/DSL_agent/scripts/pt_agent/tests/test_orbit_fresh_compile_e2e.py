"""New continuous Orbit orchestration; synthetic metadata only, no tools."""
import copy
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from pt_agent.experiments import orbit_fresh_compile_e2e as entry


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False))


def clean():
    return dict(exit_code=0, timed_out=False, signal=None, signals_sent=[], remaining_pids=[])


class ContinuousCompileTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.args = SimpleNamespace(model='vgg16', out=self.root/'fresh-vgg', seconds=entry.TOTAL_SECONDS,
            shared_build_case=self.root/'shared', go_sdk=self.root/'sdk', initialization_audit=self.root/'init.json',
            orbit_root=self.root/'orbit', registry=self.root/'registry.jsonl', checkpoint=None,
            r13_result=None, r16_result=None, poly_dir=None, deps_dir=None, intake_python=None, orbit_python=None)
        put(self.args.shared_build_case/'result.json', {'status':'ok'})
        put(self.args.shared_build_case/'completion.json', {'status':'ok'})
        put(self.args.initialization_audit, {'synthetic':True})
        self.args.go_sdk.mkdir(); self.args.orbit_root.mkdir(); self.args.registry.touch()
        patch = mock.patch.object(entry.paths, 'RESULTS_DIR', self.root); patch.start(); self.addCleanup(patch.stop)
        self.calls = []; self.transform = lambda stage, result:result

    def child_record(self, stage, cases):
        model = entry.frontend.MODEL_RECORDS[self.args.model]['factory']
        if stage in ('frontend','optimizer'):
            module = entry.frontend if stage=='frontend' else entry.optimizer
            phases = ('trace','parse') if stage=='frontend' else ('optimize','validate')
            result = dict(round=module.ROUND, mode=module.MODE, model=model, model_key=self.args.model,
                status='ok', attempted=dict.fromkeys(phases,1), completed=dict.fromkeys(phases,1),
                processes=[dict(phase=p,observation=clean()) for p in phases],
                metrics=dict(status='ok'), **dict.fromkeys(module.worker.ZERO_FIELDS,0))
            if stage=='frontend':
                result.update(orbit_model_trace_passed=True,tdag_parse_passed=True,
                    source_records=dict(model=model,original_model_modified=False,weights_modified=False,polynomial_modified=False))
            else:
                result.update(native_optimizer_completed=True,placement_artifact_checked=True,
                    source_records=dict(source_record=str(cases['frontend']/'result.json')))
        else:
            result=dict(round=entry.parser_entry.ROUND,mode=entry.parser_entry.MODE,model=model,status='ok',
                attempted=dict.fromkeys(('build','selftest','parse'),1),completed=dict.fromkeys(('build','selftest','parse'),1),
                compiler_graph_completed=True,backend_parser_consumption_verified=True,model_specific_native_codegen=None,
                formal_latency=None,accuracy_acceptance=None,security_acceptance=None,
                phases={p:dict(observation=clean()) for p in ('build','selftest','parse')},
                source=dict(shared_build_case=str(self.args.shared_build_case),optimized_source=dict(
                    model=model,model_key=self.args.model,source_result=str(cases['optimizer']/'result.json'),
                    frontend_result=str(cases['frontend']/'result.json'),placement_bootstrap_nodes=19)),
                **dict.fromkeys(entry.parser_entry.ZERO_FIELDS,0))
        return result

    def persist_child(self, stage, cases, *, duplicate=False):
        case=cases[stage]; case.mkdir(parents=True)
        result=self.transform(stage,self.child_record(stage,cases));put(case/'result.json',result)
        put(case/'completion.json',dict(status='ok',required_artifacts_persisted=True,registry_persisted=True,
            complete_case_elapsed_seconds=1000.0,timing_status='ok',exclusive_attribution_ready=True,
            overlap_and_timing_diagnostics=[]))  # Deliberately unrelated child clock, never summed.
        metrics=result.get('metrics')
        if metrics is None:
            metrics={key:result[key] for key in ('status','attempted','completed','backend_parser_consumption_verified',
                'compiler_graph_completed','model_specific_native_codegen','formal_latency','accuracy_acceptance','security_acceptance')}
            metrics.update(dict.fromkeys(entry.parser_entry.ZERO_FIELDS,0))
        row=dict(run_id=case.name,round=result['round'],model=result['model'],mode=result['mode'],
            git_rev=None,metrics=metrics,artifacts_dir=str(case),ts='synthetic')
        with self.args.registry.open('a') as stream:
            stream.write(json.dumps(row)+'\n')
            if duplicate:stream.write(json.dumps(row)+'\n')
        return result

    def runner(self, argv, **options):
        stage=entry.STAGES[len(self.calls)];self.calls.append((stage,argv,options))
        self.persist_child(stage,entry.child_cases(self.args))
        return clean()

    def run_fake(self, runner=None, observation=None):
        with mock.patch.object(entry,'run_process',side_effect=runner or self.runner), \
             mock.patch.object(entry,'resource_observation',side_effect=observation or (lambda args:dict(stop=False,reasons=[]))), \
             mock.patch('pt_agent.registry.git_rev',return_value='synthetic'):
            return entry.execute(self.args)

    def test_continuous_master_uses_real_closure_and_never_sums_child_clocks(self):
        result=self.run_fake()
        self.assertEqual(result['status'],'ok',result['error']);self.assertTrue(result['accepted'])
        self.assertEqual([x[0] for x in self.calls],list(entry.STAGES))
        self.assertEqual(result['attempted'],result['completed']);self.assertEqual(result['placement_bootstrap_nodes'],19)
        marker=entry.read(self.args.out/'completion.json')
        self.assertTrue(marker['registry_persisted']);self.assertTrue(marker['required_artifacts_persisted'])
        self.assertLess(marker['complete_case_elapsed_seconds'],1000)
        self.assertEqual(marker['ended_ns']-marker['started_ns'],marker['complete_case_elapsed_ns'])
        self.assertFalse(result['historical_times_added']);self.assertFalse(result['original_trace_reused'])
        rows=[json.loads(line) for line in self.args.registry.read_text().splitlines()]
        self.assertEqual(len(rows),4);self.assertEqual(len({r['run_id'] for r in rows}),4)
        self.assertTrue(all(set(r)==entry.REGISTRY_FIELDS for r in rows))

    def test_all_models_keep_original_commands_parameters_and_limits(self):
        for model in entry.MODELS:
            self.args.model=model; cases=entry.child_cases(self.args);commands=entry.commands(self.args,cases)
            self.assertEqual(commands['frontend'][3],entry.retry.MODULE)
            self.assertEqual(commands['frontend'][commands['frontend'].index('--memory-gib')+1],'24')
            self.assertEqual(commands['optimizer'][commands['optimizer'].index('--optimize-seconds')+1],'7200')
            self.assertEqual(commands['optimizer'][commands['optimizer'].index('--source-result')+1],str(cases['frontend']/'result.json'))
            self.assertEqual(commands['parser'][commands['parser'].index('--source-case')+1],str(cases['optimizer']))
            self.assertIn('--run',commands['parser'])
            self.assertEqual('--delegate-parser' in commands['parser'],model=='resnet20')
            self.assertTrue(all(p.is_relative_to(self.args.out) for p in cases.values()))
            self.assertTrue(all(self.args.out.name in p.name for p in cases.values()))

    def test_explicit_original_paths_propagate_without_resolving_interpreter_symlink(self):
        for name in ('checkpoint','r13_result','r16_result','poly_dir','deps_dir','intake_python','orbit_python'):
            setattr(self.args,name,self.root/name)
        commands=entry.commands(self.args,entry.child_cases(self.args))
        for name in ('checkpoint','r13_result','r16_result','poly_dir','deps_dir','intake_python','orbit_python'):
            argv=commands['frontend'];flag='--'+name.replace('_','-')
            self.assertEqual(argv[argv.index(flag)+1],str(getattr(self.args,name)))

    def test_actual_common_parser_main_dispatches_private_Res_model_record(self):
        original_records=copy.deepcopy(entry.parser_entry.MODEL_RECORDS)
        namespace=entry.parser_namespace();seen=[]
        def fake_execute(args):
            seen.append(namespace['MODEL_RECORDS'][args.model]['factory'])
            return dict(model=seen[-1],status='ok',error=None,compiler_graph_completed=True,backend_parser_consumption_verified=True)
        namespace['execute']=fake_execute
        argv=['--delegate-parser','--model','resnet20','--out',str(self.root/'new-parser'),
            '--source-case',str(self.root/'new-optimizer'),'--shared-build-case',str(self.args.shared_build_case),
            '--orbit-root',str(self.args.orbit_root),'--go-sdk',str(self.args.go_sdk),
            '--initialization-audit',str(self.args.initialization_audit),'--run']
        with mock.patch.object(entry,'parser_namespace',return_value=namespace):self.assertEqual(entry.main(argv),0)
        self.assertEqual(seen,[entry.frontend.MODEL_RECORDS['resnet20']['factory']])
        for name in ('main','source_gate','_target_source','_namespace','_completion'):
            self.assertIs(namespace[name].__globals__,namespace)
            self.assertIs(namespace[name].__code__,getattr(entry.parser_entry,name).__code__)
        self.assertIs(namespace['CLIENT'],entry.parser_entry.CLIENT)
        self.assertEqual(entry.parser_entry.MODEL_RECORDS,original_records)
        self.assertNotIn('resnet20',original_records)

    def test_actual_Res_main_enters_original_target_source_validation(self):
        namespace=entry.parser_namespace();lookups=[]
        class Records(dict):
            def __getitem__(self,key):lookups.append(key);return super().__getitem__(key)
        namespace['MODEL_RECORDS']=Records(namespace['MODEL_RECORDS'])
        namespace['read']=lambda path:dict(round='foreign optimizer')
        namespace['execute']=lambda args:namespace['source_gate'](args)
        argv=['--delegate-parser','--model','resnet20','--out',str(self.root/'new-parser'),
            '--source-case',str(self.root/'new-optimizer'),'--shared-build-case',str(self.args.shared_build_case),
            '--orbit-root',str(self.args.orbit_root),'--go-sdk',str(self.args.go_sdk),
            '--initialization-audit',str(self.args.initialization_audit),'--run']
        with mock.patch.object(entry,'parser_namespace',return_value=namespace):
            with self.assertRaisesRegex(ValueError,'Successful same-model original common optimizer required'):
                entry.main(argv)
        self.assertEqual(lookups,['resnet20'])
        self.assertNotIn('resnet20',entry.parser_entry.MODEL_RECORDS)

    def test_frontend_failure_stops_without_optimizer_or_parser(self):
        def fail(argv,**options):self.calls.append(('frontend',argv,options));return dict(clean(),exit_code=1)
        result=self.run_fake(fail)
        self.assertEqual(result['status'],'error');self.assertFalse(result['accepted'])
        self.assertEqual(result['attempted'],dict(frontend=1,optimizer=0,parser=0))
        self.assertEqual(len(self.calls),1);self.assertTrue((self.args.out/'completion.json').is_file())

    def test_parent_timeout_and_cleanup_observation_are_retained(self):
        def fail(*a,**k):return dict(clean(),exit_code=-15,signal=15,timed_out=True,signals_sent=[dict(pid=123,signal=15)])
        result=self.run_fake(fail)
        self.assertEqual(result['status'],'error');self.assertFalse(result['accepted'])
        self.assertTrue(result['processes'][0]['observation']['timed_out'])
        self.assertEqual(result['processes'][0]['observation']['signal'],15)
        self.assertEqual(result['completed'],dict.fromkeys(entry.STAGES,0))

    def test_supervision_exception_keeps_unknown_cleanup_and_stops(self):
        with mock.patch.object(entry,'run_process',side_effect=RuntimeError('synthetic runner failure')), \
             mock.patch.object(entry,'resource_observation',return_value=dict(stop=False,reasons=[])), \
             mock.patch('pt_agent.registry.git_rev',return_value='synthetic'):
            result=entry.execute(self.args)
        self.assertEqual(result['status'],'error');self.assertTrue(result['processes'][0]['cleanup_unknown'])
        self.assertEqual(result['attempted']['optimizer'],0)

    def test_wrong_optimizer_source_rejects_before_parser(self):
        def transform(stage,result):
            if stage=='optimizer':result['source_records']['source_record']=str(self.root/'old-success/result.json')
            return result
        self.transform=transform;result=self.run_fake()
        self.assertEqual(result['status'],'error');self.assertEqual(result['completed']['frontend'],1)
        self.assertEqual(result['attempted']['parser'],0)

    def test_duplicate_child_registry_rejects_before_next_stage(self):
        def run(argv,**options):
            self.persist_child('frontend',entry.child_cases(self.args),duplicate=True);return clean()
        result=self.run_fake(run)
        self.assertEqual(result['status'],'error');self.assertIn('registry identity',result['error'])
        self.assertEqual(result['attempted']['optimizer'],0)

    def test_prepared_parser_foreign_model_or_wrong_shared_binding_rejects(self):
        cases=entry.child_cases(self.args)
        for index,field in enumerate(('status','model','source','count')):
            self.args.out=self.root/('bad'+str(index));cases=entry.child_cases(self.args)
            def transform(stage,result,field=field):
                if stage=='parser':
                    if field=='status':result['status']='prepared'
                    elif field=='model':result['model']='foreign'
                    elif field=='source':result['source']['shared_build_case']=str(self.root/'foreign-build')
                    else:result['completed']['parse']=True
                return result
            self.transform=transform;self.calls=[];result=self.run_fake()
            self.assertEqual(result['status'],'error');self.assertFalse(result['accepted'])

    def test_terminal_resource_failure_retracts_accepted_preserves_completion_facts(self):
        def observation(args):return dict(stop=len(self.calls)==3,reasons=['synthetic terminal reserve']if len(self.calls)==3 else [])
        result=self.run_fake(observation=observation)
        self.assertEqual(result['status'],'error');self.assertFalse(result['accepted'])
        self.assertEqual(result['completed'],dict.fromkeys(entry.STAGES,1))
        self.assertEqual(entry.read(self.args.out/'completion.json')['status'],'error')

    def test_physical_RSS_host_and_disk_limits_are_independent(self):
        self.args.out.mkdir()
        for rss,available,free in [(16*1024**2+1,3*entry.GIB,20*entry.GIB),(0,entry.GIB,20*entry.GIB),(0,3*entry.GIB,9*entry.GIB)]:
            with self.subTest(rss=rss,available=available,free=free), \
                 mock.patch.object(entry,'_scan',return_value={os.getpid():dict(ppid=0,vmrss_kib=rss)}), \
                 mock.patch.object(entry,'_available_ram_bytes',return_value=available), \
                 mock.patch.object(entry.shutil,'disk_usage',return_value=SimpleNamespace(free=free)):
                self.assertTrue(entry.resource_observation(self.args)['stop'])

    def test_expired_master_starts_no_child(self):
        self.args.seconds=1;result=self.run_fake()
        self.assertEqual(result['status'],'error');self.assertEqual(self.calls,[])
        self.assertEqual(result['attempted'],dict.fromkeys(entry.STAGES,0))

    def test_formal_unknowns_and_no_CKKS_claim(self):
        result=self.run_fake()
        self.assertTrue(all(result[k]is None for k in ('formal_latency','accuracy_acceptance','security_acceptance','compiled_bootstraps','model_specific_native_codegen')))
        self.assertTrue(all(result[k]==0 for k in entry.ZERO))

    def test_child_bad_timing_or_missing_boot_inventory_is_not_accepted(self):
        cases=entry.child_cases(self.args)
        self.persist_child('frontend',cases)
        marker=entry.read(cases['frontend']/'completion.json');marker['exclusive_attribution_ready']=False
        put(cases['frontend']/'completion.json',marker)
        with self.assertRaisesRegex(ValueError,'durable'):entry.accept_child(self.args,'frontend',cases)
        self.persist_child('parser',cases)
        result=entry.read(cases['parser']/'result.json');result['source']['optimized_source']['placement_bootstrap_nodes']=True
        put(cases['parser']/'result.json',result)
        with self.assertRaisesRegex(ValueError,'Parser must consume'):entry.accept_child(self.args,'parser',cases)

    def test_terminal_wall_is_rechecked_after_all_children_complete(self):
        real=entry.time.monotonic
        def clock():return real()+entry.TOTAL_SECONDS if len(self.calls)==3 else real()
        with mock.patch.object(entry.time,'monotonic',side_effect=clock):result=self.run_fake()
        self.assertFalse(result['accepted']);self.assertEqual(result['status'],'error')
        self.assertEqual(result['completed'],dict.fromkeys(entry.STAGES,1))
        self.assertIn('Finite master wall/closure reserve',result['resource_completion']['reasons'])


if __name__=='__main__':unittest.main()
