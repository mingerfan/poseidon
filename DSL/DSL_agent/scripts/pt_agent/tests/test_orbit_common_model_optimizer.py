"""Synthetic generic optimizer protocol; no Gurobi/Orbit imports or solving."""
from contextlib import contextmanager
import copy
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pt_agent.experiments import orbit_common_model_optimizer as outer
from pt_agent.experiments import orbit_common_model_optimizer_worker as worker
from pt_agent.experiments.r10_paper_metrics import StageRecorder,write_json
from pt_agent.tests import test_orbit_common_model_frontend as fixtures


class FakeParams:
    def __init__(self,path,sysname,mode,**options):
        self.poly_deg=0;self.max_slot=0;self.Sf=51;self.Sw=options['Sw'];self.Csw=options['CSw']
        self.lvl_lb=1;self.lvl_ub=16;self.bts_lb=3;self.bts_ub=16;self.backend='Lattigo'
        self.sysname=sysname;self.mode=mode;self.bpsdepth=options['bpsdepth'];self.threads=options['threads']
        self.comp=options['comp'];self.part=options['part'];self.reqbp=options['reqbp'];self.netname=options['netname']
        self.trunc_val=1;self.dacapo_mlir_in=True;self.dacapo_mlir_out=True


class OrbitCommonOptimizerChecks(unittest.TestCase):
    def tearDown(self):
        worker.configure_model(outer.MODEL_RECORDS['resnet20']['factory'])
        worker.frontend_worker.configure_model(outer.MODEL_RECORDS['resnet20']['factory'])

    def fixture(self,base,model='vgg16'):
        helper=fixtures.OrbitCommonModelFrontendChecks();locations=helper.locations(base,model)
        case=base/'successful-frontend';work=case/'work';work.mkdir(parents=True)
        trace=helper._record('trace',locations,work);parsed=helper._record('parse',locations,work)
        source=dict(round='ORBIT-COMMON-FRONTEND',mode='original-orbit-common-model-trace-tdag',
            model=locations['model'],status='ok',orbit_model_trace_passed=True,tdag_parse_passed=True,
            attempted=dict(trace=1,parse=1),completed=dict(trace=1,parse=1),phase_results=dict(trace=trace,parse=parsed))
        source.update(dict.fromkeys(worker.frontend_worker.ZERO_FIELDS,0));source.update(dict.fromkeys(worker.UNMEASURED_FIELDS,None))
        write_json(case/'result.json',source);write_json(case/'completion.json',dict(status='ok',registry_persisted=True,required_artifacts_persisted=True))
        profile=dict(polynomialDegree=131072,rescalingFactor=51,levelLowerBound=1,levelUpperBound=16,
            bootstrapLevelLowerBound=3,bootstrapLevelUpperBound=16,runtime='Lattigo')
        write_json(locations['profile'],profile)
        license_file=locations['orbit_root'].parent/'orbit-venv/lib/python3.11/site-packages/gurobipy/.libs/gurobi.lic'
        license_file.parent.mkdir(parents=True);license_file.write_text('synthetic license metadata only; never parsed')
        newwork=base/'optimizer/work';inner=newwork/'optimize-worker';inner.mkdir(parents=True)
        args=SimpleNamespace(model=locations['model'],orbit_root=locations['orbit_root'],profile=locations['profile'],
            license_file=license_file,input_mlir=Path(trace['mlir']['path']),output_mlir=newwork/'optimized.mlir',
            work_dir=inner,source_record=case/'result.json')
        return args,locations,source

    def test_model_specific_params_only_change_original_benchmark_name(self):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory);args,_,_=self.fixture(base)
            for factory in worker.frontend_worker.MODEL_RECORDS:
                expected=worker.configure_model(factory)['benchmark']
                for mode in ('compile','execute'):
                    params,metadata=worker.make_params(args.profile,FakeParams,mode=mode)
                    self.assertEqual(params.netname,expected);self.assertEqual(params.mode,mode)
                    self.assertEqual((params.poly_deg,params.max_slot,params.Sf,params.Sw,params.Csw),(131072,65536,51,40,40))
                    self.assertFalse(metadata['security_validated']);self.assertEqual((params.bts_lb,params.bts_ub),(3,16))

    def test_original_profile_parameter_changes_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            args,_,_=self.fixture(Path(directory));profile=json.loads(args.profile.read_text());profile['rescalingFactor']=50;write_json(args.profile,profile)
            with self.assertRaises(worker.WorkerFailure):worker.make_params(args.profile,FakeParams,mode='compile')

    def test_generic_source_gate_accepts_actual_vgg_alex_graph_counts_without_old_install_counts(self):
        for model in ('vgg16','alexnet'):
            with self.subTest(model=model),tempfile.TemporaryDirectory() as directory:
                args,locations,_=self.fixture(Path(directory),model)
                summary=worker._check_frontend_source(args)
                self.assertEqual(summary['activation_count'],locations['activations'])
                self.assertEqual((summary['original_graph_nodes'],summary['original_graph_edges']),(2,1))
                self.assertFalse(summary['weights_reopened']);self.assertFalse(summary['source_times_included'])

    def test_source_gate_rejects_other_model_and_old_r22_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            args,_,source=self.fixture(Path(directory));source['model']=outer.MODEL_RECORDS['alexnet']['factory'];write_json(args.source_record,source)
            with self.assertRaises(worker.WorkerFailure):worker._check_frontend_source(args)
            source['model']=args.model;source['round']='R22';write_json(args.source_record,source)
            with self.assertRaises(worker.WorkerFailure):worker._check_frontend_source(args)

    def test_missing_completion_or_changed_tdag_stops_source_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            args,_,source=self.fixture(Path(directory));completion=args.source_record.parent/'completion.json'
            value=json.loads(completion.read_text());value['registry_persisted']=False;write_json(completion,value)
            with self.assertRaisesRegex(worker.WorkerFailure,'persistence'):worker._check_frontend_source(args)
            value['registry_persisted']=True;write_json(completion,value)
            source['phase_results']['parse']['graph_check']['node_count']=3;write_json(args.source_record,source)
            with self.assertRaisesRegex(worker.WorkerFailure,'TDAG'):worker._check_frontend_source(args)

    def test_native_optimizer_delegates_once_and_keeps_external_cost_unmeasured(self):
        with tempfile.TemporaryDirectory() as directory:
            args,_,_=self.fixture(Path(directory));worker.configure_model(args.model);result=worker._initial_result('optimize');runs=[];seeds=[]
            def run(input_file,output,params):runs.append((input_file,output,params.netname));Path(output).write_text('synthetic optimized IR')
            native=SimpleNamespace(Params=FakeParams,run=run,set_global_seed=lambda seed:seeds.append(seed))
            @contextmanager
            def observers(*unused):
                state=dict(counters=dict.fromkeys(worker.SOLVE_FIELDS,0),events_path='fake-events',summary_path='fake-summary',first_fatal_path='fake-fatal',restored=False)
                try:yield state
                finally:state['restored']=True
            with patch.object(worker,'_load_optimizer',return_value=(native,object())),patch.object(worker,'install_observers',side_effect=observers),\
                 patch.object(worker.shutil,'disk_usage',return_value=SimpleNamespace(free=1<<40)):
                worker.optimize_model(args,StageRecorder(args.work_dir/'events.jsonl'),result)
            self.assertEqual(len(runs),1);self.assertEqual(seeds,[42]);self.assertTrue(result['native_optimizer_completed'])
            self.assertTrue(result['observer_restored']);self.assertEqual(result['optimize_completed'],1)
            self.assertIsNone(result['external_profile_estimate']['predicted_latency'])
            self.assertEqual(runs[0][2],'VGG16SiLU64k')

    def test_placement_counts_bootstrap_nodes_and_weight_separately(self):
        graph=dict(nodes=[dict(id='arg0',op='input',level=3,scale=51),
            dict(id='0',op='bootstrap',level=16,scale=51,weight=2,op_descr=dict(targetLevel=16))],
            edges=[dict(source='arg0',target='0')],inputs=['arg0'],outputs=['0'])
        actual=worker._placement_record(graph)
        self.assertEqual(actual['placement_bootstrap_nodes'],1);self.assertEqual(actual['placement_bootstrap_weight_sum'],2)
        self.assertEqual(actual['bootstraps'][0]['predecessors'],['arg0'])

    def test_module_private_observer_retains_normal_infeasible_and_restores_references(self):
        class ILPWorker:
            def _lsabts_worker(self):pass
        original=lambda *args,**kwargs:(None,None)
        original_tb=SimpleNamespace(print_exc=lambda *a,**k:None)
        module=SimpleNamespace(solve_ilp=original,traceback=original_tb,ILP_Worker=ILPWorker)
        with tempfile.TemporaryDirectory() as directory:
            with worker.install_observers(module,None,Path(directory)) as state:
                result=module.solve_ilp(SimpleNamespace(nodes=[],edges=[]),{},None,'synthetic')
                self.assertEqual(result,(None,None));self.assertEqual(state['counters']['solve_ilp_infeasible'],1)
            self.assertIs(module.solve_ilp,original);self.assertIs(module.traceback,original_tb)
            self.assertTrue(state['restored'])

    def test_outer_command_license_and_profile_are_explicit_without_nix_or_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory);args,locations,_=self.fixture(base)
            cli=outer.build_parser().parse_args(['--model','vgg16','--source-result',str(args.source_record),'--out',str(base/'outer')])
            selected=outer.selected_paths(cli);selected.update(orbit_python=locations['orbit_python'],orbit_root=locations['orbit_root'],
                profile=args.profile,license_file=args.license_file,input_mlir=args.input_mlir)
            command=outer.phase_command(selected,base/'work','optimize',cli)
            self.assertNotIn('pt_agent.tools.nixrun',command)
            self.assertEqual(command[command.index('--model')+1],args.model)
            self.assertEqual(command[command.index('--license-file')+1],str(args.license_file))
            self.assertNotIn('--checkpoint',command)
            env,metadata=outer.child_environment(base/'work',selected,dict(HOME='/original',GRB_LICENSE_FILE='fake-old',GUROBI_HOME='fake',KEEP='safe'))
            self.assertEqual(env['GRB_LICENSE_FILE'],str(args.license_file));self.assertEqual(env['HOME'],'/original')
            self.assertNotIn('GUROBI_HOME',env);self.assertIn('GRB_LICENSE_FILE',metadata['removed_environment_keys'])

    def test_outer_source_failure_persists_once_with_no_optimizer_process(self):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory);args=outer.build_parser().parse_args(['--model','alexnet','--source-result',str(base/'absent.json'),
                '--out',str(base/'fresh'),'--registry',str(base/'registry.jsonl')])
            with patch.object(outer,'run_process') as run,patch.object(outer.shutil,'disk_usage',return_value=SimpleNamespace(free=1<<40)):
                result=outer.execute(args)
            run.assert_not_called();self.assertEqual(result['status'],'stopped_on_error')
            self.assertEqual(result['attempted'],dict(optimize=0,validate=0))
            self.assertTrue((args.out/'completion.json').is_file());self.assertEqual(len(args.registry.read_text().splitlines()),1)
            self.assertTrue(all(result[key] is None for key in worker.UNMEASURED_FIELDS))

    def test_malformed_partial_solver_evidence_is_unknown_and_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory);phase=base/'optimize-worker';phase.mkdir()
            (phase/'phase_result.json').write_text('{incomplete')
            actual=outer.solver_observations(base,True)
            self.assertIsNone(actual['attempts']);self.assertIsNone(actual['model_optimize_calls'])
            self.assertIn('record_error',actual)

    def test_complete_generic_outer_preserves_params_counts_registry_and_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory);original,locations,_=self.fixture(base)
            args=outer.build_parser().parse_args(['--model','vgg16','--source-result',str(original.source_record),
                '--out',str(base/'new-outer'),'--registry',str(base/'registry.jsonl')])
            selected=dict(model=original.model,scripts=base,repo=base,orbit_root=original.orbit_root,
                orbit_python=locations['orbit_python'],profile=original.profile,license_file=original.license_file,
                source_record=original.source_record,input_mlir=None)
            commands=[]
            def run(argv,**kwargs):
                commands.append(argv);Path(kwargs['log']).write_text('a7b4c291\n')
                observation=dict(exit_code=0,timed_out=False,signal=None,signals_sent=[],remaining_pids=[])
                if argv[0]=='git':return observation
                phase=argv[argv.index('--phase')+1];phasework=Path(argv[argv.index('--work-dir')+1]);phasework.mkdir()
                work=phasework.parent;worker.configure_model(original.model);record=worker._initial_result(phase)
                record.update(status='ok',elapsed_ns=100,params_checked=True,source_checked=True,native_optimizer_completed=True,
                    environment_restored=True,budget_monitor_completed=True,params_calls=1,params_completed=1)
                _,record['params']=worker.make_params(original.profile,FakeParams,mode='compile'if phase=='optimize'else'execute')
                output=work/'optimized.mlir'
                if phase=='optimize':
                    output.write_text('synthetic optimized IR')
                    record.update(optimize_attempts=1,optimize_completed=1,seed_calls=1,seed_completed=1,observer_restored=True,
                        solve_ilp_attempts=1,solve_ilp_completed=1,solve_ilp_feasible=1,optimizer_runs=1,solver_runs=1)
                    (phasework/'solve-ilp-events.jsonl').write_text(json.dumps(dict(type='begin',call_index=1))+'\n'+json.dumps(dict(type='end',call_index=1,result='solution'))+'\n')
                else:
                    record.update(validate_attempts=1,validate_completed=1,build_from_mlir_calls=1,build_from_mlir_completed=1,
                        check_tdag_calls=1,check_tdag_completed=1,placement_artifact_checked=True)
                    graph=dict(nodes=[dict(id='arg0',op='input',level=3,scale=51),
                        dict(id='0',op='bootstrap',level=16,scale=51,weight=2,op_descr=dict(targetLevel=16))],
                        edges=[dict(source='arg0',target='0')],inputs=['arg0'],outputs=['0'])
                    write_json(phasework/'tdag.json',graph);record['tdag']=str(phasework/'tdag.json')
                    record['graph_check']=worker._placement_record(graph)
                record['output_mlir']=dict(path=str(output),size_bytes=output.stat().st_size)
                write_json(phasework/'phase_result.json',record)
                now=time.perf_counter_ns();event=dict(event_id='fake-'+phase,stage_id='fake.'+phase,parent_id=None,start_ns=now,pid=900002,ppid=1)
                (phasework/'events.jsonl').write_text(json.dumps(dict(event,type='begin'))+'\n'+json.dumps(dict(event,type='end',end_ns=now+1,status='ok'))+'\n')
                return observation
            with patch.object(outer,'selected_paths',return_value=selected),patch.object(outer,'run_process',side_effect=run),\
                 patch.object(outer.shutil,'disk_usage',return_value=SimpleNamespace(free=1<<40)):
                result=outer.execute(args)
            self.assertEqual(result['status'],'ok',result.get('error'))
            self.assertEqual(result['attempted'],dict(optimize=1,validate=1));self.assertEqual(result['completed'],result['attempted'])
            self.assertEqual(len(commands),3);self.assertEqual(result['metrics']['solve_ilp']['attempts'],1)
            self.assertEqual(result['metrics']['solve_ilp']['completed'],1)
            self.assertIsNone(result['metrics']['solve_ilp']['model_optimize_calls'])
            rows=[json.loads(line)for line in args.registry.read_text().splitlines()]
            self.assertEqual(len(rows),1);self.assertEqual(set(rows[0]),{'run_id','round','model','mode','git_rev','metrics','artifacts_dir','ts'})
            marker=json.loads((args.out/'completion.json').read_text())
            self.assertEqual(marker['case_artifact_bytes'],outer.artifact_bytes(args.out))
            self.assertTrue(marker['registry_persisted']);self.assertTrue(marker['exclusive_attribution_ready'])
            self.assertLessEqual((args.out/'completion.json').stat().st_size,result['completion_reserved_bytes'])


if __name__=='__main__':unittest.main()
