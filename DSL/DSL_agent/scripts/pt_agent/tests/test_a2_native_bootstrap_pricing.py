"""New tiny pricing/planning and fake supervision fixtures; zero real models."""
import copy
import inspect
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pt_agent.experiments import a2_native_bootstrap_pricing as own
from pt_agent.experiments import a2_native_profile_driver as original
from pt_agent.experiments import a2_native_profile_planner as planner
from pt_agent.experiments import a2_native_profile_runtime as runtime


def config():
    keys = ('add_double','add_single','bootstrap_single','modswitch_single','negate_single',
            'mul_double','mul_single','rescale_single','rotate_single')
    table = {'earth.'+key:[1.] * 30 for key in keys}
    table['earth.bootstrap_single'] = [0.] * 30
    table['earth.bootstrap_single'][16] = 14190000.
    return dict(runtime='HEAAN-HEVM',polynomialDegree=131072,rescalingFactor=51,
        levelLowerBound=1,levelUpperBound=29,bootstrapLevelLowerBound=3,
        bootstrapLevelUpperBound=16,latencyTable=table)


def saved_candidates():
    return {'depth-dp':dict(boots=11,latency_us=1000.,seconds=1),
        'depth-dp+prune':dict(boots=11,latency_us=1001.,seconds=2),
        'stage-cut':dict(boots=14,latency_us=901.,seconds=3),
        'stage-cut+prune':dict(boots=14,latency_us=900.,seconds=4)}


def plan():
    return dict(external_inputs=['input'],values=[dict(id='input',level=16),
        dict(id='low',level=3),dict(id='fresh',level=16),dict(id='out',level=16)],
        execution=[dict(op='mod_switch',inputs=['input'],output='low'),
                   dict(op='boot',inputs=['low'],output='fresh'),
                   dict(op='relinearize',inputs=['fresh'],output='out')])


def observation():
    return dict(exit_code=0,timed_out=False,signal=None,remaining_pids=[],signals_sent=[],
                started_ns=100,ended_ns=300)


class BootstrapPricingTests(unittest.TestCase):
    def test_explicit_side_is_generic_and_qwen_style_input_remains_available(self):
        table = {'bootstrap_single':[2.,3.,4.,5.,6.,7.]}
        self.assertEqual(own.bootstrap_us(table,1,5,'input'),3.)
        self.assertEqual(own.bootstrap_us(table,1,5,'output'),7.)
        with self.assertRaises(ValueError):own.bootstrap_us(table,1,5,'alexnet')

    def test_native_output_model_preserves_legacy_default_and_rescale_reserve(self):
        profile=config();before=copy.deepcopy(profile)
        old=planner.model_for(profile);default=own.priced_model(profile);new=own.native_model(profile)
        self.assertEqual(old.boot_us,1.)
        self.assertEqual(default.boot_us,old.boot_us)
        self.assertEqual(new.boot_us,14190001.)
        self.assertEqual((new.input_level,new.boot_in_level,new.raw_boot_out_level,new.boot_out_level),
                         (16,3,16,15))
        self.assertEqual(profile,before)
        self.assertEqual(planner.model_for(profile).boot_us,old.boot_us)

    def test_saved_four_placement_scores_rank_by_explicit_price(self):
        source=saved_candidates();before=copy.deepcopy(source)
        ranked,selected=own.reprice_candidates(source,config())
        self.assertEqual(selected,'depth-dp')
        self.assertEqual(ranked['depth-dp']['latency_us'],1000+11*14190000)
        self.assertEqual(source,before)
        self.assertEqual(set(ranked),set(source))

    def test_runtime_boot_delta_keeps_missing_relinearize_full_prediction_null(self):
        original_plan=plan();before=copy.deepcopy(original_plan)
        result=own.plan_observation(original_plan,config())
        self.assertIsNone(result['predicted_latency_us'])
        self.assertEqual(result['prediction']['table_known_sum_us'],14190001.)
        self.assertEqual(result['prediction']['by_op_us']['boot'],14190000.)
        self.assertEqual(result['prediction']['missing'],[
            dict(index=2,op='relinearize',reason='operation has no profile entry')])
        self.assertEqual(result['prediction']['bootstrap_price_observations'][0]['priced_level'],16)
        self.assertEqual(original_plan,before)
        legacy=own.plan_observation(original_plan,config(),'input')
        self.assertEqual(legacy['prediction']['table_known_sum_us'],1.)

    def test_observation_uses_actual_output_level_and_original_native_abi(self):
        result,audit=own.observe_runtime(plan(),config(),1)
        self.assertEqual(audit['input_levels'],[16])
        self.assertEqual(result['compiled_bootstraps'],1)
        bad=plan();bad['values'][2]['level']=15
        with self.assertRaises(ValueError):own.observe_runtime(bad,config(),1)
        with self.assertRaises(ValueError):own.observe_runtime(plan(),config(),11)
        changed=plan();changed['values'][2]['level']=99
        with self.assertRaises(ValueError):own.plan_observation(changed,config())

    def test_finite_table_validation_still_rejects_missing_and_nonfinite(self):
        for value in (float('nan'),float('inf'),-1.):
            table={'bootstrap_single':[value]}
            with self.assertRaises(ValueError):own.bootstrap_us(table,0,0,'output')
        profile=config();profile['rescalingFactor']=43
        with self.assertRaises(ValueError):own.native_model(profile)

    def test_namespace_changes_only_private_factory_observer_and_worker_module(self):
        old=(original.worker,original.execute,planner.Model.boot_us,original.admission)
        private=own.namespace()
        self.assertIs(private['compiler_argv'].__code__,original.compiler_argv.__code__)
        self.assertIs(private['reprofile_source'].__code__,original.reprofile_source.__code__)
        self.assertIs(private['native_model'],own.native_model)
        self.assertIs(private['observe_runtime'],own.observe_runtime)
        self.assertEqual(private['ROUND'],own.ROUND)
        self.assertIn(own.MODULE,private['execute'].__code__.co_consts)
        self.assertEqual(old,(original.worker,original.execute,planner.Model.boot_us,original.admission))

    def test_tiny_real_four_plans_and_full_count_emit_use_output_priced_model(self):
        def run(active):
            active.op_context='tiny';x=active.input(0);y=x
            for i in range(14):active.stage('step'+str(i));y=y*y
            active.output([y])
        model=own.native_model(config());plans,summary,selected=runtime.plan(run,8,model)
        proofs=runtime.validate_all(run,8,model,plans)
        self.assertEqual(len(proofs),4)
        self.assertTrue(all(row['boots']==1 for row in summary.values()))
        self.assertTrue(all(row['latency_us']>=14190000 for row in summary.values()))
        self.assertEqual(summary[selected]['latency_us'],min(row['latency_us'] for row in summary.values()))
        self.assertTrue(all(row['count']['callback_bootstraps']==1 for row in proofs.values()))

    def test_real_worker_planning_dispatch_passes_output_priced_model(self):
        with tempfile.TemporaryDirectory()as temp:
            root=Path(temp);phase=root/'phases/planning';phase.mkdir(parents=True);(root/'plans').mkdir()
            profile=root/'profile.json';profile.write_text(json.dumps(config()))
            args=SimpleNamespace(phase_dir=phase,worker='planning',profile=profile,out=root,
                planning_seconds=600,state_limit=4096,frontend=Path('/fixture'))
            private=own.namespace();private['_program']=lambda args:(SimpleNamespace(graph=SimpleNamespace(weights={})),SimpleNamespace(header=SimpleNamespace(slots=8)))
            seen=[]
            def fake_plan(run,slots,model,**kwargs):
                seen.append(model);return dict.fromkeys(saved_candidates(),{}),saved_candidates(),'depth-dp'
            with patch.object(runtime,'plan',side_effect=fake_plan),patch.object(runtime,'validate_all',return_value={}):
                result=private['worker'](args)
            self.assertEqual(result['status'],'ok');self.assertEqual(seen[0].boot_us,14190001.)

    def test_real_observe_worker_uses_actual_result_level_for_both_streams(self):
        with tempfile.TemporaryDirectory()as temp:
            root=Path(temp);phase=root/'phases/observe';phase.mkdir(parents=True);(root/'emit').mkdir()
            profile=root/'profile.json';profile.write_text(json.dumps(config()))
            compiled=root/'emit/lowered.test.runtime-plan.json';compiled.write_text(json.dumps(plan()))
            native=root/'native.runtime-plan.json';native.write_text(json.dumps(plan()))
            (root/'placement.json').write_text(json.dumps(dict(selected='depth-dp',candidates={'depth-dp':dict(boots=1)})))
            args=SimpleNamespace(phase_dir=phase,worker='observe',profile=profile,out=root,native_plan=native)
            result=own.namespace()['worker'](args)
            self.assertEqual(result['status'],'ok')
            for key in ('compiled','native_reference'):
                self.assertEqual(result[key]['prediction']['boot_subtotal_us'],14190000.)
                self.assertEqual(result[key]['prediction']['bootstrap_pricing_convention'],'output')
                self.assertIsNone(result[key]['predicted_latency_us'])

    def test_fake_supervisor_preserves_five_phases_and_resource_protocol(self):
        from pt_agent.experiments import r27_orbit_lazy_backend as resources
        with tempfile.TemporaryDirectory()as temp:
            root=Path(temp);source=root/'source.ptdsl'
            source.write_text("import fhe\n@fhe.program(l_max=5,boot_out_level=5,refresh='agent',placement='program-plan')\ndef f(x):\n return x\n")
            args=SimpleNamespace(run=True,out=root/'new',model='alexnet_silu_avgpool',source_case=root/'source',
                native_case=root/'native',frontend=root/'frontend',planning_seconds=600,state_limit=4096,case_gib=64)
            private=own.namespace();private['admission']=lambda a:dict(program=str(source),model_info='info',profile='profile',native_plan='native')
            calls=[]
            def fake_run(argv,**kwargs):
                phase=argv[argv.index('--worker')+1];calls.append((phase,argv,kwargs))
                row=dict(status='ok')
                if phase=='trace':(args.out/'emit/candidate_trace.mlir').write_text('fixture')
                if phase=='compile':row['native_process']=observation()
                if phase=='observe':row.update(compiled=dict(compiled_bootstraps=99,predicted_latency_us=None),native_reference=dict(predicted_latency_us=None))
                (args.out/'phases'/phase/'result.json').write_text(json.dumps(row));return observation()
            private['run_process']=fake_run
            with patch.object(own.paths,'RESULTS_DIR',root),patch.object(resources,'_available_ram_bytes',return_value=20*1024**3),patch.object(original.shutil,'disk_usage',return_value=SimpleNamespace(free=100*1024**3)):
                result=private['execute'](args)
            self.assertEqual(result['status'],'ok');self.assertEqual(result['compiled_bootstraps'],99)
            self.assertEqual([row[0]for row in calls],list(original.PHASES))
            for phase,argv,kwargs in calls:
                self.assertIn(own.MODULE,argv);self.assertNotIn('pt_agent.experiments.a2_native_profile_driver',argv)
                self.assertEqual(kwargs['mem_gib'],24);self.assertEqual(kwargs['fsize_gib'],16)
            self.assertTrue(json.loads((args.out/'completion.json').read_text())['registry_persisted'])
            entry=json.loads((root/'registry.jsonl').read_text());self.assertEqual(set(entry),own.FIELDS)
            self.assertEqual(entry['round'],own.ROUND);self.assertIn('actual result level',result['pred_table'])

    def test_supersession_preserves_old_rows_and_is_idempotent(self):
        with tempfile.TemporaryDirectory()as temp:
            path=Path(temp)/'supersessions.jsonl'
            self.assertTrue(own._append_supersession(path,'old14','newactual','explicit result-level pricing'))
            self.assertFalse(own._append_supersession(path,'old14','newactual','explicit result-level pricing'))
            rows=[json.loads(line)for line in path.read_text().splitlines()]
            self.assertEqual(len(rows),1);self.assertTrue(rows[0]['old_compile_fact_preserved'])

    def test_closed_metadata_requires_unique_registry_and_closed_saved_births(self):
        with tempfile.TemporaryDirectory()as temp:
            root=Path(temp);case=root/'source';case.mkdir();registry=root/'registry.jsonl'
            result=dict(status='ok',model='synthetic',phases=[dict(process=observation())])
            (case/'result.json').write_text(json.dumps(result))
            (case/'completion.json').write_text(json.dumps(dict(status='ok',registry_persisted=True,required_artifacts_persisted=True)))
            (case/'phase-resources.jsonl').write_text(json.dumps(dict(processes=[dict(pid=42,starttime=123)]))+'\n')
            entry=dict(run_id=case.name,round='fixture',model='synthetic',mode='fixture',git_rev=None,
                metrics=dict(status='ok'),artifacts_dir=str(case),ts='fixture')
            registry.write_text(json.dumps(entry)+'\n')
            with patch.object(own,'_process',return_value=None):
                found,_,proof=own.closed(case,'synthetic',registry=registry)
            self.assertEqual(found,result);self.assertEqual(proof['saved_births'],1)
            with patch.object(own,'_process',return_value=dict(starttime=123,state='R')):
                with self.assertRaises(ValueError):own.closed(case,'synthetic',registry=registry)
            registry.write_text((json.dumps(entry)+'\n')*2)
            with self.assertRaises(ValueError):own.closed(case,'synthetic',registry=registry)

    def test_metadata_projection_rows_persist_before_supersession_and_never_recompile(self):
        with tempfile.TemporaryDirectory()as temp:
            root=Path(temp);case=root/'old-source';case.mkdir();native_case=root/'old-native';native_case.mkdir()
            profile=root/'profile.json';profile.write_text(json.dumps(config()))
            actual_plan=root/'ours.runtime-plan.json';actual_plan.write_text(json.dumps(plan()))
            native_plan=root/'native.runtime-plan.json';native_plan.write_text(json.dumps(plan()))
            candidates=saved_candidates();candidates['depth-dp']['boots']=1
            candidates['depth-dp+prune']['boots']=1
            (case/'placement.json').write_text(json.dumps(dict(selected='depth-dp',candidates=candidates)))
            source=dict(round=original.ROUND,model='resnet20_silu',status='ok',monitor_stopped=True,
                attempted=dict.fromkeys(original.PHASES,1),completed=dict.fromkeys(original.PHASES,1),
                compiled_bootstraps=1,compile_s=2.,
                source=dict(profile=str(profile),native_case=str(native_case),native_plan=str(native_plan)),
                phases=[dict(phase=name,status='ok',worker=dict(status='ok'))for name in original.PHASES])
            source['phases'][0]['worker']['all_four_verified']=True
            source['phases'][-1]['worker']['runtime_plan']=str(actual_plan)
            native=dict(status='ok',model='dacapo-original-resnet20-silu',bootstrap_count=1)
            completion=dict(status='ok',complete_case_elapsed_seconds=3.)
            args=SimpleNamespace(run=True,reprice_from=case,out=root/'new-projection',paper_runs=root/'runs.jsonl',
                supersessions=root/'supersessions.jsonl',model='resnet20_silu',supersedes_case=None)
            args.paper_runs.write_text(json.dumps(dict(run_id=case.name,historical_fact=14))+'\n')
            def closed(path,model):
                value=source if Path(path)==case else native
                return value,completion,dict(saved_births=1,all_saved_births_closed=True)
            real_append=own._append_supersession;seen=[]
            def supersede(path,old,new,reason):
                self.assertTrue((args.out/'completion.json').is_file())
                published=[json.loads(line)for line in args.paper_runs.read_text().splitlines()]
                self.assertIn(new,[row['run_id']for row in published]);seen.append((old,new))
                return real_append(path,old,new,reason)
            with patch.object(own.paths,'RESULTS_DIR',root),patch.object(own,'closed',side_effect=closed),patch.object(own,'_append_supersession',side_effect=supersede):
                result=own.metadata_execute(args)
            self.assertEqual(result['status'],'ok');self.assertEqual(result['compile'],0)
            self.assertEqual(len(seen),2)
            rows=[json.loads(line)for line in args.paper_runs.read_text().splitlines()]
            self.assertEqual(rows[0],dict(run_id=case.name,historical_fact=14))
            for row in rows[1:]:
                self.assertIsNone(row['compile_s']);self.assertIsNone(row['pred_latency_s'])
                self.assertEqual(row['partial_pred_latency_s'],14.190001)
                self.assertEqual(row['new_compile_attempts'],0)
                self.assertEqual(row['bootstrap_pricing_convention'],'output')

    def test_changed_saved_selection_rejects_publication_and_supersession(self):
        with tempfile.TemporaryDirectory()as temp:
            root=Path(temp);case=root/'old';case.mkdir();native_case=root/'native';native_case.mkdir()
            profile=root/'profile.json';profile.write_text(json.dumps(config()))
            (case/'placement.json').write_text(json.dumps(dict(selected='stage-cut+prune',candidates=saved_candidates())))
            source=dict(status='ok',round=original.ROUND,model='alexnet_silu_avgpool',monitor_stopped=True,
                attempted=dict.fromkeys(original.PHASES,1),completed=dict.fromkeys(original.PHASES,1),
                source=dict(profile=str(profile),native_case=str(native_case)))
            args=SimpleNamespace(run=True,reprice_from=case,out=root/'rejected',paper_runs=root/'runs.jsonl',
                supersessions=root/'supersessions.jsonl',model='alexnet_silu_avgpool',supersedes_case=None)
            with patch.object(own.paths,'RESULTS_DIR',root),patch.object(own,'closed',return_value=(source,{},{})):
                result=own.metadata_execute(args)
            self.assertEqual(result['status'],'error');self.assertIn('re-place and compile',result['error'])
            self.assertFalse(args.paper_runs.exists());self.assertFalse(args.supersessions.exists())

    def test_fresh_main_immediately_appends_distinct_success_and_failure_actual_rows(self):
        with tempfile.TemporaryDirectory()as temp:
            root=Path(temp);paper=root/'runs.jsonl'
            for outcome in ('ok','error'):
                def execute(args):
                    args.out.mkdir()
                    result=dict(status=outcome,error=None if outcome=='ok' else 'original compiler exited1',
                        compile_s=2.5,compiled_bootstraps=7 if outcome=='ok' else None,
                        pred_latency_s=None,attempted=dict(compile=1),completed=dict(compile=int(outcome=='ok')),
                        phases=[dict(phase='observe',worker=dict(compiled=dict(execution_steps=123,
                            prediction=dict(table_known_sum_us=1000000,missing=[dict(op='relinearize')]))))] if outcome=='ok' else [])
                    if outcome=='error':result['failure_class']='compiler_crash'
                    (args.out/'result.json').write_text(json.dumps(result))
                    (args.out/'completion.json').write_text(json.dumps(dict(status=outcome,complete_case_elapsed_seconds=8.)))
                    return result
                with patch.object(own,'namespace',return_value=dict(execute=execute)):
                    code=own.main(['--model','alexnet_silu_avgpool','--out',str(root/outcome),
                                  '--paper-runs',str(paper),'--run'])
                self.assertEqual(code,int(outcome!='ok'))
            rows=[json.loads(line)for line in paper.read_text().splitlines()]
            self.assertEqual([row['status']for row in rows],['ok','compiler_crash'])
            self.assertEqual([row['compile_s']for row in rows],[2.5,2.5])
            self.assertEqual(rows[0]['exec_ops'],123);self.assertEqual(rows[0]['boots_compiled'],7)
            self.assertIsNone(rows[1]['exec_ops']);self.assertIsNone(rows[1]['boots_compiled'])
            for row in rows:
                self.assertFalse(row['projection_only']);self.assertIsNone(row['generation_e2e_s'])
                self.assertEqual(row['warm_compilation_pipeline_s'],8.)
                self.assertIsNone(row['quality']['value'])


if __name__=='__main__':unittest.main(verbosity=2)
