import argparse
import copy
import csv
from datetime import datetime,timezone
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pt_agent.experiments import experiment_delivery_curated_refresh as subject


class CuratedRefreshTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.results=self.root/'results';self.results.mkdir()
        self.delivery=self.root/'delivery';self.delivery.mkdir();self.cold=self.results/'new-cold'
        self.retained=self.results/'retained-compile';self.session=self.results/'retained-session'
        self.primary_report=self.results/'cold-report';self.retained_report=self.results/'retained-report'
        self.manifest=dict(schema='predeclared-vgg-curated-refresh-v1',cold_case=str(self.cold),retained_compile_case=str(self.retained),retained_session_case=str(self.session),reports=[dict(case=str(self.primary_report),frozen_case=str(self.cold),compiled_case=str(self.cold)),dict(case=str(self.retained_report),frozen_case=str(self.session),compiled_case=str(self.retained))])
        self.manifest_path=self.root/'manifest.json';self.write(self.manifest_path,self.manifest)
        self.start=datetime(2026,11,16,tzinfo=timezone.utc).timestamp()
        for name in ('subprocess.Popen','socket.getaddrinfo','urllib.request.urlopen'):
            blocker=patch(name,side_effect=AssertionError('No external execution/transport in metadata tests'))
            blocker.start();self.addCleanup(blocker.stop)
        self.seed()

    def write(self,path,value):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value)+'\n')

    def csv(self,path,rows,fields=None):
        path.write_text(subject.csv_text(rows,fields or list(rows[0])))

    def seed(self):
        retained=subject.pending_cell(self.retained,{})
        retained.update(primary_matrix_cell=True,method_variant_role='primary-method-retained-postfreeze-compilation',approximation_variant='hybrid-session-continuation-tuner-on',bootstrap_count=215,native_compile_success=True,postfreeze_compilation_only=True,API_calls=0)
        retained['terminal'].update(result_status='ok',complete_case_seconds=94.)
        ablation=subject.pending_cell(self.results/'ablation',{})
        ablation.update(method_variant_role='no-tuner-ablation',approximation_variant='cold-LLM-no-tuner',bootstrap_count=215,native_compile_success=True,cold_full_E2E_seconds=549.,API_calls=5)
        ablation['terminal']['result_status']='ok'
        self.original_ids={retained['cell_id'],ablation['cell_id']}
        record=dict(schema='three-cnn-four-systems-experiment-delivery-v1',cells=[retained,ablation],coverage={},stage_clock_rows=[],source_metadata_paths=[str(self.retained)],attribution_exports={'rules':['Preserve saved scopes.']},registry_checks={'cases':{}},comparison_rules=['historical'],latest_scheduling_direction={'whole_model_encrypted_execution':'deferred'})
        self.write(self.delivery/(subject.BASE+'.json'),record)
        main=[subject.main_row(x,{})for x in record['cells']];self.csv(self.delivery/(subject.BASE+'.csv'),main)
        self.csv(self.delivery/'STAGE_ATTRIBUTION.csv',[dict(cell='historical',run='old',source_case='/historical',event_id='old:1',stage_id='compile',parent_id=None,status='ok',inclusive_ns=10,exclusive_ns=10,scope='Nested spans overlap.',source_completion='/historical/completion.json',source_result='/historical/result.json')])
        self.csv(self.delivery/'AGENT_ROUNDS.csv',[dict(model='ResNet20',run='old',variant='cold-hybrid',iteration=1,candidate_kind='raw',origin='llm',predicted_bootstraps=123,predicted_latency_us=1.,predicted_scope='saved',placement_variant='depth-dp',feasible=True,max_u=.8,max_u_calibrated=.8,max_u_analytic=0.,heldout_top1_percent=92.,heldout_reference_top1_percent=92.,heldout_accuracy_drop_pp=0.,heldout_argmax_agree=1000,heldout_samples=1000,quality_split='heldout',saved_evaluation_stage='complete',rejection_stage=None,rejection_reason_codes='',selected_frozen=True,selected_iteration=1,selected_origin='llm',confirmation_evaluated=True,confirmation_passed=True,confirmation_top1_percent=92.,confirmation_reference_percent=92.,confirmation_max_u=.8,confirmation_split='confirm',actual_API_response_for_this_round=True,LLM_response_seconds=1.,tuner_reported_seconds=None,tuner_reused_from=None,new_tune_executed=False,tuner_time_scope=None,missing_quality_reason=None,source_agent_trace='/historical/trace',source_worker='/historical/worker')])
        self.csv(self.delivery/'COMPILED_COUNT_SCOPES.csv',[dict(model='ResNet20',run='old',source_candidate_run='old',variant='cold-hybrid',cold_full_E2E=True,selected_iteration=1,selected_origin='llm',selected_candidate_pre_freeze_boots=123,one_time_confirmation_estimated_boots=123,postfreeze_normalized_depth_dp_boots=134,postfreeze_normalized_depth_dp_prune_boots=123,postfreeze_normalized_stage_cut_boots=132,postfreeze_selected_stage_cut_prune_boots=123,actual_compiled_final_boots=123,compiled_minus_pre_freeze_candidate=0,compiled_minus_postfreeze_normalized_depth_dp=-11,compiled_minus_postfreeze_normalized_depth_dp_prune=0,compiled_minus_postfreeze_selected_plan=0,scope='separate',source_refresh_plan='/historical/plan',source_result='/historical/result')])
        (self.delivery/'COMPILED_BOOTSTRAP_NEIGHBORHOODS_THREE_MODELS.json').write_text('retained external source')

    def args(self,**changes):
        a=dict(manifest=self.manifest_path,delivery_dir=self.delivery,results_root=self.results,poll_seconds=60,until='2026-11-16T00:02:00Z',once=True);a.update(changes);return argparse.Namespace(**a)

    def obs(self):return dict(exit_code=0,timed_out=False,signal=None,signals_sent=[],remaining_pids=[],started_ns=100,ended_ns=300,peak_rss_kib_by_pid={'7':200})

    def closure(self,path,status='ok',seconds=90.):
        self.write(path/'completion.json',dict(status=status,complete_case_elapsed_seconds=seconds,required_artifacts_persisted=True,registry_persisted=True,timing_status='ok',elapsed_scope='fixture durable closure',phase_spans=[dict(event_id=path.name+':1',stage_id='fresh_end_to_end',parent_id=None,inclusive_ns=80,exclusive_ns=10,status=status),dict(event_id=path.name+':2',stage_id='compile',parent_id=path.name+':1',inclusive_ns=70,exclusive_ns=70,status=status)]))

    def register(self,path,mode='fixture-terminal'):
        p=self.results/'registry.jsonl'
        with p.open('a')as f:f.write(json.dumps(dict(run_id=path.name,round='fixture',model=subject.MODEL,mode=mode,git_rev=None,metrics={},artifacts_dir=str(path),ts='fixture'))+'\n')

    def cold_success(self,*,count=111):
        c=self.cold;chosen=dict(iteration=1,origin='llm',boots_pred=114,latency_pred_us=100.)
        conf=dict(chosen,passed=True,total=1000,top1=91.6,exact_top1=91.6,acc_drop_pp=0.,max_u=.88,max_u_calibrated=.88,max_u_analytic=0.,agree=1000)
        self.write(c/'post-freeze/emit/lowered.fixture.runtime-plan.json',{'metadata_only':True})
        phases={p:dict(status='ok')for p in subject.PHASES}
        phases['agent']['agent_result']=str(c/'work/agent/worker-result.json')
        phases['placement'].update(refresh_plan=str(c/'post-freeze/refresh-plan.json'),boots_pred=count,latency_pred_us=90.)
        phases['compile'].update(process=self.obs(),compiled_bootstraps=count,runtime_plan=str(c/'post-freeze/emit/lowered.fixture.runtime-plan.json'),prediction=dict(table_known_sum_us=80,scope='external profile',untimed={'relinearize':1}))
        result=dict(round='R30',model=subject.MODEL,mode='fresh-common-model-agent-e2e',status='ok',variant='llm-plus-tuner',api_delta=1,compiled_bootstraps=count,monitor_error=None,historical_elapsed_reused=False,original_intake_reused=False,training_calibration_reused=False,source=dict(factory=subject.FACTORY,compiler_profile=str(subject.PROFILE)),attempted=dict.fromkeys(subject.PHASES,1),completed=dict.fromkeys(subject.PHASES,1),phases=phases,processes=[dict(phase=p,observation=self.obs())for p in subject.PHASES],quality_audit=str(c/'post-run-quality-audit.json'))
        check=dict(passed=True,finite=True,max_absolute_error=1e-14,max_error_ratio=.01,criterion='1e-12+1e-12*abs(reference)')
        worker=dict(status='ok',frozen=True,tuner_enabled=True,actual_api_calls=1,selected_program=str(c/'work/agent/program.ptdsl'),strict_math_probes=[dict(sample_id=f'test:{9000+i}',exact=check,approximate=check)for i in range(3)],agent=dict(llm_calls=1,prompt_tokens=10,completion_tokens=5,confirm=dict(selected=chosen,evals=[conf])),messages='FORBIDDEN '+('x'*10000),source='FORBIDDEN program')
        self.write(c/'result.json',result);self.closure(c);self.register(c)
        self.write(c/'work/agent/worker-result.json',worker)
        self.write(c/'work/agent/SESSION.json',dict(phase='done',calls=1,iteration=2,max_iters=6,tuner=True,conversation='FORBIDDEN '+('x'*10000)))
        self.write(c/'post-run-quality-audit.json',dict(status='quality_accepted',formal_frozen=True,quality_acceptance_passed=True,strict_math_passed=True,model=subject.MODEL,selected_program=worker['selected_program'],source_result=str(c/'work/agent/worker-result.json'),actual_api_calls=1,session_phase='done',confirmation=conf))
        self.write(c/'post-freeze/refresh-plan.json',dict(boots=count,placement='stage-cut+prune',candidates={'depth-dp':dict(boots=115),'depth-dp+prune':dict(boots=114),'stage-cut':dict(boots=113),'stage-cut+prune':dict(boots=count)}))
        trace=dict(iteration=1,seconds=1.5,model_call=1,messages='FORBIDDEN '+('x'*10000),fb=dict(stage='complete',feasible=True,cost=dict(boots_pred=114,latency_pred_us=100.,placement_variant='depth-dp+prune'),validation=conf),tuned=None,tuned_reused_from=None)
        (c/'work/agent/agent_trace.jsonl').write_text(json.dumps(trace)+'\n')
        return result

    def report(self,retained=False,policy=True,finite=8000):
        case=self.retained_report if retained else self.primary_report
        freeze=self.session if retained else self.cold
        marker=self.results/'frozen-cnn-report-access'/(freeze.name+'.json')
        self.write(marker,dict(source_case=str(freeze),out=str(case)))
        reference=7360;approx=7359 if policy else 6000
        record=dict(round='R30-FROZEN-REPORT',model=subject.MODEL,status='ok',source_case=str(freeze),report_completed=True,report_samples_read=8000,report_samples_completed=8000,finite_samples=finite,report_samples=8000,ids_exactly_test_0_through_7999=True,all_finite=finite==8000,batch_size=1,llm_calls=0,compiler_runs=0,ckks_runs=0,feedback_to_agent=False,automatic_retry=False,used_for_selection=False,independent_report_observation=True,reference_correct=reference,approximate_correct=approx,reference_top1_percent=reference/80,approximate_top1_percent=approx/80,accuracy_drop_pp=(reference-approx)/80,report_policy_passed=policy,access_marker=str(marker),processes=[dict(stage='report',observation=self.obs())],report_text='FORBIDDEN sample text')
        self.write(case/'result.json',record);self.closure(case,seconds=400.);self.register(case);return record

    def run_once(self):return subject.execute(self.args(),now=lambda:self.start)
    def load(self):return json.loads((self.delivery/(subject.BASE+'.json')).read_text())
    def cold_cell(self):return next(c for c in self.load()['cells']if c['terminal']['result_path']==str(self.cold/'result.json'))

    def test_prepared_and_unclosed_leave_metrics_empty_and_history_primary(self):
        self.write(self.cold/'result.json',dict(status='prepared',compiled_bootstraps=999,api_delta=6))
        self.run_once();c=self.cold_cell()
        self.assertFalse(c['primary_matrix_cell']);self.assertIsNone(c['bootstrap_count']);self.assertIsNone(c['API_calls']);self.assertIsNone(c['cold_full_E2E_seconds'])
        self.assertTrue(next(c for c in self.load()['cells']if c['cell_id']in self.original_ids and c.get('postfreeze_compilation_only'))['primary_matrix_cell'])

    def test_closed_error_retains_error_clock_without_fabricating_compile_success(self):
        self.write(self.cold/'result.json',dict(status='error',round='R30',model=subject.MODEL,api_delta=2,compiled_bootstraps=999,processes=[]))
        self.closure(self.cold,'error',77.);self.register(self.cold);self.run_once();c=self.cold_cell()
        self.assertEqual(c['terminal']['result_status'],'error');self.assertEqual(c['terminal']['complete_case_seconds'],77.)
        self.assertFalse(c['native_compile_success']);self.assertFalse(c['primary_matrix_cell']);self.assertIsNone(c['bootstrap_count']);self.assertIsNone(c['cold_full_E2E_seconds'])

    def test_actual_closed_cold_promotes_only_predeclared_source_and_keeps_all_variants(self):
        self.cold_success();self.run_once();a=self.load();c=self.cold_cell()
        self.assertEqual(len(a['cells']),3);self.assertTrue(self.original_ids.issubset({x['cell_id']for x in a['cells']}))
        self.assertTrue(c['primary_matrix_cell']);self.assertEqual(c['bootstrap_count'],111);self.assertEqual(c['API_calls'],1);self.assertEqual(c['cold_full_E2E_seconds'],90.)
        self.assertEqual(c['candidate_attribution']['tuner_enabled'],True)
        old=[x for x in a['cells']if x['cell_id']in self.original_ids]
        self.assertTrue(all(not x['primary_matrix_cell']for x in old));self.assertIn('no-tuner-ablation',{x['method_variant_role']for x in old})
        counts,_=subject.read_csv(self.delivery/'COMPILED_COUNT_SCOPES.csv');n=next(x for x in counts if x['run']==self.cold.name)
        self.assertEqual(n['selected_candidate_pre_freeze_boots'],'114');self.assertEqual(n['postfreeze_normalized_depth_dp_boots'],'115');self.assertEqual(n['actual_compiled_final_boots'],'111')

    def test_dirty_native_or_missing_original_math_never_promotes(self):
        record=self.cold_success();record['phases']['compile']['process']['remaining_pids']=[17];self.write(self.cold/'result.json',record)
        self.run_once();self.assertFalse(self.cold_cell()['primary_matrix_cell']);self.assertIsNone(self.cold_cell()['bootstrap_count'])
        self.assertEqual(self.cold_cell()['metadata_admission']['status'],'not_admitted')

    def test_bad_plan_pointer_and_duplicate_registry_are_rejected(self):
        record=self.cold_success();record['phases']['compile']['runtime_plan']=str(self.retained/'foreign.runtime-plan.json');self.write(self.cold/'result.json',record)
        self.register(self.cold);self.run_once();self.assertFalse(self.cold_cell()['native_compile_success'])

    def test_full_report_quality_rejection_and_nonfinite_are_independent_observations(self):
        self.cold_success();self.report(policy=False,finite=7999);self.run_once();c=self.cold_cell()
        self.assertTrue(c['primary_matrix_cell']);q=next(q for q in c['quality_observations']if q['kind']=='independent frozen report')
        self.assertFalse(q['report_policy_passed']);self.assertFalse(q['all_finite']);self.assertEqual(q['finite_samples'],7999)
        self.assertEqual(q['reference_top1_percent'],92.);self.assertEqual(q['approximate_top1_percent'],75.)
        self.assertFalse(q['feedback_to_agent']);self.assertFalse(q['used_for_selection'])

    def test_two_reports_bind_their_own_freeze_identities(self):
        self.cold_success();self.report();self.report(retained=True);self.run_once();a=self.load()
        observed=[(c['terminal']['result_path'],q['source_case'])for c in a['cells']for q in c['quality_observations']if q['kind']=='independent frozen report']
        self.assertEqual(set(observed),{(str(self.cold/'result.json'),str(self.cold)),(str(self.retained/'result.json'),str(self.session))})

    def test_wrong_report_claim_or_partial_error_never_admits_8000_accuracy(self):
        self.cold_success();r=self.report();r['report_samples_completed']=7999;r['status']='error';self.write(self.primary_report/'result.json',r);self.closure(self.primary_report,'error')
        self.run_once();c=self.cold_cell();self.assertFalse(any(q['kind']=='independent frozen report'for q in c['quality_observations']))
        self.assertFalse(c['report_terminal_observations'][0]['aggregate_accuracy_admitted'])

    def test_round_projection_skips_messages_source_and_session_conversation(self):
        self.cold_success();self.run_once()
        for p in self.delivery.glob('*.json'):
            self.assertNotIn('FORBIDDEN',p.read_text())
        rounds,_=subject.read_csv(self.delivery/'AGENT_ROUNDS.csv');self.assertEqual(len(rounds),2)
        n=next(x for x in rounds if x['run']==self.cold.name);self.assertEqual(n['origin'],'llm');self.assertEqual(n['selected_frozen'],'True')

    def test_repeated_refresh_keeps_history_rows_and_external_neighborhood_file(self):
        self.cold_success();self.run_once();self.run_once()
        stages,_=subject.read_csv(self.delivery/'STAGE_ATTRIBUTION.csv');rounds,_=subject.read_csv(self.delivery/'AGENT_ROUNDS.csv');counts,_=subject.read_csv(self.delivery/'COMPILED_COUNT_SCOPES.csv')
        self.assertEqual((len(stages),len(rounds),len(counts)),(3,2,2));self.assertEqual(len(self.load()['cells']),3)
        self.assertEqual((self.delivery/'COMPILED_BOOTSTRAP_NEIGHBORHOODS_THREE_MODELS.json').read_text(),'retained external source')

    def test_cached_tuned_round_does_not_borrow_raw_top1_or_claim_new_tune(self):
        self.cold_success();p=self.cold/'work/agent/agent_trace.jsonl';x=json.loads(p.read_text())
        x['tuned_reused_from']=1;x['tuned']=dict(source='FORBIDDEN '+('z'*10000),feasible=True,seconds=600.,cost_after=dict(boots_pred=99,latency_pred_us=50.),history=[dict(phase='result',feasible=True,agree=998,total=1000,max_u=.85,max_u_calibrated=.85,max_u_analytic=0.)])
        p.write_text(json.dumps(x)+'\n');self.run_once();rows,_=subject.read_csv(self.delivery/'AGENT_ROUNDS.csv');t=next(r for r in rows if r['candidate_kind']=='tuned')
        self.assertEqual(t['predicted_bootstraps'],'99');self.assertEqual(t['heldout_top1_percent'],'')
        self.assertEqual(t['new_tune_executed'],'False');self.assertEqual(t['tuner_reused_from'],'1');self.assertEqual(t['tuner_reported_seconds'],'600.0')

    def test_foreign_report_access_claim_and_plan_count_disagreement_are_refused(self):
        self.cold_success();r=self.report();marker=Path(r['access_marker']);self.write(marker,dict(source_case=str(self.session),out=str(self.primary_report)))
        self.run_once();self.assertFalse(any(q['kind']=='independent frozen report'for q in self.cold_cell()['quality_observations']))
        plan=self.cold/'post-freeze/refresh-plan.json';v=json.loads(plan.read_text());v['boots']=222;self.write(plan,v)
        # Fresh destination delivery preserves only the original variants before the mismatched cold admission.
        self.seed();self.run_once();self.assertFalse(self.cold_cell()['native_compile_success']);self.assertIsNone(self.cold_cell()['bootstrap_count'])

    def test_finite_clock_stops_at_cutoff_and_past_cutoff_writes_nothing(self):
        clock=[self.start];delays=[]
        def sleep(x):delays.append(x);clock[0]+=x
        calls=[]
        with patch.object(subject,'run_once',side_effect=lambda *x:calls.append(x)):
            self.assertEqual(subject.execute(self.args(once=False),now=lambda:clock[0],sleep=sleep),2)
        self.assertEqual(len(calls),2);self.assertEqual(delays,[60,60])
        with patch.object(subject,'run_once')as run:
            self.assertEqual(subject.execute(self.args(),now=lambda:self.start+120),0)
        run.assert_not_called()

    def test_main_method_coverage_counts_only_primary_agent_cells(self):
        delivery=self.load();delivery['cells']=[]
        for system in ('Agent','DaCapo-original'):
            for i in range(3):
                cell=subject.pending_cell(self.results/(system+str(i)),{})
                cell.update(system=system,primary_matrix_cell=True,native_compile_success=True,cold_full_E2E_seconds=10. if system=='Agent'and i<2 else None)
                cell['terminal']['result_status']='ok'
                if i==0:cell['quality_observations']=[dict(kind='independent frozen report')]
                delivery['cells'].append(cell)
        ablation=copy.deepcopy(delivery['cells'][0]);ablation['primary_matrix_cell']=False
        delivery['cells'].append(ablation);subject.coverage(delivery);c=delivery['coverage']
        self.assertEqual(c['successful_native_compiler_cells'],6)
        self.assertEqual(c['main_method_successful_native_compiler_cells'],3)
        self.assertEqual(c['main_method_compiler_cells_total'],3)
        self.assertEqual(c['main_method_Agent_total_cells'],3)
        self.assertEqual(c['main_method_Agent_compiled_cells'],3)
        self.assertEqual(c['main_method_Agent_cold_success_cells'],2)
        self.assertEqual(c['independent_report8000_cells'],2)
        self.assertEqual(c['main_method_independent_report8000_cells'],1)

    def test_historical_orbit_and_control_csv_quality_fields_remain_unchanged(self):
        delivery=self.load();rows,fields=subject.read_csv(self.delivery/(subject.BASE+'.csv'))
        expected={}
        for system,kind,accuracy,approximate,drop,max_u in (
                ('Orbit','optimized heldout diagnostic',92.4,92.3,.1,.82),
                ('DaCapo-same-backend-control','fixed degree heldout analysis',85.,85.,0.,.89)):
            cell=subject.pending_cell(self.results/system,{})
            cell.update(system=system,primary_matrix_cell=system=='Orbit',quality_observations=[dict(kind=kind,total=1000,reference_accuracy=accuracy,approximate_accuracy=approximate,max_u=max_u)])
            delivery['cells'].append(cell);row=subject.main_row(cell,{})
            values=dict(quality_kind=kind,quality_samples=1000,reference_accuracy_percent=accuracy,approximate_accuracy_percent=approximate,accuracy_drop_pp=drop,max_u=max_u)
            row.update(values);rows.append(row);expected[cell['cell_id']]={k:str(v)for k,v in values.items()}
        self.write(self.delivery/(subject.BASE+'.json'),delivery);self.csv(self.delivery/(subject.BASE+'.csv'),rows,fields)
        self.cold_success();self.run_once();after,_=subject.read_csv(self.delivery/(subject.BASE+'.csv'))
        for row in after:
            if row['cell_id']in expected:
                self.assertEqual({k:row[k]for k in expected[row['cell_id']]},expected[row['cell_id']])

    def test_manifest_and_publication_paths_and_poll_limits_are_finite(self):
        bad=copy.deepcopy(self.manifest);bad['reports'][0]['frozen_case']=str(self.session)
        with self.assertRaises(ValueError):subject.manifest_value(bad,self.results)
        for change in(dict(poll_seconds=True),dict(poll_seconds=1),dict(until='2026-11-18T00:00:00Z'),dict(delivery_dir=self.results)):
            with self.subTest(change=change),self.assertRaises(ValueError):subject.execute(self.args(**change),now=lambda:self.start)


if __name__=='__main__':unittest.main()
