"""Small completed25 metadata protocol fixtures; no model or Count replay."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pt_agent.experiments import qwen_completed_cost_cache as entry
from pt_agent.experiments.r10_paper_metrics import StageRecorder
from pt_agent.planners.transformer_session import load_state,save_state
from pt_agent.tests.test_qwen_component_plan_cache import fixture as accepted_fixture,write


def fixture(root):
    case,old,partition=accepted_fixture(root)
    session=case/'session';text=(session/'program.ptdsl').read_text()
    (session/'FROZEN.json').unlink();(session/'program.ptdsl').unlink()
    local=session/'sources/iteration-13.ptdsl';local.parent.mkdir();local.write_text(text)
    partial=Path(root).resolve()/'partial'
    original=partial/'session/sources/iteration-13.ptdsl';original.parent.mkdir(parents=True);original.write_text(text)
    source=dict(component_cost_policy=entry.POLICY,retained_candidate=13,
        new_provider_calls_allowed=0,original_layers=24,original_model_fixture='same-original')
    write(case/'command-1.json',['python','-m',entry.MODULE,'--worker-phase','calibration',
        '--session',str(session),'--partial-from',str(partial)])
    costroot=session/'component-cost-1';cost=entry._read(costroot/'result.json')
    counters=dict(add=25,sub=0,neg=0,mul_cc=0,mul_cp=0,rot=0,boot=787)
    cost.update(counts=counters,complete_count_replay_counts=counters,
        distinct_consts=40000,const_bytes=40000*16384*8,placement_exact=True,
        global_optimality=False,barriers=None,boot_share=.5)
    write(costroot/'result.json',cost)
    for name in entry.plans.NAMES:
        path=costroot/name/'complete-count.json';value=entry._read(path)
        boundary=19 if name=='head'else 32
        counts=dict(add=1,sub=0,neg=0,mul_cc=0,mul_cp=0,rot=0,boot=boundary)
        value['stats']=dict(counts=counts,ops_total=sum(counts.values()),distinct_consts=2000,
            max_depth=0,max_total_depth=0,min_level=5,outputs=boundary,rot_steps=[])
        write(path,value)
    rejected=dict(iteration=13,stage='resources',feasible=False,
        latency_pred_us=cost['latency_pred_us'],boots_pred=cost['boots_pred'])
    confirmation=dict(selected=None,evals=[],deduplicated=dict(before=0,after=0),duplicates=[])
    state=load_state(session/'SESSION.json')
    state.update(phase='done',iteration=14,calls=13,max_iters=13,tuner=False,
        prompt_version='r8-v11',placement='program-plan',confirm_rule='c2',
        component_cost_policy=entry.POLICY,new_provider_calls_allowed=0,
        pool=[],records={},best=None,best_program=None,confirmation=confirmation,
        history=[dict(iteration=i,stage='historical',feasible=False)for i in range(1,13)]+[rejected])
    save_state(session/'SESSION.json',state)
    write(session/'AGENT.json',dict(status='confirm_failed',confirm=confirmation,best=None,best_program=None))
    write(session/'guarded-replay.json',dict(status='committed',HTTP_added=0,committed_calls=13))
    write(session/'component-c2-access.json',dict(status='completed',calls=13,policy=entry.POLICY,report_samples=0))
    keys=('latency_pred_us','boots_pred','max_depth','ops_total','rot_keys','placement_model','placement_exact','barriers','boot_share')
    feedback=dict(stage='resources',feasible=False,validation=None,
        exact=dict(ok=True,first_mismatch=None,logit_max_diff=0.,act_count=[24,24]),
        cost={k:cost[k]for k in keys},
        errors=[dict(code='E_RESOURCE',limit='const_bytes',value=cost['const_bytes'],max=entry.OLD_CONST_BYTES)])
    write(session/'full-feedback/iteration-0013.full.json',dict(iteration=13,feedback=feedback,
        best_feasible_so_far=None,tuned=None))
    record=entry._read(case/'result.json');record.update(round=entry.SOURCE_ROUND,mode=entry.SOURCE_MODE,
        model='Qwen2-0.5B',status='quality_rejected',candidate_frozen=False,source=source,report_samples=0)
    for i,row in enumerate(record['phases'],1):
        row['worker']=dict(status='quality_rejected'if i==4 else 'ok')
        write(case/'phases'/(str(i).zfill(3)+'-'+row['phase'])/'worker-result.json',row['worker'])
    write(case/'result.json',record)
    write(case/'completion.json',dict(round=entry.SOURCE_ROUND,status='quality_rejected',
        required_artifacts_persisted=True,registry_persisted=True))
    registry=case.parent/'registry.jsonl';row=json.loads(registry.read_text())
    row.update(round=entry.SOURCE_ROUND,mode=entry.SOURCE_MODE,model='Qwen2-0.5B',metrics=dict(status='quality_rejected'))
    registry.write_text(json.dumps(row)+'\n')
    event_path=case/'phases/003-step/evaluation-stage-events.jsonl';event_path.unlink()
    ticks=iter(range(10000));recorder=StageRecorder(event_path,clock=lambda:next(ticks))
    with recorder.span('independent_exact_math'):pass
    with recorder.span('cost'):
        for name in entry.plans.NAMES:
            with recorder.span('component_cost_'+name):
                with recorder.span('cost_estimate'):
                    for stage in ('record_lower','placement','plan_express','conditional_plan_replay'):
                        with recorder.span(stage):pass
    return case,source,text


class CompletedCostChecks(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.case,self.source,self.text=fixture(self.temp.name)

    def prepare(self):
        with patch('pt_agent.experiments.r10_paper_metrics._process',return_value=None):
            return entry.prepare_outer(self.case,self.source)

    def worker_args(self):
        proof=self.prepare();directory=Path(self.temp.name).resolve()/'new';directory.mkdir()
        return entry.persist(directory,proof)

    def test_complete25_sole_resource_rejection_admits_without_quality_inheritance(self):
        proof=self.prepare();facts=proof['facts']
        self.assertEqual(list(facts['components']),list(entry.plans.NAMES))
        self.assertTrue(proof['all_process_births_closed'])
        self.assertGreater(facts['cost']['const_bytes'],4*entry.GIB)
        self.assertEqual(facts['original_C2_candidate_evaluations'],0)
        self.assertFalse(facts['heldout_or_C2_acceptance_inherited'])

    def test_worker_only_reads_saved_metadata_and_returns_cost_unchanged(self):
        path=self.worker_args();opened=Path.open
        def guarded(p,*args,**kwargs):
            if p.name=='registry.jsonl'or p.name=='compiler-config.json'or str(p).startswith('/proc/'):
                raise AssertionError('worker crossed metadata boundary')
            return opened(p,*args,**kwargs)
        with patch.object(Path,'open',guarded),patch('pt_agent.experiments.r10_paper_metrics._process',side_effect=AssertionError('process probe')):
            result=entry.load_worker(self.case,path,self.source,self.text)
        expected=entry._read(self.case/'session/component-cost-1/result.json');expected.pop('status')
        self.assertEqual(result['cost'],expected)
        self.assertFalse(result['provenance']['fresh_Count_executed'])
        self.assertEqual(result['provenance']['authority_case'],str(self.case))
        self.assertEqual(result['provenance']['original_candidate'],13)

    def test_partial_or_wrong_component_order_is_rejected(self):
        path=self.case/'session/component-cost-1/partial.json';original=entry._read(path)
        for change in (dict(completed=24),dict(components=list(reversed(entry.plans.NAMES)))):
            write(path,dict(original,**change))
            with self.assertRaises(ValueError):self.prepare()
        write(path,original)
        path=self.case/'session/component-cost-1/result.json';value=entry._read(path)
        value['refresh_plan']['components'].reverse();write(path,value)
        with self.assertRaises(ValueError):self.prepare()

    def test_only_constant_resource_error_and_no_validation_is_allowed(self):
        path=self.case/'session/full-feedback/iteration-0013.full.json';original=entry._read(path)
        changes=[dict(errors=[]),dict(errors=original['feedback']['errors']+[dict(code='E_RESOURCE',limit='rot_keys')]),
            dict(validation=dict(passed=False)),dict(exact=dict(ok=False)),dict(stage='validation')]
        for change in changes:
            value=copy.deepcopy(original);value['feedback'].update(change);write(path,value)
            with self.subTest(change=change),self.assertRaises(ValueError):self.prepare()

    def test_frozen_or_nonempty_confirmation_or_foreign_pool_is_rejected(self):
        path=self.case/'session/SESSION.json';original=load_state(path)
        for change in (dict(confirmation=dict(selected=dict(iteration=13),evals=[])),
                       dict(confirmation=dict(selected=None,evals=[dict(passed=False)])),dict(pool=[dict(iteration=13)])):
            state=copy.deepcopy(original);state.update(change);save_state(path,state)
            with self.assertRaises(ValueError):self.prepare()
        save_state(path,original);write(self.case/'session/FROZEN.json',{})
        with self.assertRaises(ValueError):self.prepare()

    def test_registry_duplicate_or_active_saved_birth_is_rejected(self):
        path=self.case.parent/'registry.jsonl';text=path.read_text();path.write_text(text+text)
        with self.assertRaises(ValueError):self.prepare()
        path.write_text(text)
        with patch('pt_agent.experiments.r10_paper_metrics._process',return_value=dict(starttime=123,state='S')):
            with self.assertRaises(ValueError):entry.prepare_outer(self.case,self.source)

    def test_cost_inventory_mismatch_and_incomplete_count_are_rejected(self):
        path=self.case/'session/component-cost-1/layer0/complete-count.json';original=entry._read(path)
        value=copy.deepcopy(original);value['stats'].pop('max_total_depth');write(path,value)
        with self.assertRaises(ValueError):self.prepare()
        write(path,original)
        path=self.case/'session/component-cost-1/result.json';value=entry._read(path)
        value['complete_count_replay_ops_total']+=1;write(path,value)
        with self.assertRaises(ValueError):self.prepare()

    def test_wrong_candidate_source_or_original_mode_is_rejected(self):
        path=self.case/'session/sources/iteration-13.ptdsl';path.write_text(self.text.replace('y=bias','y=neg'))
        with self.assertRaises(ValueError):self.prepare()
        path.write_text(self.text)
        path=self.case/'result.json';value=entry._read(path);value['mode']='other';write(path,value)
        with self.assertRaises(ValueError):self.prepare()

    def test_worker_rejects_changed_current_plan_or_candidate(self):
        path=self.worker_args()
        with self.assertRaises(ValueError):entry.load_worker(self.case,path,self.source,self.text+'\ndef foreign():\n    return 1\n')
        p=self.case/'session/component-cost-1/layer0/refresh-plan.json';value=entry._read(p)
        value['timing']['lower_s']+=1;write(p,value)
        with self.assertRaises(ValueError):entry.load_worker(self.case,path,self.source,self.text)

    def test_failed_exact_or_missing_successful_cost_event_is_rejected(self):
        path=self.case/'phases/003-step/evaluation-stage-events.jsonl'
        events=[json.loads(line)for line in path.read_text().splitlines()]
        event=next(e for e in events if e['stage_id']=='independent_exact_math'and e['type']=='end')
        event['status']='error';path.write_text('\n'.join(json.dumps(e)for e in events)+'\n')
        with self.assertRaises(ValueError):self.prepare()

    def test_proof_is_canonical_exclusive_and_rejects_invalid_birth_schema(self):
        proof=self.prepare()
        with self.assertRaises(ValueError):entry.persist(self.case/'session',proof)
        directory=Path(self.temp.name).resolve()/'new';directory.mkdir();path=entry.persist(directory,proof)
        with self.assertRaises(FileExistsError):entry.persist(directory,proof)
        proof['process_births']='foreign';write(path,proof)
        with self.assertRaises(ValueError):entry.load_worker(self.case,path,self.source,self.text)

    def test_budget_boundary_and_historical_event_records_are_preserved(self):
        self.assertEqual((entry.OLD_CONST_BYTES,entry.NEW_CONST_BYTES),(4*entry.GIB,32*entry.GIB))
        path=self.case/'phases/003-step/evaluation-stage-events.jsonl';events=path.read_text()
        proof_path=self.worker_args();result=entry.load_worker(self.case,proof_path,self.source,self.text)
        self.assertEqual(path.read_text(),events)
        self.assertIn('Historical',result['provenance']['original_clock_scope'])
        self.assertFalse(result['provenance']['physical_RAM_or_disk_limits_changed'])
        self.assertFalse(result['provenance']['heldout_or_C2_acceptance_inherited'])

    def test_only_exceeded_old_inventory_within_new_budget_is_eligible(self):
        path=self.case/'session/component-cost-1/result.json';original=entry._read(path)
        for distinct in (9000,300000):
            cost=copy.deepcopy(original);cost.update(distinct_consts=distinct,const_bytes=distinct*16384*8)
            write(path,cost)
            if distinct>50000:
                for name in entry.plans.NAMES:
                    count_path=path.parent/name/'complete-count.json';count=entry._read(count_path)
                    count['stats']['distinct_consts']=20000;write(count_path,count)
            with self.subTest(distinct=distinct),self.assertRaises(ValueError):self.prepare()


if __name__=='__main__':unittest.main()
