"""Closed-failure admission and explicit pure-worker protocol, metadata fixtures."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pt_agent.experiments import qwen_retained_component_replay as entry
from pt_agent.planners.transformer_session import load_state,save_state
from pt_agent.tests.test_qwen_retained_component_replay import state,write


class ComponentSourceChecks(unittest.TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup)
        self.root=Path(temporary.name)
        self.case=self.root/'grouped';self.case.mkdir()
        self.args=entry.parser().parse_args(['--resume-from',str(self.root/'resume'),
            '--partial-from',str(self.root/'partial'),'--grouped-from',str(self.case)])
        self.args.calibration=self.root/'fresh/calibration.json';self.args.calibration.parent.mkdir()
        self.args.session=self.root/'fresh/session';self.args.codegen=self.root/'fresh/codegen'
        self.args.work_dir=None;self.args.out=None
        self.base=dict(original_model='same-source',guarded_replay=dict(committed_calls=12,pending_reply_already_paid=True),
            resource_policy=dict(address_space_gib=24,subtree_rss_gib=16))
        self.expected=dict(copy.deepcopy(self.base),grouped_record_variant=entry.grouped.VARIANT)
        self.record=dict(round=entry.grouped.ROUND,mode=entry.grouped.MODE,model=entry.guarded.original.MODEL,status='error',candidate_frozen=False,
            termination=dict(cause='own_budget_guard'),api_calls=0,report_samples=0,source=copy.deepcopy(self.expected))
        self.done=dict(round=entry.grouped.ROUND,status='error',required_artifacts_persisted=True,registry_persisted=True)
        write(self.case/'result.json',self.record);write(self.case/'completion.json',self.done)
        write(self.case/'guard-failure.json',dict(cause='own_budget_guard'))
        value=state();value['max_iters']=15;value.pop('component_cost_policy');value.pop('new_provider_calls_allowed')
        save_state(self.case/'session/SESSION.json',value)
        for directory in (self.case,self.args.partial_from):
            target=directory/'session/sources/iteration-13.ptdsl';target.parent.mkdir(parents=True,exist_ok=True)
            target.write_text('original_paid_reply13 = 1\n')
        write(self.case/'command-1.json',['python','-m',entry.grouped.MODULE,'--resume-from',str(self.args.resume_from),
            '--partial-from',str(self.args.partial_from)])
        self.guard=self.case/'guard-samples.jsonl';self.guard.write_text(json.dumps(dict(subtree_rss_bytes=17*entry.GIB,available_bytes=3*entry.GIB))+'\n')
        (self.case/'phase-resources.jsonl').write_text(json.dumps(dict(processes=[dict(pid=101,starttime=202)],outer_process=dict(pid=103,starttime=204)))+'\n')
        self.registry=dict(run_id=self.case.name,round=entry.grouped.ROUND,model=entry.guarded.original.MODEL,mode=entry.grouped.MODE,
            git_rev='saved',metrics=dict(status='error'),artifacts_dir=str(self.case),ts='fixture')
        self.registry_path=self.root/'registry.jsonl';self.registry_path.write_text(json.dumps(self.registry)+'\n')

    def checked(self):
        with patch.object(entry.grouped,'source_gate',return_value=copy.deepcopy(self.expected)), \
                patch('pt_agent.experiments.r10_paper_metrics._process',return_value=None):
            return entry._closed_grouped(self.args,copy.deepcopy(self.base),check_processes=True)

    def test_closed_real_memory_failure_and_unique_birth_registry_proof(self):
        checked=self.checked()
        self.assertTrue(checked['actual_memory_stop']);self.assertEqual(checked['checkpoint_calls'],12)
        self.assertEqual(checked['process_births'],{'101':202,'103':204})
        self.assertFalse(checked['strict_math_or_full_cost_completion_claimed'])

    def test_running_source_or_missing_closure_rejected(self):
        self.record['status']='running';write(self.case/'result.json',self.record)
        with self.assertRaisesRegex(ValueError,'actual closed'):self.checked()
        self.record['status']='error';write(self.case/'result.json',self.record)
        self.done['registry_persisted']=False;write(self.case/'completion.json',self.done)
        with self.assertRaisesRegex(ValueError,'actual closed'):self.checked()

    def test_wall_only_or_predicted_stop_without_memory_evidence_rejected(self):
        self.guard.write_text(json.dumps(dict(subtree_rss_bytes=8*entry.GIB,available_bytes=3*entry.GIB))+'\n')
        with self.assertRaisesRegex(ValueError,'real resource stop'):self.checked()

    def test_active_original_process_identity_rejected(self):
        with patch.object(entry.grouped,'source_gate',return_value=self.expected), \
                patch('pt_agent.experiments.r10_paper_metrics._process',return_value=dict(starttime=202,state='R')):
            with self.assertRaisesRegex(ValueError,'remains active'):
                entry._closed_grouped(self.args,self.base,check_processes=True)

    def test_duplicate_registry_and_source_drift_rejected(self):
        self.registry_path.write_text((json.dumps(self.registry)+'\n')*2)
        with self.assertRaisesRegex(ValueError,'Unique original'):self.checked()
        self.registry_path.write_text(json.dumps(self.registry)+'\n')
        self.record['source']['original_model']='foreign';write(self.case/'result.json',self.record)
        with self.assertRaisesRegex(ValueError,'source fields'):self.checked()

    def test_foreign_reply_or_candidate_policy_rejected(self):
        source=self.case/'session/sources/iteration-13.ptdsl';source.write_text('different_candidate = 2\n')
        with self.assertRaisesRegex(ValueError,'same uncommitted'):self.checked()
        source.write_text('original_paid_reply13 = 1\n')
        value=load_state(self.case/'session/SESSION.json');value['tuner']=True
        save_state(self.case/'session/SESSION.json',value)
        with self.assertRaisesRegex(ValueError,'uncommitted reply13'):self.checked()

    def test_outer_exclusive_admission_then_pure_inner_has_no_registry_access(self):
        opened=[];original_open=Path.open
        def no_registry(path,*args,**kwargs):
            if path==self.registry_path:raise AssertionError('inner global registry')
            opened.append(str(path));return original_open(path,*args,**kwargs)
        with patch.object(entry.guarded,'source_gate',side_effect=lambda args:copy.deepcopy(self.base)), \
                patch.object(entry.grouped,'source_gate',side_effect=lambda args:copy.deepcopy(self.expected)), \
                patch('pt_agent.experiments.r10_paper_metrics._process',return_value=None):
            outer=entry.source_gate(self.args)
            self.assertEqual(outer['historical_paid_HTTP_count'],None)
            self.assertTrue(outer['retained_reply13_HTTP_already_occurred'])
            self.args.worker_phase='step';self.args.source_admission=self.args.calibration.parent/'component-source-admission.json'
            with patch.object(Path,'open',no_registry):inner=entry.source_gate(self.args)
        self.assertEqual(outer,inner);self.assertTrue(opened)
        proof=json.loads(self.args.source_admission.read_text());self.assertEqual(proof['module'],entry.MODULE)

    def test_forged_inner_proof_rejected(self):
        self.args.worker_phase='step';self.args.source_admission=self.args.calibration.parent/'component-source-admission.json'
        write(self.args.source_admission,dict(module=entry.MODULE,policy='wrong',checked={}))
        with patch.object(entry.guarded,'source_gate',return_value=self.base),patch.object(entry.grouped,'source_gate',return_value=self.expected):
            with self.assertRaisesRegex(ValueError,'Outer/inner'):entry.source_gate(self.args)

    def test_seed_copies_history_and_excludes_every_prior_cost_record(self):
        old=state();old['max_iters']=15;old['records']={(12,'llm','old_source'):(dict(placement_model='program-plan'),3)}
        save_state(self.args.partial_from/'session/SESSION.json',old)
        def seed(args):
            args.session.mkdir();save_state(args.session/'SESSION.json',copy.deepcopy(old))
            write(args.session/'guarded-replay.json',dict(status='pending_local_replay'))
        before=load_state(self.args.partial_from/'session/SESSION.json')
        with patch.object(entry.guarded,'seed_partial',side_effect=seed):entry.seed_partial(self.args)
        live=load_state(self.args.session/'SESSION.json')
        self.assertEqual(live['max_iters'],13);self.assertEqual(live['records'],{})
        self.assertEqual(live['messages'],before['messages']);self.assertEqual(live['history'],before['history'])
        self.assertEqual(live['calls'],12);self.assertEqual(live['pool'],[])
        marker=json.loads((self.args.session/'guarded-replay.json').read_text())
        self.assertEqual(marker['active_max_iters'],13)
        self.assertEqual(marker['active_new_provider_calls_allowed'],0)
        self.assertEqual(before,load_state(self.args.partial_from/'session/SESSION.json'))
        self.assertTrue((self.args.session/'inherited-cost-policy.json').is_file())
        self.assertTrue((self.args.calibration.parent/'prior-grouped-evidence/SESSION.json').is_file())

    def test_write_path_guard_rejects_all_original_sources(self):
        for original in (self.args.resume_from,self.args.partial_from,self.args.grouped_from):
            with self.subTest(original=original):
                self.args.out=original/'child'
                with self.assertRaises(ValueError):entry.check_write_paths(self.args)


if __name__=='__main__':unittest.main()
