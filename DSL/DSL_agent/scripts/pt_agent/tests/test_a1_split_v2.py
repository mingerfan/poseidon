"""A1 §6.13 deterministic reassignment and version-isolated sessions."""

import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pt_agent.data.samples import calibration_split_v2,read_selected_text,report_count,split_version
from pt_agent.planners import agent
from pt_agent.planners.transformer_session import session_step,save_state
from pt_agent.tests._artifacts import RESULTS_DIR,require


def partition():
    return dict(seed=20261024,calibration=list(range(23)),heldout=list(range(23,43)),
                confirm=list(range(43,63)),report=list(range(63,223)),
                sample_ids=[f'rows.jsonl:{i:08d}' for i in range(223)])


class SplitV2Tests(unittest.TestCase):
    def test_exact_prefix_order_partitions_and_original_are_preserved(self):
        original=partition();retained=copy.deepcopy(original);new=calibration_split_v2(original)
        self.assertEqual(new['calibration'],original['calibration']+original['report'][:60])
        self.assertEqual(new['report'],original['report'][60:])
        self.assertEqual((new['heldout'],new['confirm']),(original['heldout'],original['confirm']))
        names=('calibration','heldout','confirm','report')
        self.assertEqual([len(new[n]) for n in names],[83,20,20,100])
        for i,left in enumerate(names):
            for right in names[i+1:]:self.assertFalse(set(new[left])&set(new[right]))
        self.assertEqual(set().union(*(set(new[n]) for n in names)),set(range(223)))
        self.assertEqual(original,retained)
        self.assertEqual(new['sample_ids'],original['sample_ids'])
        self.assertEqual(new['version'],'split-v2')

    def test_partition_cannot_be_moved_twice_and_source_must_be_disjoint(self):
        with self.assertRaises(ValueError):calibration_split_v2(calibration_split_v2(partition()))
        original=partition();original['confirm'][0]=original['heldout'][0]
        with self.assertRaises(ValueError):calibration_split_v2(original)

    def test_only_selected_text_rows_are_parsed_and_other_rows_are_unavailable(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'rows.jsonl'
            path.write_text('not JSON\n{"text":"first","label":1}\nnot JSON\n{"text":"second","label":0}\n')
            ids=[f'rows.jsonl:{i:08d}' for i in range(4)]
            samples=read_selected_text(path,ids,[3,1])
            text,labels=samples.select([3,1])
            self.assertEqual(text,['second','first']);self.assertEqual(labels.tolist(),[0,1])
            self.assertIsNone(samples.items[0]);self.assertIsNone(samples.items[2])
            with self.assertRaises(ValueError):samples.select([0])

    def test_report_size_depends_on_version_and_legacy_defaults_to_v1(self):
        old=SimpleNamespace(split=partition());new=SimpleNamespace(split=calibration_split_v2(partition()))
        self.assertEqual((split_version(old),report_count(old)),('split-v1',160))
        self.assertEqual((split_version(new),report_count(new)),('split-v2',100))

    def test_new_state_starts_empty_and_trace_tags_do_not_change_feedback(self):
        ctx=SimpleNamespace(prompt_version='r8-v8',model_info=SimpleNamespace(split=calibration_split_v2(partition())))
        fb=dict(stage='parse',feasible=False,error='synthetic')
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(agent,'build_system_prompt',return_value='frozen system'), \
                patch.object(agent,'initial_message',return_value='summary'), \
                patch.object(agent,'evaluate',return_value=fb):
            state=agent.new_agent_state(ctx,tuner=False)
            self.assertEqual(state['split_version'],'split-v2')
            self.assertEqual((state['pool'],state['tune_cache'],state['calls']),([],{},0))
            trace=Path(tmp)/'trace.jsonl'
            agent.agent_step(ctx,state,trace_path=trace,reply=dict(content='{"program":"invalid"}',
                usage={},seconds=0,finish_reason='stop'))
            row=json.loads(trace.read_text())
            self.assertEqual(row['split_version'],'split-v2')
            self.assertEqual(row['fb'],fb)

    def test_old_checkpoint_cannot_supply_pool_or_cache_to_v2_model(self):
        ctx=SimpleNamespace(prompt_version='r8-v8',model_info=SimpleNamespace(split=calibration_split_v2(partition())))
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'SESSION.json'
            save_state(path,dict(prompt_version='r8-v8',tune_rule='v8-6.11',phase='evaluate',pool=['old'],tune_cache={'old':1}))
            with self.assertRaisesRegex(ValueError,'different sample partitions'):session_step(ctx,Path(tmp))


PREPARED=[RESULTS_DIR/name/'model/model_info.json' for name in
          ('A1-bert-prepare-20261027T123333-7967','A1-qwen-prepare-20261027T132722-12c2')]


@require(*PREPARED)
class ExistingPartitionTests(unittest.TestCase):
    def test_same_rule_for_both_retained_models(self):
        for path in PREPARED:
            with self.subTest(model=path.parent.parent.name):
                old=json.loads(path.read_text())['split'];new=calibration_split_v2(old)
                self.assertEqual(new['heldout'],old['heldout']);self.assertEqual(new['confirm'],old['confirm'])
                self.assertEqual(new['calibration'][23:],old['report'][:60])
                self.assertEqual(set().union(*(set(new[n]) for n in ('calibration','heldout','confirm','report'))),set(range(223)))


if __name__=='__main__':unittest.main()
