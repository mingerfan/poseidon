"""Small fake-counter observability checks, with no graph or Count execution."""
import gc
import inspect
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import weakref

from pt_agent.experiments import qwen_append_observation as observation
from pt_agent.experiments import qwen_agent_grouped_record as entry
from pt_agent.experiments.r10_paper_metrics import StageRecorder, read_events, summarize_events


class Counter:
    def __init__(self, value): self.value = value
    def __len__(self): return self.value


class Owner:
    def __init__(self, count=249999):
        self.kind = Counter(count)
        self.numbering = {}
        self._constants = {'k0': None}
        self._constant_objects = {}
        self.op_context = 7
        self.calls = []
        self.returned = object()

    def _append(self, kind, args, *, is_input=False):
        self.calls.append((kind, args, is_input))
        self.kind.value += 1
        return self.returned


class AppendObservationChecks(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.recorder = StageRecorder(self.directory/'evaluation-stage-events.jsonl')
        self.target = self.directory/'dag-append-progress.json'

    def install(self, owner, sequence=1):
        self.assertTrue(observation.install_append_observation(owner, self.recorder, sequence))

    def test_exact_signature_call_once_identical_result_and_fixed_interval(self):
        owner = Owner(249998); original_signature = inspect.signature(owner._append)
        self.install(owner)
        self.assertEqual(inspect.signature(owner._append), original_signature)
        arguments = object()
        self.assertIs(owner._append('mul_cc', arguments, is_input=True), owner.returned)
        self.assertFalse(self.target.exists())
        self.assertIs(owner._append('add', arguments), owner.returned)
        self.assertEqual(owner.calls, [('mul_cc',arguments,True),('add',arguments,False)])
        saved = json.loads(self.target.read_text())
        self.assertEqual((saved['nodes_after_append'],saved['interval_nodes']), (250000,250000))
        self.assertLessEqual(self.target.stat().st_size, 2048)
        self.assertFalse(self.target.with_name(self.target.name+'.tmp').exists())

    def test_original_exception_and_async_exception_propagate_without_sample(self):
        class Failed(Owner):
            def _append(self, kind, args, *, is_input=False):
                self.calls.append((kind,args,is_input)); raise ValueError('original fixture')
        owner=Failed();self.install(owner)
        with self.assertRaisesRegex(ValueError,'original fixture'):owner._append('add',())
        self.assertEqual(len(owner.calls),1);self.assertFalse(self.target.exists())
        class Interrupt(BaseException): pass
        class Interrupted(Owner):
            def _append(self, kind, args, *, is_input=False):raise Interrupt()
        owner=Interrupted();self.install(owner)
        with self.assertRaises(Interrupt):owner._append('add',())
        self.assertFalse(self.target.exists())

    def test_weak_owner_releases_with_retained_wrapper_and_gc_disabled(self):
        enabled=gc.isenabled();gc.disable()
        try:
            owner=Owner();reference=weakref.ref(owner);self.install(owner)
            wrapper=owner._append;del owner
            self.assertIsNone(reference())
            self.assertFalse(hasattr(wrapper,'__wrapped__'))
            with self.assertRaisesRegex(RuntimeError,'owner has been released'):wrapper('add',())
        finally:
            if enabled:gc.enable()

    def test_absent_recorder_is_disabled_without_instance_override(self):
        owner=Owner()
        self.assertFalse(observation.install_append_observation(owner,None,1))
        self.assertNotIn('_append',vars(owner));self.assertFalse(self.target.exists())

    def test_post_append_lag_primitive_context_and_exact_whitelist(self):
        owner=Owner();self.install(owner,3)
        owner._append('add',())
        owner.numbering[('add',(),None)]=249999
        saved=json.loads(self.target.read_text())
        self.assertEqual(saved['numbering_entries_at_append'],0)
        self.assertEqual(len(owner.numbering),1)
        self.assertEqual((saved['op_context'],saved['graph_sequence']), (7,3))
        allowed={'schema_version','kind','pid','graph_sequence','sampled_ns','interval_nodes','sampling_point',
            'nodes_after_append','numbering_entries_at_append','constant_names_at_append',
            'constant_identity_entries_at_append','op_context','op_context_type_valid',
            'complete_lowering_claim','layer_progress'}
        self.assertEqual(set(saved),allowed)
        self.assertFalse(saved['complete_lowering_claim']);self.assertIsNone(saved['layer_progress'])
        class PrivateObject:
            def __repr__(self):raise AssertionError('No object representation may be inspected')
        owner.kind.value=499999;owner.op_context=PrivateObject();owner._append('add',())
        saved=json.loads(self.target.read_text())
        self.assertIsNone(saved['op_context']);self.assertFalse(saved['op_context_type_valid'])
        owner.kind.value=749999;owner.op_context=None;owner._append('add',())
        saved=json.loads(self.target.read_text())
        self.assertIsNone(saved['op_context']);self.assertTrue(saved['op_context_type_valid'])

    def test_io_failure_disables_sampling_and_preserves_original_returns(self):
        owner=Owner();self.install(owner)
        with patch.object(Path,'write_text',side_effect=OSError('fixture disk error'))as write:
            self.assertIs(owner._append('add',()),owner.returned)
            owner.kind.value=499999
            self.assertIs(owner._append('add',()),owner.returned)
        self.assertEqual(write.call_count,1);self.assertEqual(len(owner.calls),2)
        self.assertFalse(self.target.exists())

    def test_existing_instance_override_is_preserved_and_rejected(self):
        owner=Owner();existing=lambda *a,**k:object();owner._append=existing
        with self.assertRaisesRegex(ValueError,'original unbound append'):
            observation.install_append_observation(owner,self.recorder,1)
        self.assertIs(owner._append,existing)

    def test_interrupted_error_from_snapshot_write_propagates(self):
        owner=Owner();self.install(owner)
        interruption=InterruptedError('fixture supervisor interruption')
        with patch.object(Path,'write_text',side_effect=interruption):
            with self.assertRaises(InterruptedError)as observed:owner._append('add',())
        self.assertIs(observed.exception,interruption)
        self.assertEqual(len(owner.calls),1)
        self.assertEqual(len(owner.kind),250000)
        self.assertFalse(self.target.exists())

    def test_oversized_snapshot_is_skipped_and_async_write_signal_propagates(self):
        owner=Owner();self.install(owner)
        with patch.object(observation.json,'dumps',return_value='x'*2049)as serialize:
            self.assertIs(owner._append('add',()),owner.returned)
            owner.kind.value=499999;owner._append('add',())
        self.assertEqual(serialize.call_count,1);self.assertFalse(self.target.exists())
        class Interrupt(BaseException):pass
        owner=Owner();self.install(owner)
        with patch.object(Path,'write_text',side_effect=Interrupt()):
            with self.assertRaises(Interrupt):owner._append('add',())

    def test_actual_stage_reader_stack_sequence_and_timing_stay_unchanged(self):
        owner=Owner();self.install(owner)
        with self.recorder.span('record_lower'):
            sequence,stack=self.recorder.sequence,list(self.recorder.stack)
            owner._append('add',())
            self.assertEqual(self.recorder.sequence,sequence);self.assertEqual(self.recorder.stack,stack)
        events=read_events(self.recorder.path)
        self.assertEqual([event['type']for event in events],['begin','end'])
        timing=summarize_events(events)
        self.assertEqual(timing['status'],'ok');self.assertTrue(timing['attribution_ready'])
        self.assertEqual(len(timing['spans']),1);self.assertTrue(self.target.exists())

    def test_grouped_factory_installs_observer_after_ids_and_preserves_parent_provenance(self):
        owner=Owner();owner.numbering=entry.compact.base.ScalarNumbering(owner,object)
        lifetime=entry.compact.base.Lifetime(entry.compact.base.Diagnostics(self.recorder.path))
        namespace={'CostDag':lambda *a,**k:owner}
        with patch.object(entry.compact,'private_planner',return_value=namespace):
            planner=entry.private_planner(lifetime)
        self.assertIs(planner['CostDag'](64),owner)
        self.assertIsInstance(owner.numbering.ids,entry.GroupedIds)
        self.assertIs(owner._append('add',()),owner.returned)
        self.assertEqual(json.loads(self.target.read_text())['numbering_entries_at_append'],0)
        source=dict(cost_lifetime_variant=entry.compact.base.VARIANT,compact_record_variant=entry.compact.VARIANT)
        args=SimpleNamespace(out=self.directory)
        with patch.object(entry.compact,'source_gate',return_value=source.copy())as parent:
            observed=entry.source_gate(args)
        parent.assert_called_once_with(args)
        self.assertEqual({key:observed[key]for key in source},source)
        self.assertEqual(observed['grouped_record_observation_interval_nodes'],250000)
        self.assertEqual(observed['grouped_record_observation_scheme'],observation.SCHEME)
        self.assertTrue((self.directory/'variant-source/qwen_append_observation.py').is_file())


if __name__ == '__main__':
    unittest.main()
