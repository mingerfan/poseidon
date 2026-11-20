"""Deterministic small stage plans; no model, training input or native trace."""

from contextlib import ExitStack
from dataclasses import replace
import gc
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import weakref

from pt_agent import paths
from pt_agent.tests._artifacts import require


TEMPLATE = paths.RESULTS_DIR/'E5d-agent-sessions-20261106/bert/acceptance/pars/export/Bert_PTAgent_DSL/config/compile_argv.json'


@require(TEMPLATE)
class StagedGenerationChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
            os.environ[key] = '1'
        from pt_agent.experiments.r29_local_compile_matrix import native_metadata_commands
        cls.context = ExitStack()
        cls.addClassCleanup(cls.context.close)
        cls.context.enter_context(native_metadata_commands(json.loads(TEMPLATE.read_text())))
        import numpy as np
        from pt_agent.experiments import qwen_staged_generation as entry
        from pt_agent.experiments.qwen_prefill_compile import _ephemeral_cost_dag
        from pt_agent import plan
        from pt_agent.backends import RefreshBackend
        from pt_agent.backends.base import Backend
        from pt_agent.backends.cost_dag import CostDag
        from pt_agent.backends.streaming_count import StreamingCountBackend
        from pt_agent.backends.streaming_mock import StreamingMockBackend
        from pt_agent.backends.streaming_emit import StreamingEmitBackend
        from pt_agent.placement import depth_dp
        from pt_agent.kernels.encrypted_generation import numpy_selection
        from pt_agent.experiments.qwen_monotone_selector import analytic_coefficients
        cls.np, cls.entry, cls.plan = np, entry, plan
        cls.Refresh, cls.Base = RefreshBackend, Backend
        cls.Count, cls.Mock, cls.Emit = StreamingCountBackend, StreamingMockBackend, StreamingEmitBackend
        cls.Dag = _ephemeral_cost_dag(CostDag, np)
        cls.depth_dp, cls.selection = staticmethod(depth_dp), staticmethod(numpy_selection)
        cls.coefficients = analytic_coefficients(np)
        cls.plans = {}

    def count(self):
        return self.Count(64, input_level=5, boot_out_level=5)

    def physical(self, rows=None):
        backend = self.Mock(64)
        backend.bind(self.inputs() if rows is None else rows)
        return backend

    def inputs(self):
        rows = [self.np.zeros(64, dtype=self.np.float64) for _ in range(34)]
        rows[0][:2] = [-.5, .75]
        rows[32][:9] = 1.
        rows[33][8] = 1.
        return rows

    def planned(self, callback, *, output_refresh=False):
        with patch.object(self.plan, 'levels', return_value=dict(self.entry.LEVELS)), \
                patch.object(self.plan, 'CostDag', self.Dag), self.entry.compact_scope():
            dag, recorder, _ = self.plan.record_dag(callback, 64, frontend_normalize=True)
            if output_refresh:
                node = dag._outputs[0].nid
                placed = SimpleNamespace(refreshed_uses={node: [len(dag.kind)]}, boot_nodes=[node],
                    boots=1, barriers=0, cuts=[], latency_us=0.)
            else:
                keys = ('bootstrap_single', 'mul_double', 'mul_single', 'rescale_single',
                        'rotate_single', 'negate_single', 'add_double', 'add_single')
                table = {key: {level: 1. for level in range(1, 6)} for key in keys}
                placed = self.depth_dp(dag, last_boot_level=4, boot_in_level=1, boot_out_level=5,
                    latency_table=table, boot_normalization_levels=0)
            result = self.plan.express(dag, recorder, placed)
        return result

    def fixture(self, vocabulary=33):
        np = self.np
        embedding = (np.arange(vocabulary*2, dtype=np.float64).reshape(vocabulary, 2)%11-5.)/16.
        head = np.column_stack((np.arange(vocabulary, dtype=np.float64)/64.,
                                -np.ones(vocabulary, dtype=np.float64)/64.))
        embedding.flags.writeable = False
        head.flags.writeable = False
        observations = []

        def prefill(active):
            from pt_agent.kernels import transformer
            from pt_agent.kernels.layout import FlatLayout
            values = [active.input(index) for index in range(34)]
            observations.append(dict(backend=active, token=values[9].value,
                token_level=values[9].level, mask_level=values[32].level, sel_level=values[33].level))
            output, layout = transformer.linear(active, [values[0]+values[9]], FlatLayout(64, 2), head)
            active.output(transformer.values(output, layout))

        first, terminal = self.entry.kernel_callbacks(embedding)
        if vocabulary not in self.plans:
            self.plans[vocabulary] = (self.planned(prefill), self.planned(first), self.planned(terminal))
        observations.clear()
        return embedding, head, prefill, observations, self.plans[vocabulary]

    def run_fixture(self, backend, vocabulary=33):
        embedding, head, prefill, observations, plans = self.fixture(vocabulary)
        result = self.entry.run_staged(backend, prefill=prefill, prefill_plan=plans[0],
            first_kernel_plan=plans[1], terminal_kernel_plan=plans[2], original_embedding=embedding,
            prompt_length=9)
        return result, embedding, head, observations, plans

    def test_original_output_substitution_captures_refreshed_handle_and_level(self):
        seen = []
        def callback(active):
            values = [active.input(index) for index in range(34)]
            result = values[0]*2.
            seen.append(result)
            active.output([result])
        plan = self.planned(callback, output_refresh=True)
        state = self.entry.Segments(self.count())
        outputs = state.run('output-refresh', state.inputs, plan, callback)
        self.assertEqual(outputs[0].level, 5)
        self.assertIsNot(outputs[0].value, seen[-1].value)
        row = state.metadata['segments'][0]
        self.assertEqual(row['output_refresh_positions'], [0])
        self.assertEqual(row['refresh']['boots'], 1)
        self.assertEqual(row['refresh']['calls'], plan['calls'])
        self.assertIsNone(state.backend._outputs)
        self.assertEqual(len(state.backend._inputs), 34)

    def test_logical_alias_preserves_original_input_numbering_and_restoration(self):
        def callback(active):
            values = [active.input(index) for index in range(34)]
            active.output([values[0]+values[1]])
        plan = self.planned(callback)
        physical = self.count()
        state = self.entry.Segments(physical)
        original = tuple(physical._inputs)
        first = state.run('one', state.inputs, plan, callback)
        second = state.run('two', state.inputs, plan, callback)
        self.assertEqual([row['logical_input_calls'] for row in state.metadata['segments']], [34, 34])
        self.assertTrue(all(a is b for a, b in zip(original, physical._inputs)))
        self.assertEqual(len(physical._inputs), 34)
        self.assertNotIn('input', physical.__dict__)
        self.assertNotIn('output', physical.__dict__)
        state.finish([*first, *second])
        with self.assertRaises(ValueError):
            state.finish(first)

    def test_no_unproved_relabel_or_foreign_handle_admission(self):
        state = self.entry.Segments(self.count())
        masks = state.masks(10)
        def callback(active):
            x = active.input(0)
            active.output([x+0.])
        plan = self.planned(callback)
        with self.assertRaisesRegex(ValueError, 'proved level5'):
            state.run('unrefreshed', masks[:1], plan, callback)
        with self.assertRaisesRegex(ValueError, 'original handle'):
            state.run('forged', [replace(masks[0], level=5)], plan, callback)
        other = self.entry.Segments(self.count())
        with self.assertRaisesRegex(ValueError, 'original handle'):
            state.normalize(other.inputs[:1], 'foreign')

    def test_public_masks_consume_real_layer_then_actual_boot(self):
        backend = self.count()
        state = self.entry.Segments(backend)
        masks = state.masks(10)
        self.assertEqual([x.level for x in masks], [4, 4])
        self.assertEqual([x.value.level for x in masks], [4, 4])
        self.assertEqual(backend.stats()['counts']['mul_cp'], 1)
        normalized = state.normalize(masks, 'mask-boundary')
        self.assertEqual([x.value.level for x in normalized], [5, 5])
        self.assertEqual(backend.stats()['counts']['boot'], 2)
        self.assertEqual(state.metadata['boundary_bootstraps'], 2)

    def test_missing_refresh_substitution_preserves_original_level_failure(self):
        def callback(active):
            x = active.input(0)
            for _ in range(6):
                x = x*.5
            active.output([x])
        good = self.planned(callback)
        bad = dict(good, substitutions=[], outputs=[], boots=0)
        state = self.entry.Segments(self.count())
        with self.assertRaisesRegex(Exception, 'below boot_in_level'):
            state.run('missing-real-boot', state.inputs[:1], bad, callback)
        self.assertEqual(state.metadata['segments'][-1]['status'], 'error')
        self.assertNotIn('input', state.backend.__dict__)
        self.assertNotIn('output', state.backend.__dict__)

    def test_small_full_arithmetic_matches_independent_unbooted_reference(self):
        backend = self.physical()
        state, embedding, head, observations, _ = self.run_fixture(backend)
        x = self.inputs()[0][:2]
        w0, _ = self.selection(head@x, bound=32., normalized_gap=2.**-27, coefficients=self.coefficients)
        mixed = w0@embedding
        w1, _ = self.selection(head@(x+mixed), bound=32., normalized_gap=2.**-27, coefficients=self.coefficients)
        got = [self.np.concatenate((backend.outputs[step*2][:32],
                                    backend.outputs[step*2+1][:1])) for step in range(2)]
        for observed, reference in zip(got, (w0, w1)):
            self.assertTrue(self.np.all(self.np.abs(observed-reference) <= 1e-12+1e-12*self.np.abs(reference)))
        self.assertEqual(len(observations), 2)
        handoff = state._known[state.metadata['handoff']['normalized_embedding']]
        self.assertIs(observations[1]['token'], handoff.value)
        self.assertTrue(self.np.all(self.np.abs(handoff.value.value[:2]-mixed) <= 1e-12+1e-12*self.np.abs(mixed)))
        self.assertEqual([row['token_level'] for row in observations], [5, 5])
        self.assertEqual(state.metadata['consumed_embeddings'], 1)
        self.assertTrue(state.metadata['terminal_unused_embedding_eliminated'])
        self.assertFalse(state.metadata['whole_model_verified'])

    def test_nineteen_blocks_count_all41boundary_boots_and_final38_outputs(self):
        backend = self.count()
        state, _, _, _, plans = self.run_fixture(backend, 577)
        self.assertEqual(state.metadata['vocabulary_blocks'], 19)
        self.assertEqual(state.metadata['boundary_bootstraps'], 41)
        self.assertEqual(len(backend._outputs), 38)
        expected = 2*plans[0]['boots']+plans[1]['boots']+plans[2]['boots']+41
        self.assertEqual(backend.stats()['counts']['boot'], expected)
        self.assertEqual(len(backend._inputs), 34)
        self.assertEqual(state.metadata['final_output_calls'], 1)

    def test_small_count_emit_same_segment_schedule_without_native_trace(self):
        counted = self.count()
        count_state, _, _, _, _ = self.run_fixture(counted)
        with TemporaryDirectory() as temporary:
            backend = self.Emit(64, directory=Path(temporary)/'emit', frontend=Path(temporary)/'unused.py')
            try:
                emitted, _, _, _, _ = self.run_fixture(backend)
                self.assertEqual(backend.stats()['counts'], counted.stats()['counts'])
                self.assertEqual(backend.stats()['distinct_consts'], counted.stats()['distinct_consts'])
                self.assertEqual(emitted.metadata['boundary_bootstraps'], count_state.metadata['boundary_bootstraps'])
                self.assertEqual(backend.stats()['identity_cache_entries'], 0)
                self.assertEqual(backend.stats()['retained_constant_arrays'], 0)
                self.assertTrue(backend.write().is_file())
            finally:
                if not backend.finished:
                    backend.close()

    def test_private_kernel_guards_restore_and_reject_invalid_inner(self):
        from pt_agent.kernels import compact_linear, streaming_linear, transformer
        original = (compact_linear.embedding_from_selection, streaming_linear.streaming_projection, transformer.linear)
        with self.assertRaisesRegex(RuntimeError, 'sentinel'):
            with self.entry.compact_scope():
                self.assertIsNot(compact_linear.embedding_from_selection, original[0])
                raise RuntimeError('sentinel')
        self.assertEqual((compact_linear.embedding_from_selection, streaming_linear.streaming_projection, transformer.linear), original)
        fake = self.Refresh(self.Base(64), policy='record', **self.entry.LEVELS)
        matrix = self.np.ones((3, 2), dtype=self.np.float64)
        with self.entry.compact_scope(), self.assertRaisesRegex(ValueError, 'Compact embedding'):
            compact_linear.embedding_from_selection(fake, {'metadata': {'vocabulary': 3}}, matrix)

    def test_callback_failure_restores_preexisting_instance_aliases(self):
        backend = self.count()
        state = self.entry.Segments(backend)
        old_input, old_output = backend.input, backend.output
        backend.input, backend.output = old_input, old_output
        def callback(active):
            active.input(0)
            raise RuntimeError('fixture-stop')
        plan = dict(format='pt-agent-refresh-plan-v1', levels=dict(self.entry.LEVELS), calls=1,
            boots=0, substitutions=[], outputs=[], dead=[])
        with self.assertRaisesRegex(RuntimeError, 'fixture-stop'):
            state.run('failed', state.inputs[:1], plan, callback)
        self.assertIs(backend.__dict__['input'], old_input)
        self.assertIs(backend.__dict__['output'], old_output)
        self.assertEqual(state.metadata['segments'][0]['status'], 'error')
        self.assertIsNone(backend._outputs)

    def test_plan_call_mismatch_and_nonconsecutive_logical_input_rejected(self):
        def callback(active):
            active.input(1)
        state = self.entry.Segments(self.count())
        plan = dict(format='pt-agent-refresh-plan-v1', levels=dict(self.entry.LEVELS), calls=1,
            boots=0, substitutions=[], outputs=[], dead=[])
        with self.assertRaisesRegex(ValueError, 'consecutive'):
            state.run('bad-order', state.inputs[:1], plan, callback)
        def valid(active):
            x = active.input(0)
            active.output([x+0.])
        with self.assertRaisesRegex(Exception, 'covers1calls|covers 1 calls'):
            state.run('bad-calls', state.inputs[:1], plan, valid)

    def test_fixed_coefficients_and_original_weights_are_preserved(self):
        embedding, _, _, _, _ = self.fixture()
        saved = embedding.copy()
        bad = list(self.coefficients)
        bad[-1] = bad[-1]*2.
        with self.assertRaisesRegex(ValueError, '52original'):
            self.entry.kernel_callbacks(embedding, bad)
        self.run_fixture(self.count())
        self.assertFalse(embedding.flags.writeable)
        self.assertTrue(self.np.array_equal(embedding, saved))
        for value in (True, 0, 31):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.entry.run_staged(self.count(), prefill=None, prefill_plan={}, first_kernel_plan={},
                    terminal_kernel_plan={}, original_embedding=embedding, prompt_length=value)

    def test_success_wrapper_released_without_cyclic_gc_and_raw_handle_survives(self):
        wrappers = []
        def callback(active):
            wrappers.append(weakref.ref(active))
            values = [active.input(index) for index in range(34)]
            active.output([values[0]*2.])
        plan = self.planned(callback, output_refresh=True)
        before = json.dumps(plan, sort_keys=True)
        state = self.entry.Segments(self.count())
        enabled = gc.isenabled()
        gc.disable()
        try:
            outputs = state.run('released-success', state.inputs, plan, callback)
            self.assertIsNone(wrappers[-1]())
            self.assertEqual(json.dumps(plan, sort_keys=True), before)
            self.assertEqual(outputs[0].value.level, 5)
            # Reuse the surviving actual raw handle in another original plan.
            again = state.run('raw-reuse', [outputs[0]]*34, plan, callback)
            self.assertIsNone(wrappers[-1]())
            self.assertEqual(again[0].value.level, 5)
            state.finish(again)
            self.assertIs(state.backend._outputs[0], again[0].value)
        finally:
            if enabled:
                gc.enable()

    def test_error_wrapper_private_cache_and_plan_release_without_cyclic_gc(self):
        wrappers = []
        def complete(active):
            values = [active.input(index) for index in range(34)]
            active.output([values[0]*2.])
        plan = self.planned(complete, output_refresh=True)
        before = json.dumps(plan, sort_keys=True)
        state = self.entry.Segments(self.count())
        def failed(active):
            wrappers.append(weakref.ref(active))
            complete(active)
            # Output refresh already populated a RValue->active cache cycle.
            raise RuntimeError('post-output-failure')
        def invoke():
            try:
                state.run('released-error', state.inputs, plan, failed)
            except RuntimeError:
                pass
        enabled = gc.isenabled()
        gc.disable()
        try:
            invoke()
            self.assertIsNone(wrappers[-1]())
            self.assertEqual(json.dumps(plan, sort_keys=True), before)
            self.assertEqual(state.metadata['segments'][-1]['status'], 'error')
            self.assertFalse(state._busy)
            self.assertNotIn('input', state.backend.__dict__)
            self.assertNotIn('output', state.backend.__dict__)
            outputs = state.run('next-after-error', state.inputs, plan, complete)
            self.assertEqual(outputs[0].value.level, 5)
        finally:
            if enabled:
                gc.enable()


if __name__ == '__main__':
    unittest.main()
