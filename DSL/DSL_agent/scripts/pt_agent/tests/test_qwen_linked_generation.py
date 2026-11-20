"""Small packed arithmetic and linked-handle protocol; original models unused."""

from contextlib import nullcontext
import json
import os
from pathlib import Path
import signal
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from pt_agent.experiments import qwen_linked_generation as linked


class LinkedGenerationCases(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.np = linked.embedding.base._numeric()
        from pt_agent.experiments.qwen_mock_progress import ObservedStreamingMockBackend
        from pt_agent.kernels import encrypted_generation
        cls.Backend, cls.selector = ObservedStreamingMockBackend, encrypted_generation

    def test_batch_one_adapter_and_complete_packed_cubic_tournament(self):
        np = self.np
        # More than one block and an odd carry exercise the real packed path.
        logits = np.asarray([.9, -.8, .4, -.5, .2], dtype=np.float64)
        stages = linked.monotone.analytic_coefficients(np)
        independent, _ = self.selector.numpy_selection(logits, bound=32., normalized_gap=2.**-27, coefficients=stages)
        backend = self.Backend(8, batch=1)
        try:
            inputs = linked.embedding.packed_selection(backend, logits, np)
            rows = []
            with linked.observe_packed_stages(backend, len(logits), len(stages), rows, np):
                selection = self.selector.select_maximum(backend, inputs['one_hot'], len(logits),
                    bound=32., normalized_gap=2.**-27, coefficients=stages)
            output = linked.decode_selection(selection, np)
            self.assertTrue(linked.fixed.strict_math(independent, output, np)['passed'])
            self.assertEqual(int(output.argmax()), 0)
            self.assertEqual(len(rows), 3*52)
            self.assertTrue(all(row['finite'] and row['output_finite'] for row in rows))
            self.assertTrue(all(value.value.shape == (1, 8) for value in selection['one_hot']))
            self.assertEqual(sum(row['actual_pairs'] for row in rows if row['stage'] == 0), 4)
            self.assertEqual(rows[-1]['stage'], 51)
            # Packed approximate one-hot, not its discrete argmax, enters embedding.
            matrix = np.asarray([[1., 4., 0.], [-2., 1., 7.], [8., -3., 2.], [3., 2., -4.], [5., -1., 6.]])
            values, layout = linked.embedding.compact_embedding(backend, selection, matrix)
            clear = linked.embedding.clear_mixture(output, matrix, np, block_rows=2)
            self.assertTrue(linked.fixed.strict_math(clear, values[0].value[0, :3], np)['passed'])
            self.assertEqual(layout.size, 3)
        finally:
            backend.release_batch()

    def test_stage_observer_restores_and_preserves_original_arithmetic(self):
        from pt_agent.kernels import poly
        np = self.np
        stages = linked.monotone.analytic_coefficients(np)
        original = poly.cheb_eval
        def execute(observe):
            backend = self.Backend(8, batch=1)
            try:
                values = linked.embedding.packed_selection(backend, np.asarray([.9, -.8, .4, -.5, .2]), np)['one_hot']
                context = linked.observe_packed_stages(backend, 5, 52, [], np) if observe else nullcontext()
                with context:
                    result = self.selector.select_maximum(backend, values, 5, bound=32., normalized_gap=2.**-27, coefficients=stages)
                return linked.decode_selection(result, np)
            finally:
                backend.release_batch()
        np.testing.assert_array_equal(execute(False), execute(True))
        self.assertIs(poly.cheb_eval, original)
        backend = self.Backend(8, batch=1)
        with self.assertRaises(RuntimeError):
            with linked.observe_packed_stages(backend, 5, 52, [], np):
                raise RuntimeError('synthetic error')
        self.assertIs(poly.cheb_eval, original)
        backend.release_batch()

    def test_mixed_state_uses_fractional_values_and_only_public_position(self):
        np = self.np
        matrix = np.asarray([[1., 2., 3.], [4., 5., 6.], [-2., -3., -4.], [7., 8., 9.]])
        state = dict(step=0, prefix_token_ids=[0, 1, 3, 3], prefix_length=2,
            attention_mask=[[1., 1., 0., 0.]], sel=[[0., 1., 0., 0.]])
        mixed = np.asarray([.2, .3, .5]) @ matrix[:3]
        arrays, updated = linked.mixed_inputs(state, matrix, mixed, np)
        np.testing.assert_array_equal(arrays['inputs_embeds'][0, :2], matrix[[0, 1]])
        np.testing.assert_array_equal(arrays['inputs_embeds'][0, 2], mixed)
        self.assertFalse(any(np.array_equal(mixed, row) for row in matrix))
        self.assertEqual(updated['prefix_token_ids'], [0, 1, None, 3])
        self.assertEqual(updated['attention_mask'], [[1., 1., 1., 0.]])
        self.assertEqual(updated['sel'], [[0., 0., 1., 0.]])
        self.assertEqual(state['prefix_token_ids'], [0, 1, 3, 3])
        with self.assertRaises(ValueError):
            linked.mixed_inputs(dict(state, prefix_length=3), matrix, mixed, np)

    def test_handle_binding_preserves_same_object_and_restores_after_failure(self):
        np = self.np
        backend = self.Backend(8, batch=1)
        backend.bind([np.arange(8, dtype=np.float64)[None, :]])
        original = backend.input
        handle = backend.input(0)*.5
        packed = [np.zeros((1, 8)), handle.value, np.ones((1, 8))]
        record = {}
        with linked.bind_embedding_handle(backend, packed, 1, handle, record, np):
            backend.input(0)
            self.assertIs(backend.input(1), handle)
            backend.input(2)
        self.assertTrue(record['same_object'])
        self.assertTrue(record['restored'])
        self.assertEqual(record['calls'], 1)
        self.assertEqual(backend.input, original)
        with self.assertRaises(RuntimeError):
            with linked.bind_embedding_handle(backend, packed, 1, handle, {}, np):
                raise RuntimeError('failure')
        self.assertEqual(backend.input, original)
        other = list(packed)
        other[1] = packed[1]+1.
        with self.assertRaises(ValueError):
            with linked.bind_embedding_handle(backend, other, 1, handle, {}, np):
                pass
        with self.assertRaises(ValueError):
            with linked.bind_embedding_handle(backend, packed, 1, handle, {}, np):
                backend.input(0)
        backend.release_batch()

    def test_inferred_token_input_index_obeys_real_pack_layout(self):
        # Use real packing helpers without inventing a separate layout algorithm.
        program = SimpleNamespace(header=SimpleNamespace(slots=8),
            params=[SimpleNamespace(name='attention_mask', shape=(1, 4)),
                    SimpleNamespace(name='inputs_embeds', shape=(1, 4, 3))],
            ops=[SimpleNamespace(kind='pack', inputs=('attention_mask',), attrs=dict(layout='flat')),
                 SimpleNamespace(kind='pack', inputs=('inputs_embeds',), attrs=dict(layout='token'))])
        self.assertEqual(linked.embedding_input_index(program, 2), 3)
        with self.assertRaises(ValueError):
            linked.embedding_input_index(program, 4)

    def test_phase_limits_and_rss_timer_restore_without_starting_a_child(self):
        self.assertEqual(linked.TOTAL_SECONDS, 6600)
        self.assertEqual(linked.PHASE_LIMITS['selector_embedding'], dict(rss_gib=2, as_gib=6, seconds=900))
        self.assertEqual(linked.PHASE_LIMITS['tensor'], dict(rss_gib=6, as_gib=12, seconds=300))
        self.assertEqual(linked.PHASE_LIMITS['packed_forward'], dict(rss_gib=5, as_gib=12, seconds=5400))
        original = signal.getsignal(signal.SIGALRM)
        observation = {}
        with mock.patch.object(linked.cached, '_resident_snapshot', side_effect=(1, 6*1024**2)):
            with self.assertRaises(MemoryError):
                with linked.phase_guard('packed_forward', os.getpid(), observation, apply_as=False):
                    signal.raise_signal(signal.SIGALRM)
        self.assertTrue(observation['packed_forward']['stopped'])
        self.assertEqual(signal.getsignal(signal.SIGALRM), original)
        self.assertEqual(signal.getitimer(signal.ITIMER_REAL), (0., 0.))

    def test_source_gate_preserves_all_three_evidence_chains_and_coefficients(self):
        np = self.np
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = SimpleNamespace(**{field: root/field for field in linked.PATH_FIELDS})
            hf_file = root/'hf.npy'
            hf_file.write_bytes(b'metadata only')
            rows = [dict(step=0), dict(step=1)]
            own = dict(own_result=str(args.own_output_result), selections=[dict(sample_id='train:0', step=0)])
            hf = dict(trajectories=[dict(sample_id='train:0', steps=[{}, dict(logits_file='hf.npy')])])
            artifact = dict(config=linked.monotone.config())
            with mock.patch.object(linked.cached, 'source_gate', return_value=(rows, root/'discrete.npy', {})), \
                    mock.patch.object(linked.embedding, 'check_sources', return_value=own), \
                    mock.patch.object(linked.monotone, 'load_artifact', return_value=((), artifact, {})), \
                    mock.patch.object(linked.fixed, 'source', return_value=(dict(status='ok'), root/'program', hf)):
                actual, _, summary = linked.source_gate(args, np)
                self.assertEqual(actual, rows)
                self.assertEqual(summary['original_hf_step1'], str(hf_file))
                own['own_result'] = str(root/'another')
                with self.assertRaises(ValueError):
                    linked.source_gate(args, np)

    def test_command_and_registry_keep_unique_eight_fields_and_null_unmeasured(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = SimpleNamespace(out=root/'new-run', **{field: root/field for field in linked.PATH_FIELDS})
            command = linked.command(args, 123)
            self.assertEqual(command[command.index('--seconds')+1], '6600')
            self.assertEqual(command[command.index('--supervisor-pid')+1], '123')
            self.assertEqual(command[command.index('--prefill-logits')+1], str(args.prefill_logits))
            result = linked.initial_result()
            result.update(status='ok', linked_packed_math_verified=True, training_quality_passed=False, quality_status='quality_rejected')
            observation = dict(exit_code=0, timed_out=False, remaining_pids=[])
            path = root/'registry'
            first = linked.append_registry(args.out, result, observation, path, revision='test')
            self.assertEqual(linked.append_registry(args.out, result, observation, path, revision='test'), first)
            self.assertEqual(set(first), {'run_id', 'round', 'model', 'mode', 'git_rev', 'metrics', 'artifacts_dir', 'ts'})
            self.assertEqual(first['run_id'], 'new-run')
            self.assertEqual(first['metrics']['status'], 'ok')
            self.assertFalse(first['metrics']['training_quality_passed'])
            self.assertEqual(len(path.read_text().splitlines()), 1)
            self.assertIsNone(first['metrics']['CKKS_latency'])
            with self.assertRaises(ValueError):
                linked.append_registry(args.out, result, dict(observation, exit_code=1), path, revision='test')

    def test_observed_categories_are_deltas_and_strict_math_keeps_reference(self):
        before = dict(counts={key: 1 for key in ('add', 'sub', 'neg', 'mul_cc', 'mul_cp', 'rot', 'boot')},
            distinct_consts=3, rot_steps=[1])
        after = dict(counts={key: 3 for key in before['counts']}, distinct_consts=5, rot_steps=[1, 2])
        delta = linked.stats_delta(before, after)
        self.assertEqual(delta['ops_total'], 14)
        self.assertEqual(delta['distinct_constants_added'], 2)
        np = self.np
        self.assertFalse(linked.fixed.strict_math(np.asarray([1.]), np.asarray([1.+1e-5]), np)['passed'])
        self.assertFalse(linked.fixed.strict_math(np.asarray([1.]), np.asarray([np.nan]), np)['passed'])

    def test_coefficient_cache_observation_and_error_partial_do_not_run_new_selection(self):
        from pt_agent.kernels import approx
        before = {name: getattr(approx, name).cache_info() for name in ('_remez', '_relative_remez', '_composite')}
        observed = linked.fit_cache_snapshot()
        for name, value in before.items():
            self.assertEqual(observed[name], value._asdict())
            self.assertEqual(getattr(approx, name).cache_info(), value)
        np = self.np
        from pt_agent.experiments.r10_paper_metrics import StageRecorder
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(work_dir=Path(directory), supervisor_pid=os.getpid())
            result = linked.initial_result()
            with mock.patch.object(linked, 'phase_guard', return_value=nullcontext()), \
                    mock.patch.object(linked, 'source_gate', side_effect=ValueError('source rejected')):
                with self.assertRaises(ValueError):
                    linked.run_linked(args, StageRecorder(Path(directory)/'events'), result, np)
            self.assertEqual(result['packed_selection_attempts'], 0)
            self.assertEqual(result['tensor_forward_attempts'], 0)
            self.assertEqual(result['mock_forward_attempts'], 0)

    def test_terminal_json_counts_itself_and_preserves_existing_worker_result(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'payload').write_text('tiny preserved artifact')
            result = linked.initial_result()
            result['status'] = 'ok'
            size = linked.terminal_result(root, result)
            self.assertEqual(size, sum(path.stat().st_size for path in root.rglob('*') if path.is_file()))
            self.assertEqual(json.loads((root/'result.json').read_text())['case_artifact_bytes'], size)
            (root/'worker-result.json').write_text('{"status":"preserved"}\n')
            with self.assertRaises(FileExistsError):
                linked.worker(SimpleNamespace(work_dir=root))
            self.assertEqual(json.loads((root/'worker-result.json').read_text())['status'], 'preserved')


if __name__ == '__main__':
    unittest.main()
