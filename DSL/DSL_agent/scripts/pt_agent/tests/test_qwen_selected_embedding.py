"""Small mixtures and source/header checks; no original matrix or model."""

from contextlib import nullcontext
import copy
import io
import json
import os
from pathlib import Path
import signal
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
import zipfile

from pt_agent.experiments import qwen_selected_embedding as selected


class SelectedEmbeddingCases(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.np = selected.base._numeric()
        from pt_agent.backends.streaming_mock import StreamingMockBackend
        cls.Backend = StreamingMockBackend

    def matrix(self):
        return self.np.asarray([[1., 4., 0.], [-2., 1., 7.], [8., -3., 2.],
                                [3., 2., -4.], [5., -1., 6.]], dtype=self.np.float64)

    def fixture(self, root):
        np = self.np
        model = root / 'model'
        (model / 'graph').mkdir(parents=True)
        archive = model / 'graph/weights.npz'
        archive.write_bytes(b'metadata placeholder')
        selected.base.save(model / 'model_info.json', dict(source=dict(config=dict(
            num_hidden_layers=24, hidden_size=3, vocab_size=5)), task='causal_lm'))
        own = root / 'own/result.json'
        (own.parent / 'worker').mkdir(parents=True)
        original_rows = []
        for index in range(6):
            filename = 'own-output-' + str(index) + '-logits.npy'
            np.save(own.parent / 'worker' / filename, np.asarray([.9, -.8, .4, -.5, .2]).reshape(1, 1, 5))
            original_rows.append(dict(sample_id='train:' + str(index // 2), step=index % 2,
                logits_file=filename, original_greedy_token=0, approximate_greedy_token=0))
        selected.base.save(own, dict(status='ok', job='quality-own-tensor', worker=dict(status='ok',
            own_output_generation_completed=True, tensor_forward_completed=6, state_update_completed=3,
            candidate_frozen=False, source=dict(model_info=str(model)), sample_rows=original_rows)))
        selector = root / 'selector/result.json'
        selector.parent.mkdir()
        rows = []
        weights = np.asarray([.6, .2, .1, .05, .05], dtype=np.float64)
        for index in range(6):
            filename = 'selection-' + str(index) + '.npy'
            np.save(selector.parent / filename, weights, allow_pickle=False)
            rows.append(dict(sample_id='train:' + str(index // 2), step=index % 2,
                one_hot_file=filename, completed=True, quality=dict(token_agreement=True, actual_selected_token=0),
                contract=dict(comparison_gap_contract_passed=True)))
        record = dict(status='ok', phase='evaluate', variant=selected.monotone.VARIANT,
            config=selected.monotone.config(), selection_attempts=6, selection_completed=6,
            greedy_policy_passed=True, all_comparison_contracts_passed=True, coefficient_fits=0,
            candidate_frozen=False, own_output_result=str(own), source_rows=selected.base.source_rows(own), rows=rows)
        selected.base.save(selector, record)
        out = root / 'out'
        out.mkdir()
        return SimpleNamespace(out=out, selector_result=selector, model_info=model,
                               weights_archive=archive, supervisor_pid=os.getpid()), record

    def test_independent_block_mixture_retains_fractional_and_signed_weights(self):
        np = self.np
        matrix = self.matrix()
        for weights in (np.asarray([.2, .5, .1, .1, .1]), np.asarray([-.2, 1.1, .1, -.1, .1])):
            reference = weights @ matrix
            observed = selected.clear_mixture(weights, matrix, np, block_rows=2)
            np.testing.assert_allclose(observed, reference, atol=1e-12, rtol=1e-12)
            self.assertGreater(float(np.abs(observed - matrix[int(weights.argmax())]).max()), .1)
        invalid = matrix.copy()
        invalid[3, 0] = np.nan
        with self.assertRaises(ValueError):
            selected.clear_mixture(weights, invalid, np)

    def test_explicit_compact_adapter_uses_actual_packed_values(self):
        np = self.np
        matrix = self.matrix()
        matrix.flags.writeable = False
        weights = np.asarray([.2, .5, .1, .1, .1], dtype=np.float64)
        backend = self.Backend(8)
        try:
            packed = selected.packed_selection(backend, weights, np)
            values, layout = selected.compact_embedding(backend, packed, matrix)
            observed = values[0].value[:3]
            reference = selected.clear_mixture(weights, matrix, np, block_rows=2)
            np.testing.assert_allclose(observed, reference, atol=1e-12, rtol=1e-12)
            self.assertEqual(layout.size, 3)
            self.assertEqual(len(packed['one_hot']), 2)
            self.assertFalse(matrix.flags.writeable)
            self.assertEqual(len(backend._constant_objects), 0)
            self.assertTrue(all(value.data is None for value in backend._constants.values()))
        finally:
            backend.release_batch()

    def test_header_only_inspection_does_not_map_or_read_tensor_values(self):
        np = self.np
        buffer = io.BytesIO()
        np.save(buffer, self.matrix(), allow_pickle=False)
        payload = buffer.getvalue()
        entry = SimpleNamespace(filename=selected.MEMBER, compress_type=zipfile.ZIP_STORED,
            flag_bits=0, file_size=len(payload), compress_size=len(payload), header_offset=0)
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / 'archive'
            archive.write_bytes(b'\0' * 30 + payload)
            fake = mock.MagicMock()
            fake.__enter__.return_value.infolist.return_value = [entry]
            with mock.patch.object(selected.zipfile, 'ZipFile', return_value=fake), \
                    mock.patch.object(np, 'memmap', side_effect=AssertionError('mapping forbidden')):
                metadata = selected.inspect_member(archive, selected.MEMBER, np, expected_shape=(5, 3))
            self.assertEqual(metadata['shape'], [5, 3])
            self.assertFalse(metadata['tensor_values_read'])
            self.assertFalse(metadata['tensor_mapping_created'])
            self.assertEqual(metadata['integrity_checks'], 0)

    def test_source_gate_requires_actual_selection_and_original_model_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            args, record = self.fixture(Path(directory))
            with mock.patch.object(selected, 'SHAPE', (5, 3)), \
                    mock.patch.object(selected, 'inspect_member', return_value=dict(shape=[5, 3])):
                source = selected.check_sources(args, self.np)
                self.assertEqual(len(source['selections']), 6)
                changed = copy.deepcopy(record)
                changed['rows'][0]['one_hot_file'] = '../synthetic.npy'
                selected.base.save(args.selector_result, changed)
                with self.assertRaises(ValueError):
                    selected.check_sources(args, self.np)
                selected.base.save(args.selector_result, record)
                with self.assertRaises(ValueError):
                    selected.check_sources(SimpleNamespace(**dict(vars(args), weights_archive=Path(directory) / 'other.npz')), self.np)

    def test_saved_selection_is_read_as_a_vector_without_argmax_substitution(self):
        np = self.np
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'weights.npy'
            weights = np.asarray([.2, .5, .1, .1, .1], dtype=np.float64)
            np.save(path, weights, allow_pickle=False)
            row = dict(path=str(path), recorded_quality=dict(actual_selected_token=1))
            observed = selected.read_selection(row, np, expected_size=5)
            np.testing.assert_array_equal(observed, weights)
            self.assertFalse(observed.flags.writeable)
            with self.assertRaises(ValueError):
                selected.read_selection(dict(row, recorded_quality=dict(actual_selected_token=0)), np, expected_size=5)

    def test_six_small_real_kernel_loops_and_first_strict_failure_stop(self):
        np = self.np
        with tempfile.TemporaryDirectory() as directory:
            args, _ = self.fixture(Path(directory))
            with mock.patch.object(selected, 'SHAPE', (5, 3)), \
                    mock.patch.object(selected, 'inspect_member', return_value=dict(shape=[5, 3])):
                source = selected.check_sources(args, np)
            matrix_path = Path(directory) / 'matrix.bin'
            self.matrix().tofile(matrix_path)
            def mapped(*_):
                return np.memmap(matrix_path, dtype=np.float64, mode='r', shape=(5, 3)), dict(shape=[5, 3])
            with mock.patch.object(selected, 'SHAPE', (5, 3)), mock.patch.object(selected, 'SLOTS', 8), \
                    mock.patch.object(selected, 'check_sources', return_value=source), \
                    mock.patch.object(selected, 'map_stored_member', side_effect=mapped), \
                    mock.patch.object(selected, 'resident_guard', return_value=nullcontext()):
                result = selected.worker(args)
            self.assertEqual(result['status'], 'ok', result.get('error'))
            self.assertEqual((result['clear_mixture_completed'], result['packed_embedding_completed']), (6, 6))
            self.assertEqual(result['original_weight_members_mapped'], 1)
            self.assertTrue(all(row['strict_math']['passed'] for row in result['rows']))
            self.assertTrue(all(row['hard_greedy_embedding_max_difference'] > .1 for row in result['rows']))
            from pt_agent.experiments import qwen_precision_quality
            other = Path(directory) / 'failed'
            other.mkdir()
            failed_args = SimpleNamespace(**dict(vars(args), out=other))
            with mock.patch.object(selected, 'SHAPE', (5, 3)), mock.patch.object(selected, 'SLOTS', 8), \
                    mock.patch.object(selected, 'check_sources', return_value=source), \
                    mock.patch.object(selected, 'map_stored_member', side_effect=mapped), \
                    mock.patch.object(selected, 'resident_guard', return_value=nullcontext()), \
                    mock.patch.object(qwen_precision_quality, 'strict_math', return_value=dict(passed=False)):
                failed = selected.worker(failed_args)
            self.assertEqual(failed['status'], 'error')
            self.assertEqual(failed['packed_embedding_attempts'], 1)
            self.assertEqual(failed['packed_embedding_completed'], 1)
            self.assertEqual(len(failed['rows']), 1)
            self.assertFalse(failed['rows'][0]['completed'])

    def test_two_gib_rss_guard_restores_signal_and_stops(self):
        from pt_agent.experiments import qwen_own_output_packed
        original = signal.getsignal(signal.SIGALRM)
        record = {}
        with mock.patch.object(qwen_own_output_packed, '_resident_snapshot', side_effect=(1, 3 * 1024 ** 2)):
            with self.assertRaises(MemoryError):
                with selected.resident_guard(os.getpid(), record):
                    signal.raise_signal(signal.SIGALRM)
        self.assertTrue(record['stopped'])
        self.assertEqual(signal.getsignal(signal.SIGALRM), original)
        self.assertEqual(signal.getitimer(signal.ITIMER_REAL), (0., 0.))

    def test_backup_registry_uses_each_independent_run_directory_once(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry = root/'registry.jsonl'
            for name in ('one', 'two'):
                out = root/name
                out.mkdir()
                selected.base.save(out/'result.json', dict(status='ok', clear_mixture_completed=6,
                    packed_embedding_completed=6))
                selected.base.save(out/'supervision.json', dict(observation=dict(exit_code=0, timed_out=False, remaining_pids=[])))
                kwargs = dict(variant='actual-compact-embedding', revision='test', run_id=out.name)
                first = selected.base.record_run(out, 'selected-embedding', registry, **kwargs)
                self.assertEqual(selected.base.record_run(out, 'selected-embedding', registry, **kwargs), first)
            rows = [json.loads(line) for line in registry.read_text().splitlines()]
            self.assertEqual([row['run_id'] for row in rows], ['one', 'two'])
            self.assertTrue(all(len(row)==8 for row in rows))


if __name__ == '__main__':
    unittest.main()
