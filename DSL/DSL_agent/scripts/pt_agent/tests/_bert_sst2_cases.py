"""Synthetic SST-2 and session checks; no original weights or dataset reads."""

import json
import os
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


class SST2Cases(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
            os.environ[name] = '1'
        import numpy as np
        from pt_agent.data import sst2_source
        from pt_agent.experiments import bert_sst2_worker
        bert_sst2_worker.initialize_numeric()
        cls.np = np
        cls.data_python = Path(os.environ['BERT_SST2_DATA_PYTHON'])
        cls.source, cls.worker = sst2_source, bert_sst2_worker

    def setUp(self):
        temporary = TemporaryDirectory(prefix='bert-sst2-', dir=os.environ.get('BERT_SST2_TEST_WORK'))
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.dataset = self.root / 'sst2'
        (self.dataset / 'data').mkdir(parents=True)
        for split, count in (('train', 30), ('validation', 7)):
            self.write_table(self.dataset / 'data' / (split + '-00000-of-00001.parquet'),
                [split + ' real synthetic text ' + str(i) for i in range(count)], [i % 2 for i in range(count)])
        # A malformed hidden test proves that this adapter never opens it.
        (self.dataset / 'data/test-00000-of-00001.parquet').write_text('unread hidden-label fixture')

    def raw(self):
        return self.source.SST2Samples(self.dataset, internal_samples=4, expected_report=7,
                                       data_python=self.data_python)

    def write_table(self, path, sentences, labels):
        script = ('import json, sys; import pyarrow as a; import pyarrow.parquet as p; '
                  'v=json.load(sys.stdin); p.write_table(a.table(v),sys.argv[1])')
        completed = subprocess.run([str(self.data_python), '-B', '-c', script, str(path)],
            input=json.dumps(dict(sentence=sentences, label=labels)), text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=20, env=self.source.reader_environment())
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def freeze(self, raw, text='original frozen source'):
        program = self.root / 'program.ptdsl'
        program.write_text(text)
        marker = self.root / 'FROZEN.json'
        marker.write_text(json.dumps(dict(program=str(program), split_version=raw.partition['version'])))
        return marker, program

    def test_formal_partition_is_seeded_disjoint_1000_each_and_872_report(self):
        first = self.source.make_partition(67349, 872)
        self.assertEqual(first, self.source.make_partition(67349, 872))
        self.assertEqual([len(first[name]) for name in self.source.SPLITS], [1000, 1000, 1000, 872])
        flat = [i for name in self.source.SPLITS for i in first[name]]
        self.assertEqual(len(flat), len(set(flat)))
        self.assertEqual(len(set(first['sample_ids'])), 3872)
        self.assertEqual(first['unused_training_rows'], 64349)
        self.assertTrue(all(first['sample_ids'][i].startswith('train:') for name in self.source.SPLITS[:3] for i in first[name]))

    def test_source_constructor_reads_only_metadata_and_split_returns_copy(self):
        with patch.object(self.source.ExternalReader, 'rows', side_effect=AssertionError('no contents')):
            raw = self.raw()
        changed = raw.split()
        changed['calibration'].clear()
        self.assertEqual(len(raw.partition['calibration']), 4)
        self.assertEqual(raw.accesses, [])
        self.assertFalse(raw.record()['native_test_read'])

    def test_selected_training_texts_and_labels_keep_native_row_order(self):
        raw = self.raw()
        indices = list(reversed(raw.partition['calibration']))
        texts, labels = raw.select(indices)
        rows = [int(raw.ids[index].split(':')[1]) for index in indices]
        self.assertEqual(texts, ['train real synthetic text ' + str(row) for row in rows])
        self.np.testing.assert_array_equal(labels, [row % 2 for row in rows])
        self.assertEqual(labels.dtype, self.np.int64)
        raw.select(indices)
        self.assertEqual(len(raw.accesses), 1)

    def test_validation_content_stays_unread_until_matching_freeze(self):
        raw = self.raw()
        with patch.object(raw.reader, 'rows', side_effect=AssertionError('report body')):
            with self.assertRaisesRegex(ValueError, 'after freezing'):
                raw.select(raw.partition['report'])
        marker, program = self.freeze(raw)
        raw.authorize_report(marker, program)
        texts, labels = raw.select(raw.partition['report'])
        self.assertEqual(len(texts), 7)
        self.assertEqual(raw.accesses[0]['split'], 'validation')
        self.np.testing.assert_array_equal(labels, [0, 1, 0, 1, 0, 1, 0])

    def test_changed_frozen_program_or_partition_rejects_report(self):
        raw = self.raw()
        marker, program = self.freeze(raw)
        other = self.root / 'different.ptdsl'
        other.write_text('different program')
        with self.assertRaises(ValueError):
            raw.authorize_report(marker, other)
        saved = json.loads(marker.read_text())
        saved['split_version'] = 'legacy-readme-split'
        marker.write_text(json.dumps(saved))
        with self.assertRaises(ValueError):
            raw.authorize_report(marker, program)
        self.assertEqual(raw.accesses, [])

    def test_source_record_roundtrip_and_partition_change_rejected(self):
        raw = self.raw()
        restored = self.source.SST2Samples(self.dataset, record=raw.record(), expected_report=7)
        self.assertEqual(restored.record(), raw.record())
        record = raw.record()
        record['partition']['calibration'][0] = record['partition']['heldout'][0]
        with self.assertRaisesRegex(ValueError, 'partition changed'):
            self.source.SST2Samples(self.dataset, record=record, expected_report=7)

    def test_missing_or_extra_train_shards_are_rejected(self):
        extra = self.dataset / 'data/train-00001-of-00002.parquet'
        extra.write_text('metadata fixture')
        with self.assertRaisesRegex(ValueError, 'one original train'):
            self.raw()

    def test_native_rows_require_binary_available_labels(self):
        path = self.root / 'bad.parquet'
        self.write_table(path, ['real text'], [-1])
        with self.assertRaisesRegex(ValueError, 'binary native label'):
            self.source.ExternalReader(self.data_python).rows(path, [0])

    def test_existing_numeric_report_cache_does_not_bypass_frozen_access(self):
        raw = self.raw()
        model_dir = self.root / 'prepared'
        directory = model_dir / 'samples'
        directory.mkdir(parents=True)
        (directory / 'source.json').write_text(json.dumps(raw.record()))
        self.np.savez(directory / 'report.npz', x=self.np.zeros((7, 2)), labels=self.np.zeros(7, dtype=self.np.int64))
        info = SimpleNamespace(source=dict(inputs_path=str(self.dataset), sst2=raw.record()),
                               split=raw.split(), graph=SimpleNamespace(inputs=['x']))
        with patch.object(self.source, 'SST2Samples', return_value=raw):
            provider = self.worker.NumericSplits(model_dir, info)
        with self.assertRaisesRegex(ValueError, 'frozen access'):
            provider.get('report')
        self.assertEqual(raw.accesses, [])

    def test_math_uses_original_float64_tolerance_and_finite_requirement(self):
        a = self.np.asarray([[1.0, 2.0]], dtype=self.np.float64)
        self.assertTrue(self.worker.strict_math(a, a + 1e-13)['passed'])
        self.assertFalse(self.worker.strict_math(a, a + 1e-8)['passed'])
        for b in (a.astype('float32'), a[:, :1], self.np.asarray([[1.0, self.np.inf]])):
            with self.assertRaises(ValueError):
                self.worker.strict_math(a, b)

    def test_host_parquet_environment_drops_nix_library_overrides(self):
        with patch.dict(os.environ, dict(LD_LIBRARY_PATH='/nix/synthetic', NIX_SYNTHETIC='1',
                                        IN_NIX_SHELL='pure', PYTHONHOME='/nix/synthetic')):
            environment = self.source.reader_environment()
        self.assertNotIn('LD_LIBRARY_PATH', environment)
        self.assertNotIn('NIX_SYNTHETIC', environment)
        self.assertNotIn('PYTHONHOME', environment)
        self.assertEqual(environment['MKL_NUM_THREADS'], '1')

    def test_recorded_client_preserves_exact_requests_responses_and_uses_provider_once(self):
        provider = Mock()
        provider.chat.return_value = dict(content='{"program":"source"}', usage={})
        client = self.worker.RecordedRenewingClient(provider, self.root / 'calls')
        messages = [dict(role='user', content='public structure only')]
        reply = client.chat(messages, purpose='agent-iter-1')
        self.assertEqual(client.calls, 1)
        provider.chat.assert_called_once_with(messages, purpose='agent-iter-1')
        self.assertEqual(json.loads((self.root / 'calls/call-0001.request.json').read_text())['messages'], messages)
        self.assertEqual(json.loads((self.root / 'calls/call-0001.response.json').read_text()), reply)

    def test_session_step_uses_renewable_client_and_existing_resumable_algorithm(self):
        from pt_agent.planners import transformer_session
        from pt_agent import autonomous_budget
        args = SimpleNamespace(session_dir=self.root / 'session', ledger=self.root / 'calls.jsonl',
                               allowance=self.root / 'allowance.json')
        ctx = SimpleNamespace()
        result = self.worker._initial_result('step')
        recorder = Mock()
        from contextlib import nullcontext
        recorder.span.side_effect = lambda *a, **k: nullcontext()
        with patch.object(self.worker, 'load_context', return_value=ctx), \
                patch.object(autonomous_budget, 'RenewingLLMClient') as factory, \
                patch.object(transformer_session, 'session_step', return_value=dict(next_step='tune', complete=True)) as advance:
            self.worker.step(args, recorder, result)
        self.assertEqual(factory.call_args.kwargs['budget_path'], args.ledger)
        self.assertEqual(factory.call_args.kwargs['allowance_path'], args.allowance)
        self.assertEqual(factory.call_args.kwargs['artifact_dir'], args.session_dir / 'api')
        self.assertNotIn('max_total_calls', factory.call_args.kwargs)
        self.assertEqual(advance.call_args.kwargs['confirm_rule'], 'c2')
        self.assertEqual(advance.call_args.kwargs['max_iters'], 10)
        self.assertEqual(result['report_samples'], 0)
        self.assertEqual(result['llm_calls'], 0)

    def test_recorded_failure_remains_a_reserved_attempt(self):
        from pt_agent.llm import LLMError
        provider = Mock()
        provider.chat.side_effect = LLMError('transport_timeout')
        client = self.worker.RecordedRenewingClient(provider, self.root / 'calls')
        with self.assertRaises(LLMError):
            client.chat([dict(role='user', content='public')], purpose='agent-iter-1')
        self.assertEqual(client.calls, 1)
        self.assertTrue((self.root / 'calls/call-0001.error.json').is_file())


if __name__ == '__main__':
    unittest.main()
