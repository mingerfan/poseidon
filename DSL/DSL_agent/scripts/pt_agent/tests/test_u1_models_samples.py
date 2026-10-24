"""Model information and original-sample rules from the rewritten U1 spec."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from pt_agent.data.samples import choose_sequence, encode_texts, read, split_indices
from pt_agent.models.info import ModelInfo, infer_task, metric_for


class ModelInformationTests(unittest.TestCase):
    def test_head_inference(self):
        self.assertEqual(infer_task((1, 2))[0], 'classification')
        self.assertEqual(infer_task((1, 8, 32), vocab_size=32)[0], 'causal_lm')
        self.assertEqual(infer_task((1, 8, 32))[0], 'regression')
        self.assertEqual(infer_task((1, 1))[0], 'regression')
        self.assertEqual(infer_task((1, 1), architectures=['BertForSequenceClassification'])[0], 'classification')

    def test_record_roundtrip(self):
        info = ModelInfo(dict(type='torch', path='module:build'), None, dict(mode='original tensors'),
                         [dict(name='x', shape=[1, 4], kind='features', dtype='float64')],
                         None, 'classification', 'classification_top1', {'acc_drop_pp': 0.5},
                         split_indices(223), ['inferred from output [1,2]'])
        with tempfile.TemporaryDirectory() as directory:
            info.save(directory)
            self.assertEqual(ModelInfo.load(directory), info)
            self.assertTrue((Path(directory) / 'model_info.json').is_file())

    def test_metric_selection(self):
        self.assertEqual(metric_for('classification', has_labels=True)[1], {'acc_drop_pp': 0.5})
        self.assertEqual(metric_for('classification', has_labels=False)[1], {'agreement_pct': 99.5})
        self.assertEqual(metric_for('causal_lm', has_labels=False)[1],
                         {'agreement_pct': 99.0, 'logit_rel_err': 1e-2})


class SampleTests(unittest.TestCase):
    def test_split_boundary_and_reproducibility(self):
        with self.assertRaisesRegex(ValueError, '至少需要 223 个输入样本，当前 222 个'):
            split_indices(222)
        result = split_indices(223)
        self.assertEqual(result, split_indices(223))
        groups = [result[name] for name in ('calibration', 'heldout', 'confirm', 'report')]
        self.assertEqual([len(g) for g in groups], [23, 20, 20, 160])
        flattened = [i for group in groups for i in group]
        self.assertEqual(sorted(flattened), list(range(223)))

    def test_npz_inputs_are_preserved(self):
        x = np.arange(223*4, dtype=np.float64).reshape(223, 4)
        mask = np.ones((223, 2), dtype=np.int64)
        labels = np.arange(223) % 2
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'samples.npz'
            np.savez(path, x=x, attention_mask=mask, labels=labels)
            samples = read(path)
            values, selected_labels = samples.tensors([4, 1], names=['x', 'attention_mask'])
            self.assertTrue(np.array_equal(values['x'], x[[4, 1]]))
            self.assertTrue(np.array_equal(values['attention_mask'], mask[[4, 1]]))
            self.assertTrue(np.array_equal(selected_labels, labels[[4, 1]]))

    def test_npy_single_input_alias(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'samples.npy'
            np.save(path, np.arange(12).reshape(3, 4))
            tensors, labels = read(path).tensors([2], names=['user_input'])
            self.assertEqual(tensors['user_input'].shape, (1, 4))
            self.assertIsNone(labels)

    def test_mixed_tensor_shapes_are_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            np.save(Path(directory) / 'a.npy', np.zeros((1, 4)))
            np.save(Path(directory) / 'b.npy', np.zeros((1, 5)))
            with self.assertRaisesRegex(ValueError, 'shapes differ'):
                read(directory)

    def test_jsonl_and_imagefolder_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'data.jsonl'
            path.write_text('\n'.join(json.dumps(dict(text=t, label=i)) for i, t in enumerate(('first', 'second'))))
            samples = read(path)
            self.assertEqual(samples.items, ['first', 'second'])
            self.assertEqual(samples.labels.tolist(), [0, 1])
        with tempfile.TemporaryDirectory() as directory:
            for name in ('b', 'a'):
                folder = Path(directory) / name
                folder.mkdir(); (folder / 'input.png').touch()
            samples = read(directory)
            self.assertEqual(samples.ids, ['a/input.png', 'b/input.png'])
            self.assertEqual(samples.labels.tolist(), [0, 1])

    def test_no_label_jsonl(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'data.jsonl'
            path.write_text('"original text"\n')
            self.assertEqual(read(path).items, ['original text'])
            self.assertIsNone(read(path).labels)


class SequenceTests(unittest.TestCase):
    def test_single_length_rounding(self):
        sequence = choose_sequence([9] * 100, 512)
        self.assertEqual(sequence['length'], 16)
        self.assertEqual(sequence['buckets'], [16])
        self.assertFalse(sequence['bucketed'])
        self.assertEqual(sequence['source_splits'], ['calibration', 'heldout'])

    def test_bucket_trigger_and_exact_five_percent_boundary(self):
        exact_five = choose_sequence([8] * 95 + [256] * 5, 512)
        self.assertFalse(exact_five['bucketed'])
        over_five = choose_sequence([8] * 94 + [256] * 6, 64)
        self.assertTrue(over_five['bucketed'])
        self.assertLessEqual(len(over_five['buckets']), 3)
        self.assertTrue(all(b % 8 == 0 and b <= 64 for b in over_five['buckets']))

    def test_padding_and_task_specific_truncation(self):
        class Tokenizer:
            pad_token_id = None
            eos_token_id = 2
            eos_token = '<eos>'

            def __call__(self, texts, **kwargs):
                self.last = kwargs
                return kwargs

        tokenizer = Tokenizer()
        encode_texts(tokenizer, ['text'], {'buckets': [8, 16]}, task='causal_lm')
        self.assertEqual(tokenizer.padding_side, 'right')
        self.assertEqual(tokenizer.truncation_side, 'left')
        self.assertEqual(tokenizer.pad_token, '<eos>')
        self.assertEqual(tokenizer.last['max_length'], 16)
        self.assertTrue(tokenizer.last['return_attention_mask'])
        encode_texts(tokenizer, ['text'], {'buckets': [8]}, task='classification')
        self.assertEqual(tokenizer.truncation_side, 'right')


if __name__ == '__main__':
    unittest.main()
