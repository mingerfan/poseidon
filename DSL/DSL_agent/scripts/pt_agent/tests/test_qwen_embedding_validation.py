"""Tiny NPZ fixtures for single original-member mapping; no model reads."""

import os
from pathlib import Path
import tempfile
import unittest
import zipfile


class OriginalEmbeddingMappingChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
            os.environ[name] = '1'
        import numpy as np
        from pt_agent.experiments.qwen_embedding_validation import map_stored_member
        cls.np, cls.map = np, staticmethod(map_stored_member)

    def test_one_member_maps_readonly_with_original_values_and_no_extraction(self):
        for order in ('C','F'):
            with self.subTest(order=order), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                expected = self.np.array(self.np.arange(35,dtype=self.np.float64).reshape(7,5),order=order)
                path=directory/'original.npz'
                self.np.savez(path, unrelated=self.np.ones((2,2)), original=expected)
                mapped, metadata = self.map(path,'original.npy')
                self.assertIsInstance(mapped,self.np.memmap)
                self.assertFalse(mapped.flags.writeable)
                self.np.testing.assert_array_equal(mapped,expected)
                self.assertEqual(metadata['tensor_bytes'],expected.nbytes)
                self.assertEqual(metadata['other_weight_members_loaded'],0)
                self.assertEqual(list(directory.iterdir()),[path])

    def test_compressed_wrong_dtype_missing_and_duplicate_members_reject(self):
        import io
        import warnings
        with tempfile.TemporaryDirectory() as temporary:
            directory=Path(temporary)
            compressed=directory/'compressed.npz'
            self.np.savez_compressed(compressed,original=self.np.ones((3,2)))
            with self.assertRaisesRegex(ValueError,'stored NPY'):
                self.map(compressed,'original.npy')
            wrong=directory/'wrong.npz';self.np.savez(wrong,original=self.np.ones((3,2),dtype=self.np.float32))
            with self.assertRaisesRegex(ValueError,'float64'):
                self.map(wrong,'original.npy')
            with self.assertRaisesRegex(ValueError,'exactly one'):
                self.map(wrong,'absent.npy')
            payload=io.BytesIO();self.np.save(payload,self.np.ones((3,2)))
            duplicate=directory/'duplicate.npz'
            with warnings.catch_warnings():
                warnings.simplefilter('ignore',UserWarning)
                with zipfile.ZipFile(duplicate,'w') as archive:
                    archive.writestr('original.npy',payload.getvalue());archive.writestr('original.npy',payload.getvalue())
            with self.assertRaisesRegex(ValueError,'exactly one'):
                self.map(duplicate,'original.npy')

    def test_token_probe_uses_original_saved_training_text_without_truncation(self):
        import json
        import sys
        from types import ModuleType,SimpleNamespace
        from unittest.mock import patch
        from pt_agent.experiments import qwen_embedding_validation as worker
        from pt_agent.experiments.r10_paper_metrics import StageRecorder
        calls=[]
        def tokenizer(text,**kwargs):
            calls.append((text,kwargs));return {'input_ids':[7,11,9]}
        module=ModuleType('transformers')
        module.AutoTokenizer=SimpleNamespace(from_pretrained=lambda path,**kwargs: tokenizer)
        with tempfile.TemporaryDirectory() as temporary:
            directory=Path(temporary)
            samples=directory/'samples.jsonl';samples.write_text(json.dumps({'text':'original training text'})+'\n')
            source=directory/'source_rows.json';source.write_text(json.dumps(dict(source='/synthetic/train-000.parquet',
                column='sentence',row_number_base=0,source_rows=[17])))
            args=SimpleNamespace(samples=samples,source_rows=source,tokenizer=directory/'original-tokenizer')
            result={}
            with patch.dict(sys.modules,{'transformers':module}):
                worker.tokenize_probe(args,StageRecorder(directory/'events.jsonl'),result)
            self.assertEqual(calls,[('original training text',dict(padding=False,truncation=False))])
            self.assertEqual(result['token_probe']['token_id'],7)
            self.assertEqual(result['token_probe']['original_training_row'],17)
            self.assertFalse(result['token_probe']['calibration'])


if __name__=='__main__':
    unittest.main()
