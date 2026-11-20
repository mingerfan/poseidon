"""Small independent-oracle checks; no model or real bundle reads."""
import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest


class RuntimeMathChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import numpy as np
        from pt_agent.experiments import r31_lattigo_math_worker as worker
        from pt_agent.experiments.r31_lattigo_runtime_plan import make_regression
        cls.np=np; cls.worker=worker; cls.fixture=staticmethod(make_regression)

    def test_independent_all_nonboot_operations_preserve_original_math(self):
        with TemporaryDirectory() as temp:
            plan,bundle,input_file,expected_file=self.fixture(Path(temp))
            program=json.loads(plan.read_text());manifest=json.loads((bundle/'manifest.json').read_text())
            packed=self.np.fromfile(input_file,dtype='<f8')
            actual,summary=self.worker.replay_plan(program,manifest,bundle,packed)
            expected=self.np.asarray(json.loads(expected_file.read_text())['values'],dtype=self.np.float64)
            self.assertTrue(self.worker.check_math(expected,actual)['passed'])
            self.assertEqual(summary['execution_completed'],11)
            self.assertEqual(summary['plaintext_cache_capacity'],1)
            self.assertEqual(summary['bootstrap_inputs'],[])
            self.assertLess(summary['peak_live_arrays'],len(program['values']))

    def test_boot_reference_preserves_later_original_operand_consumer(self):
        from pt_agent.experiments.r31_lattigo_input_worker import make_bootstrap_plan
        with TemporaryDirectory() as temp:
            work=Path(temp);bundle=work/'bundle';(bundle/'data').mkdir(parents=True)
            p=make_bootstrap_plan(dict(target_id='original'),branch=True)
            manifest=dict(bundle_format_version=1,version=1,bundle_id=p['plaintext_bundle']['id'],
                blobs=[dict(content='fixture:offset',byte_length=16384*8)])
            self.np.full(16384,.125,dtype='<f8').tofile(bundle/'data/offset.bin')
            x=self.np.arange(16384,dtype=self.np.float64)/16384;before=x.copy()
            actual,summary=self.worker.replay_plan(p,manifest,bundle,x)
            self.assertTrue(self.np.array_equal(x,before))
            self.assertTrue(self.worker.check_math(self.np.roll(x,-1)+x+.125,actual)['passed'])
            self.assertEqual(len(summary['bootstrap_inputs']),1)
            self.assertEqual(summary['bootstrap_inputs'][0]['input_level'],1)
            self.assertEqual(summary['execution_completed'],5)

    def test_strict_math_rejects_shape_dtype_nonfinite_and_records_real_difference(self):
        np=self.np
        for value in (np.zeros(2,dtype=np.float32),np.zeros(3),np.array([0.,np.nan])):
            with self.assertRaises(ValueError):self.worker.check_math(np.zeros(2),value)
        self.assertFalse(self.worker.check_math(np.zeros(2),np.array([0.,1e-9]))['passed'])


if __name__=='__main__':unittest.main()
