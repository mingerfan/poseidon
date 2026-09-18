"""Actual trusted helper + explicit packing transforms, plaintext only."""
import unittest

from probe_upstream_concat import upstream, run_case, shapes, CASES, PackingTransforms


class UpstreamConcatTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.api = upstream()

    def test_ten_boundary_regressions(self):
        for case in CASES:
            with self.subTest(case=case['name']):
                result = run_case(self.api, **case)
                self.assertTrue(result['passed'], result)

    def test_pinned_upstream_reproduces_four_failures(self):
        import subprocess
        from probe_upstream_concat import ROOT
        original = subprocess.run(['git','-C',str(ROOT/'third_party/dacapo'),
            'show','4616402710f39df3e5f5bd7930a6c036025aaac3:python/poly/poly/MPCB.py'],
            capture_output=True,text=True,check=True,timeout=15).stdout
        baseline = upstream(original)
        failed = {case['name'] for case in CASES if not run_case(baseline,**case)['passed']}
        self.assertEqual(failed, {'crosses_output_boundary','multi_partial_half',
                                 'multi_partial_small_tail','interleaved_multi'})
        first=shapes(baseline,16,2,1,2,1,2)
        second=shapes(baseline,16,2,1,3,1,2)
        self.assertIsInstance(baseline['CascadeConcat'](first,second),dict)

    def test_slot_channel_spatial_interleave_matrix(self):
        # nt is a plaintext vector length, not a CKKS security parameter.
        count = 0
        for nt in (8, 16, 32):
            for k in (1, 2):
                for channel_tiles in range(1, 6):
                    for h in range(1, 5):
                        for w in range(1, 5):
                            case = dict(name='matrix',nt=nt,c=channel_tiles*k*k,h=h,w=w,k=k)
                            with self.subTest(**case):
                                self.assertTrue(run_case(self.api,**case)['passed'],case)
                            count += 1
        self.assertEqual(count,480)

    def test_reject_mismatched_branch_geometry(self):
        first = shapes(self.api,16,4,2,2,1,2)
        for field in ('nt','bb','ko','ho','wo','no','po'):
            changed = dict(first)
            changed[field] += 1
            with self.subTest(field=field), self.assertRaisesRegex(ValueError,'matching branch packing'):
                self.api['CascadeConcat'](first,changed)

    def test_transform_adapter_does_not_claim_general_einops(self):
        import torch
        value=torch.arange(8,dtype=torch.float64)
        with self.assertRaisesRegex(ValueError,'Unknown diagnostic'):
            PackingTransforms.repeat(value,'a -> a')
        with self.assertRaisesRegex(ValueError,'Unknown diagnostic'):
            PackingTransforms.rearrange(value,'a -> a')

    def test_rotation_direction(self):
        from probe_upstream_concat import PlainSlots
        import numpy as np
        vector=PlainSlots([1,2,3,4])
        np.testing.assert_array_equal(vector.rotate(1).values,[2,3,4,1])
        np.testing.assert_array_equal(self.api['roll'](vector,1).values,[4,1,2,3])


if __name__=='__main__':
    unittest.main()
