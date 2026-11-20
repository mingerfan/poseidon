"""New boundary/coordinate controls; no native execution capability implied."""
import copy
import unittest
import numpy as np

from benchmark_graph import samples
from benchmark_math import evaluate
from test_conv_im2col_normalization import model
from conv_im2col_normalization import normalize as previous
from shifted_conv_window_normalization import normalize


class ShiftedWindowTests(unittest.TestCase):
    def compare(self,g):
        before=copy.deepcopy(g);new,trace=normalize(g);self.assertEqual(g,before)
        self.assertEqual(trace['changes'][0]['window_strategy'],'logical_rotation_and_boundary_mask')
        for sample in samples(g,4):
            expected,actual=evaluate(g,sample),evaluate(new,sample)
            for name in expected:np.testing.assert_allclose(actual[name],expected[name],atol=1e-12,rtol=1e-12)
        return new

    def test_dense_rectangular_spatial_shape(self):self.compare(model(kernel=(3,3)))
    def test_unbatched_same_window(self):self.compare(model(shape=(2,5,6),kernel=(3,3)))
    def test_grouped_and_depthwise_windows(self):
        self.compare(model(shape=(1,4,5,6),outc=6,groups=2,kernel=(3,3)))
        self.compare(model(outc=2,groups=2,kernel=(3,3)))
    def test_batch_boundaries_never_mix(self):self.compare(model(shape=(2,2,5,6),kernel=(3,3)))
    def test_dilated_same_window(self):self.compare(model(kernel=(3,3),padding=(2,2),dilation=(2,2)))
    def test_singleton_axis_with_all_zero_windows(self):self.compare(model(shape=(1,1,1,3),outc=1,kernel=(3,3)))
    def test_rotation_sign_with_numbered_coordinates(self):
        g=model(shape=(1,1,3,5),outc=1,kernel=(3,3));new,_=normalize(g)
        sample={'input0':np.arange(15,dtype=np.float64).reshape(1,1,3,5)/32}
        np.testing.assert_allclose(evaluate(new,sample)['out'],evaluate(g,sample)['out'],atol=1e-12,rtol=1e-12)
    def test_other_geometry_keeps_previous_mathematical_path(self):
        g=model(stride=(2,1));old,_=previous(g);new,trace=normalize(g)
        self.assertEqual(new,old);self.assertEqual(trace['changes'][0]['window_strategy'],'explicit_padded_canvas')


if __name__=='__main__':unittest.main()
