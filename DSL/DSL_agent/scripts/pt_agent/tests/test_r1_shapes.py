import unittest

from pt_agent.dsl.ir import WeightRef
from pt_agent.dsl.shapes import ShapeError, WeightError, infer


class ShapeTests(unittest.TestCase):
    def infer(self, kind, shapes, attrs=None, weights=None):
        weights = weights or {}

        def wshape(name):
            if name not in weights:
                raise WeightError("E_WEIGHT_MISSING", name)
            return weights[name]

        return infer(kind, shapes, attrs or {}, wshape)

    def test_conv2d_stride(self):
        out = self.infer("conv2d", [(1, 16, 32, 32)],
                         dict(w=WeightRef("w"), stride=(2, 2), pad=(1, 1)), {"w": (32, 16, 3, 3)})
        self.assertEqual(out, [(1, 32, 16, 16)])

    def test_depthwise_conv2d(self):
        out = self.infer("conv2d", [(1, 16, 8, 8)],
                         dict(w=WeightRef("w"), groups=16, pad=(1, 1)), {"w": (16, 1, 3, 3)})
        self.assertEqual(out, [(1, 16, 8, 8)])

    def test_conv2d_weight_shape(self):
        with self.assertRaises(WeightError) as caught:
            self.infer("conv2d", [(1, 16, 8, 8)], dict(w=WeightRef("w")), {"w": (32, 8, 3, 3)})
        self.assertEqual(caught.exception.code, "E_WEIGHT_SHAPE")

    def test_avgpool(self):
        self.assertEqual(self.infer("avgpool", [(1, 16, 32, 32)], dict(k=(2, 2), stride=None)),
                         [(1, 16, 16, 16)])

    def test_slice(self):
        self.assertEqual(self.infer("slice", [(1, 16, 32, 32)], dict(axis=2, step=2, end=None)),
                         [(1, 16, 16, 32)])

    def test_pad(self):
        self.assertEqual(self.infer("pad", [(1, 16, 16, 16)], dict(pads=(0, 0, 0, 0, 8, 8))),
                         [(1, 32, 16, 16)])

    def test_mean(self):
        self.assertEqual(self.infer("mean", [(1, 64, 8, 8)], dict(axis=(2, 3))), [(1, 64)])
        self.assertEqual(self.infer("mean", [(1, 64, 8, 8)], dict(axis=(2, 3), keepdim=True)),
                         [(1, 64, 1, 1)])

    def test_scalar_reduction(self):
        with self.assertRaises(ShapeError):
            self.infer("mean", [(64,)], dict(axis=(0,)))

    def test_linear(self):
        self.assertEqual(self.infer("linear", [(1, 64)], dict(w=WeightRef("w")), {"w": (10, 64)}), [(1, 10)])
        with self.assertRaises(WeightError) as caught:
            self.infer("linear", [(1, 64)], dict(w=WeightRef("w")), {"w": (10, 63)})
        self.assertEqual(caught.exception.code, "E_WEIGHT_SHAPE")

    def test_reshape(self):
        self.assertEqual(self.infer("reshape", [(1, 4, 4)], dict(shape=(1, -1))), [(1, 16)])
        with self.assertRaises(ShapeError):
            self.infer("reshape", [(1, 4, 4)], dict(shape=(1, 5)))

    def test_permute(self):
        self.assertEqual(self.infer("permute", [(1, 3, 4)], dict(dims=(0, 2, 1))), [(1, 4, 3)])
        with self.assertRaises(ShapeError):
            self.infer("permute", [(1, 3, 4)], dict(dims=(0, 0, 1)))

    def test_concat(self):
        self.assertEqual(self.infer("concat", [(1, 3), (1, 5)], dict(axis=1)), [(1, 8)])
        with self.assertRaises(ShapeError):
            self.infer("concat", [(1, 3), (2, 5)], dict(axis=1))

    def test_split(self):
        self.assertEqual(self.infer("split", [(1, 4)], dict(axis=1, sizes=(2, 2))), [(1, 2), (1, 2)])
        with self.assertRaises(ShapeError):
            self.infer("split", [(1, 4)], dict(axis=1, sizes=(2, 3)))

    def test_matmul(self):
        self.assertEqual(self.infer("matmul", [(2, 3, 4), (2, 4, 5)]), [(2, 3, 5)])
