import unittest

import numpy as np

from pt_agent.kernels import KernelError, mux_layout, pack_mux, unpack_mux


class LayoutTests(unittest.TestCase):
    def test_properties(self):
        first = mux_layout(256, 4, 4, 4, 1)
        self.assertEqual((first.T, first.used, first.p, first.block), (4, 64, 4, 64))
        second = mux_layout(256, 4, 4, 4, 2)
        self.assertEqual((second.T, second.used, second.p), (1, 64, 4))
        image = mux_layout(16384, 3, 32, 32, 1)
        self.assertEqual((image.T, image.used, image.p, image.block), (3, 3072, 4, 4096))

    def test_pack_copies_and_unique_slots(self):
        rng = np.random.default_rng(20261027)
        for args in ((256, 4, 4, 4, 1), (256, 6, 2, 2, 2), (16384, 32, 16, 16, 2), (16384, 64, 8, 8, 4)):
            with self.subTest(layout=args):
                lay = mux_layout(*args)
                x = rng.uniform(-1, 1, (lay.C, lay.H, lay.W))
                vector = pack_mux(x, lay)
                for copy in range(lay.p):
                    np.testing.assert_array_equal(unpack_mux(vector, lay, copy), x)
                indices = [lay.slot(c, y, x, copy) for copy in range(lay.p) for c in range(lay.C)
                           for y in range(lay.H) for x in range(lay.W)]
                self.assertEqual(len(indices), len(set(indices)))
                self.assertGreaterEqual(min(indices), 0)
                self.assertLess(max(indices), lay.n)
                unused = np.ones(lay.n, dtype=bool)
                unused[indices] = False
                self.assertTrue(np.all(vector[unused] == 0))

    def test_invalid_layouts(self):
        for args in ((256, 32, 4, 4, 1), (256, 4, 4, 4, 3), (255, 4, 4, 4, 1), (256, 0, 4, 4, 1)):
            with self.subTest(layout=args), self.assertRaises(KernelError):
                mux_layout(*args)
