"""Deterministic channel and vector partitions, including boundary slots."""

import unittest
import numpy as np

from pt_agent.kernels.layout import (FlatLayout, MuxLayout, MultiFlatLayout, MultiMuxLayout,
    KernelError, multi_mux_layout, multi_flat_layout, pack_multi, unpack_multi,
    layout_from_record, layout_record)


class LayoutTests(unittest.TestCase):
    def test_mux_boundaries(self):
        self.assertIsInstance(multi_mux_layout(64, 4, 4, 4, 1), MuxLayout)
        for c, intervals in ((5, [(0, 4), (4, 5)]), (13, [(0, 4), (4, 8), (8, 12), (12, 13)])):
            lay = multi_mux_layout(64, c, 4, 4, 1)
            self.assertIsInstance(lay, MultiMuxLayout)
            self.assertEqual([(a, z) for a, z, _ in lay.blocks], intervals)
            self.assertEqual(lay, layout_from_record(layout_record(lay)))
            self.assertEqual(len(lay.index_grid()), len(intervals))
            x = np.arange(c*16, dtype=float).reshape(c, 4, 4)
            packed = pack_multi(x, lay)
            np.testing.assert_array_equal(unpack_multi(packed, lay), x)
            for (a, z, block), v in zip(lay.blocks, packed):
                self.assertLessEqual(block.used, block.n)
                np.testing.assert_array_equal(v[block.index_grid()], np.broadcast_to(x[a:z], (block.p, z-a, 4, 4)))

    def test_interleaved_groups(self):
        lay = multi_mux_layout(128, 17, 4, 4, 2)
        self.assertEqual([b.C for _, _, b in lay.blocks], [8, 8, 1])
        self.assertEqual([b.p for _, _, b in lay.blocks], [1, 1, 2])
        x = np.arange(17*16, dtype=float).reshape(17, 4, 4)
        np.testing.assert_array_equal(unpack_multi(pack_multi(x, lay), lay), x)

    def test_flat_boundaries_and_batch(self):
        self.assertIsInstance(multi_flat_layout(32, 16), FlatLayout)
        for size in (17, 33, 49):
            lay = multi_flat_layout(32, size)
            self.assertIsInstance(lay, MultiFlatLayout)
            self.assertEqual(lay, layout_from_record(layout_record(lay)))
            x = np.arange(size, dtype=float)
            packed = pack_multi(x, lay)
            np.testing.assert_array_equal(unpack_multi(packed, lay), x)
            batch = [np.stack([v, 2*v]) for v in packed]
            np.testing.assert_array_equal(unpack_multi(batch, lay), np.stack([x, 2*x]))
            for (_, _, block), v in zip(lay.blocks, packed):
                self.assertTrue(np.all(v[block.size:] == 0))

    def test_spatial_error(self):
        for args in ((32, 3, 8, 8, 1), (32, 1, 4, 4, 2)):
            with self.assertRaisesRegex(KernelError, 'spatial tiling not supported'):
                multi_mux_layout(*args)


if __name__ == '__main__':
    unittest.main()
