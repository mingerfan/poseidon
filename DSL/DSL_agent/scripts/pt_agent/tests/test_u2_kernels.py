"""Small forced multi-block arithmetic and sharing schedules."""

from dataclasses import replace
import unittest
import numpy as np

from pt_agent.backends import CountBackend, MockBackend
from pt_agent.dsl.parser import parse
from pt_agent.kernels.layout import (KernelError, MuxLayout, MultiMuxLayout, MultiFlatLayout,
    layout_blocks, multi_mux_layout, multi_flat_layout, pack_mux, pack_flat, pack_multi, unpack_multi)
from pt_agent.kernels.multi import (MULTI, values_list, conv2d_blocks, linear_blocks,
    mean_blocks, add_blocks, downsample_blocks, relayout_blocks)
from pt_agent.kernels.conv import conv2d_mux
from pt_agent.lower import lower, pack_inputs, decode_outputs
from pt_agent.tensor_sim import run_tensor
from pt_agent.tests.test_r4_conv import conv_ref


class KernelTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(20261024)

    def bind(self, lay, x):
        vectors = pack_multi(x, lay) if isinstance(lay, MULTI) else [
            pack_mux(x, lay) if isinstance(lay, MuxLayout) else pack_flat(x, lay)]
        b = MockBackend(lay.n)
        b.bind(vectors)
        values = [b.input(i) for i in range(len(vectors))]
        return b, values if isinstance(lay, MULTI) else values[0]

    def decoded(self, value, lay):
        return decode_outputs([v.value for v in values_list(value, lay)], [lay])[0]

    def close(self, actual, expected):
        self.assertEqual(actual.shape, expected.shape)
        self.assertTrue(np.all(np.abs(actual-expected) <= 1e-9*(1+np.abs(expected))),
                        f'max diff={np.max(np.abs(actual-expected))}')

    def test_conv_four_routes_and_stride(self):
        for ci, co, stride in ((2, 5, 1), (5, 2, 1), (5, 7, 1), (7, 9, 2)):
            with self.subTest(ci=ci, co=co, stride=stride):
                lay = multi_mux_layout(32, ci, 4, 4, 1)
                x = self.rng.normal(size=(ci, 4, 4))
                weight = self.rng.normal(size=(co, ci, 3, 3))
                scale, shift = self.rng.normal(size=(2, co))
                b, value = self.bind(lay, x)
                actual, out = conv2d_blocks(b, value, lay, weight, stride=stride, scale=scale, shift=shift)
                expected = conv_ref(x, weight, stride, 1)*scale[:, None, None]+shift[:, None, None]
                self.close(self.decoded(actual, out), expected)
                for item, (start, end, block) in zip(values_list(actual, out), layout_blocks(out)):
                    for copy in range(block.p):
                        np.testing.assert_allclose(item.value[block.index_grid()[copy]],
                            expected[start:end], rtol=1e-9, atol=1e-9)

    def test_downsample_and_residual(self):
        lay = multi_mux_layout(32, 5, 4, 4, 1)
        x = self.rng.normal(size=(5, 4, 4))
        b, value = self.bind(lay, x)
        actual, out = downsample_blocks(b, value, lay, 2, 2)
        self.close(self.decoded(actual, out), np.pad(x[:, ::2, ::2], ((2, 2), (0, 0), (0, 0))))
        added, _ = add_blocks(b, actual, out, actual, out)
        self.close(self.decoded(added, out), 2*self.decoded(actual, out))
        with self.assertRaisesRegex(KernelError, 'residual layouts must match'):
            add_blocks(b, actual, out, value, lay)

    def test_mean_merge_and_keep_multi(self):
        for c, h in ((5, 4), (19, 2), (35, 1)):
            with self.subTest(c=c):
                lay = multi_mux_layout(32, c, h, h, 1)
                x = self.rng.normal(size=(c, h, h))
                b, value = self.bind(lay, x)
                actual, out = mean_blocks(b, value, lay)
                self.assertEqual(isinstance(out, MultiFlatLayout), c > 16)
                self.close(self.decoded(actual, out), x.mean(axis=(1, 2)))
                if h == 1:
                    relaid, relayout = relayout_blocks(b, value, lay)
                    self.close(self.decoded(relaid, relayout), x[:, 0, 0])
                else:
                    with self.assertRaisesRegex(KernelError, 'H = W = 1'):
                        relayout_blocks(b, value, lay)

    def test_linear_all_routes_and_larger_output(self):
        for algo in ('diag', 'bsgs'):
            for i, o in ((13, 7), (5, 35), (35, 5), (35, 51), (33, 19)):
                with self.subTest(algo=algo, i=i, o=o):
                    lay = multi_flat_layout(32, i)
                    x = self.rng.normal(size=i)
                    weight = self.rng.normal(size=(o, i))
                    bias = self.rng.normal(size=o)
                    b, value = self.bind(lay, x)
                    actual, out = linear_blocks(b, value, lay, weight, bias, algo=algo)
                    self.close(self.decoded(actual, out), weight@x+bias)
                    for item, (_, _, block) in zip(values_list(actual, out), layout_blocks(out)):
                        self.assertTrue(np.all(item.value[block.size:] == 0))
                    text = (f'@fhe.program(slots=1024)\ndef f(x: Cipher[1,{i}]):\n'
                            ' a = pack(x, layout="flat")\n'
                            f' z = linear(a, w=W["w"], b=W["bias"], algo="{algo}")\n return z\n')
                    p = parse(text); p = replace(p, header=replace(p.header, slots=32))
                    batch = x[None]
                    bk = MockBackend(32, batch=1); bk.bind(pack_inputs(p, [batch]))
                    vs, ls = lower(p, dict(w=weight, bias=bias), bk)
                    self.close(decode_outputs([v.value for v in vs], ls)[0],
                               run_tensor(p, dict(w=weight, bias=bias), batch, cache=False))

    def test_conv_shared_translations(self):
        lay = multi_mux_layout(32, 5, 4, 4, 1)
        weight = np.ones((7, 5, 3, 3))
        out = multi_mux_layout(32, 7, 4, 4, 1)
        b = CountBackend(32)
        values = [b.input(i) for i in range(len(lay.blocks))]
        result, _ = conv2d_blocks(b, values, lay, weight, stride=1)
        b.output(result)
        # Inputs: three blocks, each 8 nonzero shifts. Sum: one per input pair
        # for T=2 and zero for T=1. Projection: one per nonzero output channel.
        shifts = 3*8
        sums = sum((li.T.bit_length()-1)*((lo.C+li.p-1)//li.p)
                   for _, _, li in lay.blocks for _, _, lo in out.blocks)
        projections = sum((o*16-(o % li.p)*li.block) % lay.n != 0
                          for _, _, lo in out.blocks for _, _, li in lay.blocks for o in range(lo.C))
        fills = sum(lo.p.bit_length()-1 for _, _, lo in out.blocks)
        expected = shifts+sums+projections+fills
        self.assertEqual(b.stats()['counts']['rot'], expected)
        naive = CountBackend(32)
        xs = [naive.input(i) for i in range(len(lay.blocks))]
        for a, z, _ in out.blocks:
            for x, (c, d, li) in zip(xs, lay.blocks):
                conv2d_mux(naive, x, li, weight[a:z, c:d], stride=1)
        self.assertLess(expected, naive.stats()['counts']['rot'])

    def test_linear_shared_rotations(self):
        lay = multi_flat_layout(32, 33)
        b = CountBackend(32)
        values = [b.input(i) for i in range(3)]
        result, _ = linear_blocks(b, values, lay, np.ones((48, 33)), None, algo='diag')
        b.output(result)
        self.assertEqual(b.stats()['counts']['rot'], 3*(1+15))

    def test_lower_tensor_and_flat_act_tails(self):
        # Tiny backend sizes keep unit arithmetic small; public parser sizes
        # are checked separately. Only this local IR header is replaced.
        for method in ('method="cheb", deg=7, domain=(-2.0,2.0)',
                       'method="minimax_comp", degrees=(15,15,27), domain=(-2.0,2.0)',
                       'method="exact"'):
            fn = 'square' if method == 'method="exact"' else 'relu'
            text = (f'@fhe.program(slots=1024)\ndef f(x: Cipher[1,35]):\n'
                    ' a = pack(x, layout="flat")\n'
                    f' z = act(a, fn="{fn}", {method})\n return z\n')
            program = parse(text)
            program = replace(program, header=replace(program.header, slots=32))
            x = self.rng.uniform(-0.5, 0.5, (2, 35))
            b = MockBackend(32, batch=2)
            b.bind(pack_inputs(program, [x]))
            values, layouts = lower(program, {}, b)
            self.close(decode_outputs([v.value for v in values], layouts)[0], run_tensor(program, {}, x, cache=False))
            for v, (_, _, block) in zip(values, layouts[0].blocks):
                self.assertTrue(np.all(v.value[..., block.size:] == 0))

    def test_lower_multi_input_output_and_refresh(self):
        text = ('@fhe.program(slots=1024, refresh="agent", l_max=5, boot_out_level=5)\n'
                'def f(x: Cipher[1,35], y: Cipher[1,35]):\n'
                ' a = pack(x, layout="flat")\n b = pack(y, layout="flat")\n'
                ' c = add(a,b)\n d = refresh(c)\n return d,c\n')
        p = parse(text); p = replace(p, header=replace(p.header, slots=32))
        x, y = self.rng.normal(size=(2, 2, 35))
        b = MockBackend(32, batch=2); b.bind(pack_inputs(p, [x, y]))
        values, layouts = lower(p, {}, b)
        self.assertEqual(len(b._inputs), 6)
        self.assertEqual(len(values), 6)
        for result in decode_outputs([v.value for v in values], layouts):
            self.close(result, x+y)
        b = CountBackend(32, boot_out_level=5); lower(p, {}, b)
        self.assertEqual(b.stats()['counts']['boot'], 3)


if __name__ == '__main__':
    unittest.main()
