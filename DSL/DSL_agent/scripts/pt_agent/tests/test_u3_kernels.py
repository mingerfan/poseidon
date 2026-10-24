"""U3a CNN semantics, round trips and physical channel-block arithmetic."""

import unittest

import numpy as np
import torch
import torch.nn.functional as F

from pt_agent.backends import CountBackend, MockBackend
from pt_agent.dsl.parser import parse
from pt_agent.dsl.printer import print_program
from pt_agent.dsl.typecheck import check
from pt_agent.kernels import cnn, multi
from pt_agent.kernels.layout import (MuxLayout, ConcatMuxLayout, layout_blocks,
                                    layout_from_record, layout_record, multi_mux_layout, pack_multi, pack_mux)
from pt_agent.lower import lower, pack_inputs, decode_outputs
from pt_agent.tensor_sim import clear_caches, run_tensor


def source(slots, channels, body, height=16, width=16):
    return (f'@fhe.program(slots={slots}, refresh="compiler", placement="depth-dp")\n'
            f'def cnn_case(x: Cipher[1,{channels},{height},{width}]):\n'
            '    p = pack(x, layout="mux", gap=1)\n'+body+'\n')


def mock(program, weights, images, *, exact=False, observer=None):
    backend = MockBackend(program.header.slots, batch=len(images))
    backend.bind(pack_inputs(program, [images]))
    outputs, layouts = lower(program, weights, backend, act_mode='exact' if exact else 'program', observer=observer)
    return decode_outputs([value.value for value in outputs], layouts)[0]


class CNNKernelTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(20261024)
        torch.set_num_threads(1)
        clear_caches()

    def verify(self, text, weights, images, expected):
        program = parse(text)
        self.assertTrue(check(program, {name: value.shape for name, value in weights.items()}).ok)
        canonical = print_program(program)
        self.assertEqual(print_program(parse(canonical)), canonical)
        actual = mock(program, weights, images)
        tensor = run_tensor(program, weights, images, cache=False)
        np.testing.assert_allclose(actual, tensor, atol=1e-9, rtol=1e-9)
        np.testing.assert_allclose(mock(program, weights, images, exact=True), expected, atol=1e-9, rtol=1e-9)
        np.testing.assert_allclose(run_tensor(program, weights, images, cache=False, act_mode='exact'), expected, atol=1e-9, rtol=1e-9)
        return program

    def test_depthwise_single_and_multi_stride_1_2(self):
        x = self.rng.normal(size=(2, 8, 16, 16))
        weights = {'w': self.rng.normal(size=(8, 1, 3, 3)), 'b': self.rng.normal(size=8)}
        for slots in (4096, 1024):
            for stride in (1, 2):
                with self.subTest(slots=slots, stride=stride):
                    expected = F.conv2d(torch.tensor(x), torch.tensor(weights['w']), torch.tensor(weights['b']), stride=stride, padding=1, groups=8).numpy()
                    self.verify(source(slots, 8, f'    y = conv2d(p, w=W["w"], b=W["b"], groups=8, stride={stride}, pad=1)\n    return y'), weights, x, expected)

    def test_avgpool_three_configurations(self):
        x = self.rng.normal(size=(2, 8, 16, 16))
        for slots in (4096, 1024):
            for k, stride, pad in ((2, 2, 0), (3, 2, 1), (3, 1, 1)):
                with self.subTest(slots=slots, k=k, stride=stride):
                    expected = F.avg_pool2d(torch.tensor(x), k, stride, pad, count_include_pad=True).numpy()
                    self.verify(source(slots, 8, f'    y = avgpool(p, k={k}, stride={stride}, pad={pad})\n    return y'), {}, x, expected)

    def test_maxpool_merge_padding_and_observations(self):
        x = self.rng.normal(size=(2, 8, 16, 16))
        for slots in (4096, 1024):
            for k, pad in ((2, 0), (3, 1)):
                for method in ('cheb', 'minimax_comp'):
                    with self.subTest(slots=slots, k=k, method=method):
                        params = 'deg=7' if method == 'cheb' else 'degrees=(15,15,27)'
                        text = source(slots, 8, f'    y = maxpool(p, k={k}, stride=2, pad={pad}, method="{method}", {params}, domain=(-12.0,12.0))\n    return y')
                        expected = F.max_pool2d(torch.tensor(x), k, 2, pad).numpy()
                        p = self.verify(text, {}, x, expected)
                        left, right = [], []
                        mock(p, {}, x, observer=lambda name, value, layout, cfg: left.append((float(value.value.min()), float(value.value.max()), cfg)))
                        run_tensor(p, {}, x, observer=lambda name, value, cfg: right.append((float(value.min()), float(value.max()), cfg)), cache=False)
                        self.assertEqual(len(left), 1)
                        self.assertEqual(left[0][2], right[0][2])
                        np.testing.assert_allclose(left[0][:2], right[0][:2], atol=1e-9, rtol=1e-9)

    def test_channel_concat_two_and_three_branches(self):
        x = self.rng.normal(size=(2, 4, 16, 16))
        weights = {f'w{i}': self.rng.normal(size=(count, 4, 1, 1)) for i, count in enumerate((1, 2, 3))}
        for slots in (4096, 1024):
            for count in (2, 3):
                with self.subTest(slots=slots, count=count):
                    body = '\n'.join(f'    a{i} = conv2d(p, w=W["w{i}"], pad=0)' for i in range(count))
                    body += '\n    y = concat('+', '.join(f'a{i}' for i in range(count))+', axis=1)\n    z = add(y,y)\n    return z'
                    expected = 2*np.concatenate([F.conv2d(torch.tensor(x), torch.tensor(weights[f'w{i}'])).numpy() for i in range(count)], axis=1)
                    self.verify(source(slots, 4, body), weights, x, expected)
        blocks = [multi_mux_layout(128, c, 4, 4, 1) for c in (1, 3)]
        backend = CountBackend(128)
        before = backend.stats()
        _, layout = cnn.concat_blocks([backend.input(0), backend.input(1)], blocks)
        self.assertEqual(backend.stats(), before)
        self.assertIsInstance(layout, ConcatMuxLayout)
        self.assertEqual([end-start for start, end, _ in layout.blocks], [1, 3])
        self.assertEqual(layout_from_record(layout_record(layout)), layout)

    def test_flatten_linear_diag_bsgs(self):
        x = self.rng.normal(size=(2, 128, 4, 4))
        weights = {'w': self.rng.normal(size=(5, 2048)), 'b': self.rng.normal(size=5)}
        expected = x.reshape(2, -1)@weights['w'].T+weights['b']
        for slots in (4096, 1024):
            for algo in ('diag', 'bsgs'):
                with self.subTest(slots=slots, algo=algo):
                    self.verify(source(slots, 128, f'    f = flatten(p)\n    y = linear(f, w=W["w"], b=W["b"], algo="{algo}")\n    return y', height=4, width=4), weights, x, expected)

    def test_standalone_and_folded_batchnorm(self):
        x = self.rng.normal(size=(2, 8, 16, 16))
        weights = {'bn.weight': self.rng.normal(size=8), 'bn.bias': self.rng.normal(size=8),
                   'bn.running_mean': self.rng.normal(size=8), 'bn.running_var': self.rng.uniform(.5, 2, size=8),
                   'w': self.rng.normal(size=(8, 8, 1, 1))}
        apply = lambda x: F.batch_norm(torch.tensor(x), torch.tensor(weights['bn.running_mean']), torch.tensor(weights['bn.running_var']), torch.tensor(weights['bn.weight']), torch.tensor(weights['bn.bias']), training=False).numpy()
        for slots in (4096, 1024):
            self.verify(source(slots, 8, '    y = batchnorm(p, bn="bn")\n    return y'), weights, x, apply(x))
            conv = F.conv2d(torch.tensor(x), torch.tensor(weights['w'])).numpy()
            self.verify(source(slots, 8, '    c = conv2d(p,w=W["w"],pad=0)\n    y = batchnorm(c,bn="bn")\n    return y'), weights, x, apply(conv))

    def test_maxpool_supported_boundaries(self):
        for height in (7, 8):
            x = self.rng.normal(size=(2, 2, height, height))
            for k, stride, pad in ((2, 2, 1), (3, 3, 0), (3, 3, 1)):
                with self.subTest(height=height, k=k, stride=stride, pad=pad):
                    text = source(1024, 2, f'    y = maxpool(p, k={k}, stride={stride}, pad={pad}, method="cheb", deg=7, domain=(-12.,12.))\n    return y', height=height, width=height)
                    expected = F.max_pool2d(torch.tensor(x), k, stride, pad).numpy()
                    self.verify(text, {}, x, expected)

    def test_depthwise_count_efficiency(self):
        from pt_agent.kernels.conv import conv2d_mux
        from pt_agent.kernels.layout import mux_layout
        counts = []
        for channels in (4, 8):
            lay = mux_layout(1024, channels, 4, 4, 1)
            depthwise = self.rng.normal(size=(channels, 1, 3, 3))
            blockdiag = np.zeros((channels, channels, 3, 3))
            for c in range(channels):
                blockdiag[c, c] = depthwise[c, 0]
            new, old = CountBackend(1024), CountBackend(1024)
            cnn.depthwise_blocks(new, new.input(0), lay, depthwise)
            conv2d_mux(old, old.input(0), lay, blockdiag, stride=1)
            a, b = new.stats()['counts'], old.stats()['counts']
            self.assertLess(a['rot'], b['rot'])
            self.assertLess(a['mul_cp'], b['mul_cp'])
            counts.append(a['rot'])
        self.assertEqual(counts[0], counts[1])


if __name__ == '__main__':
    unittest.main()
