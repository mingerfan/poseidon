"""Restricted spatial calls agree with direct trusted PyTorch, including boundaries."""
import copy
import hashlib
import unittest

from restricted_model_python import parse, FORMAT_PY


def capture(expression, shape, constants=None, files=None):
    files = files or {'main.py': 'import torch.nn.functional as F\ndef forward(x):\n    return ' + expression + '\n'}
    manifest = dict(format=FORMAT_PY, id='restricted_spatial_reference',
                    entry='main.forward', files={p: hashlib.sha256(s.encode()).hexdigest() for p, s in files.items()},
                    inputs=[dict(name='x', shape=shape)], constants=constants or {}, outputs=['output'])
    return parse(manifest, files)


class RestrictedSpatialTests(unittest.TestCase):
    def compare(self, expression, shape, constants, function, keyword=None, *, positional=()):
        import numpy as np
        import torch
        from benchmark_math import evaluate
        rng = np.random.default_rng(1978)
        original, lowered, binding = capture(expression, shape, constants)
        frozen = copy.deepcopy(constants)
        for _ in range(3):
            x = rng.uniform(-.25, .25, size=shape)
            operands = [torch.tensor(x, dtype=torch.float64)]
            if function.__name__.startswith('conv'):
                operands.append(torch.tensor(constants['weight'], dtype=torch.float64))
                if 'bias' in constants:
                    operands.append(torch.tensor(constants['bias'], dtype=torch.float64))
            operands.extend(positional)
            expected = function(*operands, **(keyword or {})).numpy()
            actual = next(iter(evaluate(lowered, {'x': x}).values()))
            np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)
        self.assertEqual(constants, frozen)
        self.assertIn('python_frontend_sha256', binding)
        return original, lowered

    def test_conv1d_defaults_and_optional_bias(self):
        import torch.nn.functional as F
        self.compare('F.conv1d(x, weight)', [1, 1, 6],
                     {'weight': [[[.25, -.125, .375]]]}, F.conv1d)
        self.compare('F.conv1d(x, weight, bias=bias)', [1, 1, 6],
                     {'weight': [[[.25, -.125, .375]]], 'bias': [.125]}, F.conv1d)

    def test_grouped_conv1d_and_dilation(self):
        import torch.nn.functional as F
        self.compare('F.conv1d(x, weight, None, 2, 1, 2, 2)', [1, 2, 8],
                     {'weight': [[[.25, -.125]], [[-.375, .125]]]}, F.conv1d,
                     positional=(None, 2, 1, 2, 2))

    def test_conv2d_non_square_kernel_and_strided_padding(self):
        import torch.nn.functional as F
        self.compare('F.conv2d(x, weight, bias, stride=(2, 1), padding=(1, 0))', [1, 1, 4, 5],
                     {'weight': [[[[.25, -.125, .375], [.125, .5, -.25]]]], 'bias': [.0625]},
                     F.conv2d, {'stride': (2, 1), 'padding': (1, 0)})

    def test_depthwise_conv2d(self):
        import torch.nn.functional as F
        self.compare('F.conv2d(x, weight, groups=2, padding=1, dilation=2)', [1, 2, 4, 4],
                     {'weight': [[[[.25, -.125], [.375, .125]]], [[[-.25, .5], [.125, -.375]]]]},
                     F.conv2d, {'groups': 2, 'padding': 1, 'dilation': 2})

    def test_pool1d_stride_defaults_to_kernel(self):
        import torch.nn.functional as F
        self.compare('F.avg_pool1d(x, 2)', [1, 2, 7], {}, F.avg_pool1d, positional=(2,))

    def test_pool1d_excludes_padding(self):
        import torch.nn.functional as F
        self.compare('F.avg_pool1d(x, 3, stride=2, padding=1, count_include_pad=False)',
                     [1, 2, 7], {}, F.avg_pool1d,
                     {'kernel_size': 3, 'stride': 2, 'padding': 1, 'count_include_pad': False})

    def test_pool2d_defaults_and_explicit_none(self):
        import torch.nn.functional as F
        self.compare('F.avg_pool2d(x, (2, 3), stride=None, divisor_override=None)',
                     [1, 1, 4, 6], {}, F.avg_pool2d, {'kernel_size': (2, 3)})

    def test_pool2d_boundary_divisor(self):
        import torch.nn.functional as F
        for include in (False, True):
            self.compare(f'F.avg_pool2d(x, 3, 2, 1, False, {include})', [1, 1, 4, 4], {},
                         F.avg_pool2d, positional=(3, 2, 1, False, include))

    def test_alias_and_local_helper_have_identical_graph(self):
        constants = {'weight': [[[[.25, -.125], [.375, .125]]]]}
        direct = capture('F.conv2d(x, weight)', [1, 1, 4, 4], constants)
        files = {'main.py': 'from helper import convolve\ndef forward(x):\n    return convolve(x)\n',
                 'helper.py': 'import torch.nn.functional as ops\ndef convolve(x):\n    return ops.conv2d(x, weight)\n'}
        aliased = capture('', [1, 1, 4, 4], constants, files)
        self.assertEqual(direct[:2], aliased[:2])
        self.assertNotEqual(direct[2]['python_source_hashes'], aliased[2]['python_source_hashes'])

    def test_required_keyword_arguments(self):
        import torch.nn.functional as F
        self.compare('F.conv2d(input=x, weight=weight)', [1, 1, 4, 4],
                     {'weight': [[[[.25, -.125], [.375, .125]]]]}, F.conv2d)
        self.compare('F.avg_pool2d(x, kernel_size=(2, 2))', [1, 1, 4, 4], {},
                     F.avg_pool2d, {'kernel_size': (2, 2)})
        self.compare('F.avg_pool1d(input=x, kernel_size=2)', [1, 1, 6], {},
                     F.avg_pool1d, {'kernel_size': 2})

    def test_required_arguments_are_not_invented(self):
        for expression in ('F.conv2d(x)', 'F.avg_pool2d(x)', 'F.conv2d()', 'F.avg_pool1d(stride=2)'):
            with self.subTest(expression=expression), self.assertRaises(ValueError):
                capture(expression, [1, 1, 4, 4])

    def test_nonrepresentable_or_dynamic_options_rejected(self):
        constants = {'weight': [[[[.25, -.125], [.375, .125]]]]}
        expressions = [
            'F.conv2d(x, weight, padding="same")',
            'F.conv2d(x, weight, stride=x)',
            'F.conv2d(x, weight, stride=True)',
            'F.conv2d(x, weight, groups=True)',
            'F.conv2d(x, weight, dilation=(1,))',
            'F.conv2d(x, weight, padding=(1, 2, 3))',
            'F.conv2d(x, weight, 0, bias=weight)',
            'F.conv2d(x, weight, ceil_mode=False)',
            'F.avg_pool2d(x, 2, ceil_mode=True)',
            'F.avg_pool2d(x, 2, divisor_override=3)',
            'F.avg_pool2d(x, 2, count_include_pad=1)',
            'F.avg_pool2d(x, 2, padding=True)',
            'F.avg_pool2d(x, 2, 1, stride=2)',
            'F.avg_pool1d(x, 2, divisor_override=None)',
        ]
        for expression in expressions:
            with self.subTest(expression=expression), self.assertRaises(ValueError):
                capture(expression, [1, 1, 4, 4], constants)

    def test_input_program_is_never_executed(self):
        with self.assertRaises(ValueError):
            capture('', [1, 1, 4, 4], files={'main.py':
                'import torch.nn.functional as F\nraise RuntimeError("must not execute")\ndef forward(x):\n    return F.avg_pool2d(x, 2)\n'})


if __name__ == '__main__':
    unittest.main()
