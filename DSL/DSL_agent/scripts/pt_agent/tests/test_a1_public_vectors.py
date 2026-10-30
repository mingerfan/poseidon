"""A1 §6.10: scoped public vectors preserve arithmetic and constant order."""

from contextlib import ExitStack
import gc
import unittest
from unittest.mock import patch
import weakref

import numpy as np

from pt_agent.backends import CountBackend, MockBackend
from pt_agent.kernels import slots as public, transformer as tr
from pt_agent.kernels.layout import FlatLayout, TokenLayout
from pt_agent.kernels.linear import linear_flat


class SpyCount(CountBackend):
    def __init__(self, slots):
        super().__init__(slots)
        self.seen = []

    def const(self, value):
        self.seen.append(value)
        return super().const(value)


class SpyMock(MockBackend):
    def __init__(self, slots):
        super().__init__(slots)
        self.names = []

    def const(self, value):
        result = super().const(value)
        self.names.append(result.name)
        return result


class PublicVectorTests(unittest.TestCase):
    def test_nested_scope_reuses_readonly_vectors_and_releases_owners(self):
        backend = object()
        owners = []

        @public.scope
        def inner(b, owner):
            return public.vector(b, ('sample', id(owner)), lambda: owner.copy(), owners=(owner,))

        @public.scope
        def outer(b):
            owner = np.arange(4.)
            owners.append(weakref.ref(owner))
            first, second = inner(b, owner), inner(b, owner)
            self.assertIs(first, second)
            self.assertFalse(first.flags.writeable)
            self.assertIsNotNone(owners[0]())
            return first

        result = outer(backend)
        gc.collect()
        self.assertIsNone(owners[0]())
        self.assertIsNone(public._current.get())
        self.assertTrue(np.array_equal(result, np.arange(4.)))

    def test_scope_resets_after_error_and_different_backends_are_isolated(self):
        a, b = object(), object()

        @public.scope
        def nested(backend, number):
            return public.vector(backend, 'v', lambda: np.array([number]))

        @public.scope
        def outer(backend):
            first = nested(backend, 1.)
            other = nested(b, 2.)
            self.assertEqual(first[0], 1.)
            self.assertEqual(other[0], 2.)
            self.assertIs(first, nested(backend, 3.))
            raise ValueError('scope exit')

        with self.assertRaisesRegex(ValueError, 'scope exit'):
            outer(a)
        self.assertIsNone(public._current.get())

    def test_plain_preserves_signed_zero_and_reuses_identity(self):
        backend = SpyCount(16)

        @public.scope
        def run(b):
            tr.plain(b, 4, coefficient=0.)
            tr.plain(b, 4, coefficient=-0.)
            tr.plain(b, 4, coefficient=-0.)

        run(backend)
        positive, negative, duplicate = backend.seen
        self.assertIs(negative, duplicate)
        self.assertIsNot(positive, negative)
        self.assertFalse(np.signbit(positive[:4]).any())
        self.assertTrue(np.signbit(negative[:4]).all())
        self.assertEqual(len(backend._constants), 2)

    def test_token_linear_reuses_final_rolled_diagonals_per_token(self):
        backend = SpyCount(64)
        layout = TokenLayout(64, 3, 4)
        weight = np.arange(1., 13.).reshape(3, 4)
        bias = np.arange(1., 4.)
        xs = [backend.input(i) for i in range(3)]
        tr.linear(backend, xs, layout, weight, bias, n1=2)
        width = len(backend.seen)//3
        self.assertGreater(width, 0)
        self.assertEqual(len(backend.seen), 3*width)
        for i in range(width):
            self.assertIs(backend.seen[i], backend.seen[i+width])
            self.assertIs(backend.seen[i], backend.seen[i+2*width])
            self.assertFalse(backend.seen[i].flags.writeable)

    def test_cnn_linear_keeps_streaming_and_changed_weights_take_effect(self):
        backend = SpyCount(64)
        x = backend.input(0)
        weight = np.arange(1., 13.).reshape(3, 4)
        linear_flat(backend, x, FlatLayout(64, 4), weight, None)
        split = len(backend.seen)
        weight *= 2
        linear_flat(backend, x, FlatLayout(64, 4), weight, None)
        self.assertTrue(all(value.flags.writeable for value in backend.seen))
        self.assertIsNot(backend.seen[0], backend.seen[split])
        self.assertTrue(np.array_equal(backend.seen[0]*2, backend.seen[split]))

    def test_separate_token_calls_observe_mutated_weights(self):
        backend = SpyCount(64)
        layout = TokenLayout(64, 2, 4)
        xs = [backend.input(i) for i in range(2)]
        weight = np.arange(1., 13.).reshape(3, 4)
        tr.linear(backend, xs, layout, weight)
        split = len(backend.seen)
        weight *= 2
        tr.linear(backend, xs, layout, weight)
        self.assertIsNot(backend.seen[0], backend.seen[split])
        self.assertTrue(np.array_equal(backend.seen[0]*2, backend.seen[split]))

    def run_linear(self, shape, *, cached, algo='bsgs'):
        outputs, inputs = shape
        backend = SpyMock(32)
        layout = TokenLayout(32, 2, inputs)
        backend.bind([np.arange(32.)/32 for _ in layout.blocks])
        xs = [backend.input(i) for i in range(len(layout.blocks))]
        weight = np.arange(1., outputs*inputs+1).reshape(outputs, inputs)/100
        bias = np.arange(outputs)/10
        with ExitStack() as stack:
            if not cached:
                stack.enter_context(patch.object(public, 'memoize', side_effect=lambda b,k,f,**kw:f()))
                stack.enter_context(patch.object(tr, 'memoize', side_effect=lambda b,k,f,**kw:f()))
            result, _ = tr.linear(backend, xs, layout, weight, bias, algo=algo)
        return [value.value for value in result], backend.names

    def test_flat_multiblock_and_tall_arithmetic_and_names_identical(self):
        for shape in ((3, 4), (7, 4), (20, 4), (3, 20), (20, 20)):
            for algo in ('diag', 'bsgs'):
                with self.subTest(shape=shape, algo=algo):
                    old, old_names = self.run_linear(shape, cached=False, algo=algo)
                    new, new_names = self.run_linear(shape, cached=True, algo=algo)
                    self.assertEqual(old_names, new_names)
                    self.assertEqual(len(old), len(new))
                    for a, b in zip(old, new):
                        self.assertTrue(np.array_equal(a, b))

    def test_gather_preserves_group_order_and_reuses_masks(self):
        backend = SpyCount(16)
        xs = [backend.input(0)]
        layout = FlatLayout(16, 6)
        index, sign = (5, 0, 4, 1), (1, -1, 1, -1)

        @public.scope
        def run(b):
            tr.gather(b, xs, layout, index, sign)
            split = len(b.seen)
            tr.gather(b, xs, layout, index, sign)
            return split

        split = run(backend)
        self.assertGreater(split, 1)
        self.assertEqual(len(backend.seen), 2*split)
        for i in range(split):
            self.assertIs(backend.seen[i], backend.seen[i+split])
            self.assertFalse(backend.seen[i].flags.writeable)


if __name__ == '__main__':
    unittest.main()
