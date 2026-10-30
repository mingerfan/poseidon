"""A1 §6.10 identity reuse preserves content naming and streamed operands."""

import unittest
from unittest.mock import patch

import numpy as np

from pt_agent.backends import base
from pt_agent.backends.count import CountBackend
from pt_agent.backends.cost_dag import CostDag


class ConstantIdentityTests(unittest.TestCase):
    def test_readonly_identity_skips_copy_and_hash(self):
        backend = CountBackend(8)
        vector = np.arange(8, dtype=float)
        vector.flags.writeable = False
        first = backend.const(vector)
        with patch.object(base, '_vector', side_effect=AssertionError('copy')), \
             patch.object(base.hashlib, 'blake2b', side_effect=AssertionError('hash')):
            self.assertIs(first, backend.const(vector))
        held, plain = backend._constant_objects[id(vector)]
        self.assertIs(held, vector)
        self.assertIs(plain, first)

    def test_distinct_objects_content_dedup_and_order(self):
        backend = CountBackend(8)
        first = np.arange(8, dtype=float)
        second = first.copy()
        first.flags.writeable = second.flags.writeable = False
        self.assertEqual(backend.const(first).name, 'k0')
        self.assertEqual(backend.const([3]).name, 'k1')
        self.assertEqual(backend.const(second).name, 'k0')
        self.assertEqual(len(backend._constants), 2)

    def test_writable_inputs_keep_original_validation_and_naming(self):
        backend = CountBackend(8)
        vector = np.zeros(8)
        self.assertEqual(backend.const(vector).name, 'k0')
        vector[0] = 1
        self.assertEqual(backend.const(vector).name, 'k1')
        self.assertEqual(backend._constant_objects, {})

    def test_first_readonly_input_still_validated(self):
        for vector in (np.array([np.inf]), np.zeros(3), np.zeros((2, 4))):
            vector.flags.writeable = False
            backend = CountBackend(8)
            with self.assertRaises(base.BackendError):
                backend.const(vector)
            self.assertEqual(backend._constant_objects, {})

    def test_broadcast_and_signed_zero_remain_distinct(self):
        backend = CountBackend(8)
        positive = np.array([0.])
        negative = np.array([-0.])
        positive.flags.writeable = negative.flags.writeable = False
        self.assertEqual(backend.const(positive).name, 'k0')
        self.assertEqual(backend.const([0.]*8).name, 'k0')
        self.assertEqual(backend.const(negative).name, 'k1')
        self.assertEqual(backend.const([-0.]*8).name, 'k1')

    def test_costdag_validates_once_and_retains_operand_payload(self):
        backend = CostDag(8)
        original = base._vector
        with patch.object(base, '_vector', wraps=original) as convert:
            plain = backend.const([2.])
        self.assertEqual(convert.call_count, 1)
        np.testing.assert_array_equal(plain.data, np.full(8, 2.))
        self.assertIsNone(next(iter(backend._constants.values())).data)
        x = backend.input(0)
        backend.output([x-plain])
        self.assertEqual(backend.operands[-1], (('cipher', 0), ('plain', 'k1')))
        self.assertEqual(backend.const([-2.]).name, 'k1')
        backend.finalize()
        self.assertEqual(backend.kind, ['input', 'add'])

    def test_costdag_readonly_reuses_plain_with_payload(self):
        backend = CostDag(8)
        vector = np.array([2.])
        vector.flags.writeable = False
        first = backend.const(vector)
        self.assertIs(backend.const(vector), first)
        np.testing.assert_array_equal(first.data, np.full(8, 2.))


if __name__ == '__main__':
    unittest.main()
