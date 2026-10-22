"""Actual native frontend boundaries; opt-in, no mock ABI and no encrypted execution."""
import ctypes
import importlib.util
import os
from pathlib import Path
import sys
import unittest

from probe_native_calls import SOURCE


@unittest.skipUnless(os.environ.get('POSEIDON_NATIVE_CALLS_LIVE') == '1', 'requires isolated native probe')
class NativeFunctionTests(unittest.TestCase):
    def setUp(self):
        from hecate_python_env import VENV
        self.assertEqual(Path(sys.prefix), VENV)
        self.assertTrue(os.environ.get('IN_NIX_SHELL'))
        spec = importlib.util.spec_from_file_location('boundary_native_expr', SOURCE)
        self.hc = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = self.hc
        spec.loader.exec_module(self.hc)

    def tearDown(self):
        self.hc.lt.finalize(self.hc.ctxt)

    def test_list_result_and_trace_once(self):
        hc = self.hc
        traces = []
        @hc.func('c')
        def helper(x):
            traces.append('called')
            return [x, x + .25]
        @hc.func('c')
        def golden(x):
            a, b = helper(x), helper(x)
            self.assertIsInstance(a, list)
            self.assertIsNot(a, b)
            self.assertIsNot(a[0], b[0])
            return [a[1] + b[1]]
        golden.eval(); golden.eval(); helper.eval()
        self.assertEqual(traces, ['called'])
        self.assertEqual(golden.outputlen, 1)

    def test_tuple_result(self):
        hc = self.hc
        @hc.func('c')
        def helper(x): return x, x + .5
        @hc.func('c')
        def golden(x):
            value = helper(x)
            self.assertIsInstance(value, tuple)
            self.assertEqual(len(value), 2)
            return value
        golden.eval()
        self.assertEqual(golden.outputlen, 2)

    def test_ndarray_shape_order_and_freshness(self):
        import numpy as np
        hc = self.hc
        @hc.func('c')
        def helper(x):
            a = np.empty((2, 2), dtype=object)
            a.flat[:] = [x, x+.25, x+.5, x+.75]
            return a[:, ::-1]
        @hc.func('c')
        def golden(x):
            a, b = helper(x), helper(x)
            self.assertEqual(a.shape, (2, 2))
            self.assertEqual(a.dtype, object)
            self.assertFalse(np.shares_memory(a, b))
            return a
        golden.eval()
        self.assertEqual(golden.outputlen, 4)

    def test_rank_zero_return(self):
        import numpy as np
        hc = self.hc
        @hc.func('c')
        def helper(x):
            a = np.empty((), dtype=object); a[()] = x + .25
            return a
        @hc.func('c')
        def golden(x):
            value = helper(x)
            self.assertEqual(value.shape, ())
            self.assertIsInstance(value.item(), hc.Expr)
            return value.item()
        golden.eval()

    def test_empty_return_container(self):
        hc = self.hc
        @hc.func('c')
        def helper(x): return []
        @hc.func('c')
        def golden(x):
            self.assertEqual(helper(x), [])
            return x + .25
        golden.eval()
        self.assertEqual(helper.outputlen, 0)

    def test_zero_input_helper(self):
        hc = self.hc
        @hc.func('')
        def helper(): return hc.resolveType(.5)
        @hc.func('c')
        def golden(x): return x * helper()
        golden.eval()
        self.assertEqual(helper.inputlen, 0)

    def test_wrong_arity_rejected(self):
        hc = self.hc
        @hc.func('c')
        def helper(x): return x
        with self.assertRaises(TypeError): helper()
        with self.assertRaises(TypeError): helper(hc.Expr(0), hc.Expr(0))
        self.assertEqual(helper.evaluation_state, 'new')

    def test_wrong_signature_rejected(self):
        hc = self.hc
        for signature in ('x', 'c,', 'c,,p', 2, None):
            with self.subTest(signature=signature), self.assertRaises((TypeError, ValueError)):
                hc.func(signature)(lambda x: x)
        with self.assertRaises(TypeError): hc.func('c,p')(lambda x: x)

    def test_cipher_passed_to_plain_rejected(self):
        hc = self.hc
        @hc.func('p')
        def helper(x): return x
        @hc.func('c')
        def golden(x): return x * helper(x)
        with self.assertRaisesRegex(ValueError, 'Native function call'): golden.eval()

    def test_plain_passed_to_cipher_rejected(self):
        hc = self.hc
        @hc.func('c')
        def helper(x): return x
        @hc.func('c')
        def golden(x): return x * helper(.5)
        with self.assertRaisesRegex(ValueError, 'Native function call'): golden.eval()

    def test_direct_recursion_rejected(self):
        hc = self.hc
        @hc.func('c')
        def helper(x): return helper(x)
        with self.assertRaisesRegex(ValueError, 'Recursive'): helper.eval()
        self.assertEqual(helper.evaluation_state, 'failed')

    def test_mutual_recursion_rejected(self):
        hc = self.hc
        @hc.func('c')
        def first(x): return second(x)
        @hc.func('c')
        def second(x): return first(x)
        with self.assertRaisesRegex(ValueError, 'Recursive'): first.eval()

    def test_invalid_return_rejected(self):
        hc = self.hc
        @hc.func('c')
        def helper(x): return [x, .5]
        with self.assertRaisesRegex(TypeError, 'output cells'): helper.eval()
        with self.assertRaises(ValueError): helper.eval()

    def test_call_outside_trace_rejected_before_body(self):
        hc = self.hc; called = []
        @hc.func('p')
        def helper(x):
            called.append(True)
            return x
        with self.assertRaises(ValueError): helper(.5)
        self.assertEqual(called, [])
        self.assertEqual(helper.evaluation_state, 'new')

    def test_failed_context_cannot_continue(self):
        hc = self.hc
        @hc.func('c')
        def bad(x): raise RuntimeError('intentional failure')
        with self.assertRaises(RuntimeError): bad.eval()
        @hc.func('c')
        def other(x): return x + .5
        with self.assertRaisesRegex(ValueError, 'failed'): other.eval()

    def test_foreign_function_operand_rejected(self):
        hc = self.hc; captured = []
        @hc.func('c')
        def first(x):
            captured.append(x)
            return x
        first.eval()
        @hc.func('c')
        def helper(x): return x + .5
        @hc.func('c')
        def golden(x): return helper(captured[0])
        with self.assertRaisesRegex(ValueError, 'Native function call'): golden.eval()

    def test_caught_nested_failure_still_invalidates_trace(self):
        hc = self.hc
        @hc.func('c')
        def bad(x): raise RuntimeError('intentional nested failure')
        @hc.func('c')
        def golden(x):
            try:
                bad(x)
            except RuntimeError:
                pass
            return x
        with self.assertRaisesRegex(ValueError, 'failed'): golden.eval()
        self.assertEqual(golden.evaluation_state, 'failed')
        self.assertEqual(hc._active_traces, [])

    def test_native_bad_handles_arity_capacity(self):
        hc = self.hc; invalid = ctypes.c_size_t(-1).value
        @hc.func('c')
        def helper(x): return x + .5
        helper.eval()
        @hc.func('c')
        def golden(x):
            output = (ctypes.c_size_t * 1)(123456)
            valid_args = (ctypes.c_size_t * 1)(x.obj)
            bad_args = (ctypes.c_size_t * 1)(invalid)
            for fid, args, nargs, capacity in ((invalid, valid_args, 1, 1),
                    (helper.obj, valid_args, 0, 1), (helper.obj, valid_args, 1, 0),
                    (helper.obj, bad_args, 1, 1)):
                actual = hc.lt.createCall(hc.ctxt, fid, args, nargs, output, capacity, b'boundary', 1)
                self.assertEqual(actual, invalid)
                self.assertEqual(output[0], 123456)
            return helper(x)
        golden.eval()


class NativeAgentBoundaryTests(unittest.TestCase):
    def test_multiple_decorated_functions_still_closed_to_agent(self):
        from function_construction import normalize
        source = '@hc.func("c")\ndef helper(x):\n    return x+.25\n@hc.func("c")\ndef golden(x):\n    return [helper(x)]\n'
        with self.assertRaises(ValueError): normalize(source, {}, object_unary=True)
        # Existing ordinary Python construction helpers are a separate contract.
        normalize(source.replace('@hc.func("c")\ndef helper', 'def helper'), {}, object_unary=True)


if __name__ == '__main__': unittest.main()
