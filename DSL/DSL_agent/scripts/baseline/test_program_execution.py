"""Local execution I/O tests; these do not execute homomorphic operations."""
import copy
import unittest
import numpy as np
from benchmark_suite import Builder
from component_contract import prepare_task
from program_execution import pack_samples, named_outputs

class ProgramInputTests(unittest.TestCase):
    def request(self):
        b = Builder([(2, 3), (2, 3)])
        a = b.node("add", ["input0", "input1"])
        return prepare_task(b.finish(a))
    def test_named_inputs_and_zero_are_independent(self):
        req = self.request()
        values = dict(input0=[[1.,2.,3.],[4.,5.,6.]],input1=[[0.,-1.,0.],[1.,0.,-1.]])
        original = copy.deepcopy(values)
        packed = pack_samples(req, [values])
        self.assertEqual(packed.shape, (1, 3, 8))
        np.testing.assert_equal(packed[0, 0, :6], [1,2,3,4,5,6])
        np.testing.assert_equal(packed[0, 1, :6], [0,-1,0,1,0,-1])
        np.testing.assert_equal(packed[0, 2], np.zeros(8))
        self.assertEqual(values, original)
    def test_named_outputs_restore_tensor_shape(self):
        req = self.request()
        name = req["model"]["outputs"][0]["name"]
        result = named_outputs(req, np.arange(12).reshape(2,6), 2)
        self.assertEqual(result[1][name], [[6,7,8],[9,10,11]])
    def test_sample_count(self):
        req = self.request()
        sample = dict(input0=np.zeros((2,3)),input1=np.zeros((2,3)))
        self.assertEqual(pack_samples(req, [sample]*16).shape[0], 16)
        for values in ([], [sample]*17):
            with self.assertRaises(ValueError): pack_samples(req, values)
    def test_signature_shape_and_dtype(self):
        req = self.request()
        sample = dict(input0=np.zeros((2,3)),input1=np.zeros((2,3)))
        invalid = [
            dict(sample, extra=np.zeros((2,3))),
            dict(sample, input0=[[True,1,2],[3,4,5]]),
            dict(sample, input0=[[2**60,1.,2.],[3.,4.,5.]]),
            dict(input0=np.zeros((2,3))),
            dict(sample, input0=np.zeros(6)),
            dict(sample, input0=np.zeros((2,3), dtype=bool)),
            dict(sample, input0=np.full((2,3), "1")),
            dict(sample, input0=np.full((2,3), complex(1,0))),
            dict(sample, input0=np.full((2,3), np.inf)),
            dict(sample, input0=np.full((2,3), np.nan)),
            dict(sample, input0=np.full((2,3), 2**63, dtype=np.uint64)),
        ]
        for value in invalid:
            with self.subTest(value=str(value)):
                with self.assertRaises(ValueError): pack_samples(req,[value])
    def test_bad_output_count_and_nonfinite(self):
        req = self.request()
        for value in (np.zeros((2,6)),np.zeros((1,7)),np.full((1,6),np.inf)):
            with self.assertRaises(ValueError): named_outputs(req,value,1)
    def test_chunk_tail(self):
        b = Builder([(5,)])
        req = prepare_task(b.finish(b.node("square",["input0"])), dict(chunk_period=4))
        actual = pack_samples(req,[dict(input0=[1,2,3,4,5])])
        self.assertEqual(actual.shape,(1,3,4))
        np.testing.assert_equal(actual[0,1],[5,0,0,0])
        self.assertEqual(named_outputs(req,np.array([[1,4,9,16,25]]),1)[0][req["model"]["outputs"][0]["name"]],[1,4,9,16,25])

if __name__ == "__main__": unittest.main()
