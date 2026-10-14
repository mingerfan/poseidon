"""Source-only static ABI admission; no fake logical graph or source execution."""
import unittest,copy
import numpy as np
from source_calibration_signature import checked_samples,FORMAT
from source_calibration_contract import CalibrationError
from test_model_output_contract import tensor

class SourceSignatureTests(unittest.TestCase):
    def setUp(self):
        self.spec=dict(format="poseidon-trusted-pytorch-v38",inputs=[dict(name="x",shape=[2])],outputs=["out"])
        self.sig=dict(format=FORMAT,outputs=[dict(name="out",shape=[2])])
    def call(self,**changes):
        args=dict(manifest=self.spec,signature=self.sig,samples=[dict(x=[.5,-.25])],
            input_dtypes={"x":"float64"},tree=tensor(),resource_profile="general-tensors-v1")
        args.update(changes);return checked_samples(**args)
    def test_basic_signature_needs_no_graph(self):
        wire,types,tree,shapes=self.call()
        self.assertEqual(wire,[dict(x=[.5,-.25])]);self.assertEqual(shapes,{"out":[2]})
    def test_native_float32_wire_values_preserved(self):
        x=np.asarray([.1,.3],dtype=np.float32)
        wire,_,_,_=self.call(samples=[dict(x=x)],input_dtypes={"x":"float32"})
        self.assertEqual(wire[0]["x"],x.tolist())
    def test_named_multi_inputs_and_outputs(self):
        spec=dict(self.spec,inputs=[dict(name="right",shape=[2]),dict(name="left",shape=[1])],outputs=["out","aux"])
        sig=dict(format=FORMAT,outputs=[dict(name="out",shape=[2]),dict(name="aux",shape=[1])])
        tree=dict(kind="tuple",items=[tensor("out"),tensor("aux")])
        wire,_,_,shapes=self.call(manifest=spec,signature=sig,tree=tree,
            input_dtypes={"left":"float64","right":"float32"},samples=[dict(left=[.5],right=[1.,2.])])
        self.assertEqual(shapes,{"out":[2],"aux":[1]})
    def test_rank_32_profile_and_legacy_rank_rejection(self):
        spec=dict(self.spec,inputs=[dict(name="x",shape=[1]*32)])
        sig=dict(format=FORMAT,outputs=[dict(name="out",shape=[1]*32)])
        self.call(manifest=spec,signature=sig,samples=[dict(x=np.zeros([1]*32))])
        with self.assertRaises(CalibrationError):
            self.call(manifest=spec,signature=sig,samples=[dict(x=np.zeros([1]*32))],resource_profile="legacy-v1")
    def test_capacity_and_boolean_dimensions_rejected(self):
        for shape in ([4097],[True],[0],[-1],[1]*33):
            spec=dict(self.spec,inputs=[dict(name="x",shape=shape)])
            with self.subTest(shape=shape),self.assertRaises(CalibrationError):self.call(manifest=spec)
    def test_total_output_capacity_and_wrong_names_rejected(self):
        for sig in (dict(format=FORMAT,outputs=[dict(name="out",shape=[4097])]),
                    dict(format=FORMAT,outputs=[dict(name="wrong",shape=[2])]),
                    dict(format=FORMAT,outputs=[])):
            with self.assertRaises(CalibrationError):self.call(signature=sig)
    def test_samples_17_nonfinite_and_boolean_leaves_rejected(self):
        for samples in ([dict(x=[0.,0.])]*17,[dict(x=[float("nan"),0.])],[dict(x=[True,False])]):
            with self.assertRaises(CalibrationError):self.call(samples=samples)
    def test_wrong_shape_input_names_and_dtype_rejected(self):
        for options in (dict(samples=[dict(x=[1.])]),dict(samples=[dict(other=[1.,2.])]),
                        dict(input_dtypes={"x":"int64"}),dict(input_dtypes={"x":"float16"})):
            with self.assertRaises(CalibrationError):self.call(**options)
    def test_integer_precision_boundary_is_checked(self):
        self.call(samples=[dict(x=[2**53,-2**53])])
        with self.assertRaises(CalibrationError):self.call(samples=[dict(x=[2**53+1,0])])
    def test_signature_and_samples_are_detached(self):
        before=copy.deepcopy(self.sig);samples=[dict(x=[1.,2.])]
        wire,_,_,shapes=self.call(samples=samples);wire[0]["x"][0]=9
        shapes["out"][0]=9
        self.assertEqual(samples[0]["x"][0],1.);self.assertEqual(self.sig,before)

if __name__=="__main__":unittest.main()
