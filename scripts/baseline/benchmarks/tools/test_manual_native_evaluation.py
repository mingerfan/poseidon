import unittest
import numpy as np
from benchmark_suite import Builder
from compiler_configuration import PROFILE_SHA256
from unified_graph_contract import prepare,validate_candidate
from manual_native_evaluation import evaluate
class ManualNativeEvaluationTests(unittest.TestCase):
    def request(self):
        b=Builder([(3,)]);return prepare(b.finish(b.node("square",["input0"])),PROFILE_SHA256)
    def test_helper_arguments_lexical_constants_and_golden_selection(self):
        q=self.request();name=next(k for k,v in q["public_constants"].items() if type(v) in (int,float) and v==1)
        s='@hc.func("c")\ndef helper(u):\n    return u*u+'+name+'\n@hc.func("c,c")\ndef golden(x,zero_ct):\n    return [helper(x)-'+name+']\n'
        validate_candidate(dict(schema=1,request_id=q["request_id"],hecate_source=s),q)
        np.testing.assert_allclose(evaluate(s,q,{"input0":np.array([.2,-.3,.4])})[0][:3],np.array([.2,-.3,.4])**2,atol=2e-16,rtol=2e-15)
    def test_rotate_return_scalar(self):
        q=self.request();s='@hc.func("c,c")\ndef golden(x,zero_ct):\n    return x.rotate(1)\n'
        np.testing.assert_array_equal(evaluate(s,q,{"input0":np.array([1.,2.,3.])})[0],np.array([2.,3.,0.,1.]))
    def test_no_import_or_unsupported_method_execution(self):
        q=self.request()
        for s in ['import os\n','@hc.func("c,c")\ndef golden(x,zero_ct):\n    return x.unknown()\n']:
            with self.assertRaises((ValueError,NotImplementedError)):evaluate(s,q,{"input0":np.ones(3)})
    def test_recursive_helpers_are_bounded(self):
        q=self.request();s='@hc.func("c")\ndef h(x):\n    return h(x)\n@hc.func("c,c")\ndef golden(x,zero_ct):\n    return h(x)\n'
        with self.assertRaisesRegex(ValueError,"Helper depth"):evaluate(s,q,{"input0":np.ones(3)})
if __name__=="__main__":unittest.main()
