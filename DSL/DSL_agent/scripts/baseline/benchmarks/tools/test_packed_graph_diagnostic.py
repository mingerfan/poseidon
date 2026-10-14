import unittest
import numpy as np
from benchmark_suite import Builder
from benchmark_graph import samples
from benchmark_math import evaluate as reference
from compiler_configuration import PROFILE_SHA256
from unified_graph_contract import prepare,validate_candidate
from manual_native_evaluation import evaluate
from packed_graph_diagnostic import source
class LogicalViewTests(unittest.TestCase):
    def graph(self,n,nonlinear=False):
        b=Builder([(1,1,n)])
        z=b.node("multiply",["input0",b.const(0.)])
        tail=b.node("slice",[z],axis=-1,start=0,stop=1,step=1)
        x=b.node("concat",[tail,"input0",tail],axis=-1)
        if nonlinear:x=b.node("polynomial",[x,b.const([.1,.2])],basis="chebyshev")
        else:x=b.node("negate",[x])
        a=b.node("slice",[x],axis=-1,start=0,stop=n+2,step=2)
        c=b.node("slice",[x],axis=-1,start=1,stop=n+2,step=2)
        return b.finish(b.node("subtract",[a,c]))
    def test_padded_concat_negation_preserves_tail(self):
        for n in (4,8):
            g=self.graph(n);q=prepare(g,PROFILE_SHA256,generation_guidance="explicit-v9")
            self.assertEqual(q["layout"]["input_slot_period"],n)
            src=source(q)
            validate_candidate(dict(schema=1,request_id=q["request_id"],hecate_source=src),q)
            for xs in samples(g,16):
                expected=reference(g,xs)[g["outputs"][0]["name"]].reshape(-1)
                np.testing.assert_allclose(evaluate(src,q,xs)[0][:len(expected)],expected,atol=0,rtol=0)
    def test_unhandled_wide_materialization_rejected(self):
        q=prepare(self.graph(4,True),PROFILE_SHA256,generation_guidance="explicit-v9")
        with self.assertRaisesRegex(ValueError,"exceeds one ciphertext period"):source(q)
if __name__=="__main__":unittest.main()
