"""Balanced Chebyshev identity tests; no change to numerical reference."""
import unittest,copy,ast
import numpy as np
from chebyshev_lowering import balanced
from benchmark_suite import generate
from benchmark_graph import samples
from benchmark_math import evaluate
from benchmark_torch import evaluate as reference
from compiler_configuration import PROFILE_SHA256,configuration
from unified_graph_contract import prepare,validate_candidate
from unified_graph_lowering import lower
from test_packed_prefix_lowering import public_execute

class BalancedChebyshevTests(unittest.TestCase):
    def test_all_degrees_and_tiny_coefficients_retained(self):
        x=np.linspace(-1.,1.,65)
        for degree in range(128):
            coeff=[0.]*degree+[1.]
            np.testing.assert_allclose(balanced(x,coeff,np.zeros_like(x)),
                                       np.polynomial.chebyshev.chebval(x,coeff),atol=2e-12,rtol=2e-12)
        self.assertEqual(balanced(0.,[1e-38],0.),1e-38)
    def test_frozen_max_model_full_probe_equivalence(self):
        for row in generate():
            if row["model"]["id"] not in ("bench_helper_0040","bench_helper_0044"):continue
            g=row["model"];before=copy.deepcopy(g)
            r=prepare(g,PROFILE_SHA256,configuration("seal-cpu-eva-w45-v1"))
            src=lower(r,balanced_chebyshev=True)
            validate_candidate(dict(schema=1,request_id=r["request_id"],hecate_source=src),r)
            for inputs in samples(g,16):
                value=public_execute(src,r,inputs)[0][:1];expected=evaluate(g,inputs)["output0"].reshape(-1)
                np.testing.assert_allclose(value,expected,atol=1e-12,rtol=1e-12)
                np.testing.assert_allclose(expected,reference(g,inputs)["output0"].reshape(-1),atol=1e-12,rtol=1e-12)
            self.assertEqual(before,g)
    def test_default_and_strategies_are_explicit(self):
        from benchmark_suite import Builder
        b=Builder([(3,)]);y=b.node("polynomial",["input0",b.const([.1,.2,.3])],basis="chebyshev")
        r=prepare(b.finish(y),PROFILE_SHA256)
        self.assertEqual(lower(r),lower(r,balanced_chebyshev=False))
        source=lower(r,packed_prefix=True,balanced_chebyshev=True)
        validate_candidate(dict(schema=1,request_id=r["request_id"],hecate_source=source),r)
        for inputs in samples(r["model"],4):
            np.testing.assert_allclose(public_execute(source,r,inputs)[0][:3],
                evaluate(r["model"],inputs)["output0"].reshape(-1),atol=1e-12,rtol=1e-12)

if __name__=="__main__":unittest.main()
