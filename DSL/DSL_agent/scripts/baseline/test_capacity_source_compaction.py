"""Trusted scalar-source compaction retains calls, sharing and typing."""
import ast,unittest
import numpy as np
from benchmark_graph import samples
from benchmark_math import evaluate
from component_contract import prepare_task
from unified_capacity_lowering import compact_scalar_source
from test_unified_capacity_linear import interpreted_output

def model(twice=False):
    nodes=[dict(id="sq",op="square",inputs=["x"],attrs={},outputs=["s"]),
           dict(id="double",op="add",inputs=["s","s"],attrs={},outputs=["d"])]
    if twice:nodes.append(dict(id="twice",op="add",inputs=["d","d"],attrs={},outputs=["t"]))
    return dict(format="poseidon-model-graph-v2",resource_profile="medium-v1",id="compact",
        inputs=[dict(name="x",shape=[4])],constants={},nodes=nodes,
        outputs=[dict(name="result",value="t" if twice else "d")])

class ScalarSourceTests(unittest.TestCase):
    def compare(self,source,g):
        compact=compact_scalar_source(source);r=prepare_task(g,dict(chunk_period=256))
        for inputs in samples(g,16):
            reference=evaluate(g,inputs,resource_profile="medium-v1")["result"]
            before=interpreted_output(source,r,inputs)
            after=interpreted_output(compact,r,inputs)
            np.testing.assert_allclose(after,before,atol=0,rtol=0)
            np.testing.assert_allclose(after,reference,atol=1e-12,rtol=1e-12)
        return compact

    def test_single_call_inlined_but_shared_square_not_duplicated(self):
        src='''@hc.func("c")
def helper(a):
    v0 = a * a
    v1 = v0 + v0
    return v1
@hc.func("c,c")
def golden(x, zero_ct):
    v0 = helper(x)
    return [v0]
'''
        compact=self.compare(src,model())
        self.assertEqual(len(ast.parse(compact).body),1)
        self.assertEqual(sum(type(n) is ast.Mult for n in ast.walk(ast.parse(compact))),1)
        self.assertLess(len(list(ast.walk(ast.parse(compact)))),len(list(ast.walk(ast.parse(src)))))

    def test_repeated_calls_remain_independent_and_typed(self):
        src='''@hc.func("c")
def helper(a):
    v0 = a * a
    v1 = v0 + v0
    return v1
@hc.func("c,c")
def golden(x, zero_ct):
    v0 = helper(x)
    v1 = helper(x)
    v2 = v0 + v1
    return [v2]
'''
        compact=self.compare(src,model(True))
        self.assertEqual(len(ast.parse(compact).body),2)
        self.assertEqual(sum(type(n) is ast.Call and type(n.func) is ast.Name and n.func.id=="helper"
                             for n in ast.walk(ast.parse(compact))),2)

    def test_non_scalar_helper_not_accepted_as_trusted_emitter_output(self):
        with self.assertRaisesRegex(ValueError,"scalar return"):
            compact_scalar_source('''@hc.func("c")
def helper(a):
    return [a, a]
@hc.func("c,c")
def golden(x, zero_ct):
    v0 = helper(x)
    return v0
''')

if __name__=="__main__":unittest.main()
