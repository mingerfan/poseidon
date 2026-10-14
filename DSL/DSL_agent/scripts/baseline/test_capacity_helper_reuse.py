"""Identical linear helpers retain independent inputs and unchanged budgets."""
import ast,unittest
import numpy as np
from component_contract import prepare_task
from unified_graph_lowering import candidate_source
from unified_capacity_constants import SPATIAL_VERSION
from unified_graph_contract import prepare
from compiler_configuration import configuration

def prior_request(g):
    config=configuration("poseidon-cpu-eva-w45-v1")
    return prepare(g,config["ckks_config_sha256"],config,chunk_period=256,
                   generation_guidance="explicit-v10",capacity_packing_version=SPATIAL_VERSION)
from benchmark_graph import samples
from benchmark_math import evaluate
from test_unified_capacity_linear import interpreted_output

def model():
    def node(n,op,refs,attrs=None):return dict(id=n,op=op,inputs=refs,attrs=attrs or {},outputs=[n+"_out"])
    weights=[[(j+i*65)%17*.002-.016 for j in range(65)] for i in range(4)]
    return dict(format="poseidon-model-graph-v2",resource_profile="medium-v1",id="helper_reuse",
        inputs=[dict(name=n,shape=[2,65]) for n in ("x","y")],constants=dict(w=weights,b=[.01,-.02,.03,-.04]),
        nodes=[node("left","linear",["x","w","b"]),node("right","linear",["y","w","b"]),
               node("stacked","stack",["left_out","right_out"],dict(axis=-1)),
               node("total","sum",["stacked_out"],dict(axes=[-1],keepdims=False)),
               node("average","mean",["total_out"],dict(axes=[1],keepdims=True)),
               node("joined","concat",["left_out","right_out"],dict(axis=1))],
        outputs=[dict(name="joined",value="joined_out"),dict(name="stacked",value="stacked_out"),
                 dict(name="average",value="average_out")])

class HelperReuseTests(unittest.TestCase):
    def test_source_with_repeated_linear_fits_original_ast_budget(self):
        request=prior_request(model())
        source,_=candidate_source(request);tree=ast.parse(source)
        self.assertLessEqual(sum(1 for _ in ast.walk(tree)),4096)
        bodies=[ast.dump(ast.Module(body=n.body,type_ignores=[])) for n in tree.body if isinstance(n,ast.FunctionDef)]
        self.assertEqual(len(bodies),len(set(bodies)))
    def test_independent_inputs_and_all_named_outputs(self):
        g=model();request=prior_request(g)
        source,_=candidate_source(request)
        for inputs in samples(g,16):
            result=evaluate(g,inputs,resource_profile="medium-v1")
            expected=np.concatenate([result[o["name"]].reshape(-1) for o in g["outputs"]])
            np.testing.assert_allclose(interpreted_output(source,request,inputs),expected,atol=1e-12,rtol=1e-12)
if __name__=="__main__":unittest.main()
