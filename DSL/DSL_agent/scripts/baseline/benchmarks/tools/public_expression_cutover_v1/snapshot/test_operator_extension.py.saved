import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from benchmark_graph import validate
from model_decomposition import decompose, FORMAT
from model_resources import identity, resolve
from operator_reference import evaluate
from benchmark_math import evaluate as math_ref
from benchmark_torch import evaluate as torch_ref
from model_weight_package import load

def graph(op,shape=(2,),attrs=None,constants=None,refs=None):
    return dict(format=FORMAT,id="extension",inputs=[dict(name="x",shape=list(shape))],
                constants=constants or {},nodes=[dict(id="operation",op=op,inputs=refs or ["x"],
                attrs=attrs or {},outputs=["y"])],outputs=[dict(name="result",value="y")])

RS=dict(method="newton",domain=[.1,4.],iterations=2)
POLY=dict(method="polynomial",basis="power",domain=[-2.,2.],coefficients=[.2,.5,.1])

class OperatorExtensionTests(unittest.TestCase):
    def compare(self,g,x,profile="legacy-v1"):
        lowered,binding=decompose(g,resource_profile=profile)
        expected=evaluate(g,{"x":x})
        for runner in (math_ref,torch_ref):
            actual=runner(lowered,{"x":x},resource_profile=profile)
            np.testing.assert_allclose(actual["result"],expected["result"],rtol=1e-12,atol=1e-12)
        self.assertEqual(set(binding["source_map"].values()),{"operation"})
    def test_three_contexts_each_operator(self):
        for shape in ((2,),(2,2),(1,2,2)):
            x=np.arange(np.prod(shape),dtype=np.float64).reshape(shape)/8-.2
            cases=[
                graph("variance",shape,dict(axes=[-1],keepdims=True)),
                graph("layer_norm",shape,dict(normalized_shape=[shape[-1]],eps=.2,approximation=RS),
                      {"gain":[.7]*shape[-1],"bias":[.1]*shape[-1]},["x","gain","bias"]),
                graph("dropout",shape,dict(training=False)),
            ]
            cases += [graph(op,shape,dict(approximation=POLY)) for op in ("relu","gelu","silu","sigmoid","tanh")]
            for g in cases:
                with self.subTest(op=g["nodes"][0]["op"],shape=shape):self.compare(g,x)
            for op,approx in (("sqrt",RS),("log",dict(POLY,domain=[.1,2.]))):
                with self.subTest(op=op,shape=shape):self.compare(graph(op,shape,dict(approximation=approx)),x+.5)
        for shape in ((1,2,3),(2,2,2),(1,2,2,2)):
            x=np.arange(np.prod(shape),dtype=np.float64).reshape(shape)/16-.2
            for op in ("group_norm","instance_norm"):
                a=dict(eps=.2,approximation=RS)
                if op=="group_norm":a["groups"]=1
                with self.subTest(op=op,shape=shape):
                    self.compare(graph(op,shape,a,{"gain":[.8,1.2],"bias":[.1,-.1]},["x","gain","bias"]),x)
            self.compare(graph("global_avg_pool",shape),x)
    def test_batched_matmul(self):
        for shapes in (((2,2,3),(2,3,2)),((1,2,3),(2,3,2)),((2,3),(2,3,2))):
            g=graph("matmul",shapes[0],refs=["x","b"])
            g["inputs"].append(dict(name="b",shape=list(shapes[1])))
            x={i["name"]:np.arange(np.prod(i["shape"]),dtype=np.float64).reshape(i["shape"])/16 for i in g["inputs"]}
            lowered,_=decompose(g)
            for runner in (math_ref,torch_ref):
                np.testing.assert_allclose(runner(lowered,x)["result"],evaluate(g,x)["result"],rtol=1e-12,atol=1e-12)
    def test_approximation_and_execution_separate(self):
        g=graph("relu",attrs=dict(approximation=POLY));x={"x":np.array([-.5,.5])}
        self.assertGreater(np.max(abs(evaluate(g,x)["result"]-evaluate(g,x,ideal=True)["result"])),.01)
    def test_rejections(self):
        cases=[graph("dropout",attrs=dict(training=True)),
               graph("relu"),graph("layer_norm",attrs={}),
               graph("log",attrs=dict(approximation=POLY)),
               graph("variance",attrs=dict(axes=[0,0],keepdims=True))]
        for g in cases:
            with self.subTest(op=g["nodes"][0]["op"]),self.assertRaisesRegex(ValueError,"node operation"):decompose(g)
    def test_domain_rejection(self):
        g=graph("relu",attrs=dict(approximation=POLY))
        with self.assertRaisesRegex(ValueError,"outside"):evaluate(g,{"x":np.array([0.,3.])})
    def test_medium_admission_is_explicit(self):
        g=graph("square",(512,))
        with self.assertRaises(ValueError):decompose(g)
        lowered,b=decompose(g,resource_profile="medium-v1")
        self.assertFalse(b["runtime_layout_validated"])
        # Capacity graph v2 binds its explicit profile in the model itself.
        self.assertEqual(lowered["resource_profile"],"medium-v1")
        validate(lowered)
        unbound=copy.deepcopy(lowered);unbound["format"]="poseidon-model-graph-v1"
        unbound.pop("resource_profile")
        with self.assertRaises(ValueError):validate(unbound)
        x=np.linspace(-1,1,512)
        self.compare(g,x,"medium-v1")
    def test_resnet_shape_and_resource_identity(self):
        g=graph("square",(1,3,32,32))
        lowered,b=decompose(g,resource_profile="resnet20-v1")
        self.assertEqual(b["resource_profile"],identity("resnet20-v1"))
        with self.assertRaises(ValueError):decompose(g,resource_profile="medium-v1")
        with self.assertRaises(ValueError):resolve("arbitrary")
    def test_old_execution_binding_rejected(self):
        from benchmark_runner import DEFAULT,load
        with self.assertRaisesRegex(ValueError,"Frozen benchmark source changed"):
            load(DEFAULT)

    def test_aggregate_constants_rejected(self):
        g=graph("square",constants={f"c{i}":[0.]*4096 for i in range(17)})
        with self.assertRaisesRegex(ValueError,"Total public"):decompose(g,resource_profile="medium-v1")

class WeightPackageTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.array=np.array([[1.,2.],[3.,4.]],dtype="<f8")
        np.save(self.root/"weight.npy",self.array,allow_pickle=False)
        raw=(self.root/"weight.npy").read_bytes()
        self.spec=dict(format="poseidon-model-package-v1",resource_profile="medium-v1",
                       model=graph("linear",refs=["x","w"]),
                       weights={"w":dict(file="weight.npy",dtype="<f8",shape=[2,2],
                       bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())})
    def tearDown(self):self.tmp.cleanup()
    def run_load(self):
        path=self.root/"manifest.json";path.write_text(json.dumps(self.spec));return load(path)
    def test_load_and_bind(self):
        model,profile,binding=self.run_load()
        self.assertEqual(model["constants"]["w"],self.array.tolist())
        self.assertEqual(profile,"medium-v1");self.assertEqual(len(binding["binding_sha256"]),64)
        decompose(model,resource_profile=profile)
    def test_tampering(self):
        (self.root/"weight.npy").write_bytes(b"bad")
        with self.assertRaises(ValueError):self.run_load()
    def test_path_escape(self):
        self.spec["weights"]["w"]["file"]="../weight.npy"
        with self.assertRaises(ValueError):self.run_load()
    def test_symlink(self):
        (self.root/"link.npy").symlink_to(self.root/"weight.npy")
        self.spec["weights"]["w"]["file"]="link.npy"
        with self.assertRaises(OSError):self.run_load()
    def test_header_mismatch(self):
        self.spec["weights"]["w"]["shape"]=[4]
        with self.assertRaisesRegex(ValueError,"header"):self.run_load()
    def test_object_pickle_rejected(self):
        np.save(self.root/"weight.npy",np.array([object()],dtype=object))
        raw=(self.root/"weight.npy").read_bytes()
        self.spec["weights"]["w"].update(shape=[1],bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
        with self.assertRaisesRegex(ValueError,"header"):self.run_load()

if __name__=="__main__":unittest.main()
