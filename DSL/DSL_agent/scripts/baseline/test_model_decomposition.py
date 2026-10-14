import ast,copy,hashlib,json,unittest
from model_decomposition import FORMAT,decompose,REGISTRY
from restricted_model_python import parse,FORMAT_PY
def graph(op,inputs=None,constants=None,attrs=None,refs=None):
    return dict(format=FORMAT,id="example",inputs=inputs or [dict(name="x",shape=[2])],
                constants=constants or {},nodes=[dict(id="n",op=op,inputs=refs or ["x"],
                attrs=attrs or {},outputs=["y"])],outputs=[dict(name="output",value="y")])
def manifest(files,inputs=None,constants=None,entry="main.forward"):
    return dict(format=FORMAT_PY,id="python_example",files={p:hashlib.sha256(s.encode()).hexdigest() for p,s in files.items()},
                entry=entry,inputs=inputs or [dict(name="x",shape=[2])],constants=constants or {},outputs=["output"])
class DecompositionTests(unittest.TestCase):
    def test_core_passthrough_and_provenance(self):
        g=graph("square");original=copy.deepcopy(g);out,b=decompose(g)
        self.assertEqual(g,original);self.assertEqual(out["nodes"][0]["op"],"square")
        self.assertEqual(set(b["source_map"].values()),{"n"})
    def test_missing_approximation_rejected(self):
        with self.assertRaises(ValueError):decompose(graph("rsqrt"))
    def test_bootstrap_not_silently_lowered(self):
        with self.assertRaisesRegex(ValueError,"Unregistered"):decompose(graph("bootstrap"))
    def test_domain_rejected(self):
        with self.assertRaisesRegex(ValueError,"Positive"):
            decompose(graph("rsqrt",attrs={"approximation":{"method":"newton","domain":[0,2],"iterations":2}}))
    def test_nonfinite_rejected(self):
        with self.assertRaises(ValueError):decompose(graph("square",constants={"bad":float("nan")}))
    def test_forward_reference_rejected(self):
        with self.assertRaises(ValueError):decompose(graph("square",refs=["y"]))
    def test_duplicate_value_rejected(self):
        g=graph("square");g["nodes"][0]["outputs"]=["x"]
        with self.assertRaises(ValueError):decompose(g)
    def test_matmul_resource_budget(self):
        g=graph("matmul",inputs=[dict(name="x",shape=[8,8]),dict(name="b",shape=[8,8])],refs=["x","b"])
        with self.assertRaisesRegex(ValueError,"shape budget"):decompose(g)
    def test_registry_immutable(self):
        with self.assertRaises(TypeError):REGISTRY["unsafe"]=None

class PythonTests(unittest.TestCase):
    def test_multifile_helper_loop_matches_structure(self):
        files={"main.py":"from helper import activation\ndef forward(x):\n    for i in range(2):\n        x = activation(x)\n    return x\n",
               "helper.py":"def activation(x):\n    return x*x\n"}
        original,lowered,binding=parse(manifest(files),files)
        self.assertEqual([n["op"] for n in original["nodes"]],["multiply","multiply"])
        self.assertEqual(len(binding["python_source_hashes"]),2)
    def test_functional_linear(self):
        f={"main.py":"import torch.nn.functional as F\ndef forward(x):\n    y=F.linear(x, weight)\n    return y*y\n"}
        g,_,_=parse(manifest(f,constants={"weight":[[1.,0.],[0.,1.]]}),f)
        self.assertEqual([n["op"] for n in g["nodes"]],["linear","multiply"])
    def reject(self,code):
        f={"main.py":code}
        with self.assertRaises((ValueError,SyntaxError)):parse(manifest(f),f)
    def test_no_top_level_execution(self):self.reject("open('/tmp/should-not-exist','w')\ndef forward(x):\n return x\n")
    def test_no_dynamic_import(self):self.reject("def forward(x):\n return __import__('os').system('true')\n")
    def test_no_external_import(self):self.reject("import os\ndef forward(x):\n return x*x\n")
    def test_no_encrypted_condition(self):self.reject("def forward(x):\n if x:\n  return x*x\n return x\n")
    def test_no_unbounded_loop(self):self.reject("def forward(x):\n for i in range(1000):\n  x=x*x\n return x\n")
    def test_no_recursion(self):self.reject("def forward(x):\n return forward(x)\n")
    def test_no_silent_softmax(self):self.reject("import torch\ndef forward(x):\n return torch.softmax(x,dim=0)\n")
    def test_no_callable_shadow(self):self.reject("import torch\ndef forward(torch):\n return torch.square(torch)\n")
    def test_hash_tampering(self):
        f={"main.py":"def forward(x):\n return x*x\n"};m=manifest(f);f["main.py"]+="\n"
        with self.assertRaisesRegex(ValueError,"hash"):parse(m,f)
    def test_unlisted_file(self):
        f={"main.py":"def forward(x):\n return x*x\n"};m=manifest(f);f["other.py"]=""
        with self.assertRaises(ValueError):parse(m,f)

class ReferenceTests(unittest.TestCase):
    def compare(self,g,inputs):
        import numpy as np
        from operator_reference import evaluate
        from benchmark_math import evaluate as core
        from benchmark_torch import evaluate as torch_ref
        lowered,_=decompose(g);expected=evaluate(g,inputs)
        for actual in (core(lowered,inputs),torch_ref(lowered,inputs)):
            for n in expected:np.testing.assert_allclose(actual[n],expected[n],atol=1e-12,rtol=1e-12)
    def test_cipher_matmul(self):
        import numpy as np
        g=graph("matmul",inputs=[dict(name="x",shape=[2,3]),dict(name="b",shape=[3,2])],refs=["x","b"])
        self.compare(g,{"x":np.arange(6,dtype=np.float64).reshape(2,3)/8,"b":np.arange(6,dtype=np.float64).reshape(3,2)/7})
    def test_rmsnorm(self):
        import numpy as np
        g=graph("rms_norm",constants={"gain":[1.,.5]},refs=["x","gain"],
                attrs={"eps":.25,"approximation":{"method":"newton","domain":[.25,2.],"iterations":2}})
        self.compare(g,{"x":np.array([-.3,.7],dtype=np.float64)})
    def test_softmax(self):
        import numpy as np
        g=graph("softmax",attrs={"axis":0,"exp":{"method":"taylor","domain":[-2.,2.],"degree":6},
                                 "reciprocal":{"method":"goldschmidt","domain":[.1,8.],"iterations":3}})
        self.compare(g,{"x":np.array([-.3,.7],dtype=np.float64)})
    def test_rope(self):
        import numpy as np
        self.compare(graph("rope",attrs={"position":2,"theta":100.}),{"x":np.array([-.3,.7],dtype=np.float64)})
    def test_out_of_domain_not_clipped(self):
        import numpy as np
        from operator_reference import evaluate
        g=graph("exp",attrs={"approximation":{"method":"taylor","domain":[-.1,.1],"degree":4}})
        with self.assertRaisesRegex(ValueError,"outside"):evaluate(g,{"x":np.array([1.,2.])})
if __name__=="__main__":unittest.main()
