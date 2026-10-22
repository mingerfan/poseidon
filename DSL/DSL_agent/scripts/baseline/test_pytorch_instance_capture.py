"""InstanceNorm input mapping: independent source and graph semantics."""
import copy, importlib.util, unittest
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from benchmark_graph import canonical, samples, digest
from benchmark_math import evaluate as mathematical
from benchmark_torch import evaluate as tensor_reference
from operator_reference import evaluate as high
from pytorch_model_capture import capture, functional_input_rules
from pytorch_approximation import check_record
from pytorch_source_contract import output_tree

spec=importlib.util.spec_from_file_location("instance_sources",Path(__file__).parent/"cases/trusted-pytorch-instance-v1/models.py")
fixtures=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixtures)

def configuration(defaults=None):
    return dict(format="poseidon-pytorch-approximations-v1",
                defaults=defaults or {}, modules=[], nodes=[])

def capture_source(module, inputs, outputs, profiles=None):
    return capture(module.double().eval(),inputs,list(outputs),
        approximations=profiles or configuration(),resource_profile="general-tensors-v1",
        dtype_policy="strict-float64-v1",preserve_output_structure=True,record_state_sources=True,
        capture_policy="signature-static-index-v1",approximation_policy="node-module-default-v1",
        matrix_policy="static-tensor-matmul-v1")

class InstanceCaptureTests(unittest.TestCase):
    def compare(self,family,context):
        module=fixtures.make_model(family,context)
        inputs=fixtures.source_specs(family,context);outputs=fixtures.source_outputs(context)
        config=fixtures.source_approximations(family)
        before={k:v.detach().clone() for k,v in module.state_dict().items()}
        a,low,binding=capture_source(module,inputs,outputs,config)
        self.assertEqual((a,low,binding),capture_source(module,inputs,outputs,config))
        check_record(binding["pytorch_capture"]["approximation_selection"],a,config)
        for name,value in before.items():self.assertTrue(torch.equal(value,module.state_dict()[name]))
        for values in samples(low,16):
            with torch.no_grad():value=module(torch.from_numpy(values["x"]))
            leaves,tree=output_tree(value,lambda v:isinstance(v,torch.Tensor),outputs,
                                   scalar_tensor=lambda v:v.ndim==0)
            self.assertEqual(tree,binding["pytorch_capture"]["output_structure"])
            ideal=high(a,values,ideal=True);approximate=high(a,values)
            for name,tensor in zip(outputs,leaves):
                np.testing.assert_allclose(tensor.detach().numpy(),ideal[name],atol=1e-12,rtol=1e-12)
            for ref in (mathematical,tensor_reference):
                actual=ref(low,values)
                for name in outputs:
                    np.testing.assert_allclose(actual[name],approximate[name],atol=2e-12,rtol=2e-12)
        return a,low,binding

    def test_module1_input_three_contexts(self):
        for n in range(3):self.compare("module1_input",n)
    def test_module2_input_three_contexts(self):
        for n in range(3):self.compare("module2_input",n)
    def test_module1_frozen_three_contexts(self):
        for n in range(3):
            a,_,b=self.compare("module1_frozen",n)
            self.assertFalse(b["pytorch_capture"]["approximation_selection"]["selections"])
            self.assertIn("batch_norm",[v["op"] for v in a["nodes"]])
    def test_module2_frozen_three_contexts(self):
        for n in range(3):self.compare("module2_frozen",n)
    def test_unbatched1_three_contexts(self):
        for n in range(3):self.compare("unbatched1",n)
    def test_unbatched2_three_contexts(self):
        for n in range(3):self.compare("unbatched2",n)
    def test_unbatched1_frozen_three_contexts(self):
        for n in range(3):self.compare("unbatched1_frozen",n)
    def test_unbatched2_frozen_three_contexts(self):
        for n in range(3):self.compare("unbatched2_frozen",n)
    def test_functional_input_three_contexts(self):
        for n in range(3):self.compare("functional_input",n)
    def test_functional_frozen_three_contexts(self):
        for n in range(3):self.compare("functional_frozen",n)
    def test_unbatched_scope_keeps_main_fx_identity(self):
        a,_,b=self.compare("unbatched2",2)
        nodes={v["id"]:v for v in a["nodes"]}
        for row in b["pytorch_capture"]["approximation_selection"]["selections"]:
            self.assertEqual(nodes[row["node_id"]]["op"],"instance_norm")
    def test_real_public_state_sources_are_preserved(self):
        _,_,b=self.compare("module2_frozen",2)
        self.assertEqual(set(b["pytorch_capture"]["state_constant_sources"].values()),
                         {"norm.weight","norm.bias","norm.running_mean","norm.running_var"})
    def test_missing_approximation_refused(self):
        with self.assertRaisesRegex(ValueError,"Explicit approximation"):
            capture_source(fixtures.make_model("module1_input",0),fixtures.source_specs("module1_input",0),["out"])
    def test_unbatched_functional_input_refused(self):
        class Model(torch.nn.Module):
            def forward(self,x):return F.instance_norm(x,None,None,use_input_stats=False)
        with self.assertRaises((ValueError,RuntimeError)):
            capture_source(Model(),[dict(name="x",shape=[2,3])],["out"])
    def test_input_statistics_state_mutation_refused(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__();self.register_buffer("mean",torch.zeros(2,dtype=torch.float64))
                self.register_buffer("variance",torch.ones(2,dtype=torch.float64))
            def forward(self,x):return F.instance_norm(x,self.mean,self.variance,use_input_stats=True)
        with self.assertRaisesRegex(ValueError,"State-mutating|cannot update"):
            capture_source(Model(),[dict(name="x",shape=[1,2,3])],["out"],
                           configuration(dict(rsqrt=dict(method="newton",domain=[1e-5,8.],iterations=1))))
    def test_encrypted_statistics_refused(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__();self.register_buffer("variance",torch.ones(2,dtype=torch.float64))
            def forward(self,x,mean):return F.instance_norm(x,mean,self.variance,use_input_stats=False)
        with self.assertRaisesRegex(ValueError,"Public InstanceNorm"):
            capture_source(Model(),[dict(name="x",shape=[1,2,3]),dict(name="mean",shape=[2])],["out"])
    def test_encrypted_affine_refused(self):
        class Model(torch.nn.Module):
            def forward(self,x,weight):return F.instance_norm(x,None,None,weight=weight)
        with self.assertRaisesRegex(ValueError,"public"):
            capture_source(Model(),[dict(name="x",shape=[1,2,3]),dict(name="weight",shape=[2])],["out"],
                           configuration(dict(rsqrt=dict(method="newton",domain=[1e-5,8.],iterations=1))))
    def test_frozen_statistics_missing_refused(self):
        class Model(torch.nn.Module):
            def forward(self,x):return F.instance_norm(x,None,None,use_input_stats=False)
        with self.assertRaises((ValueError,RuntimeError)):
            capture_source(Model(),[dict(name="x",shape=[1,2,3])],["out"])
    def test_module_channel_contract_refused(self):
        class Model(torch.nn.Module):
            def __init__(self):super().__init__();self.norm=torch.nn.InstanceNorm1d(3,affine=False)
            def forward(self,x):return self.norm(x)
        with self.assertRaisesRegex(ValueError,"declared channel"):
            capture_source(Model(),[dict(name="x",shape=[1,2,3])],["out"],
                           configuration(dict(rsqrt=dict(method="newton",domain=[1e-5,8.],iterations=1))))
    def test_fixture_generator_reproduces_complete_packages(self):
        import tempfile,importlib.util
        path=Path(__file__).parent/"benchmarks/tools/generate_instance_sources.py"
        loader=importlib.util.spec_from_file_location("instance_generator",path)
        generator=importlib.util.module_from_spec(loader);loader.loader.exec_module(generator)
        base=Path(__file__).parent/"cases/trusted-pytorch-instance-v1"
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/"generated";generator.generate(out)
            expected={p.relative_to(base).as_posix():p.read_bytes() for p in base.rglob("*") if p.is_file()}
            actual={p.relative_to(out).as_posix():p.read_bytes() for p in out.rglob("*") if p.is_file()}
            self.assertEqual(expected,actual)

    def test_complete_state_reload_preserves_each_capture(self):
        import json
        from pytorch_weight_package import read,load_state
        base=Path(__file__).parent/"cases/trusted-pytorch-instance-v1"
        for path in sorted(base.glob("*/manifest.json")):
            item=json.loads(path.read_text());family,n=item["factory"]["args"]
            module=fixtures.make_model(family,n)
            original=capture_source(module,item["inputs"],item["outputs"],item["approximations"])
            arrays,_=read(path.parent,item["weights"],resource_profile="general-tensors-v1")
            self.assertEqual(set(arrays),set(module.state_dict()))
            load_state(module,arrays,item["weights"])
            self.assertEqual(original,capture_source(module,item["inputs"],item["outputs"],item["approximations"]))

    def test_state_payload_hash_tampering_refused(self):
        import json
        from pytorch_weight_package import decode
        base=Path(__file__).parent/"cases/trusted-pytorch-instance-v1/functional_frozen_1"
        item=json.loads((base/"manifest.json").read_text())["weights"]["running_var"]
        raw=(base/item["file"]).read_bytes()
        changed=raw[:-1]+bytes([raw[-1]^1])
        with self.assertRaisesRegex(ValueError,"size/hash mismatch"):decode(changed,item)

    def test_registry_states_are_declarations(self):
        row=functional_input_rules()["instance_norm"]
        self.assertEqual(row["version"],"poseidon-pytorch-instance-normalization-v1")
        self.assertEqual(row["module_ranks"],dict(one_dimensional=[2,3],two_dimensional=[3,4]))
        self.assertNotIn("encrypted_execution",row)

if __name__=="__main__":unittest.main(verbosity=2)
