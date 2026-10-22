"""Static matrix normalization versus independent references and version boundaries."""
import copy,json,sys,unittest
from pathlib import Path
import numpy as np
import torch
from benchmark_graph import samples,digest
from benchmark_math import evaluate as mathematical
from benchmark_torch import evaluate as independent
from pytorch_model_capture import capture
from pytorch_matrix_plan import POLICY,plan,check_record
from pytorch_source_contract import output_tree,validate_manifest
from hecate_python_env import ROOT
CASES=ROOT/"scripts/baseline/cases/trusted-pytorch-matrix-v7"
sys.path.insert(0,str(CASES))
from models import PARTITIONS,make_model,signature,Public,Cipher

class MatrixTensorCaptureTests(unittest.TestCase):
    def capture_model(self,module,inputs,names=("out",)):
        return capture(module,inputs,list(names),resource_profile="general-tensors-v1",
            dtype_policy="strict-float64-v1",preserve_output_structure=True,capture_policy="signature-static-index-v1",
            matrix_policy=POLICY)
    def test_ten_partitions_three_contexts_original_and_two_references(self):
        self.assertEqual(len(PARTITIONS),10)
        for partition,contexts in PARTITIONS.items():
            for index in range(3):
                with self.subTest(partition=partition,index=index):
                    structure="dict" if partition=="cipher_batch" and index==1 else "tuple" if partition=="public_batch_l" and index==1 else "tensor"
                    names=["out","extra"] if structure!="tensor" else ["out"]
                    model=make_model(partition,index,structure)
                    original,lowered,binding=self.capture_model(model,signature(partition,index),names)
                    self.assertEqual((original,lowered,binding),self.capture_model(model,signature(partition,index),names))
                    record=binding["pytorch_capture"]["matrix_capture"]
                    check_record(record,original,resource_profile="general-tensors-v1")
                    self.assertEqual(len(record["entries"]),1)
                    for inputs in samples(lowered,16):
                        raw=model(*(torch.from_numpy(inputs[n["name"]]) for n in original["inputs"]))
                        leaves,tree=output_tree(raw,lambda v:isinstance(v,torch.Tensor),names,scalar_tensor=lambda v:v.ndim==0)
                        self.assertEqual(tree,binding["pytorch_capture"]["output_structure"])
                        expected={name:v.detach().numpy().reshape(1) if v.ndim==0 else v.detach().numpy()
                                  for name,v in zip(names,leaves)}
                        for reference in (mathematical,independent):
                            actual=reference(lowered,inputs)
                            for name in names:
                                self.assertEqual(actual[name].shape,expected[name].shape)
                                np.testing.assert_allclose(actual[name],expected[name],atol=1e-12,rtol=1e-12)
    def test_normalized_graph_compounds_static_shape_queries_and_indices(self):
        class Model(torch.nn.Module):
            def forward(self,x,y):
                z=x@y
                result=z[:,1::2]
                for _ in range(z.shape[-1]):result=result+z[:,1::2]*.125
                return result,result.sum()
        spec=[dict(name="x",shape=[2,3]),dict(name="y",shape=[3,4])]
        original,lowered,binding=self.capture_model(Model(),spec,["value","total"])
        self.assertEqual(len(binding["pytorch_capture"]["index_capture"]["entries"]),5)
        for inputs in samples(lowered,16):
            expected=Model()(torch.from_numpy(inputs["x"]),torch.from_numpy(inputs["y"]))
            for reference in (mathematical,independent):
                actual=reference(lowered,inputs)
                np.testing.assert_allclose(actual["value"],expected[0].numpy(),atol=1e-12,rtol=1e-12)
                np.testing.assert_allclose(actual["total"],expected[1].numpy().reshape(1),atol=1e-12,rtol=1e-12)
    def test_old_format_policies_and_matrix_rejections_remain(self):
        from test_pytorch_matrix_capture import Product
        # This test has a portable, self-contained mathematical assertion; exact old bytes are audited by the driver.
        for kind in ("dot","mm","bmm","matmul"):
            shapes=([3],[3]) if kind=="dot" else ([2,2,3],[2,3,2]) if kind=="bmm" else ([2,3],[3,2])
            inputs=[dict(name="x",shape=shapes[0]),dict(name="y",shape=shapes[1])]
            original,lowered,binding=capture(Product(kind,0),inputs,["out"])
            self.assertNotIn("matrix_capture",binding["pytorch_capture"])
        for partition in ("cipher_vv","cipher_mv","cipher_vm","cipher_batch","public_vector_l","public_batch_r"):
            with self.subTest(partition=partition),self.assertRaises(ValueError):
                capture(make_model(partition,0),signature(partition,0),["out"],
                    resource_profile="general-tensors-v1",preserve_output_structure=True,capture_policy="signature-static-index-v1")
    def test_matrix_plan_requires_explicit_v7_policy(self):
        with self.assertRaisesRegex(ValueError,"v7"):
            capture(Cipher(0,"tensor"),signature("cipher_vv",0),["out"],matrix_policy=POLICY)
        with self.assertRaisesRegex(ValueError,"Matrix capture policy"):
            self.capture_invalid_policy("unknown")
    def capture_invalid_policy(self,policy):
        return capture(Cipher(0,"tensor"),signature("cipher_vv",0),["out"],matrix_policy=policy)
    def test_rank_contraction_broadcast_empty_type_and_materialization_rejections(self):
        invalid=[
            ([2,3],[4,2],{}),([2,2,3],[3,3,2],{}),([],[2],{}),
            ([0,2],[2,2],{}),([True,2],[2,2],{}),([1]*33,[1],{}),
            ([4,64],[64,4],dict(resource_profile="legacy-v1")),
            ([2,2,3],[3,3,2],dict(operation="bmm",resource_profile="medium-v1")),
            ([2,2],[2,2],dict(operation="dot")),([2,2,2],[2,2,2],dict(operation="mm"))]
        for a,b,opts in invalid:
            with self.subTest(a=a,b=b),self.assertRaises(ValueError):plan(a,b,**opts)
    def test_intermediate_uses_tensor_budget_before_named_output_budget(self):
        value=plan([64,64],[64,128],public_side="right",resource_profile="general-tensors-v1")
        self.assertEqual(value["result_shape"],[64,128])
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("weight",(torch.arange(8192,dtype=torch.float64)%7).reshape(64,128)*.03125)
            def forward(self,x):return (x@self.weight).sum(-1)
        model=Model()
        original,lowered,binding=self.capture_model(model,[dict(name="x",shape=[64,64])])
        check_record(binding["pytorch_capture"]["matrix_capture"],original,resource_profile="general-tensors-v1")
        for inputs in samples(lowered,4):
            expected=model(torch.from_numpy(inputs["x"])).numpy()
            for reference in (mathematical,independent):
                np.testing.assert_allclose(reference(lowered,inputs)["out"],expected,atol=1e-12,rtol=1e-12)

    def test_record_tampering_rejected_even_with_updated_graph_digest(self):
        original,_,binding=self.capture_model(make_model("public_batch_r",1),signature("public_batch_r",1))
        record=binding["pytorch_capture"]["matrix_capture"]
        for mutate in [
            lambda v:v.update(version="unknown"),
            lambda v:v["entries"][0]["plan"]["result_shape"].append(1),
            lambda v:v["entries"][0]["plan"].update(public_side=None),
            lambda v:v["entries"][0]["node_ids"].append(v["entries"][0]["node_ids"][0]),
            lambda v:v["entries"][0].update(constant=None),
            lambda v:v["entries"][0]["sources"].reverse()]:
            bad=copy.deepcopy(record);mutate(bad)
            with self.assertRaises(ValueError):check_record(bad,original,resource_profile="general-tensors-v1")
        bad=copy.deepcopy(original);entry=record["entries"][0]
        data=np.asarray(bad["constants"][entry["constant"]]);data.flat[0]+=.125
        bad["constants"][entry["constant"]]=data.tolist()
        updated=copy.deepcopy(record);updated["graph_sha256"]=digest(bad)
        with self.assertRaisesRegex(ValueError,"derived constant"):
            check_record(updated,bad,resource_profile="general-tensors-v1")
    def test_all_fixtures_have_versioned_weight_and_shape_binding(self):
        from trusted_pytorch_import import prepare,package_format
        from application_capabilities import input_sources
        manifests=sorted(CASES.glob("*/manifest.json"));self.assertEqual(len(manifests),30)
        for p in manifests:
            spec,_=prepare(p);validate_manifest(spec)
            self.assertEqual(spec["format"],"poseidon-trusted-pytorch-v7")
            self.assertEqual(package_format(spec),"poseidon-trusted-pytorch-provenance-v7")
        capability=input_sources()["trusted_pytorch"]
        self.assertIn("poseidon-trusted-pytorch-v7",capability["formats"])
        self.assertNotIn("encrypted_execution",capability["static_matrix"])
    def test_mm_bmm_and_dot_mappings_use_same_explicit_plan(self):
        from test_pytorch_matrix_capture import Product,CONTEXTS
        for kind in ("dot","mm","bmm"):
            for i,(a,b) in enumerate(CONTEXTS[kind]):
                original,lowered,binding=self.capture_model(Product(kind,i),[dict(name="x",shape=a),dict(name="y",shape=b)])
                self.assertEqual(binding["pytorch_capture"]["matrix_capture"]["entries"][0]["plan"]["operation"],kind)
                for values in samples(lowered,4):
                    actual=mathematical(lowered,values)["out"]
                    expected=Product(kind,i)(torch.from_numpy(values["x"]),torch.from_numpy(values["y"])).numpy()
                    np.testing.assert_allclose(actual,expected,atol=1e-12,rtol=1e-12)
if __name__=="__main__":unittest.main()
