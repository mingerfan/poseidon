"""V31 public state transforms: immutable source, data-only replay and legacy gates."""
import copy,hashlib,tempfile,unittest
from pathlib import Path
import numpy as np
import torch
from benchmark_graph import digest
from pytorch_public_expression_flow import _arrays
from pytorch_public_expression_capture import prepare,check_record,derived_arrays,wrapped_metadata
from pytorch_weight_package import export_state,binding
from pytorch_public_tensor_transform_source import SOURCE_FORMAT, CAPTURE_FORMAT
from pytorch_ordered_contract import package_version,reference_version

class TransformModule(torch.nn.Module):
    def __init__(self,kind,context=0,dtype=torch.float64):
        super().__init__(); self.kind=kind;self.context=context;self.retain_float32=dtype is torch.float32
        value=torch.tensor([[.125000017,-.375000019],[.625000023,.875000029]],dtype=torch.float64)
        if dtype in (torch.int32,torch.int64):value=torch.tensor([[1,-2],[3,4]],dtype=dtype)
        elif dtype is torch.bool:value=torch.tensor([[True,False],[False,True]],dtype=dtype)
        else:value=value.to(dtype)
        self.register_buffer("public",value)
    def forward(self,x):
        w=self.public
        if self.retain_float32:w=w.float()
        if self.kind=="repeat":w=w.repeat(1,2)
        elif self.kind=="tile":w=torch.tile(w,(2,1))
        elif self.kind=="repeat_interleave":w=w.repeat_interleave(2,dim=-1)
        elif self.kind=="flip":w=torch.flip(w,(-1,))
        else:w=torch.roll(w,(1,-1),(0,-1))
        w=w.double()
        if self.context==1:return x.flip((-1,))*w+x
        y=x+w
        if self.context==2:return {"primary":y,"saved":[x,None]}
        return y

def capture(kind="repeat",context=0,dtype=torch.float64,version=SOURCE_FORMAT):
    module=TransformModule(kind,context,dtype)
    shape=[4,2] if kind=="tile" else [2,4] if kind in ("repeat","repeat_interleave") else [2,2]
    with tempfile.TemporaryDirectory()as path:
        folder=Path(path);(folder/"weights").mkdir();(folder/"capture").mkdir()
        weights=export_state(module,folder/"weights",resource_profile="general-tensors-v1")
        source="def build():\n    return None\n"
        spec=dict(format=version,files={"model.py":hashlib.sha256(source.encode()).hexdigest()},
            entry="model.build",inputs=[dict(name="x",shape=shape)],
            call=dict(format="poseidon-pytorch-call-v1",args=[dict(kind="tensor",name="x")],kwargs={}),
            outputs=["y","saved"]if context==2 else["y"],factory=dict(args=[],kwargs={}),
            weights=weights,dtype_policy="convert-float64-v1"if dtype is torch.float32 else"strict-float64-v1",
            approximations=dict(format="poseidon-pytorch-approximations-v1",nodes=[],modules=[],defaults={}))
        arrays=_arrays(module)
        record=prepare(module,spec,folder/"capture",expected_weight_binding=binding(arrays,weights))
        check_record(record,spec,arrays,expected_record_binding=record["binding"])
        return module,spec,arrays,record

class PublicTransformCaptureTests(unittest.TestCase):
    def test_five_operations_three_contexts_have_original_reference_and_replay(self):
        for kind in ("repeat","tile","repeat_interleave","flip","roll"):
            for context in range(3):
                with self.subTest(kind=kind,context=context):
                    module,spec,arrays,record=capture(kind,context)
                    self.assertEqual(record["format"],CAPTURE_FORMAT)
                    self.assertEqual(record["original_reference"]["status"],"passed")
                    self.assertEqual(record["original_reference"]["groups"],16)
                    flow=record["public_expression_flow"]
                    self.assertEqual(flow["entries"][0]["plan"]["operation"],kind)
                    self.assertEqual(record["static_tensor_transform_flow"]["source_inventory"],
                                     flow["normalized_inventory"])
                    twice=[derived_arrays(record,spec,arrays,resource_profile="general-tensors-v1")for _ in range(2)]
                    self.assertEqual(set(twice[0]),set(twice[1]))
                    for name in twice[0]:np.testing.assert_array_equal(twice[0][name],twice[1][name])
                    np.testing.assert_array_equal(_arrays(module)["public"],arrays["public"])
    def test_public_logical_dtypes_remain_bound(self):
        for dtype in (torch.float32,torch.float64,torch.int32,torch.int64,torch.bool):
            with self.subTest(dtype=dtype):
                _,_,arrays,record=capture("roll",1,dtype)
                first=next(e["plan"]for e in record["public_expression_flow"]["entries"]if e["plan"]["operation"]=="roll")
                self.assertEqual(first["input"]["dtype"],arrays["public"].dtype.str)
                self.assertEqual(first["logical_output"]["dtype"],arrays["public"].dtype.str)
                self.assertEqual(record["original_reference"]["status"],"passed")
    def test_named_none_outputs_and_versions(self):
        _,spec,_,record=capture("flip",2)
        self.assertEqual(package_version(spec),"poseidon-trusted-pytorch-provenance-v31")
        self.assertEqual(reference_version(spec),"poseidon-trusted-pytorch-reference-v31")
        tree=wrapped_metadata(record)["output_structure"]
        self.assertEqual(tree["items"][1]["value"]["items"][1],{"kind":"none"})
    def test_legacy_public_transform_rejects(self):
        for kind in ("repeat","flip","roll"):
            with self.subTest(kind=kind),self.assertRaises(ValueError):
                capture(kind,version="poseidon-trusted-pytorch-v30")
    def test_original_arrays_and_declared_version_cannot_change(self):
        _,spec,arrays,record=capture()
        for changed_spec,changed_arrays in [
            ({**spec,"format":"poseidon-trusted-pytorch-v30"},arrays),
            (spec,{"public":arrays["public"]+1})]:
            with self.assertRaises(ValueError):
                check_record(record,changed_spec,changed_arrays,expected_record_binding=record["binding"])
    def test_rebound_transform_proof_cannot_change_semantics(self):
        _,spec,arrays,record=capture("roll")
        altered=copy.deepcopy(record)
        plan=altered["public_expression_flow"]["entries"][0]["plan"]
        plan["logical_output"]["dtype"]="<i8"
        for piece in (plan,altered["public_expression_flow"],altered):
            piece["binding"]=digest({k:v for k,v in piece.items()if k!="binding"})
        with self.assertRaises(ValueError):
            check_record(altered,spec,arrays,expected_record_binding=altered["binding"])
    def test_rebound_cipher_bridge_rejects(self):
        _,spec,arrays,record=capture("repeat",1)
        altered=copy.deepcopy(record);flow=altered["static_tensor_transform_flow"]
        flow["source_inventory"]=copy.deepcopy(altered["public_expression_flow"]["source_inventory"])
        flow["binding"]=digest({k:v for k,v in flow.items()if k!="binding"})
        altered["binding"]=digest({k:v for k,v in altered.items()if k!="binding"})
        with self.assertRaises(ValueError):
            check_record(altered,spec,arrays,expected_record_binding=altered["binding"])
    def test_capability_lists_explicit_version_without_promoting_default(self):
        from application_capabilities import input_sources
        from pytorch_model_intake import SOURCE_FORMATS
        self.assertIn(SOURCE_FORMAT,SOURCE_FORMATS)
        data=input_sources()["trusted_pytorch"]
        self.assertIn(SOURCE_FORMAT,data["input_call"]["versions"])
        self.assertIn(SOURCE_FORMAT,data["source_package"]["versions"])
        self.assertIn(SOURCE_FORMAT,data["static_broadcast"]["versions"])
        self.assertEqual(data["public_tensor_transforms"]["version"],SOURCE_FORMAT)
        self.assertEqual(data["public_tensor_transforms"]["encrypted_status"],"unverified")

if __name__=="__main__":unittest.main()
