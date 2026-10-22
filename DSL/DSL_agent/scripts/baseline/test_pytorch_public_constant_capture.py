"""V32 constructor intake uses immutable public state and data-only replay."""
import copy,hashlib,tempfile,unittest
from pathlib import Path
import numpy as np
import torch
from benchmark_graph import digest
from pytorch_public_constants import OPERATIONS
from pytorch_public_constant_source import SOURCE_FORMAT,CAPTURE_FORMAT
from pytorch_public_expression_capture import prepare,check_record,derived_arrays,wrapped_metadata
from pytorch_public_expression_flow import _arrays
from pytorch_weight_package import export_state,binding
from pytorch_ordered_contract import package_version,reference_version

class Module(torch.nn.Module):
    def __init__(self,kind,context=0,dtype=None):
        super().__init__();self.kind=kind;self.context=context;self.dtype=dtype
        self.register_buffer("w",torch.tensor([[.175,-.225],[.375,.125]],dtype=torch.float64))
    def forward(self,x):
        kwargs={}if self.dtype is None else dict(dtype=self.dtype)
        if self.kind=="new_full":w=x.new_full(x.shape,-.375,**kwargs)
        elif self.kind.startswith("new_"):w=getattr(x,self.kind)(x.shape,**kwargs)
        elif self.kind=="full_like":w=torch.full_like(x,.375,**kwargs)
        else:w=getattr(torch,self.kind)(x,**kwargs)
        w=w.double()
        if self.context==0:return x+w+self.w
        if self.context==1:return (x.square()+w)*self.w+x
        return {"y":x*(w+.125)+self.w,"saved":[x,None]}
def capture(kind="full_like",context=0,version=SOURCE_FORMAT,dtype=None):
    module=Module(kind,context,dtype)
    with tempfile.TemporaryDirectory()as path:
        folder=Path(path);(folder/"weights").mkdir();(folder/"capture").mkdir()
        weights=export_state(module,folder/"weights",resource_profile="general-tensors-v1")
        spec=dict(format=version,files={"model.py":hashlib.sha256(b"def make(): return None").hexdigest()},
            entry="model.make",inputs=[dict(name="x",shape=[2,2])],
            call=dict(format="poseidon-pytorch-call-v1",args=[dict(kind="tensor",name="x")],kwargs={}),
            outputs=["y","saved"]if context==2 else["y"],factory=dict(args=[],kwargs={}),
            weights=weights,dtype_policy="strict-float64-v1",
            approximations=dict(format="poseidon-pytorch-approximations-v1",nodes=[],modules=[],defaults={}))
        arrays=_arrays(module)
        record=prepare(module,spec,folder/"capture",expected_weight_binding=binding(arrays,weights))
        check_record(record,spec,arrays,expected_record_binding=record["binding"])
        return module,spec,arrays,record

class PublicConstantCaptureTests(unittest.TestCase):
    def test_six_constructors_three_contexts_and_reference_replay(self):
        for kind in OPERATIONS:
            for context in range(3):
                with self.subTest(kind=kind,context=context):
                    module,spec,arrays,record=capture(kind,context)
                    self.assertEqual(record["format"],CAPTURE_FORMAT)
                    self.assertEqual(record["original_reference"]["status"],"passed")
                    self.assertEqual(record["original_reference"]["groups"],16)
                    flow=record["public_expression_flow"]
                    proof=next(e["plan"]for e in flow["entries"]if e["plan"].get("operation")==kind)
                    self.assertEqual(proof["dependency"],"static_shape_and_declared_dtype_only")
                    self.assertEqual(record["static_tensor_transform_flow"]["source_inventory"],flow["normalized_inventory"])
                    twice=[derived_arrays(record,spec,arrays,resource_profile="general-tensors-v1")for _ in range(2)]
                    for n in twice[0]:np.testing.assert_array_equal(twice[0][n],twice[1][n])
                    np.testing.assert_array_equal(_arrays(module)["w"],arrays["w"])
    def test_five_constructor_dtypes_and_rounding_are_bound(self):
        for kind in ("full_like","new_full"):
            for dtype in (torch.float32,torch.float64,torch.int32,torch.int64,torch.bool):
                with self.subTest(kind=kind,dtype=dtype):
                    _,_,_,record=capture(kind,1,dtype=dtype)
                    self.assertEqual(record["original_reference"]["status"],"passed")
    def test_v31_constructor_still_rejects(self):
        for kind in OPERATIONS:
            with self.subTest(kind=kind),self.assertRaises(ValueError):
                capture(kind,version="poseidon-trusted-pytorch-v31")
    def test_versions_and_named_none_tree(self):
        _,spec,_,record=capture(context=2)
        self.assertEqual(package_version(spec),"poseidon-trusted-pytorch-provenance-v32")
        self.assertEqual(reference_version(spec),"poseidon-trusted-pytorch-reference-v32")
        self.assertEqual(wrapped_metadata(record)["output_structure"]["items"][1]["value"]["items"][1],dict(kind="none"))
    def test_rebound_fill_and_metadata_changes_reject(self):
        _,spec,arrays,record=capture()
        for field,value in (("fill_value",.625),("dtype","<i8")):
            changed=copy.deepcopy(record);flow=changed["public_expression_flow"]
            p=next(e["plan"]for e in flow["entries"]if e["plan"].get("operation")=="full_like")
            p[field]=value
            for part in (p,flow,changed):part["binding"]=digest({k:v for k,v in part.items()if k!="binding"})
            with self.assertRaises(ValueError):
                check_record(changed,spec,arrays,expected_record_binding=changed["binding"])
    def test_wrong_input_and_original_weights_cannot_be_rebound(self):
        _,spec,arrays,record=capture()
        for s,a in [({**spec,"inputs":[dict(name="x",shape=[4])]},arrays),
                    (spec,{"w":arrays["w"]+1})]:
            with self.assertRaises(ValueError):check_record(record,s,a,expected_record_binding=record["binding"])
    def test_capability_registration_and_legacy_limits(self):
        from application_capabilities import input_sources
        cap=input_sources()["trusted_pytorch"]
        self.assertIn(SOURCE_FORMAT,cap["formats"])
        self.assertEqual(len(cap["public_constructors"]["calls"]),6)
        self.assertEqual(cap["public_constructors"]["encrypted_status"],"unverified")
        self.assertNotIn("poseidon-trusted-pytorch-v999",cap["formats"])

    def test_actual_worker_dispatch_uses_public_adapter(self):
        import sys,json
        from unittest.mock import patch
        from pytorch_ordered_intake import capture_request
        from pytorch_ordered_contract import source_identity
        name="constructor_dispatch_fixture"
        source="import torch\nclass Model(torch.nn.Module):\n    def forward(self,x):\n        return x+torch.ones_like(x)\ndef make():\n    return Model().eval()\n"
        with tempfile.TemporaryDirectory()as td:
            folder=Path(td);(folder/(name+".py")).write_text(source)
            spec=dict(format=SOURCE_FORMAT,files={name+".py":hashlib.sha256(source.encode()).hexdigest()},
                entry=name+".make",inputs=[dict(name="x",shape=[2])],outputs=["y"],
                call=dict(format="poseidon-pytorch-call-v1",args=[dict(kind="tensor",name="x")],kwargs={}),
                factory=dict(args=[],kwargs={}),weights={},dtype_policy="strict-float64-v1",
                approximations=dict(format="poseidon-pytorch-approximations-v1",nodes=[],modules=[],defaults={}))
            import numpy as np
            runtime=dict(python=list(sys.version_info[:3]),numpy=np.__version__,torch=torch.__version__,
                         cuda=torch.version.cuda,hip=getattr(torch.version,"hip",None))
            request=folder/"request.json"
            request.write_text(json.dumps(dict(manifest=spec,resource_profile="general-tensors-v1",
                               source_hashes=source_identity(),runtime=runtime)))
            sys.path.insert(0,td)
            try:
                marker=object()
                with patch("pytorch_public_expression_capture.capture_loaded_module",return_value=marker)as selected:
                    self.assertIs(capture_request(request,folder,folder/"out"),marker)
                    self.assertEqual(selected.call_args.args[0]["manifest"],spec)
                    self.assertIsInstance(selected.call_args.args[1],torch.nn.Module)
                    self.assertEqual(selected.call_args.args[2],{})
            finally:
                sys.path.remove(td);sys.modules.pop(name,None)
