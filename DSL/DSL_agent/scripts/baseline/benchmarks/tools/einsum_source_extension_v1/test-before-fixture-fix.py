import copy,hashlib,json,math,unittest
from pathlib import Path
from benchmark_graph import digest
from pytorch_einsum_extension import POLICY,plan,normalize,check_transport
PROFILE="general-tensors-v1"
APPROX=dict(format="poseidon-pytorch-approximations-v1",defaults={},modules=[],nodes=[])
def capture(module,shapes,outputs):
    from pytorch_ordered_capture import prepare_module
    from pytorch_mha_ordered_reference import validate
    signature=[dict(name="x"if i==0 else"y"if i==1 else"z",shape=s)for i,s in enumerate(shapes)]
    prepared=prepare_module(module,signature,outputs,resource_profile=PROFILE,approximations=APPROX,
        dtype_policy="strict-float64-v1",matrix_policy="static-tensor-matmul-v1",
        approximation_policy="node-module-default-v1",functional_policy="static-adaptive-average-v1",
        common_calls=True,fused_matrix_policy="poseidon-fx-fused-matrix-draft-v1",
        gelu_source_policy="poseidon-gelu-source-variants-v1",mha_source_policy="poseidon-mha-ordered-source-v1",
        partition_source_policy="poseidon-pytorch-static-partition-v1",einsum_source_policy=POLICY)
    receipt=validate(prepared,signature,outputs,resource_profile=PROFILE,count=4)
    if receipt["status"]!="passed":raise AssertionError(receipt)
    return prepared,signature
def model(body,*,buffer=False):
    import torch
    code="class Model(torch.nn.Module):\n"
    if buffer:code+=" def __init__(self):\n  super().__init__();self.register_buffer('gain',torch.tensor(.375,dtype=torch.float64))\n"
    code+=" def forward(self,x,y=None,z=None):\n"+ "\n".join("  "+l for l in body.split("\n"))
    namespace={"torch":torch};exec(code,namespace);return namespace["Model"]()
class Views(unittest.TestCase):
    def check_source(self,body,shapes,outputs=["out"],buffer=False):
        prepared,signature=capture(model(body,buffer=buffer),shapes,outputs)
        from pytorch_ordered_contract import capture_binding,EINSUM_VIEW_SOURCE_FORMAT
        from pytorch_weight_package import binding
        # Full provenance cross-check for stateless models.
        if not buffer:
            spec=dict(format=EINSUM_VIEW_SOURCE_FORMAT,entry="model.make",files={"model.py":"0"*64},
                factory=dict(args=[],kwargs={}),weights={},dtype_policy="strict-float64-v1",
                inputs=signature,outputs=outputs,approximations=APPROX)
            high,lowered,bound,_=prepared
            checked,record=capture_binding(high,spec,resource_profile=PROFILE,capture_metadata=bound["pytorch_capture"],
                expected_weight_binding=binding({},{}))
            self.assertEqual(checked,lowered)
        return prepared
    def test_diagonal_view_output_mutates_source(self):
        self.check_source('d=torch.einsum("ii->i",x)\nd.add_(.125)\nreturn d,x',[[2,2]],["diag","updated"])
    def test_source_write_updates_diagonal_view(self):
        self.check_source('d=torch.einsum("ii->i",x)\nx.mul_(.25)\nreturn d',[[2,2]])
    def test_batched_diagonal_transpose(self):
        self.check_source('d=torch.einsum("bii->ib",x)\nd.add_(.125)\nreturn d,x',[[2,2,2]],["diag","updated"])
    def test_repeated_three_axes(self):
        self.check_source('return torch.einsum("iii->i",x)',[[2,2,2]])
    def test_repeated_labels_nonadjacent(self):
        self.check_source('return torch.einsum("iji->j",x)',[[2,3,2]])
    def test_trace_scalar(self):
        self.check_source('return torch.einsum("ii->",x)',[[2,2]])
    def test_implicit_trace(self):
        self.check_source('return torch.einsum("ii",x)',[[2,2]])
    def test_ellipsis_trace(self):
        self.check_source('return torch.einsum("...ii->...",x)',[[2,2,2]])
    def test_diagonal_matrix_composition(self):
        self.check_source('return torch.einsum("bii,ij->bj",x,y)',[[1,2,2],[2,2]])
    def test_trace_scalar_times_tensor(self):
        self.check_source('s=torch.einsum("ii->",x)\nreturn torch.einsum(",ij->ij",s,y)',[[2,2],[2,2]])
    def test_scalar_reduction_factor(self):
        self.check_source('s=x.sum()\nreturn torch.einsum("i,->",y,s)',[[2],[3]])
    def test_three_scalar_factors(self):
        self.check_source('return torch.einsum(",,->",x.sum(),y.sum(),z.sum())',[[2],[2],[2]])
    def test_scalar_identity_alias(self):
        self.check_source('s=x.sum()\nd=torch.einsum("->",s)\nd.add_(.25)\nreturn d,s',[[2]],["view","source"])
    def test_scalar_basic_index_write(self):
        self.check_source('s=x[0]\ns.mul_(.25)\nreturn torch.einsum(",i->i",s,x)',[[2]])
    def test_scalar_view_reshape_write(self):
        self.check_source('s=x[0].reshape(())\ns.add_(.125)\nreturn x,s',[[2]],["updated","scalar"])
    def test_scalar_public_weight_factor(self):
        self.check_source('return torch.einsum(",ij->ij",self.gain,x)',[[2,2]],buffer=True)
    def test_diagonal_public_weight(self):
        m=model('return torch.einsum("ii,i->i",self.weight,x)')
        import torch;m.register_buffer("weight",torch.tensor([[.2,.3],[.4,.5]],dtype=torch.float64))
        capture(m,[[2]],["out"])
    def test_incompatible_repeated_dims_reject(self):
        with self.assertRaises(ValueError):capture(model('return torch.einsum("ii->i",x)'),[[2,3]],["out"])
    def test_partial_overlap_reject(self):
        with self.assertRaisesRegex(ValueError,"overlap"):
            capture(model('d=torch.einsum("ii->i",x)\nx.add_(d)\nreturn x'),[[2,2]],["out"])
    def test_legacy_v18_still_rejects_diagonals(self):
        from pytorch_static_trace import trace,ALIAS_POLICY
        with self.assertRaisesRegex(ValueError,"Diagonal"):
            trace(model('return torch.einsum("ii->i",x)'),[dict(name="x",shape=[2,2])],
                resource_profile=PROFILE,policy=ALIAS_POLICY,einsum_source_policy="poseidon-pytorch-static-einsum-v1")
    def test_legacy_alias_schema_rejects_new_operation(self):
        from pytorch_alias_composition import SOURCE_FORMAT,plan as aliases
        source=dict(format=SOURCE_FORMAT,inputs=[dict(name="x",shape=[2,2])],constants={},
            nodes=[dict(id="n",op="diagonal",inputs=["x"],attrs=dict(offset=0,dim1=0,dim2=1),output="d")],
            outputs=[dict(name="out",value="d")])
        with self.assertRaises(ValueError):aliases(source,{})
    def test_transport_recomputes_diagonal_axes(self):
        prepared,sig=capture(model('return torch.einsum("ii->i",x)'),[[2,2]],["out"])
        r=copy.deepcopy(prepared[2]["pytorch_capture"]["ordered_alias_capture"]["capture_adapter"]["einsum_normalization"])
        r["entries"][0]["plan"]["steps"][0]["attrs"]["dim2"]=0
        r["binding"]=digest({k:v for k,v in r.items()if k!="binding"})
        with self.assertRaises(ValueError):check_transport(r,inputs=sig,resource_profile=PROFILE)
    def test_origin_tamper_rejects(self):
        prepared,_=capture(model('return torch.einsum("ii->i",x)'),[[2,2]],["out"])
        a=prepared[2]["pytorch_capture"]["ordered_alias_capture"]
        from pytorch_einsum_extension import check_alias_projection
        origin=copy.deepcopy(a["capture_source_origins"])
        name=next(k for k,v in origin.items()if v.get("action")=="diagonal")
        origin[name]["diagonal"]["dim2"]=0
        with self.assertRaises(ValueError):
            check_alias_projection(a["capture_adapter"]["einsum_normalization"],a["capture_emissions"],origin,inputs=[dict(name="x",shape=[2,2])])
    def test_old_version_dispatch(self):
        from pytorch_ordered_contract import SOURCE_FORMATS,package_version,reference_version
        for i in range(12,20):
            spec=dict(format="poseidon-trusted-pytorch-v"+str(i))
            self.assertIn(spec["format"],SOURCE_FORMATS)
            self.assertEqual(package_version(spec),"poseidon-trusted-pytorch-provenance-v"+str(i))
            self.assertEqual(reference_version(spec),"poseidon-trusted-pytorch-reference-v"+str(i))
