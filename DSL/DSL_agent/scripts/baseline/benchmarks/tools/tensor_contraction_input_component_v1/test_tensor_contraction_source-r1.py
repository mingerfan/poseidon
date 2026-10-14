"""Explicit v38 policy, actual registered tracing, security and old requests."""
import copy,json,unittest
from pathlib import Path
import torch
from benchmark_graph import digest
from pytorch_model_intake import ordered
from pytorch_ordered_contract import package_version,reference_version,SOURCE_FORMATS,implementation_paths
from pytorch_public_expression_capture import capture_format,source_policy
from pytorch_static_trace import trace,ALIAS_POLICY
from provenance_json_transport import source_selected
from contraction_fx_normalization import normalize,check,public_arrays,POLICY,arguments
from component_contract import prepare_task,reconstruct_request,request_options
class Model(torch.nn.Module):
    def __init__(self,dims):
        super().__init__()
        if isinstance(dims,torch.Tensor):self.register_buffer("dimensions",dims)
        else:self.dimensions=dims
    def forward(self,a,b):
        return torch.tensordot(a,b,dims=self.dimensions)
class SourceContractionTests(unittest.TestCase):
    def test_versioned_contract_and_parent_stability(self):
        spec=dict(format="poseidon-trusted-pytorch-v38")
        self.assertTrue(ordered(spec));self.assertIn(spec["format"],SOURCE_FORMATS)
        self.assertEqual(package_version(spec),"poseidon-trusted-pytorch-provenance-v38")
        self.assertEqual(reference_version(spec),"poseidon-trusted-pytorch-reference-v38")
        self.assertEqual(capture_format(spec),"poseidon-public-expression-source-capture-v15")
        self.assertTrue(source_selected(spec))
        parent=dict(format="poseidon-trusted-pytorch-v37")
        self.assertEqual(capture_format(parent),"poseidon-public-expression-source-capture-v14")
        self.assertEqual(source_policy(spec),source_policy(parent))
        self.assertFalse(ordered(dict(format="poseidon-trusted-pytorch-v999")))
    def test_new_implementation_files_are_trusted_and_bound(self):
        paths=implementation_paths()
        for name in ("pytorch_tensor_contraction_source.py","contraction_fx_normalization.py","tensor_contraction_plan.py"):
            self.assertIn(name,paths)
    def checked(self,a_shape,b_shape,dims):
        module=Model(dims).eval()
        inputs=[dict(name="a",shape=a_shape),dict(name="b",shape=b_shape)]
        traced=trace(module,inputs,resource_profile="general-tensors-v1",policy=ALIAS_POLICY,
            tensor_contraction_source_policy=POLICY)
        normalized,record=normalize(traced,inputs,resource_profile="general-tensors-v1")
        a=torch.arange(__import__("math").prod(a_shape),dtype=torch.float64).reshape(a_shape)/5
        b=torch.arange(__import__("math").prod(b_shape),dtype=torch.float64).reshape(b_shape)/7-.1
        torch.testing.assert_close(module(a,b),normalized(a,b),rtol=1e-12,atol=1e-12)
        self.assertEqual(len(record["entries"]),1)
        check(record,inputs,public_arrays(traced),resource_profile="general-tensors-v1")
        return record,inputs,public_arrays(traced)
    def test_metadata_and_true_fx_matrix_contraction(self):
        self.checked([2,3],[3,2],1)
    def test_public_tensor_negative_axis_pair(self):
        self.checked([2,3,2],[2,3],torch.tensor([[-1,1],[0,1]],dtype=torch.int64))
    def test_int32_public_dimension_count(self):
        self.checked([3],[3],torch.tensor(1,dtype=torch.int32))
    def test_contracted_singleton_broadcast(self):
        self.checked([2,1],[3,2],1)
    def test_scalar_and_outer_products(self):
        for ash,bsh,dims in (([3],[3],1),([2],[2],0)):
            with self.subTest(dims=dims):self.checked(ash,bsh,dims)
    def test_old_trace_policy_cannot_admit_new_call(self):
        with self.assertRaises(ValueError):
            trace(Model(1),[dict(name="a",shape=[2,3]),dict(name="b",shape=[3,2])],
                resource_profile="general-tensors-v1",policy=ALIAS_POLICY)
    def test_private_dimension_tensor_rejected(self):
        class Private(torch.nn.Module):
            def forward(self,a,b,dims):return torch.tensordot(a,b,dims=dims)
        with self.assertRaises(ValueError):
            trace(Private(),[dict(name="a",shape=[2,3]),dict(name="b",shape=[3,2]),dict(name="dims",shape=[1])],
                resource_profile="general-tensors-v1",policy=ALIAS_POLICY,tensor_contraction_source_policy=POLICY)
    def test_float_public_dimensions_and_out_mutation_rejected(self):
        with self.assertRaises(ValueError):
            trace(Model(torch.tensor(1.)),[dict(name="a",shape=[2,3]),dict(name="b",shape=[3,2])],
                resource_profile="general-tensors-v1",policy=ALIAS_POLICY,tensor_contraction_source_policy=POLICY)
        with self.assertRaises(ValueError):arguments((1,2),dict(out=torch.empty(1)))
    def test_rebound_source_inventory_rejected(self):
        r,inputs,arrays=self.checked([2,3],[3,2],1)
        r=copy.deepcopy(r);r["entries"][0]["plan"]["matrix_shapes"][0][0]=True
        r["binding"]=digest({k:v for k,v in r.items()if k!="binding"})
        with self.assertRaises(ValueError):check(r,inputs,arrays,resource_profile="general-tensors-v1")
    def test_old_request_hashes_stay_exact(self):
        base=Path(__file__).resolve().parent
        for relative in ("benchmarks/tools/linear_phase_live_v1/prepared/wide_tail_linear/request.json",
                         "benchmarks/tools/tensor_gather_live_v1/prepared/rows/request.json"):
            old=json.loads((base/relative).read_bytes())
            self.assertEqual(reconstruct_request(old),old)
            self.assertEqual(prepare_task(old["model"],request_options(old)),old)
    def test_host_description_has_same_source_capability(self):
        from application_capabilities import input_sources
        source=input_sources()["trusted_pytorch"]
        self.assertIn("poseidon-trusted-pytorch-v38",source["formats"])
        item=source["tensor_contraction"]
        self.assertEqual(item["policy"],POLICY)
        self.assertEqual(item["calls"],["torch.tensordot"])
        self.assertFalse(item["dynamic_dimensions"])
        self.assertEqual(item["encrypted_status"],"unverified")
if __name__=="__main__":unittest.main(verbosity=2)
