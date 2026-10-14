import copy,operator,unittest
from benchmark_graph import digest
from pytorch_partition_normalization import POLICY,arguments,plan,normalize,check_transport
PROFILE="general-tensors-v1"
class PlannerTests(unittest.TestCase):
    def test_uneven_chunk_count_differs_from_requested(self):
        p=plan([2,13],"chunk",axis=-1,chunks=6)
        self.assertEqual(p["sections"],[3,3,3,3,1])
        self.assertEqual(len(p["ports"]),5)
    def test_chunks_greater_than_axis(self):
        self.assertEqual(plan([2,3],"chunk",axis=1,chunks=20)["sections"],[1,1,1])
    def test_single_chunk_preserves_full_view(self):
        p=plan([2,3],"chunk",axis=1,chunks=1)
        self.assertEqual(p["ports"],[dict(port=0,start=0,stop=3,result_shape=[2,3])])
    def test_unbind_removes_selected_axis(self):
        p=plan([2,3,4],"unbind",axis=-2)
        self.assertEqual([v["result_shape"]for v in p["ports"]],[[2,4]]*3)
    def test_axis_boundaries_reject(self):
        for a in [-4,3,True,1.0]:
            with self.subTest(axis=a),self.assertRaises(ValueError):plan([2,3,4],"unbind",axis=a)
    def test_static_count_types_reject(self):
        for n in [0,-1,True,1.5,2**53+1,None]:
            with self.subTest(chunks=n),self.assertRaises(ValueError):plan([3],"chunk",chunks=n)
    def test_empty_and_zero_shapes_reject(self):
        for shape in [[],[0],[2,False],[1.5]]:
            with self.subTest(shape=shape),self.assertRaises(ValueError):plan(shape,"chunk",chunks=1)
    def test_scalar_unbind_does_not_bypass_alias_contract(self):
        with self.assertRaisesRegex(ValueError,"Scalar unbind"):plan([4],"unbind")
    def test_expansion_bound_is_checked_before_ports(self):
        with self.assertRaisesRegex(ValueError,"expansion budget"):plan([1025,1],"unbind")
    def test_rank_32_and_33_boundary(self):
        self.assertEqual(len(plan([1]*32,"chunk",chunks=1)["source_shape"]),32)
        with self.assertRaises(ValueError):plan([1]*33,"chunk",chunks=1)
    def test_positional_and_keyword_forms(self):
        self.assertEqual(arguments(("x",3,-1),{},"chunk"),dict(input="x",chunks=3,axis=-1))
        self.assertEqual(arguments((),dict(input="x",dim=1),"unbind"),dict(input="x",chunks=None,axis=1))
        self.assertEqual(arguments(("x",),dict(chunks=3),"chunk",method=True)["chunks"],3)
    def test_duplicate_and_unknown_fields_reject(self):
        for a,k,o,m in [(("x",3),dict(chunks=2),"chunk",False),
                        (("x",3),dict(out=None),"chunk",False),
                        (("x",),dict(input="y"),"unbind",True)]:
            with self.subTest(operation=o),self.assertRaises(ValueError):arguments(a,k,o,method=m)
    def test_all_static_shapes_match_torch_partition_semantics(self):
        import torch
        for width in range(1,18):
            for chunks in [1,2,3,5,6,20]:
                x=torch.empty(2,width,3)
                actual=torch.chunk(x,chunks,dim=1)
                p=plan(list(x.shape),"chunk",axis=1,chunks=chunks)
                self.assertEqual([list(v.shape)for v in actual],[v["result_shape"]for v in p["ports"]])
                self.assertEqual([v.shape[1]for v in actual],p["sections"])
        for axis in [-3,-2,-1,0,1,2]:
            x=torch.empty(2,3,4);p=plan(list(x.shape),"unbind",axis=axis)
            self.assertEqual([list(v.shape)for v in x.unbind(axis)],[v["result_shape"]for v in p["ports"]])
class GraphTests(unittest.TestCase):
    def build(self,style="chunk",shape=(2,6)):
        import torch
        from pytorch_static_trace import trace,ALIAS_POLICY
        class Model(torch.nn.Module):
            def forward(self,x):
                if style=="chunk":
                    a,b,c=torch.chunk(input=x,chunks=3,dim=-1);return a+b*2+c*3
                if style=="unbind":
                    a,b=x.unbind(0);return dict(first=a+b,second=b)
                if style=="nested":
                    a,b,c=x.chunk(3,-1);left,right=a.unbind(0);return left+right+b[0]+c[1]
                a,b,c=x.chunk(3,-1);snapshot=a.clone();b.mul_(3);return snapshot,b,x
        signature=[dict(name="x",shape=list(shape))]
        source=trace(Model(),signature,resource_profile=PROFILE,policy=ALIAS_POLICY,partition_source_policy=POLICY)
        normalized,record=normalize(source,signature,resource_profile=PROFILE)
        return source,normalized,record,signature
    def test_chunk_registered_projection_and_numerical_contribution(self):
        import torch
        a,b,record,sig=self.build()
        x=torch.arange(12,dtype=torch.float64).reshape(2,6)-3
        self.assertTrue(torch.equal(a(x.clone()),b(x.clone())))
        self.assertEqual(len(record["entries"]),1)
        self.assertFalse(any(n.target is torch.chunk for n in b.graph.nodes))
        self.assertIs(check_transport(record,inputs=sig,resource_profile=PROFILE),record)
    def test_unbind_dictionary_structure(self):
        import torch
        a,b,record,sig=self.build("unbind")
        x=torch.arange(12,dtype=torch.float64).reshape(2,6)
        for n in ["first","second"]:self.assertTrue(torch.equal(a(x.clone())[n],b(x.clone())[n]))
    def test_nested_partition_preserves_coordinate_mapping(self):
        import torch
        a,b,record,sig=self.build("nested")
        x=torch.arange(12,dtype=torch.float64).reshape(2,6)
        self.assertTrue(torch.equal(a(x.clone()),b(x.clone())))
        self.assertEqual(len(record["entries"]),2)
    def test_mutated_chunk_is_still_a_view_of_original_storage(self):
        import torch
        a,b,record,sig=self.build("mutation")
        x=torch.arange(12,dtype=torch.float64).reshape(2,6)+1
        left=a(x.clone());right=b(x.clone())
        for l,r in zip(left,right):self.assertTrue(torch.equal(l,r))
        self.assertTrue(torch.equal(right[2][:,2:4],x[:,2:4]*3))
    def test_plan_tamper_with_recomputed_binding_rejects(self):
        _,_,record,sig=self.build()
        bad=copy.deepcopy(record);bad["entries"][0]["plan"]["sections"]=[2,1,3]
        bad["binding"]=digest({k:v for k,v in bad.items()if k!="binding"})
        with self.assertRaises(ValueError):check_transport(bad,inputs=sig,resource_profile=PROFILE)
    def test_index_rewrite_tamper_rejects(self):
        _,_,record,sig=self.build()
        bad=copy.deepcopy(record);name=bad["entries"][0]["ports"][0]["name"]
        node=next(v for v in bad["normalized_inventory"]if v["name"]==name)
        node["args"]["items"][1]["items"][-1]["items"][0]["value"]=1
        bad["binding"]=digest({k:v for k,v in bad.items()if k!="binding"})
        with self.assertRaises(ValueError):check_transport(bad,inputs=sig,resource_profile=PROFILE)
    def test_missing_partition_entry_rejects(self):
        _,_,record,sig=self.build()
        bad=copy.deepcopy(record);bad["entries"]=[]
        bad["binding"]=digest({k:v for k,v in bad.items()if k!="binding"})
        with self.assertRaises(ValueError):check_transport(bad,inputs=sig,resource_profile=PROFILE)
    def test_old_trace_policy_does_not_silently_enable_partition(self):
        import torch
        from pytorch_static_trace import trace,ALIAS_POLICY
        class Model(torch.nn.Module):
            def forward(self,x):a,b=x.chunk(2,1);return a+b
        old=trace(Model(),[dict(name="x",shape=[2,4])],resource_profile=PROFILE,policy=ALIAS_POLICY)
        self.assertNotIn("partition_source_policy",old.meta)
        with self.assertRaises(ValueError):normalize(old,[dict(name="x",shape=[2,4])],resource_profile=PROFILE)
    def test_data_dependent_chunk_count_rejects(self):
        import torch
        from pytorch_static_trace import trace,ALIAS_POLICY
        class Model(torch.nn.Module):
            def forward(self,x):return torch.chunk(x,x.sum(),1)[0]
        with self.assertRaises((ValueError,TypeError)):trace(Model(),[dict(name="x",shape=[2,4])],resource_profile=PROFILE,policy=ALIAS_POLICY,partition_source_policy=POLICY)
class CaptureTests(unittest.TestCase):
    def prepare(self,module,signature,outputs):
        from pytorch_ordered_capture import prepare_module
        return prepare_module(module,signature,outputs,resource_profile=PROFILE,
            approximations=dict(format="poseidon-pytorch-approximations-v1",defaults={},modules=[],nodes=[]),
            dtype_policy="strict-float64-v1",matrix_policy="static-tensor-matmul-v1",
            approximation_policy="node-module-default-v1",common_calls=True,
            functional_policy="static-adaptive-average-v1",fused_matrix_policy="poseidon-fx-fused-matrix-draft-v1",
            gelu_source_policy="poseidon-gelu-source-variants-v1",mha_source_policy="poseidon-mha-ordered-source-v1",
            partition_source_policy=POLICY)
    def compare(self,prepared,signature,outputs):
        from pytorch_mha_ordered_reference import validate
        report=validate(prepared,signature,outputs,resource_profile=PROFILE,count=4)
        self.assertEqual(report["status"],"passed")
        self.assertLessEqual(report["max_decomposition_error"],1e-12)
        return report
    def test_chunk_then_shared_arithmetic_and_outputs(self):
        import torch
        class Model(torch.nn.Module):
            def forward(self,x):
                a,b,c=x.chunk(3,dim=1)
                return {"merge":a+b*2+c*3,"tail":c}
        signature=[dict(name="x",shape=[2,6])];outputs=["merge","tail"]
        self.compare(self.prepare(Model(),signature,outputs),signature,outputs)
    def test_unbind_then_stack_and_transpose(self):
        import torch
        class Model(torch.nn.Module):
            def forward(self,x):
                a,b=torch.unbind(input=x,dim=0)
                return torch.stack([a,b*2],dim=0).transpose(-1,-2)
        signature=[dict(name="x",shape=[2,2,2])];outputs=["y"]
        self.compare(self.prepare(Model(),signature,outputs),signature,outputs)
    def test_chunk_mutation_and_snapshot_preserve_reference(self):
        import torch
        class Model(torch.nn.Module):
            def forward(self,x):
                a,b=x.chunk(2,1);snapshot=a.clone();a.add_(b*2);return snapshot,x
        signature=[dict(name="x",shape=[2,4])];outputs=["snapshot","updated"]
        self.compare(self.prepare(Model(),signature,outputs),signature,outputs)
    def test_chunk_after_linear_and_unbind(self):
        import torch
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__();self.linear=torch.nn.Linear(2,4,bias=False,dtype=torch.float64)
                with torch.no_grad():self.linear.weight.copy_(torch.tensor([[.5,.1],[-.2,.3],[.4,-.1],[.2,.6]],dtype=torch.float64))
            def forward(self,x):
                a,b=self.linear(x).chunk(2,-1)
                p,q=torch.unbind(a,0)
                return p+q+b.sum(0)
        signature=[dict(name="x",shape=[2,2])];outputs=["y"]
        self.compare(self.prepare(Model(),signature,outputs),signature,outputs)
if __name__=="__main__":unittest.main()
