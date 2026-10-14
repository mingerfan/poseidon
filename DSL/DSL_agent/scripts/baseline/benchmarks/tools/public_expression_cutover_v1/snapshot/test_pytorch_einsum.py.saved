import copy
import string
import unittest
from benchmark_graph import digest
from pytorch_einsum_normalization import POLICY,arguments,plan,normalize,check_transport
PROFILE="general-tensors-v1"
CASES=[
 ("ij,jk->ik",[[2,3],[3,4]]),
 ("bij,bjk->bik",[[2,2,3],[1,3,2]]),
 ("bhqd,bhkd->bhqk",[[1,2,2,3],[1,2,4,3]]),
 ("...ij,...jk->...ik",[[2,1,2,3],[1,4,3,2]]),
 ("abc,cde->abde",[[2,2,3],[3,2,2]]),
 ("ij,k->jk",[[2,3],[4]]),
 ("i,j->ji",[[2],[3]]),
 ("i,i->",[[3],[3]]),
 ("ij,ij->",[[2,3],[2,3]]),
 ("ij,jk->ik",[[2,1],[3,2]]),
 ("...i,...i->",[[2,3],[1,3]]),
 ("ji",[[2,3]]),
 ("ij->j",[[2,3]]),
 ("ij,jk,k->i",[[2,3],[3,2],[2]]),
 ("ij,jk,kl,l->i",[[2,2],[2,2],[2,2],[2]]),
 ("...ij,...jk",[[2,2,3],[1,3,2]]),
 ("...ij,...jk->ik",[[2,2,3],[1,3,2]]),
 ("ij->ij",[[2,3]])
]
def replay(p,tensors):
    import torch
    values={"operand_"+str(i):v for i,v in enumerate(tensors)}
    for i,s in enumerate(p["steps"]):
        v=[values[n]for n in s["inputs"]]
        if s["op"]=="permute":out=v[0].permute(tuple(s["attrs"]["dims"]))
        elif s["op"]=="reshape":out=v[0].reshape(tuple(s["attrs"]["shape"]))
        elif s["op"]=="sum":out=torch.sum(v[0],dim=tuple(s["attrs"]["axes"]),keepdim=False)
        elif s["op"]=="matmul":out=torch.matmul(*v)
        else:out=v[0]*v[1]
        values["step_"+str(i)]=out
    return out
def data(shapes):
    import torch
    return [((torch.arange(__import__("math").prod(s),dtype=torch.float64).reshape(s)%11)-5)/13+i/37
            for i,s in enumerate(shapes)]
class PlannerTests(unittest.TestCase):
    def test_matrix_contraction_uses_existing_matmul(self):
        p=plan("ij,jk->ik",[[2,3],[3,4]])
        self.assertEqual(p["strategy"],"grouped_static_matmul")
        self.assertEqual(p["result_shape"],[2,4])
    def test_batched_ellipsis_broadcast(self):
        p=plan("...ij,...jk->...ik",[[2,1,2,3],[1,4,3,2]])
        self.assertEqual(p["result_shape"],[2,4,2,2])
    def test_head_axes_and_output_order(self):
        p=plan("bhqd,bhkd->bhqk",[[1,2,2,3],[1,2,4,3]])
        self.assertEqual(p["result_shape"],[1,2,2,4])
    def test_single_operand_transpose_is_a_view_plan(self):
        p=plan("ij->ji",[[2,3]])
        self.assertEqual([s["op"]for s in p["steps"]],["permute"])
    def test_implicit_output_uses_unique_sorted_labels(self):
        self.assertEqual(plan("ji",[[2,3]])["result_shape"],[3,2])
    def test_individual_operand_reduction_precedes_product(self):
        p=plan("ij,k->jk",[[2,3],[4]])
        self.assertEqual(p["steps"][0]["op"],"sum")
        self.assertEqual(p["result_shape"],[3,4])
    def test_contraction_size_one_broadcast_is_bounded(self):
        p=plan("ij,jk->ik",[[2,1],[3,2]])
        self.assertEqual(p["strategy"],"broadcast_product_reduce")
    def test_ellipsis_can_be_reduced_in_torch(self):
        self.assertEqual(plan("...i,...i->",[[2,3],[1,3]])["result_shape"],[])
    def test_three_and_four_operand_products(self):
        for equation,shapes in CASES[13:15]:
            self.assertEqual(plan(equation,shapes)["strategy"],"broadcast_product_reduce")
    def test_rank_32_bound(self):
        labels=(string.ascii_lowercase+string.ascii_uppercase)[:32]
        self.assertEqual(plan(labels,[[1]*32])["result_shape"],[1]*32)
        with self.assertRaises(ValueError):plan(labels+"G",[[1]*33])
    def test_malformed_and_missing_labels_reject(self):
        for equation,shapes in [("i->j",[[2]]),("i->ii",[[2]]),("i-->i",[[2]]),
                ("i->i->i",[[2]]),("i..->i",[[2]]),("i,j",[[2]]),("ii",[[2]])]:
            with self.subTest(equation=equation),self.assertRaises(ValueError):plan(equation,shapes)
    def test_diagonal_labels_have_an_explicit_blocker(self):
        with self.assertRaisesRegex(ValueError,"Diagonal"):plan("ii->i",[[2,2]])
    def test_shape_and_label_broadcast_reject(self):
        for equation,shapes in [("i,i->i",[[2],[3]]),("i",[[0]]),("i",[[True]]),
                               ("i",[[1.5]]),("",[[]])]:
            with self.subTest(equation=equation),self.assertRaises(ValueError):plan(equation,shapes)
    def test_workspace_checked_without_large_allocation(self):
        with self.assertRaisesRegex(ValueError,"workspace"):
            plan("ij,jk,ki->",[[256,256],[256,256],[256,256]])
        self.assertEqual(plan("ij,jk->ik",[[256,256],[256,256]])["result_shape"],[256,256])
    def test_operand_and_equation_budgets(self):
        with self.assertRaises(ValueError):plan("i,i,i,i,i->i",[[2]]*5)
        with self.assertRaises(ValueError):plan("i"+" "*256,[[2]])
    def test_equation_data_and_legacy_list_arguments(self):
        self.assertEqual(arguments(("ij,jk",["x","y"]),{}),("ij,jk",["x","y"]))
        for args,kwargs in [(("i",),{}),((["x",[0]],),{}),(("i","x"),dict(out=None)),
                            ((True,"x"),{})]:
            with self.subTest(args=args),self.assertRaises(ValueError):arguments(args,kwargs)
    def test_all_planned_shapes_and_values_match_pinned_torch(self):
        import torch
        for equation,shapes in CASES:
            with self.subTest(equation=equation,shapes=shapes):
                tensors=data(shapes);p=plan(equation,shapes);actual=torch.einsum(equation,*tensors)
                expected=replay(p,tensors)
                self.assertEqual(list(actual.shape),p["result_shape"])
                torch.testing.assert_close(expected,actual,atol=1e-12,rtol=1e-12)
class GraphTests(unittest.TestCase):
    def graph(self,equation,shapes,*,list_form=False):
        import torch
        from pytorch_static_trace import trace,ALIAS_POLICY
        class Model(torch.nn.Module):
            def forward(self,*values):
                return torch.einsum(equation,list(values)if list_form else values)
        # FX has a special *args proxy; use a fixed signature with the same bound equation.
        if len(shapes)==1:
            class Fixed(torch.nn.Module):
                def forward(self,x):return torch.einsum(equation,x)
            module=Fixed()
        elif len(shapes)==2:
            class Fixed(torch.nn.Module):
                def forward(self,x,y):return torch.einsum(equation,[x,y])if list_form else torch.einsum(equation,x,y)
            module=Fixed()
        elif len(shapes)==3:
            class Fixed(torch.nn.Module):
                def forward(self,x,y,z):return torch.einsum(equation,x,y,z)
            module=Fixed()
        else:
            class Fixed(torch.nn.Module):
                def forward(self,x,y,z,w):return torch.einsum(equation,x,y,z,w)
            module=Fixed()
        signature=[dict(name=n,shape=s)for n,s in zip("xyzw",shapes)]
        raw=trace(module,signature,resource_profile=PROFILE,policy=ALIAS_POLICY,einsum_source_policy=POLICY)
        normalized,record=normalize(raw,signature,resource_profile=PROFILE)
        return raw,normalized,record,signature
    def test_projection_and_numbers_preserve_original(self):
        import torch
        for equation,shapes in CASES:
            raw,normalized,record,signature=self.graph(equation,shapes)
            with self.subTest(equation=equation):
                self.assertEqual(str(raw.graph),record["source_fx_text"])
                check_transport(record,inputs=signature,resource_profile=PROFILE)
                torch.testing.assert_close(normalized(*data(shapes)),raw(*data(shapes)),atol=1e-12,rtol=1e-12)
    def test_legacy_list_form(self):
        import torch
        raw,normalized,_,_=self.graph("ij,jk->ik",[[2,3],[3,2]],list_form=True)
        torch.testing.assert_close(normalized(*data([[2,3],[3,2]])),raw(*data([[2,3],[3,2]])),atol=1e-12,rtol=1e-12)
    def tamper(self,mutate):
        _,_,record,signature=self.graph("ij,jk->ik",[[2,3],[3,2]])
        changed=copy.deepcopy(record);mutate(changed)
        changed["binding"]=digest({k:v for k,v in changed.items()if k!="binding"})
        with self.assertRaises((ValueError,KeyError,TypeError)):
            check_transport(changed,inputs=signature,resource_profile=PROFILE)
    def test_changed_equation_with_new_record_digest_rejects(self):
        self.tamper(lambda r:r["entries"][0]["plan"].update(equation="ij,jk->ki"))
    def test_changed_primitive_with_new_record_digest_rejects(self):
        self.tamper(lambda r:r["normalized_inventory"][2].update(target="torch.add"))
    def test_missing_entry_with_new_record_digest_rejects(self):
        self.tamper(lambda r:r.update(entries=[]))
    def test_extra_generated_node_rejects(self):
        self.tamper(lambda r:r["normalized_inventory"].insert(1,copy.deepcopy(r["normalized_inventory"][0])))
    def test_changed_shape_with_new_record_digest_rejects(self):
        self.tamper(lambda r:r["entries"][0]["plan"].update(result_shape=[2,3]))
    def test_changed_source_operand_rejects(self):
        self.tamper(lambda r:r["entries"][0].update(operands=["y","x"]))
    def test_old_source_policy_does_not_enable_new_equations(self):
        import torch
        from pytorch_static_trace import trace,ALIAS_POLICY
        class Model(torch.nn.Module):
            def forward(self,x,y):return torch.einsum("ij,jk->ik",x,y)
        raw=trace(Model(),[dict(name="x",shape=[2,3]),dict(name="y",shape=[3,2])],
                  resource_profile=PROFILE,policy=ALIAS_POLICY)
        with self.assertRaisesRegex(ValueError,"Explicit einsum"):
            normalize(raw,[dict(name="x",shape=[2,3]),dict(name="y",shape=[3,2])],resource_profile=PROFILE)
    def test_plaintext_equation_cannot_be_an_encrypted_proxy(self):
        import torch
        from pytorch_static_trace import trace,ALIAS_POLICY
        class Model(torch.nn.Module):
            def forward(self,x,y):return torch.einsum(x,y)
        with self.assertRaises((ValueError,TypeError)):
            trace(Model(),[dict(name="x",shape=[2]),dict(name="y",shape=[2])],
                  resource_profile=PROFILE,policy=ALIAS_POLICY,einsum_source_policy=POLICY)
class CaptureTests(unittest.TestCase):
    def compare(self,module,signature,outputs):
        from pytorch_ordered_capture import prepare_module
        from pytorch_mha_ordered_reference import validate
        prepared=prepare_module(module,signature,outputs,resource_profile=PROFILE,
            approximations=dict(format="poseidon-pytorch-approximations-v1",defaults={},modules=[],nodes=[]),
            dtype_policy="strict-float64-v1",matrix_policy="static-tensor-matmul-v1",
            approximation_policy="node-module-default-v1",common_calls=True,
            functional_policy="static-adaptive-average-v1",fused_matrix_policy="poseidon-fx-fused-matrix-draft-v1",
            gelu_source_policy="poseidon-gelu-source-variants-v1",mha_source_policy="poseidon-mha-ordered-source-v1",
            partition_source_policy="poseidon-pytorch-static-partition-v1",einsum_source_policy=POLICY)
        report=validate(prepared,signature,outputs,resource_profile=PROFILE,count=4)
        self.assertEqual(report["status"],"passed")
        self.assertLessEqual(report["max_decomposition_error"],1e-12)
        return prepared,report
    def test_two_ciphertext_matrix_contraction(self):
        import torch
        class Model(torch.nn.Module):
            def forward(self,x,y):return torch.einsum("ij,jk->ik",x,y)
        self.compare(Model(),[dict(name="x",shape=[2,3]),dict(name="y",shape=[3,2])],["out"])
    def test_head_contraction_and_named_outputs(self):
        import torch
        class Model(torch.nn.Module):
            def forward(self,x,y):
                value=torch.einsum("bhqd,bhkd->bhqk",x,y)
                return {"scores":value,"summary":value.sum(dim=-1)}
        self.compare(Model(),[dict(name="x",shape=[1,2,2,2]),dict(name="y",shape=[1,2,3,2])],["scores","summary"])
    def test_three_ciphertext_contract(self):
        import torch
        class Model(torch.nn.Module):
            def forward(self,x,y,z):return torch.einsum("ij,jk,k->i",x,y,z)
        self.compare(Model(),[dict(name="x",shape=[2,2]),dict(name="y",shape=[2,2]),dict(name="z",shape=[2])],["y"])
    def test_one_dimension_contraction_broadcast(self):
        import torch
        class Model(torch.nn.Module):
            def forward(self,x,y):return torch.einsum("ij,jk->ik",x,y)
        self.compare(Model(),[dict(name="x",shape=[2,1]),dict(name="y",shape=[3,2])],["out"])
    def test_ellipsis_reduction_returns_scalar(self):
        import torch
        class Model(torch.nn.Module):
            def forward(self,x,y):return torch.einsum("...i,...i->",x,y)
        self.compare(Model(),[dict(name="x",shape=[2,3]),dict(name="y",shape=[1,3])],["out"])
    def test_einsum_view_write_and_snapshot(self):
        import torch
        class Model(torch.nn.Module):
            def forward(self,x):
                y=torch.einsum("ij->ji",x);saved=y.clone();y.add_(x*0.25)
                return saved,x
        self.compare(Model(),[dict(name="x",shape=[2,2])],["saved","updated"])
    def test_public_weight_einsum_then_chunk_and_unbind(self):
        import torch
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("weight",torch.tensor([[.31,.13,-.19,.47],[-.23,.41,.17,-.37]],dtype=torch.float64))
            def forward(self,x):
                y=torch.einsum("bi,ij->bj",x,self.weight)
                a,b=y.chunk(2,-1);s,t=a.unbind(0)
                return {"sum":s+t,"right":b}
        self.compare(Model(),[dict(name="x",shape=[2,2])],["sum","right"])
    def test_existing_none_output_structure(self):
        import torch
        class Model(torch.nn.Module):
            def forward(self,x,y):return torch.einsum("ij,jk->ik",x,y),None
        self.compare(Model(),[dict(name="x",shape=[2,2]),dict(name="y",shape=[2,2])],["out"])
    def test_source_versions_remain_explicit(self):
        import pytorch_model_intake as source
        from pytorch_ordered_contract import package_version,reference_version
        self.assertTrue(source.ordered(dict(format=source.EINSUM_FORMAT)))
        self.assertEqual(package_version(dict(format=source.EINSUM_FORMAT)),"poseidon-trusted-pytorch-provenance-v18")
        self.assertEqual(reference_version(dict(format=source.EINSUM_FORMAT)),"poseidon-trusted-pytorch-reference-v18")
        self.assertEqual(package_version(dict(format=source.PARTITION_FORMAT)),"poseidon-trusted-pytorch-provenance-v17")
if __name__=="__main__":unittest.main()
