"""Explicit static policy and retained V22 behaviour controls."""
import copy,operator,unittest
import numpy as np
import torch
from pytorch_static_trace import trace
from pytorch_public_arithmetic_capture import TRACE_OPTIONS
from pytorch_public_value_ops import POLICY
from pytorch_public_expression_flow import normalize,_arrays,state_identity,check
from pytorch_fused_matrix_normalization import inventory
from test_public_flow import Model

PROFILE="general-tensors-v1"
COUNTERS=dict(contexts=0,source_traces=0,retained_traces=0,reference_groups=0)
def selected(module,shape):
    result=trace(module,[dict(name="x",shape=list(shape))],resource_profile=PROFILE,
                 **TRACE_OPTIONS,public_expression_policy=POLICY)
    COUNTERS["source_traces"]+=1
    return result
def prepared(module,shape):
    traced=selected(module,shape);raw=_arrays(traced)
    norm,record=normalize(traced,raw,expected_state=state_identity(raw,resource_profile=PROFILE),
                          resource_profile=PROFILE)
    check(record,raw,expected_binding=record["binding"],expected_source_inventory=inventory(traced),
          resource_profile=PROFILE)
    retained=trace(norm,[dict(name="x",shape=list(shape))],resource_profile=PROFILE,**TRACE_OPTIONS)
    COUNTERS["retained_traces"]+=1
    return traced,norm,retained,record

class Controls(unittest.TestCase):
    def failure(self,fn,code):
        with self.assertRaisesRegex(ValueError,code):fn()
    def test_public_transform_tensor_divisors_and_boolean_multiply(self):
        for operation in ("reshape","permute","slice","bool_cast","identity"):
            for context in range(3):
                class ModelWithDivision(Model):
                    def forward(self,x):
                        p=self.public
                        if self.operation=="reshape":p=p.view(*((2,3),(3,2),(1,2,3))[self.context])
                        elif self.operation=="permute":p=p.permute(*tuple(reversed(range(p.ndim))))
                        elif self.operation=="slice":p=p[...,::2]
                        elif self.operation=="bool_cast":return x*p.float()
                        else:p=p.contiguous()
                        return x/p
                with self.subTest(operation=operation,context=context):
                    module=ModelWithDivision(operation,context);public=module.mathematical_public();shape=public.shape
                    traced,norm,retained,record=prepared(module,shape)
                    self.assertEqual(len(record["entries"]),1)
                    self.assertTrue(all(n.get("tensor")is not None for n in record["normalized_inventory"]
                                        if n["op"]not in("output",)))
                    probes=[np.zeros(shape),np.linspace(-1,1,public.size).reshape(shape),
                            np.random.default_rng(33+context).normal(size=shape),np.full(shape,2.)]
                    for x in probes:
                        expected=x*public if operation=="bool_cast"else x/public
                        for m in (module,traced,norm,retained):
                            np.testing.assert_array_equal(m(torch.from_numpy(x)).detach().numpy(),expected)
                    COUNTERS["contexts"]+=1;COUNTERS["reference_groups"]+=4
    def test_prior_v22_rejects_computed_public_divisor(self):
        class Division(Model):
            def forward(self,x):return x/self.public.reshape(2,3)
        self.failure(lambda:trace(Division("reshape",0),[dict(name="x",shape=[2,3])],
            resource_profile=PROFILE,**TRACE_OPTIONS),"Only a direct declared public tensor divisor")
    def test_prior_v22_direct_divisor_unchanged(self):
        class Direct(Model):
            def forward(self,x):return x/self.public
        module=Direct("reshape",0)
        prior=trace(module,[dict(name="x",shape=[6])],resource_profile=PROFILE,**TRACE_OPTIONS)
        new=selected(module,[6])
        self.assertEqual(inventory(prior),inventory(new))
    def test_static_signature_shape_can_reshape_public(self):
        class Signature(Model):
            def forward(self,x):return x/self.public.view(x.shape)
        _,norm,_,record=prepared(Signature("reshape",0),[1,2,3])
        self.assertEqual(record["entries"][0]["plan"]["attributes"],dict(shape=[1,2,3]))
        np.testing.assert_array_equal(norm(torch.ones(1,2,3,dtype=torch.float64)).detach().numpy(),
                                      (1/np.arange(1,7)).reshape(1,2,3))
    def test_unknown_policy_is_rejected(self):
        self.failure(lambda:trace(Model("reshape",0),[dict(name="x",shape=[2,3])],resource_profile=PROFILE,
                                   **TRACE_OPTIONS,public_expression_policy="unknown"),
                     "Explicit public expression metadata policy")
    def test_policy_requires_ordered_division_contract(self):
        self.failure(lambda:trace(Model("reshape",0),[dict(name="x",shape=[2,3])],resource_profile=PROFILE,
                                   public_expression_policy=POLICY),"Explicit public expression metadata policy")
    def test_encrypted_divisor_rejected(self):
        class Encrypted(torch.nn.Module):
            def forward(self,x):return x/x.reshape(2,3)
        self.failure(lambda:selected(Encrypted(),[2,3]),"public_expression_divisor_not_closed")
    def test_divisor_zero_rejected_without_execution(self):
        class Zero(Model):
            def forward(self,x):return x/(self.public-1).reshape(2,3)
        self.failure(lambda:selected(Zero("reshape",0),[2,3]),"public_expression_operation_not_registered")
        class ZeroDeclared(torch.nn.Module):
            def __init__(self):
                super().__init__();self.register_buffer("p",torch.tensor([0.,1.],dtype=torch.float64))
            def forward(self,x):return x/self.p.reshape(1,2)
        self.failure(lambda:selected(ZeroDeclared(),[1,2]),"nonzero")
    def test_divisor_reciprocal_limit_preserved(self):
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__();self.register_buffer("p",torch.tensor([1e-9],dtype=torch.float64))
            def forward(self,x):return x/self.p.reshape(1)
        self.failure(lambda:selected(Tiny(),[1]),"reciprocal")
    def test_public_mutation_and_nonbool_cast_rejected(self):
        for mode in ("add_","float"):
            class Unsafe(Model):
                def forward(self,x):
                    p=self.public.reshape(2,3)
                    return x*p.add_(1)if mode=="add_"else x*p.float()
            module=Unsafe("reshape",0);before=module.public.clone()
            code="public_expression_operation_not_registered"if mode=="add_"else"public_expression_bool_cast"
            self.failure(lambda:selected(module,[2,3]),code)
            torch.testing.assert_close(module.public,before,rtol=0,atol=0)
    def test_cipher_cast_not_registered_as_public(self):
        class Cipher(torch.nn.Module):
            def forward(self,x):return x.float().size(0)
        self.failure(lambda:selected(Cipher(),[2]),"Static shape unavailable")
    def test_undeclared_literal_tensor_rejected(self):
        class Literal(torch.nn.Module):
            def forward(self,x):return x/torch.tensor([1.,2.],dtype=torch.float64)
        self.failure(lambda:selected(Literal(),[2]),"Static finite arithmetic parameter")
    def test_large_public_value_uses_constant_budget(self):
        class Large(torch.nn.Module):
            def __init__(self):
                super().__init__();self.register_buffer("p",torch.ones(1000,1000,dtype=torch.float64))
            def forward(self,x):
                unused=self.p.reshape(100,100,100)
                return x.square()
        module=Large();traced=selected(module,[2])
        self.assertEqual([n.meta["poseidon_static_tensor"]["shape"]for n in traced.graph.nodes
                          if n.op=="call_method"and n.target=="reshape"],[[100,100,100]])
    def test_rank_profile_bound_preserved(self):
        class Rank(torch.nn.Module):
            def __init__(self):
                super().__init__();self.register_buffer("p",torch.ones((1,)*32,dtype=torch.float64))
            def forward(self,x):return x*self.p.reshape(1)
        module=Rank();selected(module,[1])
        self.failure(lambda:trace(module,[dict(name="x",shape=[1])],resource_profile="resnet20-v1",
                          **TRACE_OPTIONS,public_expression_policy=POLICY),"public_expression_value_capacity")

if __name__=="__main__":unittest.main()
