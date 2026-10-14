"""Trusted fixtures only: FX provenance, real Torch/NumPy reference and rejection tests."""
import copy,hashlib,operator,unittest
from pathlib import Path
import numpy as np
import torch
from fx_public import normalize,check,arrays,state_identity,graph_hash
SOURCE=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
COUNTERS=dict(contexts=0,reference_groups=0,strict_traces=0)

class PublicTransform(torch.nn.Module):
    def __init__(self,operation,width):
        super().__init__();self.operation=operation;self.width=width
        if operation=="permute":raw=torch.tensor([[1,2,4],[2,4,8]],dtype=torch.float64)[:,:width]
        elif operation=="slice":raw=torch.tensor([1,2,4,8]*width,dtype=torch.float64)[:2*width]
        elif operation=="bool_cast":raw=torch.tensor([i%2==0 for i in range(width)],dtype=torch.bool)
        else:raw=torch.tensor([2**i for i in range(width)],dtype=torch.float64)
        self.register_buffer("public",raw)
    def forward(self,x):
        if self.operation=="reshape":return x/self.public.view(1,self.width)
        if self.operation=="permute":return x/self.public.permute(1,0)
        if self.operation=="slice":return x/self.public[::2]
        return x*self.public.double()

def capture(module):
    tracer=torch.fx.Tracer();tracer.proxy_buffer_attributes=True
    graph=tracer.trace(module)
    result=torch.fx.GraphModule(module,graph)
    for node in result.graph.nodes:node.meta["draft_ordered_origin"]=[node.name]
    return result

def prepare(module):
    graph=capture(module);raw=arrays(graph);expected=state_identity(raw,"general-tensors-v1")
    norm,record=normalize(graph,raw,expected_state=expected,source_sha256=SOURCE)
    check(norm,record,raw,expected_binding=record["binding"],source_sha256=SOURCE)
    return graph,raw,norm,record

def mathematical_public(module):
    a=module.public.detach().numpy()
    if module.operation=="reshape":return a.reshape(1,module.width)
    if module.operation=="permute":return a.transpose(1,0)
    if module.operation=="slice":return a[::2]
    return a.astype("float64")

class Controls(unittest.TestCase):
    def assert_failure(self,fn,code):
        with self.assertRaisesRegex(ValueError,code):fn()
    def context(self,operation,width):
        module=PublicTransform(operation,width);source,raw,norm,record=prepare(module)
        self.assertEqual(len(record["entries"]),1)
        expected_public=mathematical_public(module)
        shape=tuple(expected_public.shape)
        random=np.random.default_rng(237+width)
        probes=[np.zeros(shape),np.linspace(-1,1,expected_public.size).reshape(shape),
                random.normal(size=shape),np.full(shape,2.)]
        for x in probes:
            independent=x*expected_public if operation=="bool_cast"else x/expected_public
            original=module(torch.from_numpy(x)).detach().numpy()
            normalized=norm(torch.from_numpy(x)).detach().numpy()
            np.testing.assert_array_equal(original,independent)
            np.testing.assert_array_equal(normalized,independent)
        for node in norm.graph.nodes:
            if node.name in {n.name for n in source.graph.nodes}:
                self.assertEqual(node.meta["draft_ordered_origin"],[node.name])
        from pytorch_static_trace import trace
        from pytorch_public_arithmetic_capture import TRACE_OPTIONS
        traced=trace(norm,[dict(name="x",shape=list(shape))],resource_profile="general-tensors-v1",**TRACE_OPTIONS)
        self.assertEqual(sum(n.op=="placeholder"for n in traced.graph.nodes),1)
        self.assertFalse(any(n.op=="call_method"and n.target in("float","double","to","view","permute")for n in traced.graph.nodes))
        COUNTERS["contexts"]+=1;COUNTERS["reference_groups"]+=len(probes);COUNTERS["strict_traces"]+=1
    def test_four_semantics_three_contexts(self):
        for operation in("reshape","permute","slice","bool_cast"):
            for width in (1,2,3):
                with self.subTest(operation=operation,width=width):self.context(operation,width)
    def test_original_strict_trace_rejects_computed_divisor(self):
        from pytorch_static_trace import trace
        from pytorch_public_arithmetic_capture import TRACE_OPTIONS
        self.assert_failure(lambda:trace(PublicTransform("reshape",3),[dict(name="x",shape=[1,3])],
            resource_profile="general-tensors-v1",**TRACE_OPTIONS),"direct declared public tensor divisor")
    def test_source_graph_state_preserved(self):
        module=PublicTransform("reshape",3);before=module.public.detach().clone()
        graph,raw,norm,record=prepare(module)
        self.assertEqual(graph_hash(graph),record["source_fx_sha256"])
        torch.testing.assert_close(module.public,before,rtol=0,atol=0)
        self.assertIs(record["production_integrated"],False)
    def test_unused_large_weight_is_bound_without_expression_narrowing(self):
        graph=capture(PublicTransform("reshape",3))
        graph.register_buffer("unused",torch.arange(6000,dtype=torch.float64))
        raw=arrays(graph)
        norm,record=normalize(graph,raw,expected_state=state_identity(raw,"general-tensors-v1"),source_sha256=SOURCE)
        self.assertIn("unused",norm.state_dict())
        self.assertEqual(record["original_state"]["unused"],record["normalized_state"]["unused"])
        check(norm,record,raw,expected_binding=record["binding"],source_sha256=SOURCE)
    def test_wrong_original_public_bytes_rejected(self):
        graph=capture(PublicTransform("reshape",3));raw=arrays(graph);expected=state_identity(raw,"general-tensors-v1")
        raw["public"][0]=4.
        self.assert_failure(lambda:normalize(graph,raw,expected_state=expected,source_sha256=SOURCE),"public_state_binding")
    def test_synthetic_public_attr_rejected(self):
        module=PublicTransform("reshape",3);module.side=torch.tensor([1,2,4],dtype=torch.float64)
        graph=torch.fx.Graph();x=graph.placeholder("x");p=graph.get_attr("side");y=graph.call_function(operator.mul,(x,p));graph.output(y)
        traced=torch.fx.GraphModule(module,graph)
        self.assert_failure(lambda:normalize(traced,{},expected_state={},source_sha256=SOURCE),"public_state_binding")
    def test_synthetic_nonstate_attribute_rejected(self):
        module=torch.nn.Module();module.side=2
        graph=torch.fx.Graph();x=graph.placeholder("x");p=graph.get_attr("side");y=graph.call_function(operator.mul,(x,p));graph.output(y)
        traced=torch.fx.GraphModule(module,graph)
        self.assert_failure(lambda:normalize(traced,{},expected_state={},source_sha256=SOURCE),"public_attribute_not_declared")
    def test_derived_payload_tamper_rejected(self):
        graph,raw,norm,record=prepare(PublicTransform("reshape",3))
        attr=record["entries"][0]["attribute"];getattr(norm,attr).flatten()[0]=8
        self.assert_failure(lambda:check(norm,record,raw,expected_binding=record["binding"],source_sha256=SOURCE),"public_record_state")
    def test_record_tamper_rejected(self):
        graph,raw,norm,record=prepare(PublicTransform("reshape",3))
        binding=record["binding"];record["entries"][0]["step"]["attributes"]["shape"]=[3,1]
        self.assert_failure(lambda:check(norm,record,raw,expected_binding=binding,source_sha256=SOURCE),"public_record_authority")
    def test_wrong_source_binding_rejected(self):
        graph,raw,norm,record=prepare(PublicTransform("reshape",3))
        self.assert_failure(lambda:check(norm,record,raw,expected_binding=record["binding"],source_sha256="0"*64),"public_record_identity")
    def test_wrong_resource_profile_rejected(self):
        graph,raw,norm,record=prepare(PublicTransform("reshape",3))
        self.assert_failure(lambda:check(norm,record,raw,expected_binding=record["binding"],source_sha256=SOURCE,resource_profile="medium-v1"),"public_record_identity")
    def test_encrypted_operations_are_not_public_folds(self):
        class Model(torch.nn.Module):
            def forward(self,x):return x.reshape(1,3).square()
        graph=capture(Model());norm,record=normalize(graph,{},expected_state={},source_sha256=SOURCE)
        self.assertEqual(record["entries"],[])
        self.assertEqual(graph_hash(graph),graph_hash(norm))
    def test_public_inplace_mutation_rejected(self):
        class Model(torch.nn.Module):
            def __init__(self):super().__init__();self.register_buffer("value",torch.ones(3,dtype=torch.float64))
            def forward(self,x):return x*self.value.add_(1)
        graph=capture(Model());raw=arrays(graph)
        self.assert_failure(lambda:normalize(graph,raw,expected_state=state_identity(raw,"general-tensors-v1"),source_sha256=SOURCE),"public_operation_not_registered")
        np.testing.assert_array_equal(arrays(graph)["value"],raw["value"])
    def test_encrypted_metadata_is_not_a_public_transform(self):
        class Model(torch.nn.Module):
            def __init__(self):super().__init__();self.register_buffer("value",torch.ones(3,dtype=torch.float64))
            def forward(self,x):return x/self.value.view(x.size(0),-1)
        graph=capture(Model());raw=arrays(graph)
        self.assert_failure(lambda:normalize(graph,raw,expected_state=state_identity(raw,"general-tensors-v1"),source_sha256=SOURCE),"public_expression_dynamic_metadata")
    def test_non_boolean_precision_cast_rejected(self):
        class Model(torch.nn.Module):
            def __init__(self):super().__init__();self.register_buffer("value",torch.ones(3,dtype=torch.float64))
            def forward(self,x):return x*self.value.float()
        graph=capture(Model());raw=arrays(graph)
        self.assert_failure(lambda:normalize(graph,raw,expected_state=state_identity(raw,"general-tensors-v1"),source_sha256=SOURCE),"public_bool_cast")
    def test_function_spelling(self):
        class Model(torch.nn.Module):
            def __init__(self):super().__init__();self.register_buffer("value",torch.arange(1,4,dtype=torch.float64))
            def forward(self,x):return x/torch.reshape(self.value,(1,3))
        graph,raw,norm,record=prepare(Model());self.assertEqual(record["entries"][0]["source_target"],"torch.reshape")
    def test_static_multi_step_origin_chain(self):
        class Model(torch.nn.Module):
            def __init__(self):super().__init__();self.register_buffer("value",torch.arange(1,7,dtype=torch.float64))
            def forward(self,x):return x/self.value.reshape(2,3).permute(1,0)[::2]
        graph,raw,norm,record=prepare(Model())
        self.assertEqual(len(record["entries"]),3)
        self.assertEqual(record["entries"][1]["step"]["input"],record["entries"][0]["source_fx_node"])
        self.assertTrue(all(e["roots"]==["value"]for e in record["entries"]))
    def test_boolean_cast_spellings(self):
        for method in("float","double","to"):
            class Model(torch.nn.Module):
                def __init__(self):super().__init__();self.register_buffer("mask",torch.tensor([True,False]))
                def forward(self,x):
                    m=self.mask.float()if method=="float"else self.mask.double()if method=="double"else self.mask.to(dtype=torch.float64)
                    return x*m
            graph,raw,norm,record=prepare(Model())
            self.assertEqual(record["entries"][0]["step"]["op"],"bool_cast")
            np.testing.assert_array_equal(norm(torch.ones(2,dtype=torch.float64)).detach().numpy(),[1,0])
    def test_collision_rejected(self):
        graph=capture(PublicTransform("reshape",3))
        graph.register_buffer("_poseidon_public_expression_0",torch.ones(1,dtype=torch.float64))
        raw=arrays(graph)
        self.assert_failure(lambda:normalize(graph,raw,expected_state=state_identity(raw,"general-tensors-v1"),source_sha256=SOURCE),"public_materialization_collision")
    def test_repeated_normalization_is_deterministic(self):
        graph=capture(PublicTransform("reshape",3));raw=arrays(graph);expected=state_identity(raw,"general-tensors-v1")
        a,ra=normalize(graph,raw,expected_state=expected,source_sha256=SOURCE)
        b,rb=normalize(graph,raw,expected_state=expected,source_sha256=SOURCE)
        self.assertEqual(ra,rb)
    def test_public_unregistered_operation_rejected(self):
        class Model(torch.nn.Module):
            def __init__(self):super().__init__();self.register_buffer("value",torch.ones(3,dtype=torch.float64))
            def forward(self,x):return x*self.value.square()
        graph=capture(Model());raw=arrays(graph)
        self.assert_failure(lambda:normalize(graph,raw,expected_state=state_identity(raw,"general-tensors-v1"),source_sha256=SOURCE),"public_operation_not_registered")

if __name__=="__main__":unittest.main()
