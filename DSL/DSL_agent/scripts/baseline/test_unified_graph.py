"""Opt-in unified ABI tests. Plaintext DSL probes are not encrypted evidence."""
import copy
import json
import unittest
import numpy as np
from benchmark_suite import Builder, apply
from benchmark_graph import samples
from benchmark_math import evaluate
from unified_graph_contract import prepare, validate_request, validate_candidate, ABI
from unified_graph_lowering import lower
from compiler_configuration import configuration, PROFILE_SHA256

class Vec:
    __array_priority__=10000
    def __init__(self,v):self.v=np.asarray(v,dtype=np.float64)
    def other(self,x):return x.v if isinstance(x,Vec) else x
    def __add__(self,x):return Vec(self.v+self.other(x))
    __radd__=__add__
    def __sub__(self,x):return Vec(self.v-self.other(x))
    def __rsub__(self,x):return Vec(self.other(x)-self.v)
    def __mul__(self,x):return Vec(self.v*self.other(x))
    __rmul__=__mul__
    def __neg__(self):return Vec(-self.v)
    def rotate(self,k):return Vec(np.roll(self.v,-k))

def probe(request,source,inputs):
    from candidate_trace import evaluate_tree
    p=request["layout"]["input_slot_period"]
    args={s["dsl_name"]:Vec(np.pad(inputs[s["name"]].reshape(-1),(0,p-inputs[s["name"]].size)))
          for s in request["layout"]["inputs"]}
    args["zero_ct"]=Vec(np.zeros(p))
    constants={k:Vec(v if type(v) is list else [v]*p) for k,v in request["public_constants"].items()}
    out=evaluate_tree(source,constants,encrypted_inputs=args)
    return np.asarray([out[i].v[j] for i,j in request["layout"]["output_selectors"]])

class UnifiedTests(unittest.TestCase):
    def request(self,g):
        return prepare(g,PROFILE_SHA256,configuration("seal-cpu-eva-w45-v1"))
    def assert_pipeline(self,g):
        r=self.request(g);source=lower(r)
        validate_candidate(dict(schema=1,request_id=r["request_id"],hecate_source=source),r)
        for inputs in samples(g,4):
            expected=np.concatenate([v.reshape(-1) for v in evaluate(g,inputs).values()])
            np.testing.assert_allclose(probe(r,source,inputs),expected,atol=1e-12,rtol=1e-12)
        return r,source
    def test_different_shapes_multioutput(self):
        b=Builder([(2,),(1,2)])
        z=b.node("reshape",["input1"],shape=[2]);y=b.node("add",["input0",z])
        avg=b.node("mean",[y],axes=[0],keepdims=False)
        self.assert_pipeline(b.finish(y,avg))
    def test_split_stack_reduction(self):
        b=Builder([(6,)])
        p=b.node("split",["input0"],axis=0,sections=[3,3])
        s=b.node("stack",p,axis=0);y=b.node("sum",[s],axes=[0],keepdims=False)
        self.assert_pipeline(b.finish(s,y))
    def test_tensor_transpose_linear(self):
        b=Builder([(2,3)]);x=apply(b,"input0","transpose")
        x=apply(b,x,"linear",1);x=b.node("square",[x])
        self.assert_pipeline(b.finish(x))
    def test_spatial_bn(self):
        from benchmark_suite import spatial
        b=Builder([(1,1,3,3)]);x=spatial(b,"input0",2,0)
        x=apply(b,x,"batch_norm");x=b.node("mean",[x],axes=[-1],keepdims=True)
        self.assert_pipeline(b.finish(x))
    def test_exact_zero_uses_auxiliary_cipher(self):
        b=Builder([(2,)]);x=b.node("subtract",["input0","input0"])
        r,s=self.assert_pipeline(b.finish(x))
        self.assertIn("zero_ct",s)
    def test_request_contains_no_rule_answer(self):
        b=Builder([(2,)]);g=b.finish(apply(b,"input0","square"))
        r=self.request(g)
        self.assertNotIn("hecate_source",r)
        self.assertNotIn(lower(r),[str(v) for v in r.values()])
        self.assertNotIn("reference",r);self.assertNotIn("test_inputs",r)
        from deepseek_provider import public_request
        self.assertEqual(public_request(r),r)
    def test_layout_tampering_even_with_new_hash_rejected(self):
        from benchmark_graph import digest
        b=Builder([(2,)]);g=b.finish(apply(b,"input0","square"));r=self.request(g)
        r["layout"]["inputs"][0]["dsl_name"]="y"
        r["request_id"]=digest({k:v for k,v in r.items() if k!="request_id"})
        with self.assertRaises(ValueError):validate_request(r)
    def test_unknown_or_forged_public_constants_rejected(self):
        from benchmark_graph import digest
        b=Builder([(2,)]);g=b.finish(apply(b,"input0","square"));r=self.request(g)
        r["public_constants"]["new"]=0.25
        r["request_id"]=digest({k:v for k,v in r.items() if k!="request_id"})
        with self.assertRaises(ValueError):validate_request(r)
    def test_unsafe_source_rejected(self):
        b=Builder([(2,)]);g=b.finish(apply(b,"input0","square"));r=self.request(g)
        for source in ('import os\n','@hc.func("c,c")\ndef golden(x,zero_ct):\n return hc.bootstrap(x)\n',
                       '@hc.func("c,c")\ndef golden(x,zero_ct):\n return x.rotate(-1)\n'):
            with self.subTest(source=source),self.assertRaises(ValueError):
                validate_candidate(dict(schema=1,request_id=r["request_id"],hecate_source=source),r)
    def test_artifact_policy_keeps_old_abi_restricted(self):
        from cipher_abi import execution_options
        b=Builder([(2,),(2,),(2,),(2,)])
        x=b.node("add",["input0","input1"]);g=b.finish(x)
        r=self.request(g);opts=execution_options(r["layout"])
        self.assertEqual(opts["expected_inputs"],5);self.assertEqual(opts["execution_abi"],ABI)
        from seal_artifact_gate import inspect_artifacts
        from packed_input_abi import ABI as OLD
        with self.assertRaisesRegex(ValueError,"one model input"):
            inspect_artifacts(b"",b"",execution_abi=OLD,expected_inputs=3,input_period=4)

if __name__=="__main__":unittest.main()
