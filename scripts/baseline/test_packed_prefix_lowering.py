"""Budget-preserving packed-prefix baseline, with independent public-vector checks."""
import ast,copy,unittest
import numpy as np
from benchmark_suite import Builder,generate
from benchmark_graph import samples
from benchmark_math import evaluate
from benchmark_torch import evaluate as reference
from compiler_configuration import PROFILE_SHA256,configuration
from unified_graph_contract import prepare,validate_candidate
from unified_graph_lowering import lower,candidate_source
from unified_public_exercises import golden_variant
from unified_public_contract import CONTRACT

def public_execute(source,request,inputs):
    """Tiny test-only AST evaluator. No eval/exec and never FHE/reference evidence."""
    p=request["layout"]["input_slot_period"]
    env={k:np.asarray(v,dtype=float) for k,v in request["public_constants"].items()}
    env["zero_ct"]=np.zeros(p)
    for b in request["layout"]["inputs"]:
        v=inputs[b["name"]].reshape(-1);env[b["dsl_name"]]=np.pad(v,(0,p-len(v)))
    def expr(n):
        if isinstance(n,ast.Name):return env[n.id]
        if isinstance(n,ast.UnaryOp) and isinstance(n.op,ast.USub):return -expr(n.operand)
        if isinstance(n,ast.BinOp):
            a,b=expr(n.left),expr(n.right)
            if isinstance(n.op,ast.Add):return a+b
            if isinstance(n.op,ast.Sub):return a-b
            if isinstance(n.op,ast.Mult):return a*b
        if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=="rotate":
            return np.roll(expr(n.func.value),-n.args[0].value)
        raise AssertionError(ast.dump(n))
    f=ast.parse(source).body[0]
    for n in f.body[:-1]:
        assert isinstance(n,ast.Assign);env[n.targets[0].id]=expr(n.value)
    return [expr(n) for n in f.body[-1].value.elts]

def models():
    out=[]
    for op in ("add","subtract","multiply","negate","square","power","polynomial"):
        b=Builder([(17,)]);args=["input0"];attrs={}
        if op in ("add","subtract","multiply"):args.append(b.const([.125]))
        if op=="power":attrs=dict(exponent=4)
        if op=="polynomial":args.append(b.const([.25,-.125,.0625]));attrs=dict(basis="power")
        y=b.node(op,args,**attrs);y=b.node("mean",[y],axes=[0],keepdims=False);out.append(b.finish(y))
    b=Builder([(2,3)]);y=b.node("polynomial",["input0",b.const([.25,-.125,.0625])],basis="chebyshev")
    y=b.node("reshape",[y],shape=[6]);out.append(b.finish(y))
    b=Builder([(5,),(5,)]);y=b.node("add",["input0","input1"]);out.append(b.finish(y,"input0"))
    b=Builder([(5,),(1,)]);y=b.node("multiply",["input0","input1"]);out.append(b.finish(y))
    b=Builder([(5,)]);y=b.node("square",["input0"]);y=b.node("rotate",[y],step=-3);out.append(b.finish(y))
    for weights in ([.125]*17,[.125]*16+[.25]):
        b=Builder([(17,)]);y=b.node("add",["input0",b.const(weights)])
        y=b.node("mean",[y],axes=[0],keepdims=False);out.append(b.finish(y))
    return out

class PackedPrefixTests(unittest.TestCase):
    def test_all_lane_local_paths_and_layout_fallbacks(self):
        for g in models():
            r=prepare(g,PROFILE_SHA256)
            src=lower(r,packed_prefix=True)
            validate_candidate(dict(schema=1,request_id=r["request_id"],hecate_source=src),r)
            for inputs in samples(g,4):
                actual=public_execute(src,r,inputs);expected=evaluate(g,inputs);ref=reference(g,inputs)
                for i,o in enumerate(g["outputs"]):
                    y=expected[o["name"]].reshape(-1)
                    np.testing.assert_allclose(actual[i][:len(y)],y,atol=1e-12,rtol=1e-12)
                    np.testing.assert_allclose(y,ref[o["name"]].reshape(-1),atol=1e-12,rtol=1e-12)
    def test_original_scalar_source_preserved_when_valid(self):
        for g in models()[:7]:
            r=prepare(g,PROFILE_SHA256);src,record=candidate_source(r)
            self.assertEqual(src,lower(r));self.assertFalse(record["fallback"])
    def test_three_frozen_directed_budgets_without_expansion(self):
        from benchmark_runner import load,DEFAULT
        import json
        rows,_=load(DEFAULT);models_by_id={r["model"]["id"]:r["model"] for r in rows}
        tasks=json.loads((DEFAULT/"coverage.json").read_text())["directed_tasks"]
        for t in tasks:
            if t["model_id"] not in ("bench_boundary_0044","bench_boundary_0061","bench_boundary_0097"):continue
            r=prepare(models_by_id[t["model_id"]],PROFILE_SHA256,configuration("seal-cpu-eva-w45-v1"),
                      t["exercise"],construction_profile=CONTRACT);before=copy.deepcopy(r)
            with self.assertRaisesRegex(ValueError,"Expanded ciphertext operation budget exceeded"):
                validate_candidate(dict(schema=1,request_id=r["request_id"],
                    hecate_source=golden_variant(lower(r),t["exercise"],r)),r)
            src,record=candidate_source(r);self.assertTrue(record["fallback"]);self.assertEqual(before,r)
            checked=validate_candidate(dict(schema=1,request_id=r["request_id"],hecate_source=src),r)
            from unified_public_contract import normalize
            expanded=normalize(src,r)
            self.assertLessEqual(len(ast.parse(expanded["source"]).body[0].body)-1,256)
            self.assertLessEqual(checked["functions"]["golden"]["expanded_cost"],1024)
    def test_native_capacity_fallback_keeps_request_and_reference(self):
        for size in (64,128,256):
            b=Builder([(size,)])
            y=b.node("square",["input0"])
            y=b.node("add",[y,"input0"])
            g=b.finish(y)
            r=prepare(g,PROFILE_SHA256);before=copy.deepcopy(r)
            src,record=candidate_source(r)
            self.assertTrue(record["fallback"])
            self.assertEqual(before,r)
            validate_candidate(dict(schema=1,request_id=r["request_id"],hecate_source=src),r)
            for inputs in samples(g,4):
                actual=public_execute(src,r,inputs)[0][:size]
                expected=evaluate(g,inputs)[g["outputs"][0]["name"]].reshape(-1)
                ref=reference(g,inputs)[g["outputs"][0]["name"]].reshape(-1)
                np.testing.assert_allclose(actual,expected,atol=1e-12,rtol=1e-12)
                np.testing.assert_allclose(expected,ref,atol=1e-12,rtol=1e-12)
    def test_nonbudget_error_not_hidden_by_fallback(self):
        g=models()[0];r=prepare(g,PROFILE_SHA256);r["request_id"]="0"*64
        with self.assertRaises(ValueError):candidate_source(r)
    def test_chunk_path_remains_explicit(self):
        r=prepare(models()[0],PROFILE_SHA256,chunk_period=8)
        with self.assertRaisesRegex(ValueError,"chunk ABI"):lower(r,packed_prefix=True)

if __name__=="__main__":unittest.main()
