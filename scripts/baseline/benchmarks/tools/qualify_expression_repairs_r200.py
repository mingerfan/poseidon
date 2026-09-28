"""Bounded offline diagnosis only: frozen models, no paid provider, no source substitution in Agent scores."""
import ast,copy,json,sys,shlex,time,hashlib,shutil
from pathlib import Path
BASE=Path(__file__).resolve().parents[2];ROOT=BASE.parents[1]
sys.path[:0]=[str(BASE),str(Path(__file__).parent)]
if __name__=="__main__" and "--inside" not in sys.argv:
    from hecate_python_env import enter_nix,VENV
    raise SystemExit(enter_nix('LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+
        shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"]),seconds=1800))
def double_add(source,public):
    tree=ast.parse(source);names={n.id for n in ast.walk(tree) if isinstance(n,ast.Name)}
    counter=0
    class Transform(ast.NodeTransformer):
        def visit_BinOp(self,n):
            nonlocal counter
            n=self.generic_visit(n)
            if not isinstance(n.op,ast.Mult):return n
            def two(x):
                return (isinstance(x,ast.Name) and type(public.get(x.id)) in (int,float) and public[x.id]==2 or
                        isinstance(x,ast.Constant) and type(x.value) in (int,float) and x.value==2)
            value=n.right if two(n.left) else n.left if two(n.right) else None
            if value is None:return n
            name="sharedDouble"+str(counter);counter+=1
            assert name not in names
            self.statements.append(ast.Assign(targets=[ast.Name(id=name,ctx=ast.Store())],value=value))
            return ast.BinOp(left=ast.Name(id=name,ctx=ast.Load()),op=ast.Add(),right=ast.Name(id=name,ctx=ast.Load()))
    for fn in tree.body:
        assert isinstance(fn,ast.FunctionDef)
        body=[]
        for stmt in fn.body:
            assert isinstance(stmt,(ast.Assign,ast.Return))
            tr=Transform();tr.statements=[];new=tr.visit(stmt);body.extend(tr.statements);body.append(new)
        fn.body=body
    return ast.unparse(ast.fix_missing_locations(tree))+"\n",counter

def fused_source(q):
    from exact_series_diagnostics import multiply_argument,finite_literals
    from chebyshev_lowering import balanced
    import unified_graph_lowering as lowering
    nodes=q["model"]["nodes"];g=q["model"]
    assert len(g["inputs"])==1 and len(g["outputs"])==1 and len(nodes)==3
    poly,add,mul=nodes
    assert [n["op"] for n in nodes]==["polynomial","add","multiply"]
    assert poly["attrs"]=={"basis":"chebyshev"} and poly["inputs"][0]==g["inputs"][0]["name"]
    assert add["inputs"][0]==poly["outputs"][0]
    assert set(mul["inputs"])=={poly["inputs"][0],add["outputs"][0]}
    assert g["outputs"][0]["value"]==mul["outputs"][0]
    coeff=g["constants"][poly["inputs"][1]];offset=g["constants"][add["inputs"][1]]
    assert type(offset) in (float,int)
    derived=finite_literals(multiply_argument(coeff,offset))
    em=DerivedEmitter(q);binding=q["layout"]["inputs"][0]
    y=balanced(lowering.Scalar(em,binding["dsl_name"]),derived,em.zero)
    return '@hc.func("c,c")\ndef golden('+binding["dsl_name"]+',zero_ct):\n'+"\n".join(em.lines)+"\n    return ["+y.name+"]\n"

# Only the manual diagnostic emitter permits derived public literals within the
# existing source validator budget. Production emitter and request stay unchanged.
from unified_graph_lowering import Emitter
class DerivedEmitter(Emitter):
    def literal(self,x):
        import math
        x=float(x)
        if x in self.constants:return super().literal(x)
        if not math.isfinite(x) or abs(x)>1024:raise ValueError("Existing literal budget")
        return repr(x)

def main():
    from hecate_python_env import enter_nix,VENV,WORK
    if "--inside" not in sys.argv:
        return enter_nix('LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+
            shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"]),seconds=1800)
    from semantic_benchmark_execution import runtime_sources
    from component_contract import request_options,reconstruct_request
    from unified_graph_contract import prepare,validate_candidate,static_repair_hint
    from component_backend import qualify
    from benchmark_graph import digest,samples
    from benchmark_math import evaluate as maths
    from benchmark_torch import evaluate as torch_reference
    from test_packed_prefix_lowering import public_execute
    from probe_native_diagnostics_r195 import repair
    from exact_series_diagnostics import parity_evaluate
    import unified_graph_lowering as lowering
    import chebyshev_lowering
    import numpy as np
    R=WORK/"results";out=R/"stage2-expression-repairs-r200/offline";out.mkdir()
    previous=json.loads((R/"seal-gate-r195/prepared.json").read_text())
    tracker=json.loads((ROOT/"docs/baseline/seal-artifact-gate-closure-r198.json").read_text())
    original={s["id"]:s for s in previous["cases"]};requests={};rows=[]
    guards=json.loads((R/"stage2-expression-repairs-r199/before.json").read_text())["compiler"]
    sources=runtime_sources();started=time.monotonic()
    def guard():
        assert runtime_sources()==sources,"Runtime source drift"
        assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in guards.items()),"Compiler drift"
        assert time.monotonic()-started<1700,"Offline time budget"
        assert shutil.disk_usage(R).free>4*1024**3,"Disk reserve"
        from retained_artifact_usage import size_bytes
        assert size_bytes([R])<32*1024**3,"Global retained budget"
        assert size_bytes([out])<4*1024**3,"Local retained budget"
    def save():
        report=dict(format="poseidon-expression-repairs-r200",source_hashes=sources,rows=rows,
                    paid_calls=0,new_agent_successes=0,compiler_changed=False,seconds=time.monotonic()-started)
        report["binding"]=digest(report);(out/"report.json").write_text(json.dumps(report,indent=2))
    for row in tracker["rows"]:
        if row["classification"]=="passed":continue
        old=original[row["id"]]["request"];opts=request_options(old);opts.pop("compiler_configuration")
        opts["generation_guidance"]="explicit-v9"
        q=prepare(old["model"],old["compiler_profile_sha256"],old.get("compiler_configuration"),
                  constant_policy=old["constant_origins"].get("policy"),**opts)
        assert reconstruct_request(q)==q
        omit={"generation_guidance","request_id"}
        assert {k:v for k,v in q.items() if k not in omit}=={k:v for k,v in old.items() if k not in omit}
        requests[row["id"]]=q
    (out/"prepared.json").write_text(json.dumps(dict(requests=requests,source_hashes=sources),indent=2))
    def check(task,strategy,source,execute=False):
        guard();q=requests[task];folder=out/(task+"-"+strategy);folder.mkdir()
        candidate=dict(schema=1,request_id=q["request_id"],hecate_source=source)
        (folder/"job.json").write_text(json.dumps(dict(request=q,candidate=candidate,manual_fixture=True),indent=2))
        item=dict(id=task,strategy=strategy,request_id=q["request_id"],source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                  manual_only=True,job=str(folder/"job.json"),paid_calls=0)
        try:
            validate_candidate(candidate,q);item["static_passed"]=True
            # Independent original graph references; emitted DSL is only the compared subject.
            maximum=0.;groups=0
            for xs in samples(q["model"],16):
                actual=public_execute(source,q,xs)
                a=maths(q["model"],xs);b=torch_reference(q["model"],xs)
                for index,spec in enumerate(q["model"]["outputs"]):
                    expected=a[spec["name"]].reshape(-1)
                    np.testing.assert_allclose(expected,b[spec["name"]].reshape(-1),atol=1e-12,rtol=1e-12)
                    got=actual[index][:expected.size]
                    np.testing.assert_allclose(got,expected,atol=2e-12,rtol=2e-12)
                    maximum=max(maximum,float(np.max(np.abs(got-expected))))
                groups+=1
            item["plaintext"]=dict(groups=groups,max_abs=maximum,passed=True)
            if execute:item["result"]=qualify(q,candidate)
        except (ValueError,AssertionError,TypeError) as error:
            item["rejected"]=type(error).__name__+": "+str(error)
        rows.append(item);save();print(json.dumps(item),flush=True);guard()
        if item.get("result",{}).get("failure",{}).get("layer") in ("integrity","environment"):
            raise ValueError("Integrity/environment stop")
    # Seven original static failures: syntax/type repair remains distinct from execution.
    for row in tracker["rows"]:
        if row["classification"]!="static_check":continue
        folder=Path(row["evidence"]);rep=json.loads((folder/"report.json").read_text());idx=rep["attempts"][-1]["index"]
        c=json.loads((folder/("attempt-%02d"%idx)/"response.txt").read_text())
        source,changes=repair(c["hecate_source"],requests[row["id"]]["public_constants"])
        if row["id"]=="free_bench_helper_0045":
            # The original slicing is invalid; independently emit the SAME graph with explicit extraction/masks.
            source=lowering.lower(requests[row["id"]],packed_prefix=True,balanced_chebyshev=True)
            strategy="trusted-layout-baseline"
        else:strategy="manual-type-scope"
        check(row["id"],strategy,source,execute=False)
    task="free_bench_helper_0113";q=requests[task]
    source=(R/"agent-deepseek-wvg3br4p/attempt-01/candidate.py").read_text()
    doubled,count=double_add(source,q["public_constants"]);assert count>0
    check(task,"shared-add-doubling",doubled,execute=True)
    check(task,"exact-series-fusion",fused_source(q),execute=True)
    # New parity representation, not a rerun of the prior balanced or split strategy.
    original_emitter=lowering.Emitter;original_balanced=chebyshev_lowering.balanced
    try:
        lowering.Emitter=DerivedEmitter;chebyshev_lowering.balanced=parity_evaluate
        for task in ("free_bench_helper_0104","free_bench_helper_0032"):
            check(task,"exact-parity-basis",lowering.lower(requests[task],packed_prefix=True,balanced_chebyshev=True),execute=True)
    finally:
        lowering.Emitter=original_emitter;chebyshev_lowering.balanced=original_balanced
    guard();save()
    return 0
if __name__=="__main__":raise SystemExit(main())
