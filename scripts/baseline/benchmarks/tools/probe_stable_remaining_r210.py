"""Bounded manual feasibility only; zero API calls and immutable original models."""
import sys,shlex,json,hashlib,time
from pathlib import Path
BASE=Path(__file__).resolve().parents[2];ROOT=BASE.parents[1]
sys.path[:0]=[str(BASE),str(Path(__file__).parent)]
def main():
    from hecate_python_env import VENV,enter_nix,WORK
    if "--inside" not in sys.argv:
        return enter_nix('LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+
            shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"]),seconds=1800)
    import numpy as np
    import chebyshev_lowering
    from packed_graph_diagnostic import source
    from exact_power_diagnostics import estrin
    from manual_native_evaluation import evaluate as actual
    from benchmark_math import evaluate
    from benchmark_torch import evaluate as reference
    from benchmark_graph import samples,digest
    from unified_graph_contract import validate_candidate
    from component_backend import qualify
    from semantic_benchmark_execution import runtime_sources
    from unified_graph_lowering import Emitter,Scalar
    R=WORK/"results";out=R/"stage2-expression-repairs-r210";report_path=out/"report.json"
    if report_path.exists():raise ValueError("Refuse to overwrite existing evidence")
    requests=json.loads((R/"stage2-expression-repairs-r201/offline/prepared.json").read_text())["requests"]
    sources=runtime_sources();guard=json.loads((out/"before.json").read_text())["compiler"]
    original_literal=Emitter.literal
    def literal(self,x):
        if float(x) in self.constants:return original_literal(self,x)
        if not np.isfinite(x) or abs(x)>1024:raise ValueError("Literal bound")
        return repr(float(x))
    Emitter.literal=literal
    rows=[]
    for task,strategy in (("free_bench_helper_0037","balanced-chebyshev"),("free_bench_helper_0039","balanced-chebyshev")):
        q=requests[task];item=dict(id=task,manual_only=True,paid_calls=0,strategy=strategy)
        started=time.monotonic()
        try:
            if strategy=="balanced-chebyshev":src=source(q)
            else:
                em=Emitter(q);binding=q["layout"]["inputs"][0];original=Scalar(em,binding["dsl_name"])
                values=dict(q["model"]["constants"]);values[binding["name"]]=original
                assert [n["op"] for n in q["model"]["nodes"]]==["polynomial"]*3+["add","multiply"]
                for i,n in enumerate(q["model"]["nodes"]):
                    args=[values[k] for k in n["inputs"]]
                    if n["op"]=="polynomial":y=estrin(args[0],args[1],em.zero,weight=original if i==2 else None)
                    elif n["op"]=="add":y=args[0]+original*args[1]
                    else:y=args[1]
                    values[n["outputs"][0]]=y
                src='@hc.func("c,c")\ndef golden('+binding["dsl_name"]+',zero_ct):\n'+"\n".join(em.lines)+"\n    return ["+y.name+"]\n"
            candidate=dict(schema=1,request_id=q["request_id"],hecate_source=src)
            path=out/(task+"-"+strategy+".json")
            path.write_text(json.dumps(dict(request=q,candidate=candidate,manual_fixture=True),indent=2))
            item["job"]=str(path);item["source_sha256"]=hashlib.sha256(src.encode()).hexdigest()
            check=validate_candidate(candidate,q);item["static_passed"]=True
            item["expanded_cost"]=check["functions"]["golden"]["expanded_cost"]
            maximum=0.;reference_max=0.;groups=0
            for xs in samples(q["model"],16):
                a=actual(src,q,xs);b=evaluate(q["model"],xs);c=reference(q["model"],xs)
                for i,o in enumerate(q["model"]["outputs"]):
                    expected=b[o["name"]].reshape(-1)
                    np.testing.assert_allclose(expected,c[o["name"]].reshape(-1),atol=1e-12,rtol=1e-12)
                    delta=np.abs(a[i][:len(expected)]-expected)
                    if not np.all(delta<=1e-5+1e-4*np.abs(expected)):raise ValueError("Candidate plaintext differs beyond frozen tolerance")
                    maximum=max(maximum,float(np.max(delta)))
                    reference_max=max(reference_max,float(np.max(np.abs(expected))))
                groups+=1
            item["plaintext"]=dict(groups=groups,passed=True,max_abs=maximum,reference_max_abs=reference_max,
                                  tolerance="1e-5 + 1e-4*abs(reference)")
            item["result"]=qualify(q,candidate)
        except (ValueError,AssertionError,NotImplementedError) as e:item["rejected"]=str(e)
        item["elapsed_seconds"]=time.monotonic()-started
        rows.append(item)
        assert runtime_sources()==sources
        assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in guard.items())
        report=dict(rows=rows,source_hashes=sources,paid_calls=0,new_agent_successes=0)
        report["binding"]=digest(report);report_path.write_text(json.dumps(report,indent=2))
        print(json.dumps(item),flush=True)
        if item.get("result",{}).get("failure",{}).get("layer") in ("environment","integrity"):raise ValueError("Integrity/environment stop")
    return 0
if __name__=="__main__":raise SystemExit(main())
