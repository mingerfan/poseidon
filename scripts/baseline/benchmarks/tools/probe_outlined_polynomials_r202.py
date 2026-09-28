"""Offline function outlining for identical public polynomial calls; no model/limit changes."""
import sys,json,shlex,hashlib
from pathlib import Path
BASE=Path(__file__).resolve().parents[2];ROOT=BASE.parents[1]
sys.path[:0]=[str(BASE),str(Path(__file__).parent)]
def main():
    from hecate_python_env import enter_nix,VENV,WORK
    if "--inside" not in sys.argv:
        return enter_nix('LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+
            shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"]),seconds=900)
    import numpy as np
    import unified_graph_lowering as lowering
    import chebyshev_lowering
    from exact_series_diagnostics import parity_evaluate
    from qualify_expression_repairs_r201 import DerivedEmitter
    from manual_native_evaluation import evaluate as source_evaluate
    from benchmark_graph import digest,samples
    from benchmark_math import evaluate
    from benchmark_torch import evaluate as reference
    from unified_graph_contract import validate_candidate
    from component_backend import qualify
    from semantic_benchmark_execution import runtime_sources
    R=WORK/"results";out=R/"stage2-expression-repairs-r202/outlined";out.mkdir()
    requests=json.loads((R/"stage2-expression-repairs-r201/offline/prepared.json").read_text())["requests"]
    sources=runtime_sources()
    guard=json.loads((R/"stage2-expression-repairs-r199/before.json").read_text())["compiler"]
    original=chebyshev_lowering.balanced;emitter=lowering.Emitter
    rows=[]
    for task,base_evaluate,label in [
        ("free_bench_helper_0045",original,"outlined-balanced"),
        ("free_bench_helper_0032",parity_evaluate,"outlined-parity")]:
        q=requests[task];definitions={};item=dict(id=task,strategy=label,manual_only=True,paid_calls=0)
        def outlined(x,coeff,zero):
            key=tuple(float(v) for v in coeff)
            if key not in definitions:
                name="polyScope"+str(len(definitions))
                em=DerivedEmitter(q);em.zero=lowering.Scalar(em,"polyZero",True)
                y=base_evaluate(lowering.Scalar(em,"polyInput"),coeff,em.zero)
                definitions[key]=(name,'@hc.func("c,c")\ndef '+name+'(polyInput,polyZero):\n'+"\n".join(em.lines)+"\n    return "+y.name+"\n")
            name=definitions[key][0]
            return x.e.emit(name+"("+x.name+","+zero.name+")")
        try:
            lowering.Emitter=DerivedEmitter;chebyshev_lowering.balanced=outlined
            source=lowering.lower(q,packed_prefix=True,balanced_chebyshev=True)
            source="\n".join(text for _,text in definitions.values())+"\n"+source
            job=out/(task+".json");candidate=dict(schema=1,request_id=q["request_id"],hecate_source=source)
            job.write_text(json.dumps(dict(request=q,candidate=candidate,manual_fixture=True),indent=2))
            item.update(job=str(job),functions=len(definitions),source_sha256=hashlib.sha256(source.encode()).hexdigest())
            check=validate_candidate(candidate,q);item["static_passed"]=True
            item["expanded_cost"]=check["functions"]["golden"]["expanded_cost"]
            max_error=0.
            for xs in samples(q["model"],16):
                a=source_evaluate(source,q,xs);b=evaluate(q["model"],xs);c=reference(q["model"],xs)
                for i,o in enumerate(q["model"]["outputs"]):
                    expected=b[o["name"]].reshape(-1)
                    np.testing.assert_allclose(expected,c[o["name"]].reshape(-1),atol=1e-12,rtol=1e-12)
                    actual=a[i][:expected.size]
                    np.testing.assert_allclose(actual,expected,atol=2e-12,rtol=2e-12)
                    max_error=max(max_error,float(np.max(np.abs(actual-expected))))
            item["plaintext"]=dict(groups=16,passed=True,max_abs=max_error)
            item["result"]=qualify(q,candidate)
        except (ValueError,AssertionError,NotImplementedError) as e:item["rejected"]=str(e)
        finally:
            lowering.Emitter=emitter;chebyshev_lowering.balanced=original
        assert runtime_sources()==sources
        assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in guard.items())
        rows.append(item);report=dict(rows=rows,source_hashes=sources,paid_calls=0,new_agent_successes=0)
        report["binding"]=digest(report);(out/"report.json").write_text(json.dumps(report,indent=2))
        print(json.dumps(item),flush=True)
        if item.get("result",{}).get("failure",{}).get("layer") in ("environment","integrity"):
            raise ValueError("Environment/integrity stop")
    return 0
if __name__=="__main__":raise SystemExit(main())
