"""Two original packed slicing cases; all evidence is manual, never Agent scoring."""
import sys,shlex,json,hashlib
from pathlib import Path
BASE=Path(__file__).resolve().parents[2];ROOT=BASE.parents[1]
sys.path[:0]=[str(BASE),str(Path(__file__).parent)]
def main():
    from hecate_python_env import VENV,enter_nix,WORK
    if "--inside" not in sys.argv:
        return enter_nix('LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+
            shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"]),seconds=900)
    import numpy as np
    from packed_graph_diagnostic import source
    from manual_native_evaluation import evaluate as actual
    from benchmark_math import evaluate
    from benchmark_torch import evaluate as reference
    from benchmark_graph import samples,digest
    from unified_graph_contract import validate_candidate
    from component_backend import qualify
    from semantic_benchmark_execution import runtime_sources
    R=WORK/"results";out=R/"stage2-expression-repairs-r202/packed";out.mkdir()
    requests=json.loads((R/"stage2-expression-repairs-r201/offline/prepared.json").read_text())["requests"]
    sources=runtime_sources();guard=json.loads((R/"stage2-expression-repairs-r199/before.json").read_text())["compiler"]
    rows=[]
    for task in ("free_bench_helper_0045","free_bench_helper_0032"):
        q=requests[task];item=dict(id=task,manual_only=True,paid_calls=0,strategy="packed-logical-slice")
        try:
            src=source(q);candidate=dict(schema=1,request_id=q["request_id"],hecate_source=src)
            path=out/(task+".json");path.write_text(json.dumps(dict(request=q,candidate=candidate,manual_fixture=True),indent=2))
            item["job"]=str(path);item["source_sha256"]=hashlib.sha256(src.encode()).hexdigest()
            check=validate_candidate(candidate,q);item["static_passed"]=True
            item["expanded_cost"]=check["functions"]["golden"]["expanded_cost"]
            maximum=0.
            for xs in samples(q["model"],16):
                a=actual(src,q,xs);b=evaluate(q["model"],xs);c=reference(q["model"],xs)
                for i,o in enumerate(q["model"]["outputs"]):
                    expected=b[o["name"]].reshape(-1)
                    np.testing.assert_allclose(expected,c[o["name"]].reshape(-1),atol=1e-12,rtol=1e-12)
                    np.testing.assert_allclose(a[i][:len(expected)],expected,atol=2e-12,rtol=2e-12)
                    maximum=max(maximum,float(np.max(np.abs(a[i][:len(expected)]-expected))))
            item["plaintext"]=dict(groups=16,passed=True,max_abs=maximum)
            item["result"]=qualify(q,candidate)
        except (ValueError,AssertionError,NotImplementedError) as e:item["rejected"]=str(e)
        rows.append(item)
        assert runtime_sources()==sources
        assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in guard.items())
        report=dict(rows=rows,source_hashes=sources,paid_calls=0,new_agent_successes=0)
        report["binding"]=digest(report);(out/"report.json").write_text(json.dumps(report,indent=2))
        print(json.dumps(item),flush=True)
        if item.get("result",{}).get("failure",{}).get("layer") in ("environment","integrity"):raise ValueError("Integrity/environment stop")
    return 0
if __name__=="__main__":raise SystemExit(main())
