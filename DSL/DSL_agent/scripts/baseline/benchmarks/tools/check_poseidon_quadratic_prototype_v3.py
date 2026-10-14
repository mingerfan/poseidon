"""Prototype numerical/reference acceptance; no API, no default-path rewrite."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shlex
import sys
import time
ROOT=(Path(__file__).resolve().parents[6] if Path(__file__).resolve().parents[4].name == "DSL_agent" and Path(__file__).resolve().parents[5].name == "DSL" else Path(__file__).resolve().parents[4])
sys.path[:0]=[str(ROOT/"scripts/baseline"),str(Path(__file__).resolve().parent)]

def main():
    from hecate_python_env import WORK,VENV,enter_nix
    p=argparse.ArgumentParser();p.add_argument("--inside",action="store_true");p.add_argument("--execute",action="store_true")
    p.add_argument("--output",required=True,type=Path)
    a=p.parse_args()
    if not a.inside:
        args=[str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside","--output",str(a.output)]
        if a.execute:args.append("--execute")
        return enter_nix('LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join(args),seconds=1000,environment="poseidon-cpu")
    import numpy as np
    from benchmark_graph import require,canonical,digest,samples
    from benchmark_math import evaluate as mathematical
    from benchmark_torch import evaluate as torch_reference
    from compiler_configuration import configuration
    from unified_graph_contract import prepare,validate_candidate
    from unified_graph_lowering import candidate_source
    from polynomial_correlation_prototype import simplify,estrin
    import unified_graph_lowering as lowering
    lowering.power_polynomial=estrin
    from check_poseidon_helpers import snapshot
    from component_backend import qualify
    from benchmark_suite import Builder
    require(not a.output.exists() and a.output.resolve().parent==WORK/"results","Fresh bounded output")
    a.output.mkdir();start=time.monotonic();frozen=snapshot()
    prototype=Path(__file__).with_name("polynomial_correlation_prototype.py")
    script_hash=hashlib.sha256(prototype.read_bytes()).hexdigest()
    config=configuration("poseidon-cpu-eva-w45-v1")
    proposal=json.loads((ROOT/"docs/baseline/poseidon-whole-models-live-proposal-r354.json").read_text())
    models=[dict(id=c["id"],model=json.loads((ROOT/c["model_path"]).read_text()),whole=True,
        prior_request=Path(c["prepare_evidence"])/"request.json") for c in proposal["cases"]]
    for size in (2,4,6):
        for kind in ("norm","dot"):
            b=Builder([(size,)] if kind=="norm" else [(size,),(size,)])
            v=b.node("square",["input0"]) if kind=="norm" else b.node("multiply",["input0","input1"])
            y=b.node("sum",[v],axes=[0],keepdims=True)
            models.append(dict(id=kind+"_"+str(size),model=b.finish(y),whole=False))
    plan=dict(format="poseidon-quadratic-prototype-plan-v1",sources=frozen,prototype_sha256=script_hash,
        driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),paid_calls=0,agent_generated=False,
        max_seconds=900,max_bytes=1024**3,planned=len(models),execute=a.execute)
    (a.output/"plan.json").write_bytes(canonical(plan));rows=[]
    for item in models:
        require(snapshot()==frozen and time.monotonic()-start<900,"Source/time boundary")
        model=item["model"];request=prepare(model,config["ckks_config_sha256"],config,generation_guidance="explicit-v10")
        if item.get("prior_request"):require(request==json.loads(item["prior_request"].read_text()),"Original whole request")
        before=copy.deepcopy(request)
        reduction=lowering.has_packed_reduction(request)
        original=lowering.lower(request,packed_prefix=reduction,packed_reductions=reduction,balanced_power=True)
        lowering_record=dict(strategy="raw-grouped-polynomial",packed_reduction=reduction)
        source,detail=simplify(original,request)
        require(request==before,"Immutable public binding")
        candidate=dict(schema=1,request_id=request["request_id"],hecate_source=source)
        folder=a.output/item["id"];folder.mkdir()
        (folder/"candidate.py").write_text(source)
        static=validate_candidate(candidate,request)
        from test_packed_prefix_lowering import public_execute
        error=0.
        for inputs in samples(model,16):
            values=public_execute(source,request,inputs)
            for ref in (mathematical,torch_reference):
                expected=ref(model,inputs)
                for i,out in enumerate(model["outputs"]):
                    target=expected[out["name"]].reshape(-1)
                    np.testing.assert_allclose(values[i][:len(target)],target,atol=1e-12,rtol=1e-12)
                    error=max(error,float(np.max(np.abs(values[i][:len(target)]-target))))
        (folder/"model.json").write_bytes(canonical(model))
        (folder/"request.json").write_bytes(canonical(request))
        (folder/"original.py").write_text(original);(folder/"candidate.py").write_text(source)
        row=dict(id=item["id"],status="reference_passed",request_id=request["request_id"],
            model_sha256=digest(model),source_sha256=hashlib.sha256(source.encode()).hexdigest(),
            reference_groups=16,reference_max_error=error,quadratic=detail,lowering=lowering_record)
        if a.execute:
            result=qualify(request,candidate)
            row.update(result)
        rows.append(row)
        (a.output/"report.json").write_bytes(canonical(dict(plan_sha256=digest(plan),rows=rows,planned=len(models),
            passed=sum(r["status"] in ("passed","reference_passed") for r in rows),
            failed=sum(r["status"] not in ("passed","reference_passed") for r in rows),
            not_run=len(models)-len(rows),paid_calls=0,agent_generated=False,seconds=time.monotonic()-start)))
        print("QUADRATIC",item["id"],row["status"],len(detail.get("quadratic_rewrites",[])),row.get("failure"),flush=True)
        require(row.get("failure",{}).get("layer") not in ("environment","sandbox","integrity"),"Shared backend fault")
    require(snapshot()==frozen,"Final source boundary")
    return int(any(r["status"] not in ("passed","reference_passed") for r in rows))

if __name__=="__main__":raise SystemExit(main())
