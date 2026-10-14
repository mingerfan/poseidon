"""No-API automatic baseline acceptance for the three unchanged whole models."""
import json
import hashlib
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path
ROOT=(Path(__file__).resolve().parents[6] if Path(__file__).resolve().parents[4].name == "DSL_agent" and Path(__file__).resolve().parents[5].name == "DSL" else Path(__file__).resolve().parents[4])
sys.path.insert(0,str(ROOT/"scripts/baseline"))

def main():
    from hecate_python_env import VENV,WORK,enter_nix
    output=Path(sys.argv[1])
    if "--inside" not in sys.argv[2:]:
        command='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join(
            [str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),str(output),"--inside"])
        return enter_nix(command,seconds=1000,environment="poseidon-cpu")
    from benchmark_graph import require,canonical,digest,samples
    from semantic_benchmark_execution import runtime_sources,verify_numerical_report
    from benchmark_math import evaluate as mathematical
    from benchmark_torch import evaluate as torch_reference
    from audit_unified_candidate import verify_candidate
    from unified_graph_contract import prepare
    from compiler_configuration import configuration
    import numpy as np
    require(not output.exists() and output.parent.resolve()==WORK/"results","Fresh evidence")
    output.mkdir()
    proposal=json.loads((ROOT/"docs/baseline/poseidon-whole-models-live-proposal-r286.json").read_text())
    frozen=runtime_sources()
    cases=proposal["cases"]
    plan=dict(format="poseidon-scheduled-whole-model-baseline-plan-v1",cases=cases,source_hashes=frozen,
        old_proposal_binding=proposal["binding"],compiler_configuration="poseidon-cpu-eva-w45-v1",
        limits=dict(batch_seconds=900,case_seconds=300,artifact_bytes=1024**3,native_tasks=1,
                    global_bytes=32*1024**3,free_bytes=4*1024**3),
        paid_calls=0,agent_generated=False,manual_candidate_file=False)
    (output/"plan.json").write_bytes(canonical(plan))
    begin=time.monotonic()
    rows=[dict(id=c["id"],status="not_run") for c in cases]
    def save():
        (output/"report.json").write_bytes(canonical(dict(format="poseidon-scheduled-whole-model-baseline-v1",
            binding=digest(plan),rows=rows,planned=len(cases),passed=sum(r["status"]=="passed" for r in rows),
            failed=sum(r["status"]=="failed" for r in rows),not_run=sum(r["status"]=="not_run" for r in rows),
            seconds=time.monotonic()-begin,paid_calls=0,agent_generated=False)))
    save()
    config=configuration(plan["compiler_configuration"])
    for i,case in enumerate(cases):
        require(runtime_sources()==frozen and time.monotonic()-begin<900,"Source/time boundary")
        path=ROOT/case["model_path"]
        model=json.loads(path.read_text())
        require(digest(model)==case["model_sha256"] and hashlib.sha256(path.read_bytes()).hexdigest()==case["model_file_sha256"],"Original whole model")
        prior_request=json.loads((Path(case["prepare_evidence"])/"request.json").read_text())
        request=prepare(model,config["ckks_config_sha256"],config,generation_guidance="explicit-v10")
        require(request==prior_request,"Original public request unchanged")
        reference_error=0.
        for inputs in samples(model,16):
            a,b=mathematical(model,inputs),torch_reference(model,inputs)
            for name in a:
                np.testing.assert_allclose(a[name],b[name],atol=1e-12,rtol=1e-12)
                reference_error=max(reference_error,float(np.max(np.abs(a[name]-b[name]))))
        command=[str(VENV/"bin/python"),"-B",str(ROOT/"scripts/baseline/run_candidate.py"),"--inside",
            "--case",str(path),"--self-test","--max-repairs","0","--unified-guidance","explicit-v10",
            "--compiler-configuration",plan["compiler_configuration"]]
        log=output/(case["id"]+".log")
        with log.open("x") as handle:
            process=subprocess.Popen(command,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
            try:code=process.wait(timeout=300)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGINT)
                try:code=process.wait(timeout=20)
                except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);code=process.wait(timeout=10)
        paths=re.findall(r"^Candidate evidence: (.+)$",log.read_text(),re.M)
        row=dict(id=case["id"],status="failed",exit_code=code,model_sha256=digest(model),
            reference_groups=16,reference_max_error=reference_error)
        require(len(paths)==1,"One controlled evidence")
        evidence=Path(paths[0])
        report=json.loads((evidence/"report.json").read_text())
        verify_numerical_report(report,plan["compiler_configuration"])
        row.update(evidence=str(evidence),status=report["status"],report_sha256=hashlib.sha256((evidence/"report.json").read_bytes()).hexdigest(),
            rule_baseline_lowering=report.get("rule_baseline_lowering"),
            failure_layer=report.get("failure_layer"),
            compiled=any(a.get("compiled") for a in report["attempts"]),
            executed=any(a.get("executed") for a in report["attempts"]))
        if report["status"]=="passed":
            audit=verify_candidate(evidence)
            row["comparison"]={k:audit["comparison"][k] for k in ("compared_values","max_absolute_error")}
        else:
            row["failure_layer"]=next((a.get("failure_layer") for a in report["attempts"] if a.get("failure_layer")),None)
        rows[i]=row
        save()
        print("WHOLE_MODEL_RESULT",case["id"],row["status"],row.get("failure_layer"),flush=True)
        require(row.get("failure_layer") not in ("environment","integrity","sandbox"),"Shared execution fault")
    require(runtime_sources()==frozen,"Final source boundary")
    save()
    return 0 if all(r["status"]=="passed" for r in rows) else 1

if __name__=="__main__":raise SystemExit(main())
