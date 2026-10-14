"""Serial no-API final regression and package replay in one existing Nix shell."""
import contextlib
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import time
ROOT=(Path(__file__).resolve().parents[6] if Path(__file__).resolve().parents[4].name == "DSL_agent" and Path(__file__).resolve().parents[5].name == "DSL" else Path(__file__).resolve().parents[4])
sys.path.insert(0,str(ROOT/"scripts/baseline"))

def main():
    from hecate_python_env import WORK,VENV,enter_nix
    if "--inside" not in sys.argv[1:]:
        cmd='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join(
            [str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"])
        return enter_nix(cmd,seconds=1850,environment="poseidon-cpu")
    from benchmark_graph import canonical,digest,require
    from semantic_benchmark_execution import runtime_sources,verify_numerical_report
    from poseidon_backend_identity import identity
    from candidate_bundle import export_bundle,check_bundle,replay_bundle
    from audit_unified_candidate import verify_candidate
    roots={k:WORK/"results"/v for k,v in (
        ("operators","poseidon-scheduled-operators-r363"),
        ("whole","poseidon-scheduled-whole-r364"),
        ("packages","poseidon-scheduled-packages-r365"),
        ("summary","poseidon-scheduling-final-r366"))}
    require(all(not p.exists() for p in roots.values()),"Fresh final evidence")
    roots["summary"].mkdir()
    frozen=runtime_sources();backend=identity();begin=time.monotonic()
    plan=dict(format="poseidon-trusted-schedule-acceptance-plan-v1",source_hashes=frozen,
        backend_identity=backend,roots={k:str(p) for k,p in roots.items()},
        limits=dict(seconds=1800,bytes=1024**3,global_bytes=32*1024**3,free_bytes=4*1024**3,
                    native_tasks=1,compile_jobs=2,link_jobs=1),paid_calls=0,agent_generated=False)
    (roots["summary"]/"plan.json").write_bytes(canonical(plan))
    def boundary():
        require(runtime_sources()==frozen and identity()==backend,"Source/backend boundary")
        require(time.monotonic()-begin<1800 and shutil.disk_usage(WORK).free>=4*1024**3,"Time/free-space boundary")
        total=int(subprocess.check_output(["du","-sb",str(WORK/"results")],timeout=30,text=True).split()[0])
        require(total<=32*1024**3,"Global evidence boundary")
    stages=[]
    def launch(label,command,seconds):
        boundary()
        with (roots["summary"]/(label+".log")).open("x") as log:
            child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:code=child.wait(timeout=seconds)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGINT)
                try:code=child.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid,signal.SIGKILL);code=child.wait(timeout=10)
        stages.append(dict(stage=label,exit_code=code))
        print("FINAL_STAGE",label,code,flush=True)
        boundary()
        return code
    launch("operators",[str(VENV/"bin/python"),"-B",str(ROOT/"scripts/baseline/check_operator_regression.py"),
        "--inside","--run-local","--manifest",str(ROOT/"scripts/baseline/cases/operator-extension-v5/manifest.json"),
        "--compiler-configuration","poseidon-cpu-eva-w45-v1","--output",str(roots["operators"])],1200)
    op=json.loads((roots["operators"]/"report.json").read_text())
    require(not any(r.get("failure_layer") in ("integrity","environment","sandbox") for r in op["rows"]),"Shared operator fault")
    launch("whole",[str(VENV/"bin/python"),"-B",str(ROOT/"scripts/baseline/benchmarks/tools/check_poseidon_scheduled_whole_v2.py"),
        str(roots["whole"]),"--inside"],500)
    whole=json.loads((roots["whole"]/"report.json").read_text())
    require(not any(r.get("failure_layer") in ("integrity","environment","sandbox") for r in whole["rows"]),"Shared whole-model fault")
    roots["packages"].mkdir();package_rows=[]
    selected=[("rmsnorm",whole),("attention_decode",whole),("layer_norm_2",op)]
    for name,report in selected:
        boundary()
        row=next(r for r in report["rows"] if r["id"]==name)
        if row["status"]!="passed":
            package_rows.append(dict(id=name,status="not_run",reason="Source did not pass numerical qualification"));continue
        evidence=Path(row["evidence"])
        source_report=json.loads((evidence/"report.json").read_text())
        verify_numerical_report(source_report,"poseidon-cpu-eva-w45-v1")
        require(source_report["agent_calls"]==0 and not source_report["llm_generation_validated"],"No Agent success")
        package=roots["packages"]/name/"program";package.parent.mkdir()
        manifest=export_bundle(evidence,package);checked=check_bundle(package)
        require(checked["validation"]["level"]=="numerical" and checked["format"]=="poseidon-dsl-bundle-v4","Numerical package")
        before=set((WORK/"results").glob("poseidon-package-replay-*"))
        with (package.parent/"reload.log").open("x") as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
            require(replay_bundle(package)==0,"Direct package replay")
        added=set((WORK/"results").glob("poseidon-package-replay-*"))-before
        require(len(added)==1,"Unique reload evidence")
        replay=next(iter(added));result=json.loads((replay/"report.json").read_text())
        require(result["status"]=="passed" and result["encrypted_execution"] and
                result["agent_calls"]==result["compiler_calls"]==0,"Actual encrypted replay")
        package_rows.append(dict(id=name,status="passed",program=str(package),package_binding=manifest["binding"],
            source_evidence=str(evidence),reload_evidence=str(replay),
            comparison={k:result["comparison"][k] for k in ("compared_values","max_absolute_error")},
            actual_compile=False,actual_encrypted_execution=True,paid_calls=0))
        print("FINAL_PACKAGE",name,"passed",flush=True)
    audited=[];paths=set(roots.values())
    for label,report in (("operators",op),("whole",whole)):
        for row in report["rows"]:
            if row.get("evidence"):paths.add(Path(row["evidence"]))
            normalized=dict(id=row["id"],group=label,status=row["status"],failure_layer=row.get("failure_layer"),
                evidence=row.get("evidence"))
            if row["status"]=="passed":
                audit=verify_candidate(Path(row["evidence"]))
                normalized["comparison"]={k:audit["comparison"][k] for k in ("compared_values","max_absolute_error")}
                normalized["encrypted_input_groups"]=4
            audited.append(normalized)
    for row in package_rows:
        if row.get("reload_evidence"):paths.add(Path(row["reload_evidence"]))
    used=sum(p.stat().st_size for root in paths for p in root.rglob("*") if p.is_file())
    require(used<=1024**3,"Batch artifact boundary")
    report=dict(format="poseidon-trusted-schedule-acceptance-v1",binding=digest(plan),stages=stages,
        rows=audited,packages=package_rows,planned=len(audited),
        passed=sum(r["status"]=="passed" for r in audited),failed=sum(r["status"]=="failed" for r in audited),
        not_run=sum(r["status"]=="not_run" for r in audited),
        passed_input_groups=sum(r.get("encrypted_input_groups",0) for r in audited),
        compared_values=sum(r.get("comparison",{}).get("compared_values",0) for r in audited),
        max_absolute_error=max(r.get("comparison",{}).get("max_absolute_error",0) for r in audited),
        package_replay_values=sum(r.get("comparison",{}).get("compared_values",0) for r in package_rows),
        package_replay_max_error=max(r.get("comparison",{}).get("max_absolute_error",0) for r in package_rows),
        seconds=time.monotonic()-begin,artifact_bytes=used,agent_generated=False,paid_calls=0)
    boundary();(roots["summary"]/"report.json").write_bytes(canonical(report))
    print("FINAL_SUMMARY",report["passed"],report["failed"],report["compared_values"],report["max_absolute_error"],flush=True)
    return int(any(r["status"]!="passed" for r in audited))

if __name__=="__main__":raise SystemExit(main())
