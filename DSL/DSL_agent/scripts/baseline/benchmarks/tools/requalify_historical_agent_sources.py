"""Requalify exact historical Agent sources on Poseidon; never call a Provider."""
import argparse
import contextlib
import copy
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import sys
import time
ROOT=(Path(__file__).resolve().parents[6] if Path(__file__).resolve().parents[4].name == "DSL_agent" and Path(__file__).resolve().parents[5].name == "DSL" else Path(__file__).resolve().parents[4])
sys.path[:0]=[str(ROOT/"scripts/baseline"),str(Path(__file__).resolve().parent)]

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def verify_frozen_files(folder, hashes, limit=8*1024**2):
    from benchmark_graph import require
    checked={}
    for name,expected in hashes.items():
        rel=Path(name);file=folder/rel
        require(not rel.is_absolute() and ".." not in rel.parts and file.resolve().is_relative_to(folder.resolve()),
                "Historical evidence path")
        require(file.is_file() and not file.is_symlink() and file.stat().st_size<=limit,"Historical evidence file")
        require(sha(file)==expected,"Historical evidence hash mismatch: "+name)
        checked[name]=expected
    return checked

def historical(folder):
    from benchmark_graph import require,canonical,digest
    from agent_response_provenance import verify_answers
    from candidate_contract import strict_json
    report=strict_json((folder/"report.json").read_text())
    request=strict_json((folder/"request.json").read_text())
    model=strict_json((folder/"model.json").read_text())
    require(model==request["model"],"Historical model binding")
    metrics=report["provider_metrics"]
    approved=dict(provider=metrics["service_provider"],model=metrics["model"],
        reasoning_effort=metrics["reasoning_effort"],max_tokens=metrics["max_tokens"],
        max_repairs=metrics["max_calls"]-1,provider_retries=metrics["provider_retries"],
        api_timeout_seconds=metrics["timeout_seconds"])
    provenance=verify_answers(folder,report,request,approved)
    frozen=verify_frozen_files(folder,report["frozen_hashes"])
    passed=next(a for a in report["attempts"] if a["status"]=="passed")
    attempt=folder/("attempt-%02d"%passed["index"])
    artifacts=verify_frozen_files(attempt/"output",passed["artifact_hashes"])
    raw=(attempt/"response.txt").read_bytes()
    candidate=strict_json(raw.decode())
    require(candidate["hecate_source"].encode()==(attempt/"candidate.py").read_bytes(),"Exact historical source")
    for key in ("report.json","request.json","model.json"):
        frozen[key]=sha(folder/key)
    frozen[str((attempt/"response.txt").relative_to(folder))]=sha(attempt/"response.txt")
    frozen[str((attempt/"candidate.py").relative_to(folder))]=sha(attempt/"candidate.py")
    return dict(report=report,request=request,model=model,candidate=candidate,
        provenance=provenance,frozen=frozen,artifacts=artifacts,attempt=attempt,
        source_sha256=sha(attempt/"candidate.py"))

def rebind(old):
    from component_contract import request_options
    from compiler_configuration import configuration
    from unified_graph_contract import prepare,COMPACT
    from benchmark_graph import require
    request=old["request"]
    options=request_options(request)
    options.pop("compiler_configuration")
    config=configuration("poseidon-cpu-eva-w45-v1" if request["compiler_configuration"]["waterline"]==45
                         else "poseidon-cpu-eva-w40-v1")
    policy=request["constant_origins"].get("policy")
    current=prepare(copy.deepcopy(old["model"]),config["ckks_config_sha256"],config,
        constant_policy=COMPACT if policy==COMPACT else None,**options)
    for field in ("model","public_constants","constant_origins","layout"):
        require(current[field]==request[field],"Historical semantic binding drift: "+field)
    candidate=copy.deepcopy(old["candidate"])
    candidate["request_id"]=current["request_id"]
    require(candidate["hecate_source"]==old["candidate"]["hecate_source"],"Historical candidate rewritten")
    return current,candidate

def compare_arrays(old_folder,current_folder):
    import numpy as np
    from benchmark_graph import require
    for name in ("arrays.npz","weights.npz"):
        with np.load(old_folder/name,allow_pickle=False) as a,np.load(current_folder/name,allow_pickle=False) as b:
            require(set(a.files)==set(b.files),"Reference/weights array names changed")
            for key in a.files:
                require(a[key].dtype==b[key].dtype and a[key].shape==b[key].shape and
                        np.array_equal(a[key],b[key]),"Reference/input/weights values changed")
    return dict(inputs_unchanged=True,reference_unchanged=True,weights_unchanged=True)

def main():
    from hecate_python_env import WORK,VENV,enter_nix
    p=argparse.ArgumentParser()
    p.add_argument("--inside",action="store_true")
    p.add_argument("--selection",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    if not a.inside:
        cmd='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join(
            [str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside",
             "--selection",str(a.selection),"--output",str(a.output)])
        return enter_nix(cmd,seconds=1900,environment="poseidon-cpu")
    import numpy as np
    from benchmark_graph import require,canonical,digest
    from check_poseidon_helpers import snapshot
    from semantic_benchmark_execution import runtime_sources
    from poseidon_backend_identity import identity
    from unified_graph_prepare import prepare_case
    from component_backend import qualify
    from candidate_bundle import export_bundle,check_bundle,replay_bundle
    require(not a.output.exists() and a.output.parent.resolve()==WORK/"results","Fresh bounded output")
    choices=json.loads(a.selection.read_text())
    require(0<len(choices)<=48,"Historical selection size")
    require(len({r["folder"] for r in choices})==len(choices),"Duplicate historical source")
    a.output.mkdir();begin=time.monotonic()
    frozen=runtime_sources();core=snapshot();backend=identity()
    plan=dict(format="poseidon-historical-source-acceptance-plan-v1",selection=choices,
        selection_sha256=sha(a.selection),source_hashes=frozen,application_source_hashes=core,
        backend_identity=backend,driver_sha256=sha(Path(__file__)),paid_calls=0,
        new_agent_generation=False,limits=dict(seconds=1800,case_seconds=300,artifact_bytes=1024**3,
            global_bytes=32*1024**3,free_bytes=4*1024**3,native_workers=1,compile_jobs=2,link_jobs=1))
    (a.output/"plan.json").write_bytes(canonical(plan))
    rows=[dict(id=r["label"],status="not_run",origin_category="historical_agent_source") for r in choices]
    def save():
        (a.output/"report.json").write_bytes(canonical(dict(format="poseidon-historical-source-acceptance-v1",
            binding=digest(plan),planned=len(rows),rows=rows,passed=sum(r["status"]=="passed" for r in rows),
            failed=sum(r["status"]=="failed" for r in rows),blocked=sum(r["status"]=="blocked" for r in rows),
            not_run=sum(r["status"]=="not_run" for r in rows),paid_calls=0,new_agent_generation=False,
            seconds=time.monotonic()-begin)))
    def boundary():
        require(runtime_sources()==frozen and identity()==backend and sha(a.selection)==plan["selection_sha256"],
                "Frozen migration binding changed")
        require(time.monotonic()-begin<1800 and shutil.disk_usage(WORK).free>=4*1024**3,"Time/free-space boundary")
        import subprocess
        total=int(subprocess.check_output(["du","-sb",str(WORK/"results")],timeout=30,text=True).split()[0])
        require(total<=32*1024**3,"Global artifact boundary")
    save()
    for i,item in enumerate(choices):
        boundary();case=a.output/item["label"];case.mkdir();stage="historical_provenance"
        row=rows[i];folder=Path(item["folder"]);start=time.monotonic()
        try:
            old=historical(folder)
            require(old["source_sha256"]==item["source_sha256"],"Selected historical source changed")
            row.update(model_id=old["model"]["id"],historical_evidence=str(folder),
                historical_backend=old["report"]["backend"],historical_request_id=old["request"]["request_id"],
                source_sha256=old["source_sha256"],historical_provenance=old["provenance"],
                historical_files=old["frozen"],historical_artifacts=old["artifacts"])
            history=case/"historical";history.mkdir()
            for name,path in (("report.json",folder/"report.json"),("request.json",folder/"request.json"),
                              ("model.json",folder/"model.json"),("response.txt",old["attempt"]/"response.txt"),
                              ("candidate.py",old["attempt"]/"candidate.py")):
                shutil.copyfile(path,history/name)
            stage="request_binding"
            request,candidate=rebind(old)
            (case/"request.json").write_bytes(canonical(request))
            (case/"model.json").write_bytes(canonical(old["model"]))
            (case/"response.json").write_bytes(canonical(candidate))
            ref=case/"reference-check";ref.mkdir()
            prepare_case(old["model"],ref,request["layout"]["input_slot_period"] if "logical_inputs" in request["layout"] else None)
            row.update(compare_arrays(folder,ref))
            row.update(request_id=request["request_id"],candidate_unchanged=True)
            stage="backend_qualification"
            result=qualify(request,candidate)
            row.update(result);row["backend_terminal_status"]=result["status"]
            row["status"]="passed" if result["status"]=="passed" else "failed"
            if result.get("evidence"):
                row.update(compare_arrays(folder,Path(result["evidence"])))
                executed=Path(result["evidence"])/"attempt-00/candidate.py"
                require(sha(executed)==old["source_sha256"],"Executed source changed")
            if row["status"]=="passed":
                stage="package_replay"
                package=case/"program"
                manifest=export_bundle(Path(result["evidence"]),package)
                require(sha(package/"candidate.py")==old["source_sha256"],"Exported source changed")
                require(check_bundle(package)["validation"]["level"]=="numerical","Numerical package")
                before=set((WORK/"results").glob("poseidon-package-replay-*"))
                with (case/"reload.log").open("x") as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
                    require(replay_bundle(package)==0,"Historical program reload")
                added=set((WORK/"results").glob("poseidon-package-replay-*"))-before
                require(len(added)==1,"Unique reload")
                reload=next(iter(added));rr=json.loads((reload/"report.json").read_text())
                require(rr["status"]=="passed" and rr["encrypted_execution"] and
                        rr["agent_calls"]==rr["compiler_calls"]==0,"Actual repeated encrypted execution")
                row.update(program=str(package),package_binding=manifest["binding"],reload_evidence=str(reload),
                    reload_comparison=rr["comparison"],package_generation_calls=0,package_compiler_calls=0)
                # The v4 replay flag remains false. Authorship is carried by this
                # separate delivery binding, not invented live-call metadata.
                envelope=dict(format="poseidon-historical-agent-delivery-v1",origin_category="historical_agent_source",
                    new_agent_generation=False,manual_source_repair=False,source_sha256=old["source_sha256"],
                    historical_request_id=old["request"]["request_id"],poseidon_request_id=request["request_id"],
                    historical_files={str(p.relative_to(case)):sha(p) for p in history.iterdir()},
                    program_manifest_sha256=sha(package/"manifest.json"),package_binding=manifest["binding"],
                    validation_level="numerical",backend_identity=backend,paid_calls=0)
                envelope["binding"]=digest(envelope)
                (case/"delivery.json").write_bytes(canonical(envelope));row["delivery_binding"]=envelope["binding"]
        except (ValueError,AssertionError,KeyError,TypeError,OSError) as error:
            row.update(status="failed",failure_stage=stage,diagnostic=str(error)[:2000])
            (case/"failure.json").write_bytes(canonical(row))
        row["seconds"]=time.monotonic()-start
        require(row["seconds"]<300,"Case time boundary")
        print("HISTORICAL",item["label"],row["status"],row.get("failure",row.get("diagnostic")),flush=True)
        save();boundary()
        if row.get("failure",{}).get("layer") in ("integrity","environment","sandbox") or row.get("failure_stage")=="historical_provenance":
            raise ValueError("Shared environment or provenance fault; retained completed records")
    paths={a.output}
    for row in rows:
        for key in ("evidence","reload_evidence"):
            if row.get(key):paths.add(Path(row[key]))
    used=sum(p.stat().st_size for root in paths for p in root.rglob("*") if p.is_file())
    require(used<=1024**3,"Migration artifact boundary")
    summary=json.loads((a.output/"report.json").read_text())
    summary.update(artifact_bytes=used,compared_values=sum(r.get("compared_values",0) for r in rows if r["status"]=="passed"),
        max_absolute_error=max((r.get("max_absolute_error",0) for r in rows if r["status"]=="passed"),default=0),
        package_replay_values=sum(r.get("reload_comparison",{}).get("compared_values",0) for r in rows),
        package_replay_max_error=max((r.get("reload_comparison",{}).get("max_absolute_error",0) for r in rows),default=0))
    (a.output/"report.json").write_bytes(canonical(summary))
    return int(any(r["status"]!="passed" for r in rows))

if __name__=="__main__":raise SystemExit(main())
