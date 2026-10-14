"""Freeze requests and scripted candidates only; no compilation or execution."""
import os,sys,shlex,json
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parents[2];sys.path.insert(0,str(BASE))
from hecate_python_env import VENV,enter_nix
from scope import *
def inside():
    require(os.environ.get("IN_NIX_SHELL")=="pure"and Path(sys.prefix)==VENV,"Pinned preparation")
    require(not PROPOSAL.exists()and not OUT.exists(),"Fresh scope preparation")
    from application_execution.context import WorkerContext
    from application_execution.backend import application
    from unified_graph_lowering import candidate_source
    from candidate_bundle import check_provenance
    controls=read(HERE.parent/"public_expression_cutover_v1/source-controls-v2.json")
    require(controls["tests"]==167 and not any(controls[k]for k in("failed","errors","skipped"))and
            controls["source_capture_passed"]==6 and controls["source_capture_failed"]==0,"Source qualification")
    require(measure(WORK/"results")+SCOPE["new_bytes"]<=SCOPE["global_bytes"]and
            shutil.disk_usage(WORK).free>=SCOPE["reserve_bytes"]+SCOPE["new_bytes"],"Offline preflight space")
    OUT.mkdir(mode=0o700)
    sdk=identity();rows=[]
    capture=HERE.parent/"public_expression_cutover_v1/capture-evidence-v3"
    models=read(HERE.parent/"public_expression_cutover_v1/capture-models-v3.json")
    require(tuple(m["case"]for m in models)==CASES,"Exact source model order")
    for model_row in models:
        name=model_row["case"];directory=OUT/name;directory.mkdir()
        model=read(capture/name/"model.json");provenance=read(capture/name/"provenance.json")
        check_provenance(provenance,model)
        task_store=directory/"tasks";app=application(WorkerContext.capture(task_store))
        task=app.submit(model,model_provenance=provenance,options=OPTIONS,
                        limits=dict(generations=1,http_attempts=1,wall_seconds=SCOPE["case_seconds"],
                                    api_timeout_seconds=1200,max_tokens=384000))
        require(app.status(task)["status"]=="queued","Prepared component acceptance")
        meta=app._checked_task_metadata(task);request=meta["request"];source=candidate_source(request)[0]
        request_path=directory/"request.json";write(request_path,request)
        candidate=directory/"candidate.py";candidate.write_text(source)
        files={str(p):sha(p)for p in(request_path,candidate,capture/name/"model.json",capture/name/"provenance.json",
                                        Path(model_row["manifest_path"]))}
        manifest=read(model_row["manifest_path"])
        root=Path(model_row["manifest_path"]).parent
        files.update({str(root/n):sha(root/n)for n in manifest["files"]})
        files.update({str(root/w["file"]):sha(root/w["file"])for w in manifest["weights"].values()})
        rows.append(dict(case=name,task_id=task,task_store=str(task_store),request=str(request_path),request_id=request["request_id"],
                         request_sha256=digest(request),candidate=str(candidate),model_sha256=digest(model),
                         manifest=model_row["manifest_path"],files=files))
    require(identity()["manifest_sha256"]==sdk["manifest_sha256"],"SDK changed during preparation")
    plan=dict(format="poseidon-public-expression-offline-proposal-v1",source_contract="poseidon-trusted-pytorch-v23",
              backend="poseidon_cpu",scope=SCOPE,options=OPTIONS,candidate_origin="deterministic_rule",
              cases=rows,reuse_cases=list(REUSE),sdk_identity=sdk["manifest_sha256"],
              stage_identities={k:sdk[k]["sha256"]for k in("generation","compilation","execution","validation")},
              tools={str(p.relative_to(ROOT)):sha(p)for p in HERE.glob("*.py")},
              numeric_tolerance=dict(atol=1e-5,rtol=1e-4),approved=False,
              compiler_calls=0,encrypted_calls=0,provider_calls=0,http_attempts=0,
              source_controls_binding=controls["binding"])
    plan["binding"]=digest(plan);verify(plan);write(PROPOSAL,plan);write(OUT/"proposal.json",plan)
    print(json.dumps(dict(status="prepared",cases=len(rows),binding=plan["binding"],sdk=plan["sdk_identity"],
                         source_controls=controls["binding"],provider_calls=0,compiler_calls=0,encrypted_calls=0)))
if __name__=="__main__":
    if "--inside"in sys.argv:raise SystemExit(inside())
    command='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"])
    raise SystemExit(enter_nix(command,seconds=180,keep_env=(),environment="poseidon-cpu"))
