"""Resume prepared data after the configuration-schema check; no case is rerun."""
import os,sys,shlex
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parents[2];sys.path.insert(0,str(BASE))
from hecate_python_env import VENV,enter_nix
from scope import *
def inside():
    require(os.environ.get("IN_NIX_SHELL")=="pure"and Path(sys.prefix)==VENV,"Pinned request recovery")
    require(not PROPOSAL.exists()and OUT.is_dir()and not(OUT/"launch.json").exists(),
            "Only unexecuted prepared records")
    from application_execution.context import WorkerContext
    from application_execution.backend import application
    from candidate_bundle import check_provenance
    from unified_graph_lowering import candidate_source
    controls=read(HERE.parent/"public_expression_cutover_v1/source-controls-v2.json")
    require(controls["tests"]==167 and controls["source_capture_passed"]==6 and
            not any(controls[k]for k in("failed","errors","skipped","source_capture_failed")),"Current source controls")
    models=read(HERE.parent/"public_expression_cutover_v1/capture-models-v3.json")
    require(tuple(r["case"]for r in models)==CASES,"Exact captured case inventory")
    source=HERE.parent/"public_expression_cutover_v1/capture-evidence-v3";sdk=identity();rows=[]
    for model_row in models:
        name=model_row["case"];directory=OUT/name;task_store=directory/"tasks"
        stored=list(task_store.glob("*/task.json"));require(len(stored)==1,"One untouched prepared task")
        task=stored[0].parent.name;app=application(WorkerContext.capture(task_store))
        require(app.status(task)["status"]=="queued"and not list(task_store.rglob("qualification.json")),"No execution occurred")
        meta=app._checked_task_metadata(task);request=meta["request"]
        require(read(directory/"request.json")==request and
                Path(directory/"candidate.py").read_text()==candidate_source(request)[0],"Unchanged frozen data and rule candidate")
        model=read(source/name/"model.json");provenance=read(source/name/"provenance.json")
        check_provenance(provenance,model)
        require(meta["model_provenance"]==provenance,"Prepared provenance bridge")
        files={str(p):sha(p)for p in(directory/"request.json",directory/"candidate.py",
                                   source/name/"model.json",source/name/"provenance.json",Path(model_row["manifest_path"]))}
        manifest=read(model_row["manifest_path"]);root=Path(model_row["manifest_path"]).parent
        files.update({str(root/n):sha(root/n)for n in manifest["files"]})
        files.update({str(root/w["file"]):sha(root/w["file"])for w in manifest["weights"].values()})
        rows.append(dict(case=name,task_id=task,task_store=str(task_store),request=str(directory/"request.json"),
                         request_id=request["request_id"],request_sha256=digest(request),
                         candidate=str(directory/"candidate.py"),model_sha256=digest(model),
                         manifest=model_row["manifest_path"],files=files))
    require(identity()["manifest_sha256"]==sdk["manifest_sha256"],"Stable preparation identity")
    plan=dict(format="poseidon-public-expression-offline-proposal-v1",source_contract="poseidon-trusted-pytorch-v23",
              backend="poseidon_cpu",scope=SCOPE,options=OPTIONS,candidate_origin="deterministic_rule",
              cases=rows,reuse_cases=list(REUSE),sdk_identity=sdk["manifest_sha256"],
              stage_identities={k:sdk[k]["sha256"]for k in("generation","compilation","execution","validation")},
              tools={str(p.relative_to(ROOT)):sha(p)for p in HERE.glob("*.py")},
              numeric_tolerance=dict(atol=1e-5,rtol=1e-4),approved=False,
              compiler_calls=0,encrypted_calls=0,provider_calls=0,http_attempts=0,
              source_controls_binding=controls["binding"],preparation_recovery="data_validation_only_no_case_rerun")
    plan["binding"]=digest(plan);verify(plan);write(PROPOSAL,plan);write(OUT/"proposal.json",plan)
    print(json.dumps(dict(status="prepared",cases=len(rows),binding=plan["binding"],sdk=plan["sdk_identity"],
                         compiler_calls=0,encrypted_calls=0,provider_calls=0)))
if __name__=="__main__":
    if "--inside"in sys.argv:raise SystemExit(inside())
    command='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"])
    raise SystemExit(enter_nix(command,seconds=120,keep_env=(),environment="poseidon-cpu"))
