"""One serial Poseidon assessment of an existing historical Agent-source package."""
import hashlib,json,os,shutil,sys,time
from pathlib import Path
import numpy as np
ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5])
sys.path.insert(0,str(ROOT/"scripts/baseline"))
from benchmark_graph import canonical,digest
from application_execution.context import WorkerContext
from application_execution.backend import application
from task_numerical_assessment import TaskNumericalAssessment,implementation
from application_identity import current_identity_manifest as application_identity
from workspace_paths import work_root
SOURCE_STORE=Path("/home/lhohy/poseidon-work/platforms/aarch64-linux/results/agent-mha-live-delivery-recovery-v2/tasks")
TASK="5384076bf9c04dc097254d1b1db30886"
BINDING="8865c3fd84d28d756446ca53770bee50c72e81c2f38032a468870f83c3ad04ad"
CANDIDATE="292bff0b21f31c77036cd063c8e641df458058b9599cba4d5096b3fce98077da"
RESULTS=work_root()/"platforms/aarch64-linux/results"
OUT=RESULTS/"agent-task-assessment-offline-v2"
STORE=OUT/"tasks"
def hashes(folder):
    return {str(p.relative_to(folder)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in folder.rglob("*") if p.is_file() and not p.is_symlink()}
def size(folder):
    return sum(p.stat().st_size for p in folder.rglob("*") if p.is_file() and not p.is_symlink())
def main():
    before_stage=application_identity()
    # The separately reviewed paid scope is only observed; no calls are made.
    assert before_stage["manifest_sha256"]=="aea83c53afb1bd7456c2c5fc90c56b45cb2886d1a47b9739e3da156381e7c2ad"
    assert size(RESULTS)<32*1024**3 and shutil.disk_usage(RESULTS).free>=4*1024**3
    OUT.mkdir(mode=0o700)
    source_task=SOURCE_STORE/TASK
    entries=list(source_task.rglob("*"))
    assert all(not p.is_symlink() for p in entries)
    assert all(p.name!=".env" and p.suffix not in (".sk",".pem") for p in entries if p.is_file())
    assert size(source_task)<=128*1024**2
    original_task_snapshot=hashes(source_task)
    # A byte-identical snapshot receives its own current worker receipt.
    # The historical store and worker-context are never updated.
    STORE.mkdir(mode=0o700)
    shutil.copytree(source_task,STORE/TASK)
    assert hashes(STORE/TASK)==original_task_snapshot
    package=STORE/TASK/"program"
    task_before=hashlib.sha256((STORE/TASK/"task.json").read_bytes()).hexdigest()
    package_before=hashes(package)
    sources={"scripts/baseline/task_numerical_assessment.py",
             "scripts/baseline/test_task_numerical_assessment.py",
             str(Path(__file__).relative_to(ROOT))}
    source_hashes={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sorted(sources)}
    assert any(h==CANDIDATE for h in package_before.values())
    rng=np.random.default_rng(1321)
    shapes={"q":(1,2,2),"k":(1,3,2),"v":(1,3,2)}
    samples=[{n:np.zeros(s).tolist() for n,s in shapes.items()},
             {n:(np.arange(np.prod(s),dtype=np.float64).reshape(s)%3-1).tolist()
               for n,s in shapes.items()},
             {n:rng.uniform(-1,1,size=s).tolist() for n,s in shapes.items()}]
    for index in (1,2):
        for n in shapes:
            a=np.asarray(samples[index][n])
            samples[index][n]=(a*({"q":.08,"k":.05,"v":.3}[n])).tolist()
    with (OUT/"input-samples.json").open("x") as f:json.dump(samples,f,allow_nan=False)
    context=WorkerContext.capture(STORE)
    app=application(context)
    prior=app.result(TASK)
    assert prior["status"]=="succeeded" and prior["package"]["binding"]==BINDING
    reference_before=implementation()
    started=time.monotonic()
    assessment=TaskNumericalAssessment(app).assess(TASK,samples,expected_package_binding=BINDING)
    with (OUT/"assessment.json").open("x") as f:json.dump(assessment,f,allow_nan=False,indent=2)
    if assessment.get("status")!="passed":
        print(json.dumps({"status":assessment.get("status"),"failure":assessment.get("failure")}))
        raise SystemExit(2)
    assert assessment["encrypted_execution"] and assessment["supplied_inputs_numerically_validated"]
    assert not assessment["original_source_accuracy_certified"]
    assert assessment["agent_calls"]==assessment["compiler_calls"]==0
    assert assessment["package_validation"]==dict(level="numerical",encrypted_execution=True,numerically_validated=True)
    assert app.result(TASK)==prior and hashes(package)==package_before
    assert hashes(source_task)==original_task_snapshot
    assert hashlib.sha256((STORE/TASK/"task.json").read_bytes()).hexdigest()==task_before
    assert application_identity()==before_stage and implementation()==reference_before
    assert source_hashes=={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sorted(sources)}
    assert size(RESULTS)<32*1024**3 and shutil.disk_usage(RESULTS).free>=4*1024**3
    assert size(OUT)<=512*1024**2
    report=dict(format="poseidon-task-assessment-backend-acceptance-v1",status="passed",
        task_id=TASK,bundle_binding=BINDING,candidate_sha256=CANDIDATE,
        origin="Unmodified historical real-Agent DSL, delivered by separately labelled scripted replay",
        new_agent_generation=False,api_calls=0,compiler_calls=0,
        encrypted_execution=True,samples=assessment["samples"],
        compared_values=assessment["ckks_comparison"]["compared_values"],
        ckks_comparison={k:v for k,v in assessment["ckks_comparison"].items()
                         if k not in ("actual","reference","absolute_errors","relative_errors")},
        max_absolute_error=assessment["ckks_comparison"]["max_absolute_error"],
        decomposition_max_absolute_error=assessment["decomposition_comparison"]["max_absolute_error"],
        approximation_max_absolute_error=assessment["approximation_comparison"]["max_absolute_error"],
        total_max_absolute_error=assessment["total_comparison"]["max_absolute_error"],
        original_source_accuracy_certified=False,
        application_criterion=assessment["application_criterion"],reference_policy=assessment["reference_policy"],
        assessment_binding=assessment["binding"],reference_identity=reference_before,
        stage_identity=before_stage["manifest_sha256"],worker_context_binding=context.binding,historical_task_unchanged=True,
        task_snapshot_sha256=digest(original_task_snapshot),
        sources=source_hashes,package_unchanged=True,task_unchanged=True,
        elapsed_seconds=time.monotonic()-started,artifact_bytes=size(OUT))
    # Count values from the checked declared output shape, independently of report formatting.
    metadata=app._checked_task_metadata(TASK)
    from benchmark_graph import validate
    checked=validate(metadata["request"]["model"])
    report["compared_values"]=len(samples)*sum(int(np.prod(checked["shapes"][o["value"]]))
                                              for o in metadata["request"]["model"]["outputs"])
    report["binding"]=digest(report)
    with (OUT/"report.json").open("x") as f:json.dump(report,f,allow_nan=False,indent=2)
    print(json.dumps({k:report[k] for k in ("status","samples","compared_values","max_absolute_error",
        "decomposition_max_absolute_error","approximation_max_absolute_error","total_max_absolute_error",
        "original_source_accuracy_certified","api_calls","compiler_calls","encrypted_execution",
        "elapsed_seconds","binding")}))
if __name__=="__main__":main()
