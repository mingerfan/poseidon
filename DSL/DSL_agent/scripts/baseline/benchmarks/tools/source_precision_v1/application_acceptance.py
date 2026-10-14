"""Raw-source reference for one retained MHA package; no new FHE or generation."""
import hashlib,json,sys,shutil,time
from pathlib import Path
import numpy as np
ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5]);sys.path.insert(0,str(ROOT/"scripts/baseline"))
from benchmark_graph import digest
from application_identity import current_identity_manifest
from application_execution.context import WorkerContext
from application_execution.backend import application
from task_source_reference import TaskSourceReference
from task_numerical_assessment import mathematical,flatten,implementation as numerical_identity
from numerical_results import compare
from workspace_paths import work_root
SOURCE=Path("/home/lhohy/poseidon-work/platforms/aarch64-linux/results/agent-mha-live-delivery-recovery-v2/tasks")
TASK="5384076bf9c04dc097254d1b1db30886"
BINDING="8865c3fd84d28d756446ca53770bee50c72e81c2f38032a468870f83c3ad04ad"
PREVIOUS=Path("/home/lhohy/poseidon-work/platforms/aarch64-linux/results/agent-task-assessment-offline-v2")
OUT=work_root()/"platforms/aarch64-linux/results/agent-source-precision-acceptance-v1"
def hashes(folder):
    return {str(p.relative_to(folder)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in folder.rglob("*") if p.is_file()}
def main():
    identity=current_identity_manifest()
    assert identity["manifest_sha256"]=="aea83c53afb1bd7456c2c5fc90c56b45cb2886d1a47b9739e3da156381e7c2ad"
    previous=json.loads((PREVIOUS/"assessment.json").read_text())
    binding=previous.pop("binding")
    phase=json.loads((ROOT/"docs/baseline/agent-task-numerical-assessment-r1321.json").read_text())
    phase_binding=phase.pop("binding")
    assert phase_binding=="7356a93136b5af1458e243ab2c427b9cfa6dd58399439b8612aa4fb5739a76da" and digest(phase)==phase_binding
    assert binding==phase["real_backend"]["assessment_binding"] and digest(previous)==binding
    # Exact historical decrypted values are used as data, without claiming a new execution.
    previous["binding"]=binding
    assert previous["bundle_binding"]==BINDING and previous["reference_identity_sha256"]==numerical_identity()
    samples=json.loads((PREVIOUS/"input-samples.json").read_text())
    assert digest(samples)==previous["supplied_input_sha256"] and previous["samples"]==3
    assert shutil.disk_usage(SOURCE).free>=4*1024**3
    original=hashes(SOURCE/TASK)
    OUT.mkdir(mode=0o700);store=OUT/"tasks";store.mkdir(mode=0o700)
    shutil.copytree(SOURCE/TASK,store/TASK)
    assert hashes(store/TASK)==original
    app=application(WorkerContext.capture(store))
    before=app.result(TASK)
    assert before["status"]=="succeeded" and before["package"]["binding"]==BINDING
    started=time.monotonic()
    report=TaskSourceReference(app).reference(TASK,samples,OUT/"native-source",
        expected_package_binding=BINDING,input_dtypes={n:"float64" for n in ("q","k","v")},
        trust_local_code=True)
    with (OUT/"task-reference.json").open("x") as f:json.dump(report,f,allow_nan=False,indent=2)
    if report["status"]!="evaluated":
        print(json.dumps({"status":report["status"],"failure":report.get("failure")}))
        raise SystemExit(2)
    assert report["supplied_input_sha256"]==previous["supplied_input_sha256"]
    metadata=app._checked_task_metadata(TASK);model=metadata["request"]["model"]
    from benchmark_graph import validate
    checked=validate(model)
    native=[{o["name"]:np.asarray(sample[o["name"]],dtype=np.float64)
             .reshape(checked["shapes"][o["value"]]) for o in model["outputs"]}
            for sample in report["source_reference"]["outputs"]]
    original_graph=metadata["model_provenance"]["original"]
    ideal=[mathematical(original_graph,s,ideal=True) for s in samples]
    native_flat=flatten(native,model);ideal_flat=flatten(ideal,model)
    semantic=compare(native_flat,ideal_flat,1e-12,1e-12)
    assert semantic["compared_values"]==30
    assert semantic["passed"],"Bound native source does not match its independent mathematical graph"
    decrypted=np.asarray(previous["ckks_comparison"]["actual"],dtype=np.float64)
    approximate=np.asarray(previous["ckks_comparison"]["reference"],dtype=np.float64)
    approximation=compare(approximate,native_flat,1e-5,1e-4)
    total=compare(decrypted,native_flat,1e-5,1e-4)
    assert app.result(TASK)==before and hashes(SOURCE/TASK)==original and hashes(store/TASK)==original
    assert current_identity_manifest()==identity
    source_files=["source_precision_reference.py","task_source_reference.py"]
    result=dict(format="poseidon-source-precision-application-acceptance-v1",status="passed",
        source_reference_evaluated=True,new_encrypted_execution=False,new_compiler_calls=0,new_api_calls=0,
        samples=3,compared_values=30,native_dtype="float64",
        native_vs_mathematical_max_absolute_error=semantic["max_absolute_error"],
        native_vs_mathematical=semantic,approximate_vs_native=approximation,historical_decrypted_vs_native=total,
        application_accuracy_certified=False,original_task_and_package_unchanged=True,
        package_binding=BINDING,task_reference_binding=report["binding"],historical_assessment_binding=binding,
        stage_identity=identity["manifest_sha256"],source_identity=report["source_reference"]["source_identity"],
        sources={str((ROOT/"scripts/baseline"/n).relative_to(ROOT)):
                 hashlib.sha256((ROOT/"scripts/baseline"/n).read_bytes()).hexdigest() for n in source_files},
        elapsed_seconds=time.monotonic()-started)
    result["binding"]=digest(result)
    with (OUT/"report.json").open("x") as f:json.dump(result,f,allow_nan=False,indent=2)
    print(json.dumps({k:result[k] for k in ("status","samples","compared_values",
        "native_vs_mathematical_max_absolute_error","new_encrypted_execution","new_api_calls","binding")}))
if __name__=="__main__":main()
