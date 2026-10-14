"""NPY dataset loading, cache-only scheduling and two genuine Poseidon calls."""
import hashlib,json,shutil,sys,time
from pathlib import Path
import numpy as np
ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5]);sys.path.insert(0,str(ROOT/"scripts/baseline"))
from benchmark_graph import canonical,digest
from application_identity import current_identity_manifest
from application_execution.context import WorkerContext
from application_execution.backend import application
from task_dataset_assessment import reference_anchor
from task_dataset_evaluation import TaskDatasetEvaluation
from assessment_array_dataset import export_dataset,load_dataset
from workspace_paths import work_root
HIST=Path("/home/lhohy/poseidon-work/platforms/aarch64-linux/results/agent-native-criterion-acceptance-v1")
INPUTS=Path("/home/lhohy/poseidon-work/platforms/aarch64-linux/results/agent-task-assessment-offline-v2/input-samples.json")
TASK="5384076bf9c04dc097254d1b1db30886"
PACKAGE="8865c3fd84d28d756446ca53770bee50c72e81c2f38032a468870f83c3ad04ad"
OUT=work_root()/"platforms/aarch64-linux/results/agent-dataset-io-acceptance-v2"
def hashes(folder):
    result={}
    for p in folder.rglob("*"):
        assert not p.is_symlink()
        assert p.name!=".env" and p.suffix not in (".sk",".pem")
        if p.is_file():result[str(p.relative_to(folder))]=hashlib.sha256(p.read_bytes()).hexdigest()
    return result
def write(path,value):
    with path.open("xb")as file:file.write(canonical(value))
def pinned(path,expected):
    v=json.loads(path.read_text())
    assert v["binding"]==expected and digest({k:x for k,x in v.items() if k!="binding"})==expected
    return v
def total(folder):
    return sum(p.stat().st_size for p in folder.rglob("*") if p.is_file())
def main():
    start=time.monotonic()
    identity=current_identity_manifest()
    assert identity["manifest_sha256"]=="aea83c53afb1bd7456c2c5fc90c56b45cb2886d1a47b9739e3da156381e7c2ad"
    historical=hashes(HIST)
    receipt=pinned(HIST/"assessment.json","a49082ff553bea124677e2374e70dabbe4449e3f374482efd9d91d3f80c78142")
    task=pinned(HIST/"tasks"/TASK/"task.json",receipt["task_metadata_binding"])
    request=task["request"];assert request["request_id"]==receipt["request_id"]
    values=json.loads(INPUTS.read_text());assert len(values)==3 and digest(values)==receipt["supplied_input_sha256"]
    anchor=reference_anchor(receipt,expected_binding=receipt["binding"])
    results=work_root()/"platforms/aarch64-linux/results"
    assert total(results)<32*1024**3 and shutil.disk_usage(results).free>=4*1024**3
    OUT.mkdir(mode=0o700);store=OUT/"tasks";store.mkdir(mode=0o700)
    shutil.copytree(HIST/"tasks"/TASK,store/TASK)
    assert hashes(store/TASK)==hashes(HIST/"tasks"/TASK)
    ctx=WorkerContext.capture(store);app=application(ctx);outcome=app.result(TASK)
    assert outcome["status"]=="succeeded" and outcome["package"]["binding"]==PACKAGE
    inputs={spec["name"]:np.asarray([v[spec["name"]] for v in values],dtype=np.float64)
            for spec in request["model"]["inputs"]}
    manifest=export_dataset(request,inputs,OUT/"input-dataset",sample_ids=["zero","signed","seeded"])
    dataset=load_dataset(request,OUT/"input-dataset/dataset.json",expected_binding=manifest["binding"])
    scope=dict(format="poseidon-dataset-io-local-acceptance-scope-v1",
        task_id=TASK,package_binding=PACKAGE,task_snapshot_sha256=digest(hashes(store/TASK)),
        historical_receipt_binding=receipt["binding"],stage_identity=identity["manifest_sha256"],
        dataset_file_binding=manifest["binding"],samples=3,values=30,part_samples=2,
        max_new_assessments=2,max_source_forward_calls=2,max_encrypted_calls=2,
        criterion=dict(kind="max_absolute_error",limit=.01),
        purpose="Expected original-source accuracy failure stays separate from CKKS numerical success.",
        expected_application_outcome="validation_failed",
        limits=dict(total_wall_seconds=600,new_output_bytes=128*1024**2,
            total_result_bytes=32*1024**3,minimum_free_bytes=4*1024**3,
            native_address_space_bytes=4*1024**3,native_cpu_seconds=150),
        api_calls=0,compiler_calls=0)
    scope["binding"]=digest(scope);write(OUT/"scope.json",scope)
    cached=TaskDatasetEvaluation(app,TASK,dataset,anchor=anchor,
        application_criterion=scope["criterion"],
        cached_receipts={"part_0":dict(receipt=receipt,binding=receipt["binding"])})
    cached_result=cached.advance()
    assert cached_result["status"]=="validation_failed" and cached_result["new_assessments"]==0
    assert cached_result["summary"]["evaluated_samples"]==3
    write(OUT/"cached-state.json",cached_result)
    evaluation=TaskDatasetEvaluation(app,TASK,dataset,anchor=anchor,
        application_criterion=scope["criterion"],max_new_assessments=2,part_samples=2,
        native_source=dict(input_dtypes={"q":"float64","k":"float64","v":"float64"},
            trust_local_code=True,evidence_root=str(OUT/"native-source")))
    write(OUT/"declaration.json",evaluation.declaration)
    quiet=evaluation.advance(max_new_parts=0)
    assert quiet["status"]=="ready" and quiet["new_assessments"]==0
    first=evaluation.advance();write(OUT/"state.first.json",first)
    assert first["status"]=="ready" and first["new_assessments"]==1
    assert first["summary"]["evaluated_samples"]==2
    final=evaluation.advance();write(OUT/"state.final.json",final)
    for name,r in evaluation.receipts.items():write(OUT/(name+"-receipt.json"),r)
    assert final["status"]=="validation_failed" and final["new_assessments"]==2
    summary=final["summary"]
    assert summary["complete"]is True and summary["evaluated_samples"]==summary["numerically_validated_samples"]==3
    assert summary["ckks_comparison"]["compared_values"]==30
    assert summary["supplied_inputs_numerically_validated"]is True
    assert summary["application_criterion"]["passed"]is False
    assert summary["original_source_accuracy_certified"]is False
    assert all(r["encrypted_execution"]is True and r["source_reference_evaluated"]is True and
        r["decomposition_comparison"]["passed"]is True and r["ckks_comparison"]["passed"]is True
        for r in evaluation.receipts.values())
    repeat=evaluation.advance();assert repeat["binding"]==final["binding"]
    assert final["new_agent_calls"]==final["new_compiler_calls"]==0
    assert app.result(TASK)==outcome and hashes(HIST)==historical
    assert hashes(store/TASK)==hashes(HIST/"tasks"/TASK)
    assert current_identity_manifest()==identity
    size=total(OUT);assert size<=scope["limits"]["new_output_bytes"] and time.monotonic()-start<=600
    report=dict(format="poseidon-dataset-io-local-acceptance-v1",status="passed",
        acceptance_kind="Data IO and admission controls passed; original-source criterion accurately fails.",
        dataset_status=final["status"],samples=3,compared_values=30,
        ckks_max_absolute_error=summary["ckks_comparison"]["max_absolute_error"],
        ckks_mae=summary["ckks_comparison"]["mae"],
        original_source_max_absolute_error=summary["application_criterion"]["max_absolute_error"],
        original_source_accuracy_passed=False,new_encrypted_calls=2,new_source_forward_calls=2,
        cached_receipts_reused=1,cache_only_new_calls=0,new_agent_calls=0,new_compiler_calls=0,
        historical_evidence_unchanged=True,original_task_and_package_unchanged=True,
        stage_identity=identity["manifest_sha256"],scope_binding=scope["binding"],
        dataset_file_binding=manifest["binding"],evaluation_binding=final["evaluation_binding"],
        final_state_binding=final["binding"],cached_state_binding=cached_result["binding"],
        assessment_bindings={n:r["binding"] for n,r in evaluation.receipts.items()},
        bytes=size,elapsed_seconds=time.monotonic()-start)
    report["binding"]=digest(report);write(OUT/"report.json",report)
    print(json.dumps(report))
if __name__=="__main__":main()
