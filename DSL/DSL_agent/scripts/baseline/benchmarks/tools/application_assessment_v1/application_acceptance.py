"""One new real assessment through the persistent application entry."""
import hashlib,json,shutil,sys,time
from pathlib import Path
ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5]);sys.path.insert(0,str(ROOT/"scripts/baseline"))
from benchmark_graph import canonical,digest
from application_identity import current_identity_manifest
from application_execution.context import WorkerContext
from application_execution.backend import application
from application_assessment import ApplicationAssessment
from task_numerical_assessment import NATIVE_POLICY
from workspace_paths import work_root
PREVIOUS=Path("/home/lhohy/poseidon-work/platforms/aarch64-linux/results/agent-dataset-io-acceptance-v2")
TASK="5384076bf9c04dc097254d1b1db30886"
PACKAGE="8865c3fd84d28d756446ca53770bee50c72e81c2f38032a468870f83c3ad04ad"
OUT=work_root()/"platforms/aarch64-linux/results/agent-application-assessment-acceptance-v1"
def hashes(folder):
    files={}
    for p in folder.rglob("*"):
        assert not p.is_symlink()
        assert p.name!=".env" and p.suffix not in (".sk",".pem")
        if p.is_file():files[str(p.relative_to(folder))]=hashlib.sha256(p.read_bytes()).hexdigest()
    return files
def total(folder):return sum(p.stat().st_size for p in folder.rglob("*")if p.is_file())
def write(path,value):
    with path.open("xb")as f:f.write(canonical(value))
def pinned(path,expected):
    value=json.loads(path.read_text())
    assert value["binding"]==expected and digest({k:v for k,v in value.items()if k!="binding"})==expected
    return value
def main():
    start=time.monotonic();old=hashes(PREVIOUS)
    prior=pinned(PREVIOUS/"report.json","c9f36f5ed36a60263cefba98b5605c7fc8a6287b73e758ddc101d38469c8ddac")
    receipt=pinned(PREVIOUS/"part_0-receipt.json",prior["assessment_bindings"]["part_0"])
    identity=current_identity_manifest()
    assert identity["manifest_sha256"]==prior["stage_identity"]
    results=work_root()/"platforms/aarch64-linux/results"
    assert total(results)<32*1024**3 and shutil.disk_usage(results).free>=4*1024**3
    OUT.mkdir(mode=0o700);store=OUT/"tasks";store.mkdir(mode=0o700)
    shutil.copytree(PREVIOUS/"tasks"/TASK,store/TASK)
    assert hashes(store/TASK)==hashes(PREVIOUS/"tasks"/TASK)
    app=application(WorkerContext.capture(store));before=app.result(TASK)
    assert before["status"]=="succeeded" and before["package"]["binding"]==PACKAGE
    scope=dict(format="poseidon-application-assessment-local-scope-v1",
        program_task=TASK,package_binding=PACKAGE,stage_identity=identity["manifest_sha256"],
        source_receipt_binding=receipt["binding"],source_dataset_binding=prior["dataset_file_binding"],
        samples=3,values=30,reused_samples=2,new_samples=1,new_values=10,
        max_new_assessments=1,max_encrypted_calls=1,max_native_workers=1,
        limits=dict(total_wall_seconds=600,new_output_bytes=128*1024**2,
            total_result_bytes=32*1024**3,reserve_bytes=4*1024**3,
            native_address_space_bytes=4*1024**3,native_cpu_seconds=150),
        criterion=dict(kind="max_absolute_error",limit=.01),
        expected_result="validation_failed",
        purpose="Controller acceptance preserves original-source accuracy failure.",
        api_calls=0,compiler_calls=0)
    scope["binding"]=digest(scope);write(OUT/"scope.json",scope)
    service=ApplicationAssessment(app,OUT/"assessment-store")
    handle=service.submit(TASK,PREVIOUS/"input-dataset/dataset.json",
        expected_dataset_binding=prior["dataset_file_binding"],reference_policy=NATIVE_POLICY,
        application_criterion=scope["criterion"],max_new_assessments=1,part_samples=2,
        input_dtypes={"q":"float64","k":"float64","v":"float64"},trust_local_code=True,
        cached_receipts={"part_0":dict(receipt=receipt,binding=receipt["binding"])})
    write(OUT/"handle.json",handle)
    initial=service.status(handle);write(OUT/"status.initial.json",initial)
    assert initial["status"]=="queued" and initial["completed_parts"]==1 and initial["budget"]["used"]==0
    ready=service.advance(handle,max_new_parts=0)
    assert ready["status"]=="ready" and ready["budget"]["used"]==0
    write(OUT/"status.ready.json",ready)
    # Re-create the control entry before advancing. Same existing Application
    # and worker context; no source/budget rebinding.
    restored=ApplicationAssessment(app,OUT/"assessment-store")
    result=restored.advance(handle);write(OUT/"result.json",result)
    assert result["status"]=="validation_failed" and result["budget"]["used"]==1
    assert result["completed_parts"]==2 and result["summary"]["evaluated_samples"]==3
    assert result["acceptance"]["numerical_on_declared_dataset"]is True
    assert result["acceptance"]["application_criterion_validated"]is False
    assert result["failure"]["code"]=="assessment_application_criterion_failed"
    summary=result["summary"]
    assert summary["ckks_comparison"]["compared_values"]==30
    assert summary["original_source_accuracy_certified"]is False
    actual=json.loads((OUT/"assessment-store"/handle["evaluation_id"]/"receipts/part_1.json").read_text())
    assert actual["encrypted_execution"]is True and actual["source_reference_evaluated"]is True
    assert actual["samples"]==1 and actual["ckks_comparison"]["compared_values"]==10
    assert actual["ckks_comparison"]["passed"]is True and actual["decomposition_comparison"]["passed"]is True
    assert actual["ckks_comparison"]["atol"]==1e-5 and actual["ckks_comparison"]["rtol"]==1e-4
    repeat=ApplicationAssessment(app,OUT/"assessment-store").result(handle)
    assert repeat["binding"]==result["binding"]
    assert restored.advance(handle)["binding"]==result["binding"]
    assert restored.cancel(handle)["binding"]==result["binding"]
    assert app.result(TASK)==before and hashes(PREVIOUS)==old
    assert hashes(store/TASK)==hashes(PREVIOUS/"tasks"/TASK)
    assert current_identity_manifest()==identity
    size=total(OUT);assert size<=scope["limits"]["new_output_bytes"] and time.monotonic()-start<=600
    report=dict(format="poseidon-application-assessment-local-acceptance-v1",status="passed",
        evaluation_status=result["status"],application_failure=result["failure"]["code"],
        original_source_accuracy_passed=False,numerically_validated_samples=3,compared_values=30,
        current_encrypted_samples=1,current_encrypted_values=10,reused_samples=2,reused_values=20,
        ckks_max_absolute_error=summary["ckks_comparison"]["max_absolute_error"],
        current_ckks_max_absolute_error=actual["ckks_comparison"]["max_absolute_error"],
        original_source_max_absolute_error=summary["application_criterion"]["max_absolute_error"],
        controller_handle=handle,current_assessment_binding=actual["binding"],result_binding=result["binding"],
        scope_binding=scope["binding"],stage_identity=identity["manifest_sha256"],
        task_and_package_unchanged=True,historical_evidence_unchanged=True,
        reload_without_execution=True,completed_cancel_idempotent=True,
        new_encrypted_calls=1,new_source_reference_workers=1,new_agent_calls=0,new_compiler_calls=0,
        bytes=size,elapsed_seconds=time.monotonic()-start)
    report["binding"]=digest(report);write(OUT/"report.json",report)
    print(json.dumps(report))
if __name__=="__main__":main()
