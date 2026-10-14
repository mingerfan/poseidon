"""Recalculate a declared dataset from pinned historical real Poseidon evidence."""
import hashlib,json,sys,time
from pathlib import Path
ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5]);sys.path.insert(0,str(ROOT/"scripts/baseline"))
from benchmark_graph import digest
from task_dataset_assessment import reference_anchor,declare_dataset,DatasetNumericalAssessment,DatasetAssessmentError
from application_identity import current_identity_manifest
from workspace_paths import work_root
PREVIOUS=Path("/home/lhohy/poseidon-work/platforms/aarch64-linux/results/agent-native-criterion-acceptance-v1")
INPUTS=Path("/home/lhohy/poseidon-work/platforms/aarch64-linux/results/agent-task-assessment-offline-v2/input-samples.json")
OUT=work_root()/"platforms/aarch64-linux/results/agent-dataset-receipt-audit-v2"
def hashes(folder):
    result={}
    for p in folder.rglob("*"):
        assert not p.is_symlink()
        assert p.name!=".env" and p.suffix not in (".sk",".pem")
        if p.is_file():result[str(p.relative_to(folder))]=hashlib.sha256(p.read_bytes()).hexdigest()
    return result
def write(path,value):
    with path.open("x") as f:json.dump(value,f,allow_nan=False,indent=2)
def bound(value,expected):
    assert value["binding"]==expected and digest({k:v for k,v in value.items() if k!="binding"})==expected
def main():
    started=time.monotonic();original=hashes(PREVIOUS)
    phase=json.loads((ROOT/"docs/baseline/agent-native-source-assessment-r1323.json").read_text())
    bound(phase,"b420a4b717a2996ca6437721e3d0eaf3c77790486fb82ac7f0296d8bbd74b7fd")
    backend=json.loads((PREVIOUS/"report.json").read_text())
    bound(backend,phase["real_backend"]["binding"])
    receipt=json.loads((PREVIOUS/"assessment.json").read_text())
    bound(receipt,backend["assessment_binding"])
    task=json.loads((PREVIOUS/"tasks/5384076bf9c04dc097254d1b1db30886/task.json").read_text())
    bound(task,receipt["task_metadata_binding"])
    request=task["request"];assert request["request_id"]==receipt["request_id"]
    from model_source_provenance import build
    source=json.loads((PREVIOUS/"tasks/5384076bf9c04dc097254d1b1db30886/source.json").read_text())
    bound(source,receipt["source_snapshot_binding"])
    assert build(source,request,task["model_provenance"])==task["model_source"]
    inputs=json.loads(INPUTS.read_text());assert len(inputs)==3 and digest(inputs)==receipt["supplied_input_sha256"]
    anchor=reference_anchor(receipt,expected_binding=backend["assessment_binding"])
    batches=[dict(id="fixed_inputs",sample_ids=["zero","signed","seeded"],samples=inputs)]
    declaration=declare_dataset(request,batches,anchor=anchor,
        application_criterion=dict(kind="max_absolute_error",limit=.01))
    collector=DatasetNumericalAssessment(request,declaration,expected_declaration_binding=declaration["binding"])
    pending=collector.summary();assert pending["status"]=="incomplete" and pending["evaluated_samples"]==0
    collector.record("fixed_inputs",receipt,expected_binding=backend["assessment_binding"])
    summary=collector.summary()
    duplicate=collector.record("fixed_inputs",receipt,expected_binding=backend["assessment_binding"])
    assert duplicate["status"]=="already_recorded" and collector.summary()==summary
    assert summary["status"]=="validation_failed" and summary["complete"]is True
    assert summary["planned_samples"]==summary["evaluated_samples"]==summary["numerically_validated_samples"]==3
    assert summary["numerical_receipts"]==1 and summary["ckks_comparison"]["compared_values"]==30
    assert summary["application_criterion"]["passed"]is False
    assert summary["application_criterion"]["max_absolute_error"]==backend["decrypted_vs_native_max_absolute_error"]
    assert summary["ckks_comparison"]["max_absolute_error"]==backend["ckks_max_absolute_error"]
    assert summary["part_status_counts"]=={"validation_failed":1}
    assert summary["original_source_accuracy_certified"]is False and summary["dataset_criterion_validated"]is False
    assert summary["agent_calls"]==summary["compiler_calls"]==summary["new_encrypted_calls"]==0
    identity=current_identity_manifest()
    assert identity["manifest_sha256"]==backend["stage_identity"]
    assert hashes(PREVIOUS)==original
    OUT.mkdir(mode=0o700)
    write(OUT/"dataset-declaration.json",declaration)
    write(OUT/"summary.json",summary);write(OUT/"pending-summary.json",pending)
    result=dict(format="poseidon-dataset-receipt-real-evidence-audit-v1",
        status="passed",audit_kind="Original real application-accuracy failure retained; no new execution",
        dataset_status=summary["status"],dataset_binding=declaration["binding"],summary_binding=summary["binding"],
        source_phase_binding=phase["binding"],historical_assessment_binding=receipt["binding"],
        samples=3,compared_values=30,numerical_receipts=1,
        ckks_max_absolute_error=summary["ckks_comparison"]["max_absolute_error"],
        ckks_mae=summary["ckks_comparison"]["mae"],
        source_total_max_absolute_error=summary["application_criterion"]["max_absolute_error"],
        source_application_accuracy_passed=False,repeated_receipt_counted_once=True,
        original_evidence_unchanged=True,source_forward_calls=0,new_encrypted_calls=0,compiler_calls=0,api_calls=0,
        stage_identity=identity["manifest_sha256"],elapsed_seconds=time.monotonic()-started,
        sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (
            ROOT/"scripts/baseline/task_dataset_assessment.py",Path(__file__))})
    result["binding"]=digest(result);write(OUT/"report.json",result)
    print(json.dumps(result))
if __name__=="__main__":main()
