"""Current native-source criterion plus one actual cached Poseidon execution."""
import hashlib,json,sys,shutil,time
from pathlib import Path
ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5]);sys.path.insert(0,str(ROOT/"scripts/baseline"))
from benchmark_graph import digest
from application_identity import current_identity_manifest
from application_execution.context import WorkerContext
from application_execution.backend import application
from task_numerical_assessment import TaskNumericalAssessment,SOURCE_SELECTION,NATIVE_POLICY,implementation
from source_precision_reference import source_identity
from workspace_paths import work_root
SOURCE=Path("/home/lhohy/poseidon-work/platforms/aarch64-linux/results/agent-mha-live-delivery-recovery-v2/tasks")
TASK="5384076bf9c04dc097254d1b1db30886"
PACKAGE="8865c3fd84d28d756446ca53770bee50c72e81c2f38032a468870f83c3ad04ad"
PREVIOUS=Path("/home/lhohy/poseidon-work/platforms/aarch64-linux/results/agent-task-assessment-offline-v2")
OUT=work_root()/"platforms/aarch64-linux/results/agent-native-criterion-acceptance-v1"
def hashes(folder):
    result={}
    for p in folder.rglob("*"):
        assert not p.is_symlink(),"Task snapshot must not contain symbolic links"
        assert p.name!=".env" and p.suffix not in (".sk",".pem"),"Credential/private-key file rejected"
        if p.is_file():result[str(p.relative_to(folder))]=hashlib.sha256(p.read_bytes()).hexdigest()
    return result
def write(path,value):
    with path.open("x") as f:json.dump(value,f,allow_nan=False,indent=2)
def main():
    before_identity=current_identity_manifest()
    assert before_identity["manifest_sha256"]=="aea83c53afb1bd7456c2c5fc90c56b45cb2886d1a47b9739e3da156381e7c2ad"
    previous=json.loads((PREVIOUS/"assessment.json").read_text());prior_binding=previous.pop("binding")
    phase=json.loads((ROOT/"docs/baseline/agent-task-numerical-assessment-r1321.json").read_text())
    phase_binding=phase.pop("binding")
    assert phase_binding=="7356a93136b5af1458e243ab2c427b9cfa6dd58399439b8612aa4fb5739a76da" and digest(phase)==phase_binding
    assert prior_binding==phase["real_backend"]["assessment_binding"] and digest(previous)==prior_binding
    samples=json.loads((PREVIOUS/"input-samples.json").read_text())
    assert digest(samples)==previous["supplied_input_sha256"] and previous["samples"]==len(samples)==3
    assert shutil.disk_usage(SOURCE).free>=4*1024**3
    original=hashes(SOURCE/TASK)
    OUT.mkdir(mode=0o700);store=OUT/"tasks";store.mkdir(mode=0o700)
    shutil.copytree(SOURCE/TASK,store/TASK)
    assert hashes(store/TASK)==original
    app=application(WorkerContext.capture(store));before=app.result(TASK)
    assert before["status"]=="succeeded" and before["package"]["binding"]==PACKAGE
    frozen_reference=implementation(policy=NATIVE_POLICY);frozen_native=source_identity()
    declaration=dict(format="poseidon-native-criterion-local-acceptance-scope-v1",
        task_id=TASK,package_binding=PACKAGE,task_snapshot_sha256=digest(original),
        stage_identity=before_identity["manifest_sha256"],
        native_reference_identity=frozen_native,numerical_reference_identity=frozen_reference,
        supplied_input_sha256=digest(samples),samples=3,values=30,
        criterion=dict(kind="max_absolute_error",limit=.01),
        purpose="Interface negative control; not a network accuracy target or tolerance change",
        expected_application_outcome="validation_failed",
        limits=dict(total_wall_seconds=300,native_address_space_bytes=4*1024**3,native_cpu_seconds=150,
                    new_output_bytes=64*1024**2,total_result_bytes=32*1024**3,minimum_free_bytes=4*1024**3),
        compiler_calls=0,api_calls=0,backend="poseidon_cpu")
    declaration["binding"]=digest(declaration);write(OUT/"scope.json",declaration)
    started=time.monotonic()
    result=TaskNumericalAssessment(app).assess(TASK,samples,expected_package_binding=PACKAGE,
        application_criterion=declaration["criterion"],
        native_source=dict(format=SOURCE_SELECTION,trust_local_code=True,
            input_dtypes={n:"float64" for n in ("q","k","v")},
            evidence_directory=str(OUT/"native-source")))
    write(OUT/"assessment.json",result)
    if not result.get("encrypted_execution") or not result.get("source_reference_evaluated"):
        print(json.dumps(dict(status=result["status"],failure=result.get("failure"))));raise SystemExit(2)
    assert result["status"]=="validation_failed" and result["failure"]["code"]=="assessment_application_criterion_failed"
    assert result["decomposition_comparison"]["passed"] and result["ckks_comparison"]["passed"]
    assert result["source_vs_registered_mathematics"]["passed"]
    assert result["source_total_comparison"]["max_absolute_error"]>.7
    assert result["application_criterion"]["passed"]is False and result["application_criterion"]["scope"]==NATIVE_POLICY
    assert result["supplied_inputs_numerically_validated"]is True
    assert result["original_source_accuracy_certified"]is False
    assert result["source_application_criterion_validated"]is False
    assert result["source_application_validation_scope"]["dataset_accuracy_certified"]is False
    assert result["ckks_comparison"]["atol"]==1e-5 and result["ckks_comparison"]["rtol"]==1e-4
    assert result["ckks_comparison"]["compared_values"]==30
    assert result["agent_calls"]==result["compiler_calls"]==0
    assert result["reference_identity_sha256"]==frozen_reference==implementation(policy=NATIVE_POLICY)
    assert source_identity()==frozen_native
    assert app.result(TASK)==before and hashes(SOURCE/TASK)==original and hashes(store/TASK)==original
    assert current_identity_manifest()==before_identity
    size=sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file())
    assert size<=64*1024**2 and time.monotonic()-started<=300
    report=dict(format="poseidon-native-criterion-local-acceptance-v1",
        status="passed",acceptance_kind="Expected application-accuracy failure is reported correctly",
        application_status=result["status"],application_failure=result["failure"]["code"],
        encrypted_execution=True,source_reference_evaluated=True,
        samples=result["samples"],compared_values=result["ckks_comparison"]["compared_values"],
        ckks_max_absolute_error=result["ckks_comparison"]["max_absolute_error"],
        ckks_mae=result["ckks_comparison"]["mae"],
        native_vs_registered_max_absolute_error=result["source_vs_registered_mathematics"]["max_absolute_error"],
        approximation_and_conversion_max_absolute_error=result["source_conversion_and_approximation_comparison"]["max_absolute_error"],
        decrypted_vs_native_max_absolute_error=result["source_total_comparison"]["max_absolute_error"],
        application_criterion=result["application_criterion"],original_source_accuracy_certified=False,
        case_origin="Byte-identical retained real-Agent DSL; offline package reused; no new Agent generation",
        package_binding=PACKAGE,assessment_binding=result["binding"],scope_binding=declaration["binding"],
        original_task_and_package_unchanged=True,stage_identity=before_identity["manifest_sha256"],
        new_api_calls=0,new_compiler_calls=0,new_encrypted_calls=1,bytes=size,
        elapsed_seconds=time.monotonic()-started)
    report["binding"]=digest(report);write(OUT/"report.json",report)
    print(json.dumps(report))
if __name__=="__main__":main()
