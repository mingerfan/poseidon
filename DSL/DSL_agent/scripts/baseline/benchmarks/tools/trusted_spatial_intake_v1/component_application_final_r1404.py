"""Production source-to-request component checks; no generation or FHE."""
import sys,json,shlex,time,traceback,unittest
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parents[2];sys.path.insert(0,str(BASE))
from hecate_python_env import VENV,enter_nix,WORK
if "--inside"not in sys.argv:
    command='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join(
      [str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"])
    raise SystemExit(enter_nix(command,seconds=300,environment="poseidon-cpu"))
from application_execution.context import WorkerContext
from application_execution.backend import application
from application_identity import current_identity_manifest
from benchmark_graph import digest
from candidate_contract import strict_json
from candidate_bundle import check_provenance
start=time.monotonic();sdk=current_identity_manifest()
out=WORK/"results"/("agent-spatial-source-application-r1404-"+str(time.time_ns()));out.mkdir()
source_root=WORK/"results"/"agent-spatial-source-sandbox-stage-1791728344527028739"
plan=json.loads(((Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5])/"docs/baseline/agent-spatial-offline-proposal-r1400.json").read_text())
options=plan["cases"][0]["options"];rows=[]
for name in("transpose_1_0","structured_dict","float32_public_weight"):
    row=dict(case=name,status="running");rows.append(row)
    app=application(WorkerContext.capture(out/name/"tasks"))
    task=app.submit_source(dict(format="poseidon-model-source-request-v1",kind="trusted_pytorch",
        manifest=str(source_root/(name+"-source")/"manifest.json"),trust_local_code=True),
        resource_profile="general-tensors-v1",options=options,admission_policy="structured-v1",
        operational_limits=plan["request_operational_limits"],limits=plan["generation_limits"])
    try:
        result=app.prepare_source(task)
        assert result["status"]=="queued"and result["intake"]["reference_verified"]is True,result
        meta=app._checked_task_metadata(task);check_provenance(meta["model_provenance"],meta["request"]["model"])
        assert meta["model_provenance"]["trusted_pytorch"]["manifest"]["format"]=="poseidon-trusted-pytorch-v27"
        assert result["generations"]==result["http_attempts"]==0
        row.update(status="passed",task=task,request_id=meta["request"]["request_id"],
          original_sha256=digest(meta["model_provenance"]["original"]),
          model_sha256=digest(meta["request"]["model"]),reference_verified=True,
          compilation_verified=False,encrypted_execution=False)
    except Exception as e:
        row.update(status="failed",task=task,error_type=type(e).__name__,error=str(e))
        (out/(name+"-failure.txt")).write_text(traceback.format_exc())
    print(json.dumps(row),flush=True)
tests=["test_pytorch_spatial_source","test_pytorch_input_binding","test_pytorch_public_dtype_cast",
       "test_pytorch_closed_public_arithmetic","test_operator_spatial_graph","test_unified_public_arithmetic","test_pytorch_spatial_cli"]
suite=unittest.defaultTestLoader.loadTestsFromNames(tests)
with(out/"production-tests.log").open("x")as stream:
    unit=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
assert current_identity_manifest()==sdk,"SDK changed during source admission"
report=dict(format="poseidon-spatial-source-application-checkpoint-v1",result=str(out),rows=rows,
    passed=sum(r["status"]=="passed"for r in rows),failed=sum(r["status"]=="failed"for r in rows),
    unit_tests=dict(run=unit.testsRun,failed=len(unit.failures),errors=len(unit.errors),skipped=len(unit.skipped)),
    sdk_identity=sdk["manifest_sha256"],elapsed_seconds=time.monotonic()-start,
    source_isolation="existing bubblewrap unshare-all",api_calls=0,compilations=0,encrypted_calls=0)
report["binding"]=digest(report);(out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps({k:v for k,v in report.items()if k!="rows"}),flush=True)
raise SystemExit(1 if report["failed"]or not unit.wasSuccessful()else 0)
