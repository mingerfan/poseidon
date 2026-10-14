"""Finite current controls and legacy/source binding audit; no API/FHE."""
import copy,hashlib,json,os,shlex,sys,time,unittest
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5]);BASE=ROOT/"scripts/baseline"
os.environ["POSEIDON_PLATFORM"]="aarch64-linux";os.environ["POSEIDON_WORK_ROOT"]="/home/lhohy/poseidon-work"
sys.path.insert(0,str(BASE))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 if not sys.argv[1:]:
  from hecate_python_env import VENV,enter_nix
  cmd='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"])
  return enter_nix(cmd,seconds=120,keep_env=(),environment="poseidon-cpu")
 assert sys.argv[1:]==["--inside"]
 from application_identity import current_identity_manifest
 from component_contract import prepare_task,request_options,reconstruct_request,failure
 from candidate_check_diagnostic import CandidateCheckError
 from unified_graph_contract import validate_candidate,validate_request
 from unified_construction_guidance import cost_facts
 from application_diagnostics import attach
 from deepseek_provider import public_request
 from public_weight_context import CONTEXT_BYTES
 from benchmark_graph import digest,canonical
 start=time.monotonic();identity=current_identity_manifest()
 names=["test_unified_construction_guidance","test_unified_tensor_axis_guidance",
 "test_unified_generation_guidance","test_application_diagnostics","test_native_reserved_binding_feedback",
 "test_native_diagnostics","test_native_public_loops.NativeLoopTests"]
 suite=unittest.defaultTestLoader.loadTestsFromNames(names)
 with (HERE/"controls.log").open("x") as f:
  result=unittest.TextTestRunner(stream=f,verbosity=2).run(suite)
 summary=dict(format="poseidon-construction-feedback-controls-v1",status="passed" if result.wasSuccessful() and not result.skipped else"failed",
 tests_run=result.testsRun,passed=result.testsRun-len(result.failures)-len(result.errors)-len(result.skipped),
 failed=len(result.failures),errors=len(result.errors),skipped=len(result.skipped),seconds=time.monotonic()-start,
 provider_calls=0,http_attempts=0,compiler_calls=0,encrypted_calls=0,
 test_modules=names,identity=identity)
 if not result.wasSuccessful() or result.skipped:
  with (HERE/"controls.json").open("x") as f:json.dump(summary,f,sort_keys=True,indent=2)
  print(json.dumps({k:v for k,v in summary.items() if k!="identity"}))
  return 1
 baseline=json.loads((HERE/"baseline.json").read_text())
 for record in baseline["legacy_requests"]:
  assert prepare_task(record["model"],record["options"])==record["request"],"legacy_request_changed"
 for n,h in baseline["historical_requests"].items():
  assert sha(n)==h,"old_request_file_changed"
  reconstruct_request(json.loads(Path(n).read_text()))
 assert sha(BASE/"decorated_functions.py")==baseline["original_checker_sha256"],"counting_or_limit_changed"
 assert all(sha(ROOT/n)==h for n,h in baseline["protected_sources"].items()),"protected_source_changed"
 proposal=json.loads((ROOT/"docs/baseline/agent-mha-tensor-axis-live-proposal-r1333.json").read_text())
 contexts=[];oldroot=Path("/home/lhohy/poseidon-work/platforms/aarch64-linux/results/agent-mha-tensor-axis-live-v1")
 for i,c in enumerate(proposal["cases"]):
  old=json.loads(Path(c["request_file"]).read_text());options=request_options(old);options["generation_guidance"]="explicit-v13"
  new=prepare_task(old["model"],options)
  validate_request(new);assert reconstruct_request(new)==new
  left=copy.deepcopy(old);right=copy.deepcopy(new)
  for v in (left,right):v.pop("request_id");v.pop("generation_guidance")
  assert left==right,"public_model_or_frozen_math_changed"
  context=public_request(new);count=len(canonical(context));assert count<=CONTEXT_BYTES
  path=HERE/(c["id"]+".request-v13.json")
  with path.open("x") as f:json.dump(new,f,sort_keys=True,indent=2)
  row=dict(case=c["id"],request_id=new["request_id"],public_context_bytes=count,request_file=str(path),
   request_file_sha256=sha(path),mathematical_binding_unchanged=True)
  if i==0:
   directory=oldroot/c["id"]/"tasks"/"a2f5ba0de3954fd3b3e8106aae6d6a07"
   failures=[]
   for index in (0,1,2,3):
    raw=json.loads((directory/f"response-{index}.json").read_text());candidate=json.loads(raw["raw"])
    original_source=hashlib.sha256(candidate["hecate_source"].encode()).hexdigest()
    candidate["request_id"]=new["request_id"]
    try:validate_candidate(candidate,new)
    except CandidateCheckError as error:
     supplied=attach(dict(status="failed",failure=failure("static_check",error)),candidate=candidate,request=new)
     record=dict(round=index,reason=error.reason,source_sha256=original_source,source_changed=False,
      diagnostic=supplied["public_feedback"]["diagnostic"])
     if error.reason=="expanded_work_budget":
      facts=cost_facts(new,candidate["hecate_source"],supplied["failure"])
      assert facts is not None and "Static work accounting:" in record["diagnostic"]
      record["cost_facts"]=facts
     elif error.reason=="rotation_step_not_allowed":
      assert "Admitted rotation steps:" in record["diagnostic"]
     failures.append(record)
    else:raise AssertionError("original_invalid_source_accepted")
   row["historical_failed_source_replays"]=failures
  contexts.append(row)
 assert current_identity_manifest()==identity,"active_sources_changed"
 summary.update(legacy_requests_unchanged=len(baseline["legacy_requests"]),historical_requests_unchanged=2,
 native_counter_unchanged=True,protected_files_preserved=len(baseline["protected_sources"]),
 frozen_new_contexts=contexts,seconds=time.monotonic()-start)
 summary["binding"]=digest(summary)
 with (HERE/"controls.json").open("x") as f:json.dump(summary,f,sort_keys=True,indent=2)
 print(json.dumps({k:v for k,v in summary.items() if k not in ("identity","frozen_new_contexts")}))
 return 0
if __name__=="__main__":raise SystemExit(main())
