"""Recheck only original blocked models, preserving original model identities.

Default is plan only. Each execution shard has at most 48 models and rechecks
both independent plaintext references. Explicit --fhe-id selects existing models
before observing new results; no paid provider and no automatic full rerun.
"""
import argparse,hashlib,json,os,re,shlex,signal,subprocess,sys,time
from pathlib import Path
BASE=Path(__file__).resolve().parents[2];ROOT=BASE.parents[1]
sys.path.insert(0,str(BASE))
from benchmark_runner import strict_file,dump,load
from benchmark_graph import digest,samples
from semantic_benchmark_execution import runtime_sources,preflight,CONFIG
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument("--output",type=Path,required=True);p.add_argument("--shard-index",type=int,required=True)
 p.add_argument("--original-compiler-failures",action="store_true",help="Recheck the two original compile failures; separate from 112 preflight blockers");p.add_argument("--selected-only",action="store_true",help="Recheck only explicitly selected original FHE IDs; preserve full original denominator");p.add_argument("--execute",action="store_true");p.add_argument("--fhe-id",action="append",default=[])
 p.add_argument("--compiler-configuration",choices=("seal-cpu-eva-w45-v1","seal-cpu-eva-w40-v1"),default=CONFIG)
 p.add_argument("--inside",action="store_true",help=argparse.SUPPRESS);a=p.parse_args()
 from workspace_paths import RESULTS
 from hecate_python_env import enter_nix,VENV
 suite=BASE/"benchmarks/semantic-v1-chunk-helpers-r29"
 rows,index=load(suite,check_sources=False) # Keep frozen data, explicitly use current validators/references.
 historical=RESULTS/"benchmark-r30-full-baseline/shard-000/plan.json"
 old=strict_file(historical,8*1024**2)
 by_id={r["model"]["id"]:r for r in rows};selected=[]
 for r in old["preflight"]:
  if r["status"]=="ready":continue
  row=by_id[r["id"]]
  if row["model_sha256"]!=r["model_sha256"]:raise ValueError("Original identity drift")
  selected.append(dict(row,old_status=r["status"],old_reason=r["reason"]))
 if len(selected)!=110:raise ValueError("Original blocked denominator changed")
 bundle=BASE/"benchmarks/model-contexts-draft-v1/tasks.json"
 supplement=strict_file(bundle,2*1024**2)
 supplemental_ids={"semantic_context_331124fd5981bd48a0c2971f":"AST node limit",
                  "semantic_context_89458ebf15d877d81d6b8b2d":"Deterministic baseline operation budget"}
 for row in supplement["supplemental_models"]:
  name=row["model"]["id"]
  if name in supplemental_ids:
   if digest(row["model"])!=row["model_sha256"]:raise ValueError("Supplement identity drift")
   selected.append(dict(row,category="supplemental",old_status="blocked_rule_preflight",old_reason=supplemental_ids[name]))
 if len(selected)!=112 or not 0<=a.shard_index<3:p.error("Original shard bounds")
 if a.original_compiler_failures:
  if a.shard_index!=0:p.error("Original compiler failures form one two-case shard")
  failures=strict_file(ROOT/"docs/baseline/benchmark-full-baseline-r29.json",4*1024**2)["compiler_failed_models"]
  if set(failures)!={"bench_helper_0040","bench_helper_0044"}:raise ValueError("Original compiler failure identities")
  selected=[dict(by_id[name],old_status="compiler_failed",old_reason="Original r29 compiler failure") for name in failures]
 window=selected[a.shard_index*48:(a.shard_index+1)*48];ids={r["model"]["id"] for r in window}
 if len(set(a.fhe_id))!=len(a.fhe_id) or not set(a.fhe_id)<=ids:p.error("Choose distinct FHE IDs from this frozen shard")
 original_window_ids=sorted(ids)
 if a.selected_only:
  if not a.fhe_id:p.error("Selected-only requires explicit model IDs")
  window=[r for r in window if r["model"]["id"] in set(a.fhe_id)];ids={r["model"]["id"] for r in window}
 if not a.execute:
  print(json.dumps(dict(planned=len(window),original_blocked=112,original_compile_failed=2,ids=sorted(ids),fhe_ids=a.fhe_id,paid_calls=0)));return 0
 if a.output.exists() or not a.output.resolve().is_relative_to(RESULTS.resolve()):p.error("New platform output required")
 if not a.inside:
  return enter_nix('LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),*sys.argv[1:],"--inside"]),seconds=1860)
 if os.environ.get("IN_NIX_SHELL")!="pure" or Path(sys.prefix)!=VENV:p.error("Pinned pure environment")
 import numpy as np,torch
 from platform_config import require_python_packages
 require_python_packages(torch,np);torch.set_num_threads(1)
 from benchmark_math import evaluate as mathematical
 from benchmark_torch import evaluate as independent
 from audit_unified_candidate import verify_candidate
 sources=runtime_sources();a.output.mkdir(parents=True);(a.output/"runner.py").write_bytes(Path(__file__).read_bytes())
 plan=dict(format="poseidon-original-retry-plan-v1",models=window,fhe_ids=a.fhe_id,
  original_blocked_denominator=112,original_compile_failed_denominator=2,compiler_failure_only=a.original_compiler_failures,original_window_ids=original_window_ids,selected_only=a.selected_only,
  runtime_sources=sources,runner_sha256=sha(Path(__file__)),historical_plan_sha256=sha(historical),
  historical_runtime_sha256=digest(old["source_hashes"]),
  disk_measurement="Retained evidence checked between candidates; transient private keys follow existing runner cleanup; no peak disk claim",frozen_index_sha256=sha(suite/"index.json"),
  supplemental_bundle_sha256=sha(bundle),max_wall_seconds=1800,max_result_mib=256,
  compiler_configuration=a.compiler_configuration,concurrency=1,paid_calls=0,original_models_unchanged=True)
 plan["binding"]=digest(plan);dump(a.output/"plan.json",plan);start=time.monotonic();result=[]
 def boundary():
  if runtime_sources()!=sources or sha(Path(__file__))!=plan["runner_sha256"]:raise ValueError("Source integrity failure")
  if time.monotonic()-start>=1800:raise TimeoutError("Batch time budget")
  evidence=[Path(x["evidence"]) for x in result if x.get("evidence")]
  if sum(f.stat().st_size for folder in [a.output,*evidence] for f in folder.rglob("*") if f.is_file())>256*1024**2:raise ValueError("Batch disk budget")
 for source in window:
  boundary();model=source["model"];name=model["id"]
  row=dict(id=name,model_sha256=source["model_sha256"],old_status=source["old_status"],old_reason=source["old_reason"],
           reference_status="not_run",preflight_status="not_run",execution_status="not_selected",paid_calls=0)
  # Both reference implementations must agree before even considering DSL execution.
  ref_errors=[]
  try:
   for inputs in samples(model,16):
    expected=mathematical(model,inputs);other=independent(model,inputs)
    for output in expected:
     np.testing.assert_allclose(other[output],expected[output],atol=1e-12,rtol=1e-12)
     ref_errors.append(float(np.max(np.abs(other[output]-expected[output]))))
   row.update(reference_status="passed",reference_probes=16,max_reference_error=max(ref_errors))
  except (ValueError,AssertionError,TypeError) as error:
   row.update(reference_status="failed",diagnostic=str(error))
  if row["reference_status"]=="passed":
   from unified_graph_contract import prepare,validate_candidate
   from unified_graph_lowering import candidate_source
   from compiler_configuration import PROFILE_SHA256,configuration
   check=dict(status="ready")
   try:
    request=prepare(model,PROFILE_SHA256,configuration(a.compiler_configuration))
    generated,lowering=candidate_source(request)
    validate_candidate(dict(schema=1,request_id=request["request_id"],hecate_source=generated),request)
    check["lowering"]=lowering
   except (ValueError,TypeError,IndexError) as error:
    check.update(status="blocked_rule_preflight",reason=str(error))
   row["lowering"]=check.get("lowering")
   row.update(preflight_status=check["status"],preflight_reason=check.get("reason"))
   if name in a.fhe_id:
    row["execution_status"]="blocked_preflight"
    if check["status"]=="ready":
     case=a.output/(name+".model.json");dump(case,model)
     command=[str(VENV/"bin/python"),"-B",str(BASE/"run_candidate.py"),"--inside","--case",str(case),
              "--self-test","--max-repairs","0","--compiler-configuration",a.compiler_configuration]
     log=a.output/(name+".log");remaining=1800-(time.monotonic()-start)
     with log.open("w") as stream:
      child=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
      try:code=child.wait(timeout=min(600,remaining))
      except BaseException:
       if child.poll() is None:
        os.killpg(child.pid,signal.SIGINT)
        try:child.wait(timeout=15)
        except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=10)
       raise
     row.update(execution_status="failed",exit_code=code,log_sha256=sha(log))
     matches=re.findall(r"^Candidate evidence: (.+)$",log.read_text(),re.M)
     if not matches:raise ValueError("Execution environment or launch failed; no candidate evidence")
     evidence=Path(matches[-1]).resolve()
     if not evidence.is_relative_to(RESULTS.resolve()):raise ValueError("Evidence outside platform")
     report=strict_file(evidence/"report.json",8*1024**2)
     if report.get("agent_calls")!=0:raise ValueError("Unexpected provider call")
     row.update(evidence=str(evidence),report_sha256=sha(evidence/"report.json"),
                reported_status=report["status"],failure_layer=report.get("failure_layer") or next((x.get("failure_layer") for x in report.get("attempts",[]) if x.get("failure_layer")),None),
                attempt_failures=[x.get("diagnostic") for x in report.get("attempts",[]) if x.get("status")!="passed"])
     if code==0 and report["status"]=="passed":
      audit=verify_candidate(evidence)
      if audit["model_sha256"]!=source["model_sha256"]:raise ValueError("Executed model identity drift")
      audit_path=a.output/(name+".audit.json");dump(audit_path,audit)
      row.update(execution_status="passed",audit_sha256=sha(audit_path),comparison=audit["comparison"])
  result.append(row);dump(a.output/"progress.json",result);boundary()
 report=dict(format="poseidon-original-retry-v1",plan_binding=plan["binding"],rows=result,
  seconds=time.monotonic()-start,paid_calls=0,original_models_unchanged=True,
  new_encrypted_passes=sum(r["execution_status"]=="passed" for r in result))
 report["binding"]=digest(report);dump(a.output/"report.json",report)
 from collections import Counter
 print(json.dumps(dict(models=len(result),preflight=dict(Counter(r["preflight_status"] for r in result)),
   encrypted=dict(Counter(r["execution_status"] for r in result)),paid_calls=0,seconds=report["seconds"])))
 return int(any(r["reference_status"]!="passed" or r["execution_status"]=="failed" for r in result))
if __name__=="__main__":raise SystemExit(main())
