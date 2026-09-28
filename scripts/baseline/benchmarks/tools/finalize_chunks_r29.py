"""Checked phase acceptance; keep full benchmark/Agent goal explicitly incomplete."""
import json,hashlib,subprocess,ast
from pathlib import Path
from benchmark_graph import require,digest
from benchmark_runner import DEFAULT,load,dump
from semantic_benchmark_execution import runtime_sources
from hecate_python_env import ROOT,WORK
base=WORK/"results";evidence={}
def read(name):
 p=base/name;evidence[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest();return json.loads(p.read_text())
source=digest(runtime_sources())
complete=read("upstream-chunks-r29-complete.json");extra=read("upstream-chunks-r29-all-helper-complete.json")
require(all(r["status"]=="passed" and r["source_sha256"]==source for r in (complete,extra)),"Current-source phases incomplete")
unit=read("upstream-chunks-r29-unit.json");require(unit["passed"]==264 and unit["success"] and not unit["errors"] and not unit["failures"],"Unit gate")
plain=read("upstream-chunks-r29-plaintext/report.json");require(plain["passed"]==1200 and plain["failed"]==0 and plain["tests"]==19200,"Dual reference gate")
reports={}
for name,n in (("helper-directed",13),("regression",2),("chunk-regression",18),("legacy-helpers",3),("helper-remainder",44),("boundaries",3),("silu-w40",1)):
 r=read("upstream-chunks-r29-"+name+"/report.json")
 require(r["source_sha256"]==source and r["passed"]==n and r["failed"]==0,"Native gate: "+name)
 reports[name]=r
negative=read("upstream-chunks-r29-negative/report.json")
require(negative["source_sha256"]==source and negative["rejection_test_passed"] and negative["positive_model_passes"]==0,"Negative gate")
a=negative["rows"][0];require(a["compiled"] and a["executed"] and not a["numerically_correct"] and a["failure_layer"]=="numerical_comparison","Negative failure layer")
audit=read("upstream-chunks-r29-compiler-evidence.json");rest=read("upstream-chunks-r29-remainder-compiler-evidence.json")
require(audit["audited"]==audit["numerical_passed"]==36 and not audit["numerical_failed"],"Main artifact gate")
require(rest["audited"]==48 and rest["numerical_passed"]==47 and rest["numerical_failed"]==1,"Remainder artifact gate")
silu_audit=read("upstream-chunks-r29-silu-compiler-evidence.json")
require(silu_audit["audited"]==silu_audit["numerical_passed"]==1 and not silu_audit["numerical_failed"],"Chunk SiLU artifact gate")
silu_failure=read("upstream-chunks-r29-silu/report.json")
require(silu_failure["source_sha256"]==source and silu_failure["failed"]==1 and silu_failure["passed"]==0,"Retain w45 failure")
failed_row=silu_failure["rows"][0];fa=failed_row["attempts"][0]
require(fa["failure_layer"]=="compiler" and not fa["compiled"] and not fa["executed"],"w45 failure layer")
failed_path=Path(failed_row["evidence"]);control_path=Path(reports["silu-w40"]["rows"][0]["evidence"])
require((failed_path/"attempt-00/candidate.py").read_bytes()==(control_path/"attempt-00/candidate.py").read_bytes(),"SiLU source changed between controls")
import numpy as np
with np.load(failed_path/"arrays.npz",allow_pickle=False) as left,np.load(control_path/"arrays.npz",allow_pickle=False) as right:
 require(set(left.files)==set(right.files) and all(np.array_equal(left[k],right[k]) for k in left.files),"SiLU inputs/reference changed")
compat=read("upstream-chunks-r29-compatibility.json")
require(compat["old_requests"]==508 and compat["all_identical"] and compat["old_helpers_identical"]==47 and compat["new_helpers"]==13,"Compatibility")
rows,index=load(DEFAULT);coverage=json.loads((DEFAULT/"coverage.json").read_text())
require(len(coverage["helper_directed_tasks"])==60 and len(coverage["directed_tasks"])==627 and len(coverage["rejection_tasks"])==48,"Ledger denominator")
from upstream_helper_directed_cases import cases
expected={r["name"]:r for r in cases()};actual={}
for name in ("helper-directed","legacy-helpers","helper-remainder"):
 for item in reports[name]["rows"]:
  task_id=item["id"].removeprefix("legacy_")
  require(task_id not in actual,"Duplicate helper task credit")
  folder=Path(item["evidence"]);request=json.loads((folder/"request.json").read_text())
  require(digest(request["model"])==expected[task_id]["model_sha256"] and
          (folder/"attempt-00/candidate.py").read_text()==expected[task_id]["source"],"Helper model/source binding")
  actual[task_id]=str(folder)
require(set(actual)==set(expected) and len(actual)==60,"Incomplete helper task coverage")
comparisons=[record["comparison"] for r in reports.values() for record in r["records"]]
count=sum(r["compared_values"] for r in comparisons)
summary=dict(schema=1,phase="chunk_helper_r29",status="phase_complete_overall_plan_incomplete",
 checkout=str(ROOT),head=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
 branch=subprocess.check_output(["git","branch","--show-current"],text=True).strip(),
 runtime_source_sha256=source,benchmark_suite=str(DEFAULT.relative_to(ROOT)),models=1200,topologies=779,
 construction_tasks=627,helper_tasks=60,rejection_tasks=48,unchanged_old_helper_tasks=47,new_chunk_helper_tasks=13,
 helper_tasks_executed=60,helper_evidence=actual,unit=unit,plaintext=dict(passed=1200,failed=0,tests=19200),
 positive_encrypted_executions=84,positive_failed=1,compiler_failed=1,execution_skipped=0,
 silu_configuration_controls=dict(w45_failure=failed_row,w40=reports["silu-w40"]["rows"][0],candidate_and_reference_identical=True),
 intentional_negative=negative,
 compiler_artifacts_audited=85,compared_values=count,max_absolute_error=max(r["max_absolute_error"] for r in comparisons),
 weighted_mae=sum(r["mae"]*r["compared_values"] for r in comparisons)/count,
 old_requests_identical=508,main_chunk_batch={k:reports["helper-directed"][k] for k in ("planned","passed","failed","compared_values","max_absolute_error","seconds")},
 boundary_rows=reports["boundaries"]["rows"],configuration="per immutable task; existing w40/w45 profiles, no defaults changed",
 tolerance="abs(actual-reference) <= 1e-5 + 1e-4*abs(reference)",reference_changed=False,security_changed=False,
 old_w40_failure_preserved=True,actual_compile=True,actual_seal_cpu=True,actual_decrypt=True,
 paid_api_calls=0,installed=False,downloaded=False,commit=False,push=False,pr=False,x86_retested=False,gpu_retested=False,
 evidence=evidence,remaining=["final-source 48-case pilot and whole 1200/627 encrypted evaluation",
 "ledger requirement-by-requirement completion audit","separately approved real Agent evaluation"],
 limitations=["three real bootstrap helper families blocked","v9 chunk helpers limited to SiLU/MPBN/Linear/ReshapeLinear",
 "other chunk/helper combinations retain explicit rejections","two deep Max mathematical compiler failures retained from r28",
 "old w40 numerical failure not erased by later passing runs","new two-chunk SiLU w45 compiler capacity failure; separate w40 control passed","finite numerical/intervention evidence, not all-input proof"])
target=ROOT/"docs/baseline/upstream-chunk-helper-acceptance.json";require(not target.exists(),"Preserve acceptance");dump(target,summary)
print(json.dumps({k:summary[k] for k in ("status","runtime_source_sha256","helper_tasks_executed","positive_encrypted_executions","compiler_artifacts_audited","max_absolute_error")}))
