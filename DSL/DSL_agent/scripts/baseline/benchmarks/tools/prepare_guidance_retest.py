"""Offline CLI preparation and approval-boundary checks; never run --live."""
import json,re,subprocess,sys
from pathlib import Path
from stage2_guidance_retest_plan import build_plan,ROOT,IDS
from run_stage2_guidance_retest import require_approval,terminal_failure_layer
from benchmark_graph import digest
from benchmark_runner import dump
R=Path("/home/lhohy/poseidon-work/platforms/aarch64-linux/results/stage2-guidance-retest-prepare-r100")
R.mkdir(exist_ok=False)
plan=build_plan()
assert plan==build_plan()
assert plan["entrypoint"].endswith("/scripts/baseline/run_candidate.py")
assert [r["id"] for r in plan["cases"]]==list(IDS)
assert len(plan["cases"])==6 and plan["limits"]["maximum_generations"]==24 and plan["limits"]["maximum_http_attempts"]==96
for bad in ("","incorrect",json.loads((ROOT/"docs/baseline/stage2-agent-pilot-r97.json").read_text())["binding"]):
 try:require_approval(plan,bad)
 except ValueError:pass
 else:raise AssertionError("Unapproved binding accepted")
assert require_approval(plan,plan["binding"])==plan # gate test only; no dispatch
assert terminal_failure_layer(dict(status="passed",attempts=[dict(failure_layer="static_check"),dict(status="passed")])) is None
assert terminal_failure_layer(dict(status="failed",attempts=[dict(failure_layer="static_check"),dict(failure_layer="seal_runtime")]))=="seal_runtime"
rows=[]
for spec in plan["cases"]:
 folder=R/spec["id"];folder.mkdir()
 model=folder/"model.json";dump(model,spec["model"])
 args=[x for x in spec["candidate_arguments"] if x!="--live"]
 assert not any(x in ("--live","--deepseek") for x in args)
 command=[plan["executable"],"-B",plan["entrypoint"],"--inside","--case",str(model),*args,"--prepare"]
 with (folder/"prepare.log").open("x") as f:
  proc=subprocess.run(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,timeout=90)
 text=(folder/"prepare.log").read_text()
 assert proc.returncode==0,text
 evidence=Path(re.search(r"Candidate evidence: (.+)",text).group(1))
 actual=json.loads((evidence/"request.json").read_text())
 report=json.loads((evidence/"report.json").read_text())
 assert actual==spec["request"]
 assert report["agent_calls"]==0 and report["status"]=="request_prepared_not_generated"
 rows.append(dict(id=spec["id"],evidence=str(evidence),request_id=actual["request_id"],status="prepared_not_generated"))
assert build_plan()==plan
out=ROOT/"docs/baseline/stage2-guidance-retest-r100.json"
assert not out.exists()
dump(out,plan)
report=dict(format="poseidon-guidance-retest-preflight-r100-v1",plan_binding=plan["binding"],
            rows=rows,passed=6,failed=0,skipped=0,paid_calls=0,compiled=False,encrypted_execution=False,
            repeated_plan_equal=True,missing_wrong_old_approval_rejected=True,terminal_failure_layer_checks_passed=True)
report["binding"]=digest(report);dump(R/"report.json",report)
print(json.dumps({"plan_binding":plan["binding"],"preflight_binding":report["binding"],"prepared":6,"paid_calls":0}))
