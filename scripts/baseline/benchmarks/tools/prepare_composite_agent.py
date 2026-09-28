"""Freeze nine actual CLI requests and verify budget gates; never call a provider."""
import json,re,subprocess,sys
from pathlib import Path
from stage2_composite_agent_plan import build_plan,ROOT
from run_stage2_composite_agent import require_approval,terminal_failure_layer
from benchmark_graph import digest
from benchmark_runner import dump
from workspace_paths import RESULTS
r=RESULTS/"stage2-composite-agent-prepare-r105";r.mkdir()
plan=build_plan();assert plan==build_plan()
assert len(plan["cases"])==9 and plan["limits"]["maximum_generations"]==36 and plan["limits"]["maximum_http_attempts"]==144
for bad in ("","wrong"):
 try:require_approval(plan,bad)
 except ValueError:pass
 else:raise AssertionError("Missing/wrong binding accepted")
require_approval(plan,plan["binding"])
assert terminal_failure_layer(dict(status="passed",attempts=[dict(failure_layer="static_check"),dict(status="passed")])) is None
rows=[]
for spec in plan["cases"]:
 f=r/spec["id"];f.mkdir();dump(f/"model.json",spec["model"])
 args=[x for x in spec["candidate_arguments"] if x!="--live"]
 command=[plan["executable"],"-B",plan["entrypoint"],"--inside","--case",str(f/"model.json"),*args,"--prepare"]
 with (f/"prepare.log").open("x") as log:p=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,cwd=ROOT,timeout=90)
 text=(f/"prepare.log").read_text();assert p.returncode==0,text
 e=Path(re.search(r"Candidate evidence: (.+)",text).group(1))
 assert json.loads((e/"request.json").read_text())==spec["request"]
 report=json.loads((e/"report.json").read_text());assert report["agent_calls"]==0 and report["status"]=="request_prepared_not_generated"
 rows.append(dict(id=spec["id"],evidence=str(e),request_id=spec["request_id"],status="prepared_not_generated"))
assert plan==build_plan()
path=ROOT/"docs/baseline/stage2-composite-agent-r105.json";assert not path.exists();dump(path,plan)
body=dict(format="poseidon-composite-agent-preflight-r105-v1",plan_binding=plan["binding"],rows=rows,
          passed=9,failed=0,skipped=0,paid_calls=0,compiled=False,encrypted_execution=False,
          wrong_missing_binding_rejected=True,repeated_plan_equal=True)
body["binding"]=digest(body);dump(r/"report.json",body)
print(json.dumps(dict(plan_binding=plan["binding"],preflight_binding=body["binding"],cases=[c["id"] for c in plan["cases"]],limits=plan["limits"])))
