"""Actual plan and CLI preparation checks across campaign modes; no provider."""
import json,re,subprocess,sys
from pathlib import Path
from stage2_agent_campaign import ROOT,identity
from run_stage2_agent_campaign import validate_plan
from stage2_agent_pilot_plan import check_binding,sha
from benchmark_graph import digest
from benchmark_runner import strict_file,dump
from workspace_paths import RESULTS
campaign=RESULTS/"stage2-agent-campaign-prepare-r106"
index=strict_file(campaign/"index.json",16*1024**2);check_binding(index)
out=RESULTS/"stage2-campaign-preflight-r108";out.mkdir()
plans=[];specs=[]
for ref in index["shards"]:
 p=campaign/ref["file"];assert sha(p)==ref["sha256"]
 plan=strict_file(p,8*1024**2);assert plan["binding"]==ref["binding"]
 validate_plan(plan,plan["binding"]);plans.append(plan);specs.extend(plan["cases"])
assert len(specs)==2030 and len({s["id"] for s in specs})==2030
selected=[]
for group in sorted({s["group"] for s in specs}):selected.append(next(s for s in specs if s["group"]==group))
selected.append(next(s for s in specs if s.get("chunk_period")))
selected.append(next(s for s in specs if s["group"]=="directed_construction" and s.get("construction_profile")!="hecate-unified-public-v1"))
assert len({s["id"] for s in selected})==8
rows=[]
for spec in selected:
 folder=out/spec["id"];folder.mkdir();dump(folder/"model.json",spec["model"])
 args=[x for x in spec["candidate_arguments"] if x!="--live"]
 command=[sys.executable,"-B",str(ROOT/"scripts/baseline/run_candidate.py"),"--inside","--case",str(folder/"model.json"),*args,"--prepare"]
 with (folder/"prepare.log").open("x") as f:p=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,timeout=90)
 text=(folder/"prepare.log").read_text();assert p.returncode==0,text
 e=Path(re.search(r"Candidate evidence: (.+)",text).group(1))
 assert strict_file(e/"request.json",131072)==spec["request"]
 report=strict_file(e/"report.json",8*1024**2)
 assert report["agent_calls"]==0 and report["status"]=="request_prepared_not_generated"
 rows.append(dict(id=spec["id"],evidence=str(e),request_id=spec["request_id"],status="prepared_not_generated"))
body=dict(format="poseidon-campaign-preflight-r108-v1",campaign_binding=index["binding"],
          validated_plans=len(plans),validated_requests=2030,actual_cli_prepared=len(rows),rows=rows,
          paid_calls=0,compiled=False,encrypted_execution=False,stage2_complete=False)
body["binding"]=digest(body);dump(out/"report.json",body)
print(json.dumps({k:v for k,v in body.items() if k!="rows"}))
