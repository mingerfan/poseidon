"""Bounded fresh packed-prefix acceptance; manual candidates only."""
import json,subprocess,time,hashlib,tarfile
from pathlib import Path
import numpy as np
from benchmark_runner import DEFAULT,load,dump
from benchmark_graph import require,digest,samples
from benchmark_math import evaluate
from benchmark_torch import evaluate as reference
from unified_graph_contract import prepare,validate_candidate
from unified_graph_lowering import lower,candidate_source
from compiler_configuration import PROFILE_SHA256,configuration
from semantic_benchmark_execution import runtime_sources
from hecate_python_env import WORK,VENV,ROOT
from audit_unified_candidate import verify_candidate

out=WORK/"results/upstream-chunks-r29-legacy-helpers";require(not out.exists(),"Preserve batch");out.mkdir()
start=time.monotonic();sources=runtime_sources()
from upstream_virtual_cases import cases as helper_cases
selected=[]
for row in helper_cases():
 if row["name"] in ("bench_helper_0008","bench_helper_0072","bench_helper_0080"):
  selected.append(dict(id="legacy_"+row["name"],model=row["model"],kind="helper",row=row,profile=None))
require(len(selected)==3,"Fixed legacy helper denominator")
dump(out/"plan.json",dict(source_sha256=digest(sources),sources=sources,items=selected,agent_calls=0,
 max_wall_seconds=1800,max_result_bytes=512*1024**2,native_concurrency=1,
 runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
with tarfile.open(out/"frozen-source.tar.gz","w:gz") as tar:
 for name in sources:tar.add(ROOT/name,arcname=name)
 tar.add(Path(__file__),arcname=str(Path(__file__).relative_to(ROOT)))
rows=[];records=[]
for item in selected:
 require(runtime_sources()==sources and time.monotonic()-start<1800,"Frozen source/time boundary")
 require(sum(p.stat().st_size for p in out.rglob("*") if p.is_file())+sum(r.get("evidence_bytes",0) for r in rows)<=512*1024**2,"Evidence budget")
 folder=out/item["id"];folder.mkdir();g=item["model"];dump(folder/"model.json",g)
 config=item.get("row",{}).get("configuration","seal-cpu-eva-w45-v1")
 kwargs=dict(construction_profile=item["profile"])
 if item["kind"]=="directed":kwargs["construction"]=item["task"]["exercise"]
 if item["kind"]=="helper":
  row=item["row"];kwargs.update(helper_profile=row.get("profile","upstream-poly-bn-silu-v2"),helper_exercise=row["required_helpers"])
 request=prepare(g,PROFILE_SHA256,configuration(config),**kwargs)
 if item["kind"]=="directed":
  source,lowering=candidate_source(request);require(lowering["fallback"],"Expected unchanged-limit fallback")
 elif item["kind"]=="forced_prefix":source=lower(request,packed_prefix=True);lowering=dict(strategy="forced_prefix_manual")
 else:source=item["row"]["source"];lowering=dict(strategy="manual_helper_regression")
 candidate=dict(schema=1,request_id=request["request_id"],hecate_source=source);check=validate_candidate(candidate,request)
 dump(folder/"candidate.json",candidate);dump(folder/"responses.json",[json.dumps(candidate)])
 for inputs in samples(g,16):
  a,b=evaluate(g,inputs),reference(g,inputs)
  for name in a:np.testing.assert_allclose(a[name],b[name],atol=1e-12,rtol=1e-12)
 cmd=[str(VENV/"bin/python"),"-B","scripts/baseline/run_candidate.py","--inside","--case",str(folder/"model.json"),
      "--max-repairs","0","--compiler-configuration",config]
 if item["kind"]=="directed":
  cmd+=["--self-test","--unified-profile","public-v1","--unified-exercise",item["task"]["exercise"]]
 else:cmd+=["--replay",str(folder/"responses.json")]
 if item["kind"]=="helper":
  cmd+=["--unified-helpers",kwargs["helper_profile"]]
  for name in kwargs["helper_exercise"]:cmd+=["--unified-helper-exercise",name]
 with (folder/"run.log").open("w") as f:proc=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=300)
 log=(folder/"run.log").read_text();matches=[s.split("Candidate evidence: ",1)[1] for s in log.splitlines() if s.startswith("Candidate evidence: ")]
 result=dict(id=item["id"],kind=item["kind"],exit_code=proc.returncode,lowering=lowering,status="runner_failed")
 if matches:
  evidence=Path(matches[-1]);report=json.loads((evidence/"report.json").read_text())
  result.update(status=report["status"],evidence=str(evidence),evidence_bytes=sum(p.stat().st_size for p in evidence.rglob("*") if p.is_file()))
  if report["status"]=="passed":
   if item["kind"]=="directed":require(report["rule_baseline_lowering"]["strategy"]=="packed-prefix-v1","Actual fallback was not used")
   record=verify_candidate(evidence);record["case_id"]=item["id"];records.append(record)
 rows.append(result);dump(out/"progress.json",dict(rows=rows,completed=len(rows)));print(item["id"],result["status"],flush=True)
require(runtime_sources()==sources,"Final source drift")
count=sum(r["comparison"]["compared_values"] for r in records)
summary=dict(planned=len(selected),passed=len(records),failed=len(rows)-len(records),skipped=0,rows=rows,records=records,
 compared_values=count,max_absolute_error=max(r["comparison"]["max_absolute_error"] for r in records),
 weighted_mae=sum(r["comparison"]["mae"]*r["comparison"]["compared_values"] for r in records)/count,
 source_sha256=digest(sources),seconds=time.monotonic()-start,agent_calls=0)
dump(out/"report.json",summary);require(summary["failed"]==0,"Manual batch failures preserved")
