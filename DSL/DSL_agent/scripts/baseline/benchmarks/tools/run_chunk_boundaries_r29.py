"""Three period/capacity controls; separate from frozen corpus and tasks."""
import json,subprocess,time,hashlib,tarfile
from pathlib import Path
import numpy as np
from upstream_chunk_cases import cases,source
from upstream_candidate_helpers import manifest,CHUNK_PROFILE
from benchmark_suite import Builder
import copy
from unified_graph_contract import prepare,validate_candidate
from benchmark_runner import dump
from benchmark_graph import require,digest,samples
from benchmark_math import evaluate
from benchmark_torch import evaluate as reference
from semantic_benchmark_execution import runtime_sources
from compiler_configuration import PROFILE_SHA256,configuration
from hecate_python_env import WORK,VENV,ROOT
from audit_unified_candidate import verify_candidate
out=WORK/"results/upstream-chunks-r29-boundaries";require(not out.exists(),"Preserve evidence");out.mkdir()
sources=runtime_sources();start=time.monotonic();selected=[]
g=copy.deepcopy(cases()[1]["model"])
items=[("HE_Linear",g,8)]
b=Builder([(1,4,64)])
v=b.node("batch_norm",["input0",b.const([.125,-.25,.375,-.5]),b.const([1.,1.5,2.,2.5]),
    b.const([.25,-.5,.75,-1.]),b.const([.01,-.02,.03,-.04])],eps=1e-5)
g=b.finish(v);items.extend([("HE_MPBN",copy.deepcopy(g),64),("HE_MPBN",copy.deepcopy(g),256)])
for i,(family,g,p) in enumerate(items):
 g["id"]="chunk_boundary_"+str(i);src,required=source(g,manifest(g,CHUNK_PROFILE,p),p,family)
 selected.append(dict(name=g["id"],model=g,source=src,required_helpers=required,
                      configuration="seal-cpu-eva-w45-v1",profile=CHUNK_PROFILE,chunk_period=p))
require(len(selected)==3,"Three fixed boundary controls")
dump(out/"plan.json",dict(sources=sources,source_sha256=digest(sources),cases=selected,native_concurrency=1,
    max_seconds=1800,max_result_bytes=512*1024**2,agent_calls=0,
    runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
with tarfile.open(out/"source.tar.gz","w:gz") as tar:
 for name in sources:tar.add(ROOT/name,arcname=name)
rows=[];records=[]
for item in selected:
 require(runtime_sources()==sources and time.monotonic()-start<1800,"Source/time boundary")
 folder=out/item["name"];folder.mkdir();g=item["model"];dump(folder/"model.json",g)
 row=dict(id=item["name"],status="preflight_blocked")
 try:
  r=prepare(g,PROFILE_SHA256,configuration(item["configuration"]),helper_profile=item["profile"],
            helper_exercise=item["required_helpers"],chunk_period=item["chunk_period"])
  candidate=dict(schema=1,request_id=r["request_id"],hecate_source=item["source"])
  check=validate_candidate(candidate,r);row["static_check"]=check
  dump(folder/"responses.json",[json.dumps(candidate)])
  for inputs in samples(g,16):
   a,b=evaluate(g,inputs),reference(g,inputs)
   for key in a:np.testing.assert_allclose(a[key],b[key],atol=1e-12,rtol=1e-12)
 except ValueError as error:
  row["reason"]=str(error);rows.append(row);dump(out/"progress.json",dict(rows=rows));continue
 cmd=[str(VENV/"bin/python"),"-B","scripts/baseline/run_candidate.py","--inside","--case",str(folder/"model.json"),
      "--replay",str(folder/"responses.json"),"--max-repairs","0","--compiler-configuration",item["configuration"],
      "--unified-helpers",item["profile"],"--unified-chunk-period",str(item["chunk_period"])]
 for name in item["required_helpers"]:cmd+=["--unified-helper-exercise",name]
 with (folder/"run.log").open("w") as f:proc=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=300)
 row["exit_code"]=proc.returncode;row["status"]="runner_failed"
 matches=[s.split("Candidate evidence: ",1)[1] for s in (folder/"run.log").read_text().splitlines() if s.startswith("Candidate evidence: ")]
 if matches:
  evidence=Path(matches[-1]);report=json.loads((evidence/"report.json").read_text())
  row.update(status=report["status"],evidence=str(evidence),
     evidence_bytes=sum(p.stat().st_size for p in evidence.rglob("*") if p.is_file()))
  row["attempts"]=[{k:a[k] for k in ("status","failure_layer","diagnostic","compiled","executed","numerically_correct") if k in a} for a in report["attempts"]]
  if report["status"]=="passed":
   record=verify_candidate(evidence);record["case_id"]=item["name"];records.append(record)
 rows.append(row);dump(out/"progress.json",dict(rows=rows));print(item["name"],row["status"],flush=True)
 require(sum(p.stat().st_size for p in out.rglob("*") if p.is_file())+sum(r.get("evidence_bytes",0) for r in rows)<=512*1024**2,"Evidence budget")
require(runtime_sources()==sources,"Final source drift")
count=sum(r["comparison"]["compared_values"] for r in records)
dump(out/"report.json",dict(planned=3,passed=len(records),failed=len(rows)-len(records),skipped=0,rows=rows,records=records,
 source_sha256=digest(sources),agent_calls=0,seconds=time.monotonic()-start,compared_values=count,
 max_absolute_error=max((r["comparison"]["max_absolute_error"] for r in records),default=None)))
require(len(records)==3,"Chunk helper failures preserved")
