"""Bounded balanced-polynomial positive controls; no API."""
import json,subprocess,time,hashlib,unittest
from pathlib import Path
import numpy as np
from benchmark_runner import dump
from benchmark_suite import Builder
from benchmark_graph import require,digest,samples
from benchmark_math import evaluate
from benchmark_torch import evaluate as reference
from unified_graph_contract import prepare,validate_candidate
from unified_graph_lowering import lower
from compiler_configuration import PROFILE_SHA256,configuration
from semantic_benchmark_execution import runtime_sources
from hecate_python_env import WORK,VENV
from audit_unified_candidate import verify_candidate
out=WORK/"results/chebyshev-r28-positive";require(not out.exists(),"Preserve evidence");out.mkdir()
sources=runtime_sources();start=time.monotonic()
dump(out/"plan.json",dict(sources=sources,source_sha256=digest(sources),agent_calls=0,max_seconds=600,
    max_result_bytes=128*1024**2,native_concurrency=1,
    runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
suite=unittest.defaultTestLoader.loadTestsFromNames(["test_chebyshev_lowering","test_packed_prefix_lowering"])
result=unittest.TextTestRunner(verbosity=2).run(suite)
dump(out/"unit.json",dict(run=result.testsRun,failed=len(result.failures),errors=len(result.errors),
    skipped=len(result.skipped),passed=result.testsRun-len(result.failures)-len(result.errors)-len(result.skipped)))
require(result.wasSuccessful(),"Unit failures preserved")
rows=[];records=[]
for degree in (2,7,15):
 require(runtime_sources()==sources and time.monotonic()-start<600,"Source/time boundary")
 b=Builder([(1,)]);coeff=[(-1.)**i/(8*(i+1)) for i in range(degree+1)]
 y=b.node("polynomial",["input0",b.const(coeff)],basis="chebyshev")
 g=b.finish(y);g["id"]="balanced_degree_"+str(degree)
 folder=out/g["id"];folder.mkdir();dump(folder/"model.json",g)
 r=prepare(g,PROFILE_SHA256,configuration("seal-cpu-eva-w45-v1"))
 src=lower(r,balanced_chebyshev=True);candidate=dict(schema=1,request_id=r["request_id"],hecate_source=src)
 validate_candidate(candidate,r);dump(folder/"responses.json",[json.dumps(candidate)])
 for inputs in samples(g,16):
  a,c=evaluate(g,inputs),reference(g,inputs)
  for key in a:np.testing.assert_allclose(a[key],c[key],atol=1e-12,rtol=1e-12)
 cmd=[str(VENV/"bin/python"),"-B","scripts/baseline/run_candidate.py","--inside","--case",str(folder/"model.json"),
      "--replay",str(folder/"responses.json"),"--max-repairs","0","--compiler-configuration","seal-cpu-eva-w45-v1"]
 with (folder/"run.log").open("w") as f:proc=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=180)
 matches=[s.split("Candidate evidence: ",1)[1] for s in (folder/"run.log").read_text().splitlines() if s.startswith("Candidate evidence: ")]
 row=dict(id=g["id"],status="runner_failed",exit_code=proc.returncode)
 if matches:
  evidence=Path(matches[-1]);report=json.loads((evidence/"report.json").read_text())
  row.update(status=report["status"],evidence=str(evidence))
  require((evidence/"attempt-00/candidate.py").read_text()==src,"Candidate substitution")
  if report["status"]=="passed":records.append(verify_candidate(evidence))
 rows.append(row);dump(out/"progress.json",dict(rows=rows));print(row,flush=True)
 size=sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
 size+=sum(sum(p.stat().st_size for p in Path(r["evidence"]).rglob("*") if p.is_file()) for r in rows if "evidence" in r)
 require(size<=128*1024**2,"Evidence budget")
require(runtime_sources()==sources,"Source drift")
dump(out/"report.json",dict(rows=rows,records=records,passed=len(records),failed=len(rows)-len(records),
    source_sha256=digest(sources),agent_calls=0,seconds=time.monotonic()-start))
require(len(records)==3,"Positive encrypted controls failed")
