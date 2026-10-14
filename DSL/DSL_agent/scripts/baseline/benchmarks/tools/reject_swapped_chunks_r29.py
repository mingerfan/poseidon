"""Intentional semantic miswiring: must compile/execute but fail frozen comparison."""
import json,subprocess,time,hashlib
from pathlib import Path
from upstream_chunk_cases import cases
from unified_graph_contract import prepare,validate_candidate
from benchmark_graph import require,digest
from benchmark_runner import dump
from semantic_benchmark_execution import runtime_sources
from compiler_configuration import PROFILE_SHA256,configuration
from hecate_python_env import WORK,VENV
out=WORK/"results/upstream-chunks-r29-negative";require(not out.exists(),"Preserve negative evidence");out.mkdir()
sources=runtime_sources();row=cases()[0]
r=prepare(row["model"],PROFILE_SHA256,configuration(row["configuration"]),helper_profile=row["profile"],
          helper_exercise=row["required_helpers"],chunk_period=row["chunk_period"])
src=row["source"];require(len(row["required_helpers"])==2,"Fixed negative call count")
for name in row["required_helpers"]:
    old=name+"(x,y,zero_ct)";require(src.count(old)==1,"Fixed negative wiring")
    src=src.replace(old,name+"(y,x,zero_ct)")
require("def golden(x,y,zero_ct):" in src,"Preserve declared input order")
candidate=dict(schema=1,request_id=r["request_id"],hecate_source=src)
validate_candidate(candidate,r)
dump(out/"model.json",row["model"]);dump(out/"responses.json",[json.dumps(candidate)])
dump(out/"plan.json",dict(source_sha256=digest(sources),expected_failure_layer="numerical_comparison",
    mutation="swap two physical input chunks; model/reference/contract unchanged",agent_calls=0,
    runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
args=[str(VENV/"bin/python"),"-B","scripts/baseline/run_candidate.py","--inside","--case",str(out/"model.json"),
      "--replay",str(out/"responses.json"),"--max-repairs","0","--compiler-configuration",row["configuration"],
      "--unified-helpers",row["profile"],"--unified-chunk-period",str(row["chunk_period"])]
for name in row["required_helpers"]:args+=["--unified-helper-exercise",name]
with (out/"run.log").open("w") as f:proc=subprocess.run(args,stdout=f,stderr=subprocess.STDOUT,timeout=300)
paths=[s.split("Candidate evidence: ",1)[1] for s in (out/"run.log").read_text().splitlines() if s.startswith("Candidate evidence: ")]
require(len(paths)==1,"Expected one negative candidate")
evidence=Path(paths[0]);report=json.loads((evidence/"report.json").read_text());a=report["attempts"][0]
observed=dict(evidence=str(evidence),status=report["status"],failure_layer=a.get("failure_layer"),
              compiled=a["compiled"],executed=a["executed"],numerically_correct=a["numerically_correct"],
              comparison=a.get("comparison"))
expected=proc.returncode!=0 and a.get("failure_layer")=="numerical_comparison" and a["compiled"] and a["executed"] and not a["numerically_correct"]
require(runtime_sources()==sources,"Source drift")
dump(out/"report.json",dict(rows=[observed],rejection_test_passed=expected,positive_model_passes=0,
                           source_sha256=digest(sources),agent_calls=0))
require(expected,"Incorrect wiring was not rejected at the expected numerical layer")
