"""Fresh five-case diagnosis, preserving failed and budget-blocked evidence."""
import json,subprocess,time,hashlib,sys
from pathlib import Path
from benchmark_suite import generate
from benchmark_runner import dump
from benchmark_graph import require,digest
from unified_graph_contract import prepare,validate_candidate
from unified_graph_lowering import lower
from compiler_configuration import PROFILE_SHA256,configuration
from semantic_benchmark_execution import runtime_sources
from hecate_python_env import WORK,VENV,ROOT
out=WORK/"results/baseline-boundaries-r27-initial";require(not out.exists(),"Preserve diagnosis");out.mkdir()
sources=runtime_sources();wanted={"bench_helper_0040","bench_helper_0044","bench_boundary_0044","bench_boundary_0061","bench_boundary_0097"}
models=[r for r in generate() if r["model"]["id"] in wanted];require(len(models)==5,"Frozen cases")
dump(out/"plan.json",dict(source_sha256=digest(sources),sources=sources,case_ids=sorted(wanted),agent_calls=0,seconds_limit=600,
    runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
rows=[];start=time.monotonic()
for item in models:
    g=item["model"];row=dict(id=g["id"],metadata=item["metadata"]);folder=out/g["id"];folder.mkdir();dump(folder/"model.json",g)
    try:
        r=prepare(g,PROFILE_SHA256,configuration("seal-cpu-eva-w45-v1"));dump(folder/"request.json",r);row["prepared"]=True
        src=lower(r);(folder/"candidate.py").write_text(src);row["rule_statements"]=len(src.splitlines())-3
        validate_candidate(dict(schema=1,request_id=r["request_id"],hecate_source=src),r);row["rule_validated"]=True
    except ValueError as e:row.update(status="preflight_blocked",reason=str(e));rows.append(row);dump(out/"progress.json",dict(rows=rows));continue
    cmd=[str(VENV/"bin/python"),"-B","scripts/baseline/run_candidate.py","--inside","--case",str(folder/"model.json"),
         "--self-test","--max-repairs","0","--compiler-configuration","seal-cpu-eva-w45-v1"]
    with (folder/"run.log").open("w") as f:proc=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=180)
    row["exit_code"]=proc.returncode;log=(folder/"run.log").read_text()
    paths=[l.split("Candidate evidence: ",1)[1] for l in log.splitlines() if l.startswith("Candidate evidence: ")]
    if paths:
        folder=Path(paths[-1]);r=json.loads((folder/"report.json").read_text())
        row.update(status=r["status"],evidence=str(folder),attempts=[{k:a[k] for k in ("status","failure_layer","diagnostic","compiled","executed","numerically_correct") if k in a} for a in r["attempts"]])
    else:row["status"]="runner_failed"
    rows.append(row);dump(out/"progress.json",dict(rows=rows))
    require(runtime_sources()==sources,"Source changed")
require(runtime_sources()==sources,"Final source changed")
dump(out/"report.json",dict(rows=rows,source_sha256=digest(sources),agent_calls=0,seconds=time.monotonic()-start))
print(json.dumps(rows))
