"""All frozen construction tasks: static preparation, not execution."""
import json,time,hashlib
from collections import Counter
from pathlib import Path
from benchmark_runner import DEFAULT,load,dump
from benchmark_graph import require,digest
from compiler_configuration import PROFILE_SHA256,configuration
from unified_graph_contract import prepare
from unified_graph_lowering import candidate_source
from unified_public_contract import CONTRACT
from semantic_benchmark_execution import runtime_sources
from hecate_python_env import WORK
sources=runtime_sources();start=time.monotonic();models,index=load(DEFAULT)
models={r["model"]["id"]:r["model"] for r in models}
coverage=json.loads((DEFAULT/"coverage.json").read_text());rows=[]
for task in coverage["directed_tasks"]:
 row=dict(id=task["id"],task_sha256=task["task_sha256"],model=task["model_id"],exercise=task["exercise"],profile=task["profile"])
 try:
  request=prepare(models[task["model_id"]],PROFILE_SHA256,configuration("seal-cpu-eva-w45-v1"),task["exercise"],
                  construction_profile=CONTRACT if task["profile"]==CONTRACT else None)
  source,record=candidate_source(request)
  row.update(status="ready",request_id=request["request_id"],source_sha256=hashlib.sha256(source.encode()).hexdigest(),lowering=record)
 except (ValueError,TypeError,IndexError) as e:row.update(status="blocked",reason=str(e))
 rows.append(row)
require(runtime_sources()==sources,"Preflight source drift")
out=WORK/"results/packed-prefix-r27-all-constructions.json";require(not out.exists(),"Preserve preflight")
dump(out,dict(planned=len(rows),counts=dict(Counter(r["status"] for r in rows)),rows=rows,
              seconds=time.monotonic()-start,source_sha256=digest(sources),actual_compile=False,actual_encrypted_execution=False,agent_calls=0))
print("Full construction static preparation",len(rows),dict(Counter(r["status"] for r in rows)),flush=True)
