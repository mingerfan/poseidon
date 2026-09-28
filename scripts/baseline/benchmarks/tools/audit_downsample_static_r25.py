
"""Static eligibility only, never encrypted success."""
import hashlib
from pathlib import Path
from collections import Counter
from benchmark_runner import DEFAULT,load,dump
from benchmark_graph import digest,require
from unified_graph_contract import layout
from upstream_adapters.downsample_node import bind_node
from semantic_benchmark_execution import runtime_sources
from hecate_python_env import WORK
rows,index=load(DEFAULT);accepted=[];blocked=[]
for row in rows:
    model=row["model"];p=layout(model)["input_slot_period"]
    for node in model["nodes"]:
        if node["op"]!="slice":continue
        for helper in ("HE_DS",):
            try:
                binding=bind_node(model,node["id"],p)
                accepted.append(dict(model_id=model["id"],node_id=node["id"],helper=helper,binding_sha256=digest(binding),
                                     work=binding["work"],inner_calls=len(binding["calls"]),split=row["split"]))
            except ValueError as error:
                blocked.append(dict(model_id=model["id"],node_id=node["id"],helper=helper,reason=str(error)))
out=WORK/"results/upstream-ds-r25-static-mappings.json";require(not out.exists(),"Preserve evidence")
dump(out,dict(schema=1,accepted_nodes=dict(Counter(r["helper"] for r in accepted)),accepted=accepted,blocked=blocked,
              execution_status="not_run",encrypted_success_inferred=False,model_set_sha256=index["model_set_sha256"],
              source_sha256=digest(runtime_sources()),runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
print(dict(Counter(r["helper"] for r in accepted)))
