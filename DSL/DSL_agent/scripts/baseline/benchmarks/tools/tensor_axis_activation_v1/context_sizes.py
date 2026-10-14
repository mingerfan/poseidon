import sys,copy,json
from pathlib import Path
sys.path.insert(0,str(Path("scripts/baseline").resolve()))
from unified_graph_contract import prepare
from component_contract import request_options,reconstruct_request
from unified_capacity_constants import request_version
from compiler_configuration import configuration
from public_weight_context import project
from benchmark_graph import canonical
p=json.loads(Path("docs/baseline/agent-mha-feedback-live-proposal-r1330.json").read_text())
for row in p["cases"]:
 old=reconstruct_request(json.loads(Path(row["request_file"]).read_text()));opts=request_options(old)
 name=opts.pop("compiler_configuration");opts["generation_guidance"]="explicit-v12"
 opts["capacity_packing_version"]=request_version(old);opts["capacity_abi"]=old["layout"]["execution_abi"]
 r=prepare(old["model"],old["compiler_profile_sha256"],configuration(name),
 constant_policy=old["constant_origins"].get("policy"),**opts)
 ctx=project(r);guide=ctx["generation_guidance"];expr=guide["poseidon_expression_rules"]
 print(json.dumps(dict(case=row["id"],context=len(canonical(ctx)),
 prior=len(canonical(project(old))),guide=len(canonical(guide)),
 expression=len(canonical(expr)),annotations=len(canonical(expr["tensor_axis_annotations"])),
 expression_keys={k:len(canonical(v))for k,v in expr.items()})))
