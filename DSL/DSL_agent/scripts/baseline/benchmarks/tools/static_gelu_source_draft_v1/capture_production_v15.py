import json,time,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from benchmark_graph import digest
from hecate_python_env import ROOT,WORK
from pytorch_model_intake import run
root=ROOT/"scripts/baseline/cases/trusted-pytorch-gelu-v15"
output=WORK/"results/agent-gelu-source-v15-capture"
if output.exists():raise RuntimeError("Preserve existing capture evidence")
output.mkdir()
started=time.monotonic();rows=[]
for case in json.loads((root/"catalog.json").read_text())["cases"]:
    answer=run(root/case["manifest"],output/case["id"],resource_profile="general-tensors-v1",trusted=True)
    report=json.loads((output/case["id"]/"capture-reference.json").read_text())
    finite=report["finite_reference"]
    rows.append(dict(id=case["id"],status=answer["status"],groups=finite["groups"],
        values=finite["compared_values"],source_variant=case["source_variant"],
        max_source_error=finite["max_original_torch_vs_independent_ideal"],
        max_decomposition_error=finite["max_decomposition_error"],
        max_approximation_error=finite["max_approximation_error"],
        source_function_binding=finite["gelu_source_binding"]))
report=dict(format="poseidon-gelu-source-production-capture-v1",status="passed",cases=rows,
    groups=sum(r["groups"] for r in rows),values=sum(r["values"] for r in rows),
    elapsed_seconds=time.monotonic()-started,compiled=False,encrypted_execution=False,
    agent_generated=False,paid_calls=0,downloads=0,installs=0)
report["binding"]=digest(report)
(output/"report.json").write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
