"""Freeze pre-change identities, requests and protected sources; no execution."""
import json,os,shlex,sys,hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5]);BASE=ROOT/"scripts/baseline"
os.environ["POSEIDON_PLATFORM"]="aarch64-linux";os.environ["POSEIDON_WORK_ROOT"]="/home/lhohy/poseidon-work"
sys.path.insert(0,str(BASE))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 if not sys.argv[1:]:
  from hecate_python_env import VENV,enter_nix
  cmd='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"])
  return enter_nix(cmd,seconds=60,keep_env=(),environment="poseidon-cpu")
 assert sys.argv[1:]==["--inside"]
 from component_contract import prepare_task
 from benchmark_suite import Builder
 from benchmark_graph import digest
 from application_identity import current_identity_manifest
 b=Builder([(4,)]);h=b.node("linear",["input0",b.const([[.2,-.1,.3,.4],[-.2,.1,.5,-.3]])])
 graph=b.finish(b.node("square",[h]))
 options=[{}]+[dict(generation_guidance=f"explicit-v{i}") for i in range(1,13)]
 records=[dict(options=o,request=prepare_task(graph,o)) for o in options]
 identity=current_identity_manifest()
 protected=json.loads((ROOT/"docs/baseline/agent-v12-live-rebound-proposal-r1304.json").read_text())["protected_sources"]
 assert all(sha(ROOT/n)==h for n,h in protected.items())
 old=json.loads((ROOT/"docs/baseline/agent-mha-tensor-axis-live-proposal-r1333.json").read_text())
 frozen={str(Path(c["request_file"])):c["request_file_sha256"] for c in old["cases"]}
 assert all(sha(n)==h for n,h in frozen.items())
 data=dict(format="poseidon-construction-feedback-baseline-v1",model=graph,legacy_requests=records,
 identity=identity,protected_sources=protected,historical_requests=frozen,
 original_checker_sha256=sha(BASE/"decorated_functions.py"),source_before={n:sha(BASE/n) for n in
 ["unified_graph_contract.py","application_diagnostics.py","validation_adapter.py","poseidon_sandbox_policy.py","sandbox_policy.py","run_candidate.py"]})
 data["binding"]=digest(data)
 with (HERE/"baseline.json").open("x") as f:json.dump(data,f,sort_keys=True,indent=2)
 print(json.dumps(dict(status="frozen",legacy_requests=len(records),protected_files=len(protected),binding=data["binding"],provider_calls=0)))
 return 0
if __name__=="__main__":raise SystemExit(main())
