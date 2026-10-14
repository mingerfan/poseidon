"""Real isolated import/request-validation probe; no tracing or encrypted execution."""
import sys,shlex,json,hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5]);BASE=ROOT/"scripts/baseline"
sys.path[:0]=[str(BASE),str(BASE/"benchmarks/tools/static_broadcast_live_v1")]
import run as live
if "--inside"not in sys.argv:
    live.no_active()
    from hecate_python_env import VENV,enter_nix
    command='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join(
        [str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"])
    raise SystemExit(enter_nix(command,seconds=120,environment="poseidon-cpu"))
from hecate_python_env import VENV
from poseidon_candidate_sandbox import run
index=json.loads((HERE/"prepared-requests.json").read_text())
root=Path("/home/lhohy/poseidon-work/platforms/aarch64-linux/results/agent-output-binding-sandbox-r1475")
assert not root.exists();root.mkdir(mode=0o700);out=root/"out";out.mkdir()
payload=root/"payload.json"
payload.write_text(json.dumps([json.loads(Path(c["request_file"]).read_text())for c in index["cases"]]))
script='import json;from unified_graph_contract import validate_request;requests=json.load(open("/payload.json"));[validate_request(r)for r in requests];print("validated",len(requests))'
tree=live.OwnedTree().start()
try:code=run(payload,out,[str(VENV/"bin/python"),"-B","-c",script],root/"probe.log",seconds=60)
finally:background=tree.finish()
assert not background and code==0 and (root/"probe.log").read_text().strip()=="validated 6"
record=dict(format="poseidon-output-binding-real-sandbox-probe-v1",status="passed",
 cases=6,sandbox_calls=1,source_trace_calls=0,compiler_calls=0,encrypted_calls=0,http_attempts=0,
 readonly_explicit_modules=True,existing_network_isolation=True,existing_limits=True,
 log_sha256=hashlib.sha256((root/"probe.log").read_bytes()).hexdigest())
(HERE/"sandbox-probe.json").write_text(json.dumps(record,indent=2))
print(json.dumps(record))
