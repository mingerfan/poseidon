"""Verify the phase documentation against frozen evidence and current SDK."""
import sys,shlex,json,hashlib,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5]);BASE=ROOT/"scripts/baseline"
sys.path[:0]=[str(BASE/"benchmarks/tools/static_broadcast_live_v1"),str(BASE),str(HERE)]
import run as live
import qualification_scope as q
if "--inside"not in sys.argv:
 live.no_active()
 from hecate_python_env import VENV,enter_nix
 cmd='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join(
  [str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"])
 raise SystemExit(enter_nix(cmd,seconds=180,environment="poseidon-cpu"))
from application_backend import PoseidonBackend
identity=PoseidonBackend().identity_manifest()
assert identity==q.read(HERE/"identity-snapshot.json")
record=q.read(ROOT/"docs/baseline/agent-public-mask-output-binding-acceptance-r1476.json");q.bound(record)
assert identity["manifest_sha256"]==record["sdk"] and record["all_http_resolved"] and record["source_freeze_released"]
for mode in("offline","live"):
 plan=q.verify(mode);out=Path(plan["campaign"])
 assert q.read(out/"terminal.json")["exit_code"]==0
 owner=q.read(out/"owner.json");assert live.process_info(owner["pid"])is None
 for filename in("audit.json","numerical-audit.json"):
  x=q.read(out/filename);q.bound(x);assert x["status"]=="passed"
changes=q.read(HERE/"documentation-changes.json")
for rel,pair in changes.items():
 assert q.sha(ROOT/rel)==pair["after"]
 if pair["before"]is not None:assert q.sha(HERE/"documentation-prechange"/rel)==pair["before"]
for rel in("agent-public-mask-output-binding-acceptance-r1476.json","operator-decomposition-and-python.md","agent-public-mask-acceptance-r1472.json"):
 assert (ROOT/"docs/baseline"/rel).is_file()
assert "## PyTorch v33 公开条件选择"in(ROOT/"docs/baseline/operator-decomposition-and-python.md").read_text()
branch=subprocess.check_output(["git","branch","--show-current"],cwd=ROOT,text=True).strip()
assert branch=="lhy-agent-dsl"
status=subprocess.check_output(["git","status","--porcelain=v1","-z"],cwd=ROOT)
result=dict(format="poseidon-phase-documentation-check-v1",status="passed",sdk=record["sdk"],
 phase_binding=record["binding"],documents=changes,branch=branch,git_entries=len([s for s in status.split(bytes([0]))if s]),
 frozen_scope_unchanged=True,actual_owners_exited=True,old_failures_preserved=True,
 backend_identity_unchanged=True,new_compiler_calls=0,new_encrypted_calls=0,new_provider_calls=0)
result["binding"]=q.digest(result);q.write(HERE/"documentation-check.json",result)
print(json.dumps({k:result[k]for k in("status","sdk","git_entries","binding")}))
