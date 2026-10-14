"""Source-kind correction; existing package reuse with zero generation/backend calls."""
import sys,json,shlex
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5]);BASE=ROOT/"scripts/baseline"
sys.path[:0]=[str(BASE/"benchmarks/tools/static_broadcast_live_v1"),str(BASE),str(HERE)]
import run as live
import qualification_scope as s
from qualification import source_for_case
OUT=s.RESULTS/"agent-generic-capacity-cache-followup-r1503"
if "--inside"not in sys.argv:
 live.no_active()
 from hecate_python_env import VENV,enter_nix
 assert not OUT.exists();OUT.mkdir(mode=0o700)
 command='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"])
 raise SystemExit(enter_nix(command,seconds=180,environment="poseidon-cpu"))
from application_execution.context import WorkerContext
from application_execution.backend import application
from candidate_bundle import check_bundle
from application_backend import PoseidonBackend
assert PoseidonBackend().identity_manifest()==s.read(HERE/"identity-snapshot.json")
parent=s.RESULTS/"agent-generic-capacity-offline-r1502/weights_linear_10k"
case=next(c for c in s.read(HERE/"prepared-requests.json")["cases"]if c["id"]=="weights_linear_10k")
task=s.read(parent/"task-id.json")["task_id"]
app=application(WorkerContext.capture(parent/"tasks"))
before=check_bundle(app._dir(task)/"program");snapshot=s.read(app._dir(task)/"source.json")
source=source_for_case(case);assert source["kind"]=="graph"
again=app.submit_source(source,resource_profile="general-tensors-v1",options=snapshot["requirements"]["options"],
 limits=dict(snapshot["budget"],wall_seconds=120))
class Never:
 kind="scripted_replay";agent_calls=0
 def generate(self,*args):raise AssertionError("cache_must_not_generate")
result=app.run(again,Never())
s.write(OUT/"result.json",result)
assert result["status"]=="succeeded"and result["package"]["reused"]is True
assert result["generations"]==result["http_attempts"]==0 and result["package"]["binding"]==before["binding"]
assert check_bundle(app._dir(task)/"program")==before
assert s.read(app._dir(again)/"source.json")["files"]==snapshot["files"]
report=dict(format="poseidon-generic-capacity-cache-followup-v1",status="passed",case=case["id"],
 sdk=PoseidonBackend().identity_manifest()["manifest_sha256"],parent_report=str(parent/"case-report.json"),
 task_id=task,reuse_task_id=again,package_binding=before["binding"],model_sha256=case["model_sha256"],
 source_kind="graph",same_frozen_source=True,compiler_calls=0,encrypted_calls=0,generations=0,http_attempts=0,
 originals_or_history_modified=False)
report["binding"]=s.digest(report);s.write(OUT/"report.json",report)
print(json.dumps(report))
