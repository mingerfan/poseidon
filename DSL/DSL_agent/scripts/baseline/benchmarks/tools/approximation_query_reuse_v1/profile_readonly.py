"""Bounded read-only profile of completed dataset result retrieval."""
from pathlib import Path
import sys,shlex,time,cProfile,pstats,json
HERE=Path(__file__).resolve().parent;B=HERE.parents[2];T=B/"benchmarks/tools"
sys.path[:0]=[str(B),str(T/"tensor_contraction_live_v1"),str(T/"static_broadcast_live_v1")]
import qualification_scope as q,run as live
from hecate_python_env import VENV,enter_nix
if sys.argv[1:]!=["--inside"]:
 live.no_active()
 command='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"])
 raise SystemExit(enter_nix(command,seconds=240,environment="poseidon-cpu"))
from application_execution.context import WorkerContext
from application_execution.backend import application
from application_approximation_assessment import ApplicationApproximationAssessment
prior=q.read(T/"tensor_contraction_live_v1/live-plan.json")
folder=q.RESULTS/"agent-approximation-query-reuse-r1616/matrix"
task_store=Path(prior["campaign"])/"matrix/tasks"
app=application(WorkerContext.capture(task_store))
service=ApplicationApproximationAssessment(app,folder/"assessment-store")
handle=q.read(folder/"handle.json")
expected=q.read(folder/"result.json")["binding"]
profile=cProfile.Profile();started=time.monotonic();profile.enable()
actual=service.result(handle)
profile.disable();seconds=time.monotonic()-started
q.require(actual["binding"]==expected,"read_only_result_unchanged")
stats=pstats.Stats(profile)
rows=[]
for (filename,line,function),values in stats.stats.items():
 primitive,total,own,cumulative,callers=values
 rows.append(dict(file=filename,line=line,function=function,primitive_calls=primitive,total_calls=total,
                  self_seconds=own,cumulative_seconds=cumulative))
rows.sort(key=lambda r:r["cumulative_seconds"],reverse=True)
report=dict(format="poseidon-approximation-readonly-profile-v1",seconds=seconds,top=rows[:24],
 result_binding=expected,api_calls=0,compiler_calls=0,encrypted_calls=0,native_reference_calls=0,
 source_sha256=q.sha(Path(__file__).resolve()),diagnosis_is_profiling_only=True)
report["binding"]=q.digest(report);q.write(HERE/"readonly-profile.json",report)
print(json.dumps(report),flush=True)
