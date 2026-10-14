"""Actual system-Python command path over a current qualified package; zero execution quota."""
import hashlib,json,os,shlex,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5]);BASE=ROOT/"scripts/baseline"
WORK=Path("/home/lhohy/poseidon-work/platforms/aarch64-linux")
os.environ["POSEIDON_PLATFORM"]="aarch64-linux";os.environ["POSEIDON_WORK_ROOT"]="/home/lhohy/poseidon-work"
sys.path.insert(0,str(BASE))
def main():
 if sys.argv[1:]!=["--inside"]:
  from hecate_python_env import VENV,enter_nix
  command='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"])
  return enter_nix(command,seconds=120,keep_env=(),environment="poseidon-cpu")
 from application_identity import current_identity_manifest
 from application_execution.context import WorkerContext
 from application_execution.backend import application
 from assessment_array_dataset import export_dataset
 from candidate_bundle import check_bundle
 from benchmark_graph import digest
 import numpy as np
 proposal=json.loads((ROOT/"docs/baseline/agent-mha-tensor-axis-live-proposal-r1333.json").read_text())
 assert current_identity_manifest()==proposal["source_stage_identity"]
 tasks=WORK/"results/agent-tensor-axis-context-budget-acceptance-r1334/tasks"
 ids=[p.name for p in tasks.iterdir()if p.is_dir()and(p/"program/manifest.json").exists()]
 if not ids:
  ids=[p.name for p in tasks.iterdir()if p.is_dir()and(p/"program").is_dir()]
 assert len(ids)==1,"exact_current_qualified_program"
 task=ids[0];app=application(WorkerContext.capture(tasks))
 original=app.result(task);assert original["status"]=="succeeded"
 request=app._checked_task_metadata(task)["request"];before=check_bundle(tasks/task/"program")
 root=WORK/"results/agent-assessment-supervision-cli-v1"
 assert not root.exists();root.mkdir(mode=0o700)
 arrays={s["name"]:np.zeros([2]+s["shape"],dtype=np.float64)for s in request["model"]["inputs"]}
 manifest=export_dataset(request,arrays,root/"dataset")
 common=["/usr/bin/python3","-B",str(ROOT/"scripts/agent_assessment.py"),
  "--tasks",str(tasks),"--store",str(root/"assessment-store"),
  "--platform","aarch64-linux","--work-root","/home/lhohy/poseidon-work","--timeout","30"]
 rows=[];start=time.monotonic()
 def invoke(action,args,expected):
  begin=time.monotonic()
  r=subprocess.run(common+[action,*args],stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=35)
  raw=r.stdout
  assert len(raw)<=16*1024**2+4096 and len(raw.splitlines())==1,"single_public_json"
  response=json.loads(raw);assert response["binding"]==digest({k:v for k,v in response.items()if k!="binding"})
  assert response["status"]==expected,response.get("failure",{}).get("code")
  assert response["process_cleanup"]["confirmed"]is True and response["process_cleanup"]["status"]=="completed","owned_command_cleanup"
  if"result"in response:
   value=response["result"];assert value["budget"]["used"]==0 and not value["worker_active"]
   assert value["new_agent_calls"]==value["new_compiler_calls"]==0
  (root/(action+".json")).write_bytes(raw)
  rows.append(dict(action=action,status=response["status"],exit_code=r.returncode,seconds=time.monotonic()-begin,
   cleanup_confirmed=True,supervision_binding=response["process_cleanup"]["binding"]))
  return response
 submitted=invoke("submit",["--task",task,"--dataset",str(root/"dataset/dataset.json"),
  "--dataset-binding",manifest["binding"],"--max-new-assessments","0"],"submitted")
 handle=submitted["handle"]
 args=["--evaluation-id",handle["evaluation_id"],"--submission-binding",handle["binding"]]
 invoke("status",args,"queued")
 invoke("advance",args+["--max-new-parts","0"],"budget_exhausted")
 invoke("result",args,"budget_exhausted")
 assert app.result(task)==original and check_bundle(tasks/task/"program")==before,"source_program_and_original_result_preserved"
 assert current_identity_manifest()==proposal["source_stage_identity"],"pending_paid_identity_changed"
 report=dict(format="poseidon-assessment-supervision-cli-acceptance-v1",status="passed",
  system_python="/usr/bin/python3",rows=rows,command_calls=len(rows),seconds=time.monotonic()-start,
  source_task=task,package_binding=before["binding"],package_preserved=True,
  generations=0,http_attempts=0,compiler_calls=0,encrypted_calls=0,decryptions=0,
  numerical_validation_performed=False,sdk_and_component_path_real=True,quota=0,
  pending_live_identity_unchanged=True)
 report["binding"]=digest(report);(root/"report.json").write_text(json.dumps(report,sort_keys=True,indent=2)+"\n")
 print(json.dumps(report),flush=True)
 return 0
if __name__=="__main__":raise SystemExit(main())
