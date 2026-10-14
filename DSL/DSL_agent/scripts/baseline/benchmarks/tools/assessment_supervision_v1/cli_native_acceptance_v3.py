"""Native-shell assessment CLI checks; prepared once in the pinned environment."""
import hashlib,json,os,shlex,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5]);BASE=ROOT/"scripts/baseline"
WORK=Path("/home/lhohy/poseidon-work/platforms/aarch64-linux")
OUT=WORK/"results/agent-assessment-supervision-cli-v3"
os.environ["POSEIDON_PLATFORM"]="aarch64-linux"
os.environ["POSEIDON_WORK_ROOT"]="/home/lhohy/poseidon-work"
sys.path.insert(0,str(BASE))
def canonical(v):return json.dumps(v,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
def digest(v):return hashlib.sha256(canonical(v)).hexdigest()
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write_new(path,value):
    with os.fdopen(os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600),"wb")as f:
        f.write(canonical(value));f.flush();os.fsync(f.fileno())
def tree_hashes(folder):
    return {str(p.relative_to(folder)):sha(p)for p in folder.rglob("*")if p.is_file()}
def frozen():
    p=json.loads((ROOT/"docs/baseline/agent-mha-tensor-axis-live-proposal-r1333.json").read_text())
    for stage in ("generation","compilation","execution","validation"):
        for rel,want in p["source_stage_identity"][stage]["sources"].items():
            assert sha(BASE/rel)==want,"pending_paid_source_changed"
    return p
def prepare():
    from hecate_python_env import VENV
    assert os.environ.get("IN_NIX_SHELL")=="pure"and Path(sys.prefix)==VENV,"pinned_preparation"
    from application_identity import current_identity_manifest
    from application_execution.context import WorkerContext
    from application_execution.backend import application
    from assessment_array_dataset import export_dataset
    from candidate_bundle import check_bundle
    import numpy as np
    proposal=frozen()
    assert current_identity_manifest()==proposal["source_stage_identity"]
    tasks=WORK/"results/agent-tensor-axis-context-budget-acceptance-r1334/tasks"
    ids=[p.name for p in tasks.iterdir()if p.is_dir()and(p/"program").is_dir()]
    assert len(ids)==1,"exact_qualified_program"
    task=ids[0];app=application(WorkerContext.capture(tasks))
    original=app.result(task);assert original["status"]=="succeeded"
    request=app._checked_task_metadata(task)["request"]
    bundle=check_bundle(tasks/task/"program")
    assert not OUT.exists();OUT.mkdir(mode=0o700)
    arrays={s["name"]:np.zeros([2]+s["shape"],dtype=np.float64)for s in request["model"]["inputs"]}
    manifest=export_dataset(request,arrays,OUT/"dataset")
    evidence={"format":"poseidon-assessment-native-preparation-v1","task":task,"tasks":str(tasks),
      "dataset_binding":manifest["binding"],"package_binding":bundle["binding"],
      "program_hashes":tree_hashes(tasks/task/"program"),"original_result_sha256":digest(original),
      "original_state_sha256":sha(tasks/task/"state.json"),"request_id":request["request_id"],
      "proposal_binding":proposal["binding"]}
    evidence["binding"]=digest(evidence);write_new(OUT/"preparation.json",evidence)
    return 0
def main():
    if sys.argv[1:]==["--prepare"]:return prepare()
    assert not sys.argv[1:],"explicit_single_scope"
    assert not os.environ.get("IN_NIX_SHELL"),"native_parent_required"
    started=time.monotonic();proposal=frozen()
    assert not OUT.exists(),"fresh_acceptance_root"
    from hecate_python_env import VENV,enter_nix
    cmd='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join(
        [str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--prepare"])
    assert enter_nix(cmd,seconds=30,keep_env=(),environment="poseidon-cpu")==0,"preparation_exit"
    prepared=json.loads((OUT/"preparation.json").read_text())
    assert prepared["binding"]==digest({k:v for k,v in prepared.items()if k!="binding"})
    common=["/usr/bin/python3","-B",str(ROOT/"scripts/agent_assessment.py"),
      "--tasks",prepared["tasks"],"--store",str(OUT/"assessment-store"),"--platform","aarch64-linux",
      "--work-root","/home/lhohy/poseidon-work","--timeout","20"]
    rows=[]
    def invoke(action,args,expected):
        assert time.monotonic()-started<120,"acceptance_total_window"
        begin=time.monotonic()
        r=subprocess.run(common+[action,*args],stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=25)
        assert len(r.stdout)<=16*1024**2+4096 and len(r.stdout.splitlines())==1,"single_json_result"
        response=json.loads(r.stdout)
        assert response["binding"]==digest({k:v for k,v in response.items()if k!="binding"})
        write_new(OUT/(action+".json"),response)
        assert response["status"]==expected,(response.get("failure")or{}).get("code")
        assert response["process_cleanup"]["confirmed"]is True
        assert response["process_cleanup"]["status"]=="completed"
        if"result"in response:
            value=response["result"]
            assert value["budget"]["used"]==0 and not value["worker_active"]
            assert value["new_agent_calls"]==value["new_compiler_calls"]==0
        rows.append({"action":action,"status":response["status"],"exit_code":r.returncode,
          "seconds":time.monotonic()-begin,"cleanup_confirmed":True,
          "supervision_binding":response["process_cleanup"]["binding"]})
        return response
    submitted=invoke("submit",["--task",prepared["task"],"--dataset",str(OUT/"dataset/dataset.json"),
      "--dataset-binding",prepared["dataset_binding"],"--max-new-assessments","0"],"submitted")
    handle=submitted["handle"]
    args=["--evaluation-id",handle["evaluation_id"],"--submission-binding",handle["binding"]]
    invoke("status",args,"queued")
    invoke("advance",args+["--max-new-parts","0"],"budget_exhausted")
    invoke("result",args,"budget_exhausted")
    task=Path(prepared["tasks"])/prepared["task"]
    assert tree_hashes(task/"program")==prepared["program_hashes"],"original_package_changed"
    assert sha(task/"state.json")==prepared["original_state_sha256"],"original_task_changed"
    frozen();assert time.monotonic()-started<120,"acceptance_total_window"
    report={"format":"poseidon-assessment-native-cli-acceptance-v1","status":"passed","rows":rows,
      "command_calls":len(rows),"seconds":time.monotonic()-started,
      "package_preserved":True,"original_task_preserved":True,"source_task":prepared["task"],
      "package_binding":prepared["package_binding"],"preparation_binding":prepared["binding"],
      "generations":0,"http_attempts":0,"compiler_calls":0,"encrypted_calls":0,"decryptions":0,
      "quota":0,"numerical_validation_performed":False,"sdk_and_component_path_real":True,
      "parent_is_native":True,"pending_paid_identity_unchanged":True}
    report["binding"]=digest(report);write_new(OUT/"report.json",report)
    print(json.dumps(report));return 0
if __name__=="__main__":raise SystemExit(main())
