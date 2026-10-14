"""Prepare one new task from bound historical model/source; no execution."""
import os,sys,shlex
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE));sys.path.insert(1,str(HERE.parents[2]))
from hecate_python_env import enter_nix,VENV
from scope import *
def inside():
 from application_execution.context import WorkerContext
 from application_execution.backend import application
 from candidate_bundle import check_provenance
 require(os.environ.get("IN_NIX_SHELL")=="pure"and Path(sys.prefix)==VENV,"Pinned one-case preparation")
 require(not OUT.exists()and not PROPOSAL.exists(),"Fresh one-case scope")
 old=read(ORIGINAL);item=next(x for x in old["cases"]if x["case"]=="named")
 capture=HERE.parent/"public_expression_cutover_v1/capture-evidence-v3/named"
 model=read(capture/"model.json");provenance=read(capture/"provenance.json")
 check_provenance(provenance,model)
 require(measure(WORK/"results")+SCOPE["new_bytes"]<=SCOPE["global_bytes"]and
  shutil.disk_usage(WORK).free>=SCOPE["reserve_bytes"]+SCOPE["new_bytes"],"Prepared space available")
 OUT.mkdir(mode=0o700)
 app=application(WorkerContext.capture(OUT/"tasks"))
 task=app.submit(model,model_provenance=provenance,options=OPTIONS,
  limits=dict(generations=1,http_attempts=1,wall_seconds=180,api_timeout_seconds=1200,max_tokens=384000))
 meta=app._checked_task_metadata(task);request=meta["request"]
 require(request==read(item["request"]),"One-case mathematical request unchanged")
 write(OUT/"request.json",request)
 (OUT/"candidate.py").write_bytes(Path(item["candidate"]).read_bytes())
 sdk=current()
 plan=dict(format="poseidon-public-expression-output-offline-proposal-v1",scope=SCOPE,options=OPTIONS,
  case="named",source_contract="poseidon-trusted-pytorch-v23",backend="poseidon_cpu",
  candidate_origin="unchanged_previous_rule",sdk_identity=sdk["manifest_sha256"],
  stage_identities={s:sdk[s]["sha256"]for s in ("generation","compilation","execution","validation")},
  original_proposal_binding=old["binding"],original_report_binding=read(ROOT/"docs/baseline/agent-public-expression-offline-results-r1370.json")["binding"],
  task_id=task,task_store=str(OUT/"tasks"),request=str(OUT/"request.json"),
  request_id=request["request_id"],candidate=str(OUT/"candidate.py"),
  numeric_tolerance=dict(atol=1e-5,rtol=1e-4),
  tools={str(HERE.joinpath(n).relative_to(ROOT)):sha(HERE/n)for n in ("scope.py","prepare.py","run.py")},
  files={**item["files"],str(OUT/"request.json"):sha(OUT/"request.json"),str(OUT/"candidate.py"):sha(OUT/"candidate.py")},
  compiler_calls=0,encrypted_calls=0,provider_calls=0,http_attempts=0,approved=False)
 plan["binding"]=digest(plan);verify(plan);write(PROPOSAL,plan);write(OUT/"proposal.json",plan)
 print(json.dumps(dict(status="prepared",binding=plan["binding"],request_id=plan["request_id"],
  sdk=plan["sdk_identity"],task=task,compiler_calls=0,encrypted_calls=0,provider_calls=0)))
 return 0
if __name__=="__main__":
 if "--inside"in sys.argv:raise SystemExit(inside())
 command='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"])
 raise SystemExit(enter_nix(command,seconds=120,keep_env=(),environment="poseidon-cpu"))
