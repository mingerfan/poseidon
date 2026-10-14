"""Prepare immutable current-source tasks and a bounded live proposal. No API."""
import pathlib,json,hashlib,os,sys,shlex,time,subprocess
HERE=pathlib.Path(__file__).resolve().parent;ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5]);BASE=ROOT/"scripts/baseline"
sys.path.insert(0,str(BASE))
os.environ["POSEIDON_PLATFORM"]="aarch64-linux";os.environ["POSEIDON_WORK_ROOT"]="/home/lhohy/poseidon-work"
from hecate_python_env import VENV,enter_nix
WORK=pathlib.Path("/home/lhohy/poseidon-work/platforms/aarch64-linux")
OUT=WORK/"results/agent-common-source-live-preparation-v1"
PROPOSAL=ROOT/"docs/baseline/agent-common-source-live-proposal-r1341.json"
OPTIONS=dict(generation_guidance="explicit-v13",public_weight_policy="descriptors-math-roles-v2",chunk_period=4)
PROFILE="general-tensors-v1"
def read(p):return json.loads(pathlib.Path(p).read_text())
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
def write(p,v):
 with pathlib.Path(p).open("x")as f:json.dump(v,f,sort_keys=True,indent=2,allow_nan=False);f.write("\n")
def main():
 if sys.argv[1:]!=["--inside"]:
  assert not OUT.exists()and not PROPOSAL.exists()
  OUT.mkdir(mode=0o700)
  write(OUT/"preparation-plan.json",dict(format="poseidon-common-source-preparation-v1",created_wall=time.time(),
       deadline_wall=time.time()+300,provider_calls=0,compiler_calls=0,encrypted_calls=0,new_bytes=64*1024**2))
  cmd='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join(
      [str(VENV/"bin/python"),"-B",str(pathlib.Path(__file__).resolve()),"--inside"])
  return enter_nix(cmd,seconds=300,keep_env=(),environment="poseidon-cpu")
 assert os.environ.get("IN_NIX_SHELL")=="pure"and pathlib.Path(sys.prefix)==VENV
 from application_identity import current_identity_manifest
 from application_execution.context import WorkerContext
 from application_execution.backend import application
 from candidate_contract import request_file_bytes
 from public_weight_context import project
 from application_model_source import FORMAT
 identity=current_identity_manifest()
 previous=read(BASE/"benchmarks/tools/einsum_input_v1/baseline.json")
 for group in["protected_sources","historical_request_files"]:
  for n,h in previous[group].items():assert sha(pathlib.Path(n)if pathlib.Path(n).is_absolute()else ROOT/n)==h,n
 app=application(WorkerContext.capture(OUT/"tasks"));rows=[]
 fixtures=read(BASE/"benchmarks/tools/static_partition_v1/fixtures.json")+read(BASE/"benchmarks/tools/einsum_input_v1/fixtures.json")
 for c in read(ROOT/"docs/baseline/agent-mha-tensor-axis-live-proposal-r1333.json")["cases"]:
  fixtures.append(dict(case=c["id"],manifest=c["source_manifest"],manifest_sha256=c["source_manifest_sha256"]))
 limits=dict(generations=4,http_attempts=16,wall_seconds=14400,api_timeout_seconds=1200,max_tokens=384000)
 deadline=read(OUT/"preparation-plan.json")["deadline_wall"]
 for item in fixtures:
  assert time.time()<deadline,"preparation_time_boundary"
  assert sha(pathlib.Path(item["manifest"]))==item["manifest_sha256"],"original_source_manifest_changed"
  assert current_identity_manifest()==identity,"SDK_source_changed"
  task=app.submit_source(dict(format=FORMAT,kind="trusted_pytorch",manifest=item["manifest"],trust_local_code=True),
                         resource_profile=PROFILE,options=OPTIONS,limits=limits)
  state=app.prepare_source(task)
  write(OUT/(item["case"]+".source-state.json"),state)
  assert state["status"]=="queued"and state["phase"]=="queued"and state.get("intake",{}).get("reference_verified")is True,(
         "source_prepare_rejected",item["case"],state.get("failure"))
  meta=app._checked_task_metadata(task);request=meta["request"]
  full=HERE/(item["case"]+".request.json")
  full.write_bytes(request_file_bytes(request))
  public=project(request)
  public_path=HERE/(item["case"]+".public.json")
  public_path.write_bytes(request_file_bytes(public))
  spec=read(item["manifest"])
  source_files={str((pathlib.Path(item["manifest"]).parent/n).relative_to(ROOT)):h for n,h in spec["files"].items()}
  weight_files={str((pathlib.Path(item["manifest"]).parent/v["file"]).relative_to(ROOT)):v["sha256"]for v in spec["weights"].values()}
  row=dict(id=item["case"],task_id=task,source_manifest=item["manifest"],source_manifest_sha256=item["manifest_sha256"],
           source_format=spec["format"],source_files=source_files,weight_files=weight_files,request_id=request["request_id"],
           full_request_file=str(full),full_request_file_sha256=sha(full),public_request_file=str(public_path),
           public_request_file_sha256=sha(public_path),public_context_bytes=public_path.stat().st_size,
           model_sha256=digest(request["model"]),max_generations=4,max_http_attempts=16,max_repairs=3,
           validation_level="numerical",resource_profile=PROFILE,options=OPTIONS,approximation_accuracy_certified=False)
  rows.append(row);write(OUT/(item["case"]+".request-binding.json"),row)
  print(json.dumps(dict(case=item["case"],status="prepared",public_bytes=row["public_context_bytes"],provider_calls=0)),flush=True)
 assert len(rows)==16 and len({r["id"]for r in rows})==16
 proposal=dict(format="poseidon-common-source-live-proposal-v1",revision="r1341",approved=False,executed=False,
    development_checkout=str(ROOT),branch="lhy-agent-dsl",source_stage_identity=identity,
    identity_sha256=identity["manifest_sha256"],store=str(OUT/"tasks"),cases=rows,
    provider=dict(name="deepseek",model="deepseek-flash",reasoning_effort="high",stream=True,
                  timeout_seconds=1200,max_tokens=384000,transport_retries=3),
    budget=dict(max_generations=64,max_http_attempts=256,max_repairs=3,api_concurrency=1,native_concurrency=1,
                compile_jobs=2,link_jobs=1,batch_wall_seconds=43200,case_wall_seconds=14400,
                new_batch_bytes=4*1024**3,per_case_bytes=4*1024**3,global_result_bytes=32*1024**3,
                reserve_bytes=4*1024**3,native_address_bytes=4*1024**3,native_cpu_seconds=150,currency_limit=None),
    validation=["source_capture","real_agent_generation","true_compile","artifact_integrity",
                "poseidon_cpu_encrypted_numerical","package_reload","new_inputs","cache_reuse"],
    disclosure=["public mathematical model","public weights or descriptors","layout","DSL rules","controlled feedback"],
    excluded_disclosure=["scripted answers","test arrays","reference outputs","credentials"],
    previous_unknown_http_retried=False,provider_calls=0,compiler_calls=0,encrypted_calls=0,
    tool_sha256=sha(pathlib.Path(__file__)),historical_files_preserved=True)
 proposal["binding"]=digest(proposal);write(PROPOSAL,proposal)
 write(OUT/"preparation-report.json",dict(format="poseidon-common-source-live-preparation-report-v1",status="prepared",
       cases=len(rows),proposal_binding=proposal["binding"],proposal_sha256=sha(PROPOSAL),
       current_identity=identity["manifest_sha256"],provider_calls=0,compiler_calls=0,encrypted_calls=0))
 print(json.dumps(dict(prepared=16,proposal_binding=proposal["binding"],provider_calls=0)),flush=True)
 return 0
if __name__=="__main__":raise SystemExit(main())
