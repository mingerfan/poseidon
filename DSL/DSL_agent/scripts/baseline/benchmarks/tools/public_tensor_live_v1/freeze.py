"""Build frozen public requests from existing qualification data; never generates."""
import json,hashlib,sys,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5]);BASE=ROOT/"scripts/baseline"
sys.path.insert(0,str(BASE))
from hecate_python_env import VENV,enter_nix
def main():
 if "--inside"not in sys.argv:
  import shlex
  cmd='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" '+shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"])
  return enter_nix(cmd,seconds=60,keep_env=(),environment="poseidon-cpu")
 from scope import CASES,PROVIDER,BOUNDS,OPTIONS,PROFILE,PROPOSAL,digest,read,sha
 from public_weight_context import project
 from deepseek_provider import public_request
 from pytorch_ordered_contract import source_identity
 results=Path("/home/lhohy/poseidon-work/platforms/aarch64-linux/results")
 latest=results/"agent-public-tensor-v22-cpu-continuation-v1/report.json"
 old=results/"agent-public-tensor-v22-cpu-v2/observed-terminal-v2.json"
 rows={r["case"]:r for r in read(latest)["rows"]};rows["bool_multiply"]=read(old)["first_success"]
 assert not PROPOSAL.exists()
 prepared=HERE/"requests";assert not prepared.exists();prepared.mkdir()
 stage=read(Path(rows["mask_index_residual"]["package"]).parent/"task.json")["identity"]
 from application_backend import PoseidonBackend
 backend=PoseidonBackend();assert backend.identity_manifest()==stage and backend.identity()["tools"]==stage["backend"]
 cases=[]
 for name in CASES:
  row=rows[name];package=Path(row["package"]);meta=read(package.parent/"task.json");q=meta["request"]
  assert row["status"]=="passed"and row["compiled"]and row["encrypted_execution"]and row["reloaded"]and row["repeated"]
  manifest=ROOT/"scripts/baseline/benchmarks/tools/public_tensor_activation_v1/fixtures"/name/"manifest.json"
  m=read(manifest);assert m["format"]=="poseidon-trusted-pytorch-v22"
  full=prepared/(name+".request.json");public=prepared/(name+".public.json")
  full.write_text(json.dumps(q,sort_keys=True,indent=2,allow_nan=False)+"\n")
  public.write_text(json.dumps(public_request(q),sort_keys=True,indent=2,allow_nan=False)+"\n")
  case=dict(id=name,source_manifest=str(manifest.relative_to(ROOT)),source_manifest_sha256=sha(manifest),
   source_format=m["format"],input_signature=m["inputs"],output_names=m["outputs"],dtype_policy=m["dtype_policy"],
   resource_profile=PROFILE,generation_guidance="explicit-v13",slot_period=4,validation_level="numerical",track="free_generation",
   max_repairs=3,max_generations=4,max_http_attempts=16,approximation_accuracy_certified=False,
   source_files={str(manifest.parent/f):h for f,h in m["files"].items()},
   declared_weight_files={str(manifest.parent/v["file"]):v["sha256"]for v in m["weights"].values()},
   request_id=q["request_id"],model_sha256=digest(q["model"]),request_file=str(full),request_file_sha256=sha(full),
   public_request_file=str(public),public_request_sha256=sha(public),
   qualification=dict(candidate_origin="deterministic_rule",package=str(package),package_binding=row["package_binding"],
    producer_stage_manifest=meta["identity"]["manifest_sha256"],not_current_agent_generation=True))
  cases.append(case)
 p=dict(format="poseidon-public-tensor-live-proposal-v1",status="awaiting_user_approval",source_root=str(ROOT),
  head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),branch="lhy-agent-dsl",
  provider=PROVIDER,bounds=BOUNDS,numeric_tolerance=dict(atol=1e-5,rtol=1e-4),monetary_cap=None,retries_may_duplicate_billing=True,
  source_stage_identity=stage,source_stage_counts={s:len(stage[s]["sources"])for s in("generation","compilation","execution","validation")},
  source_capture_implementation=source_identity(),cases=cases,qualification_records={str(f):sha(f)for f in(latest,old)},
  paid_calls=0,http_attempts=0,old_paid_campaigns_closed=True,no_retry_of_historical_unknown_calls=True,
  external_content=["public model/decomposed graph","public weights or descriptors","layout","DSL rules","bounded diagnostics"],
  excluded_content=["rule candidate/answer","private test inputs","reference outputs","credentials","private keys"],
  protocol_evidence={"checked_date":"2026-10-10","model_and_effort":"https://api-docs.deepseek.com/api/create-chat-completion/",
   "documented_max_tokens":393216,"requested_max_tokens":384000},
  network_changes=False,dependency_installation=False,backend_changes=False,parameter_changes=False)
 p["binding"]=digest(p)
 PROPOSAL.write_text(json.dumps(p,sort_keys=True,indent=2,allow_nan=False)+"\n")
 source=(HERE/"scope.py").read_text().replace("FROZEN_BINDING",p["binding"]).replace("FROZEN_SHA",sha(PROPOSAL))
 (HERE/"scope.py").write_text(source)
 print(json.dumps(dict(proposal=str(PROPOSAL),binding=p["binding"],case_count=len(cases),api_calls=0,backend_calls=0)))
 return 0
if __name__=="__main__":raise SystemExit(main())
