"""Freeze public spatial source requests; no provider construction or compile/FHE calls."""
import copy,json,sys,time,subprocess,shlex,hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5]);BASE=ROOT/"scripts/baseline"
sys.path.insert(0,str(BASE))
from hecate_python_env import VENV,enter_nix,WORK
def write(path,data):
 with Path(path).open("x")as f:json.dump(data,f,indent=2,allow_nan=False);f.write("\n")
def main():
 if "--inside"not in sys.argv:
  command='PYTHONPATH='+shlex.quote(str(BASE))+' LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join(
    [str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"])
  return enter_nix(command,seconds=600,keep_env=(),environment="poseidon-cpu")
 import scope
 from scope import CASES,PROVIDER,BOUNDS,OPTIONS,PROFILE,PROPOSAL,digest,read,sha
 from deepseek_provider import public_request
 from pytorch_ordered_contract import source_identity
 from application_backend import PoseidonBackend
 from application_execution.context import WorkerContext
 from application_execution.backend import application
 from candidate_bundle import check_provenance
 from component_contract import reconstruct_request
 assert not PROPOSAL.exists()
 sdk=PoseidonBackend().identity_manifest()
 prior=read(ROOT/"docs/baseline/agent-spatial-source-offline-proposal-r1405.json")
 assert sdk["manifest_sha256"]==prior["sdk_identity"] and sdk["backend"]==prior["backend_identity"]
 staged=WORK/"results/agent-spatial-source-component-stage-1791728068527392953"
 stage_report=read(staged/"report.json")
 assert stage_report["failed"]==0 and tuple(r["case"]for r in stage_report["rows"])==CASES
 earlier={c["case"]:c for c in prior["cases"]}
 request_root=HERE/"requests";request_root.mkdir()
 receipts=WORK/"results"/("agent-spatial-source-live-prepare-r1407-"+str(time.time_ns()));receipts.mkdir()
 start=time.monotonic();cases=[];options=copy.deepcopy(OPTIONS)
 options["capacity_packing_version"]=prior["options"]["capacity_packing_version"]
 for name in CASES:
  folder=request_root/name;folder.mkdir();source=folder/"source";source.mkdir()
  old_manifest=Path(earlier[name]["source_manifest"])if name in earlier else staged/name/"source/manifest.json"
  m=copy.deepcopy(read(old_manifest));assert m["format"]=="poseidon-trusted-pytorch-v27"
  for filename in m["files"]:
   text=(old_manifest.parent/filename).read_text()
   text=text.replace("from pytorch_input_binding import CALL_FORMAT",'CALL_FORMAT="poseidon-pytorch-call-v1"')
   dest=source/filename;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(text)
  for info in m["weights"].values():
   dest=source/info["file"];dest.parent.mkdir(parents=True,exist_ok=True)
   if not dest.exists():dest.write_bytes((old_manifest.parent/info["file"]).read_bytes())
  m["files"]={filename:sha(source/filename)for filename in m["files"]}
  manifest=source/"manifest.json";write(manifest,m)
  app=application(WorkerContext.capture(receipts/name/"tasks"))
  source_request=dict(format="poseidon-model-source-request-v1",kind="trusted_pytorch",manifest=str(manifest),trust_local_code=True)
  task=app.submit_source(source_request,resource_profile=PROFILE,options=options,
   limits=dict(generations=4,http_attempts=16,wall_seconds=14400,api_timeout_seconds=1200,max_tokens=384000))
  state=app.prepare_source(task)
  assert state["status"]=="queued"and state["intake"]["reference_verified"]is True,(name,state)
  assert state["generations"]==state["http_attempts"]==0
  meta=app._checked_task_metadata(task);q=meta["request"]
  assert meta["identity"]==sdk and reconstruct_request(q)==q
  check_provenance(meta["model_provenance"],q["model"])
  full=folder/"request.json";public=folder/"public-request.json"
  write(full,q);payload=public_request(q)
  assert not any(k in payload for k in ("hecate_source","reference_outputs","test_inputs","model_provenance","api_key"))
  write(public,payload)
  case=dict(id=name,source_manifest=str(manifest.relative_to(ROOT)),source_manifest_sha256=sha(manifest),
   source_format=m["format"],input_signature=m["inputs"],output_names=m["outputs"],dtype_policy=m["dtype_policy"],
   resource_profile=PROFILE,generation_guidance="explicit-v13",slot_period=4,validation_level="numerical",
   track="free_generation",max_repairs=3,max_generations=4,max_http_attempts=16,approximation_accuracy_certified=False,
   source_files={str(source/f):h for f,h in m["files"].items()},
   declared_weight_files={str(source/v["file"]):v["sha256"]for v in m["weights"].values()},
   request_id=q["request_id"],model_sha256=digest(q["model"]),request_file=str(full),request_file_sha256=sha(full),
   public_request_file=str(public),public_request_sha256=sha(public),
   qualification=dict(current_source_capture="passed",current_independent_reference="passed",compilations=0,encrypted_calls=0,
      source_task_directory=str(app._dir(task)),historical_plaintext_report=str(staged/"report.json"),
      old_rule_poseidon_case=name if name in earlier else None,not_current_agent_generation=True))
  cases.append(case)
  print(json.dumps(dict(case=name,status="frozen",provider_calls=0)),flush=True)
 source_impl=dict(source_identity(),**__import__("application_execution.context_identity",fromlist=["authority"]).authority(ROOT)["sources"])
 proposal=dict(format="poseidon-spatial-source-live-proposal-v1",status="frozen_plan",
  source_root=str(ROOT),head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True,timeout=10).strip(),
  branch="lhy-agent-dsl",provider=PROVIDER,bounds=BOUNDS,numeric_tolerance=dict(atol=1e-5,rtol=1e-4),
  monetary_cap=None,retries_may_duplicate_billing=True,source_stage_identity=sdk,
  source_stage_counts={s:len(sdk[s]["sources"])for s in ("generation","compilation","execution","validation")},
  source_capture_implementation=source_impl,cases=cases,
  qualification_records={str(f):sha(f)for f in (staged/"report.json",ROOT/"docs/baseline/agent-spatial-source-offline-results-r1406.json")},
  paid_calls=0,http_attempts=0,old_paid_campaigns_closed=True,no_retry_of_historical_unknown_calls=True,
  external_content=["public model/decomposed graph","public weights or descriptors","layout","DSL rules","bounded diagnostics"],
  excluded_content=["rule candidate/answer","private test inputs","reference outputs","credentials","private keys"],
  protocol_evidence=dict(local_config="generation_protocol.py",checked_date="2026-10-11",
   chat_completion="https://api-docs.deepseek.com/api/create-chat-completion/",
   documented_models=["deepseek-flash","deepseek-v4-pro"],documented_max_tokens=393216,requested_max_tokens=384000,
   reasoning_effort="high",stream=True,verification="Official Chat Completions documentation rechecked before this freeze"),
  network_changes=False,dependency_installation=False,backend_changes=False,parameter_changes=False,
  preparation_result=str(receipts),preparation_seconds=time.monotonic()-start)
 proposal["binding"]=digest(proposal);write(PROPOSAL,proposal)
 text=(HERE/"scope.py").read_text().replace("FROZEN_BINDING",proposal["binding"]).replace("FROZEN_SHA",sha(PROPOSAL))
 (HERE/"scope.py").write_text(text)
 import importlib;scope=importlib.reload(scope)
 runner=dict(format="poseidon-spatial-source-live-runner-v1",proposal_binding=proposal["binding"],
  proposal_sha256=sha(PROPOSAL),tool_hashes=scope.tools());runner["binding"]=digest(runner);write(scope.MANIFEST,runner)
 assert PoseidonBackend().identity_manifest()==sdk
 scope.verify()
 print(json.dumps(dict(proposal=str(PROPOSAL),binding=proposal["binding"],cases=len(cases),
  sdk_identity=sdk["manifest_sha256"],provider_calls=0,compiler_calls=0,encrypted_calls=0,
  seconds=time.monotonic()-start)),flush=True)
 return 0
if __name__=="__main__":sys.exit(main())
