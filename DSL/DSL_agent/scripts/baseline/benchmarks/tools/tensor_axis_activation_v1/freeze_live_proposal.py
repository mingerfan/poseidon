"""Freeze a new two-case review scope. Does not create a live transport."""
import copy,hashlib,json,os,shlex,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5]);BASE=ROOT/"scripts/baseline"
os.environ["POSEIDON_PLATFORM"]="aarch64-linux";os.environ["POSEIDON_WORK_ROOT"]="/home/lhohy/poseidon-work"
sys.path.insert(0,str(BASE))
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):p.write_text(json.dumps(v,sort_keys=True,indent=2)+"\n")
def main():
 if sys.argv[1:]!=["--inside"]:
  from hecate_python_env import VENV,enter_nix
  cmd='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join(
   [str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"])
  return enter_nix(cmd,seconds=60,keep_env=(),environment="poseidon-cpu")
 from application_identity import current_identity_manifest
 from unified_graph_contract import prepare,validate_request
 from component_contract import request_options,reconstruct_request
 from unified_capacity_constants import request_version
 from compiler_configuration import configuration
 from benchmark_graph import canonical,digest
 from public_weight_context import verify_context
 from generation_protocol import Config
 old=read(ROOT/"docs/baseline/agent-mha-feedback-live-proposal-r1330.json")
 doc=ROOT/"docs/baseline/agent-mha-tensor-axis-live-proposal-r1333.json"
 assert not doc.exists(),"preserve_frozen_proposal"
 folder=HERE/"requests-v12";assert not folder.exists();folder.mkdir()
 cases=[]
 for row in old["cases"]:
  file=Path(row["request_file"]);assert sha(file)==row["request_file_sha256"]
  prior=reconstruct_request(read(file));opts=request_options(prior)
  name=opts.pop("compiler_configuration");opts["generation_guidance"]="explicit-v12"
  opts["capacity_packing_version"]=request_version(prior);opts["capacity_abi"]=prior["layout"]["execution_abi"]
  c=configuration(name,schema=prior["compiler_configuration"]["schema"])
  current=prepare(copy.deepcopy(prior["model"]),prior["compiler_profile_sha256"],c,
    constant_policy=prior["constant_origins"].get("policy"),**opts)
  validate_request(current);assert reconstruct_request(current)==current
  before=copy.deepcopy(prior);after=copy.deepcopy(current)
  for value in(before,after):value.pop("generation_guidance");value.pop("request_id")
  assert canonical(before)==canonical(after),"only_guidance_binding_may_change"
  public=verify_context(current)
  text=json.dumps(public)
  assert not any(k in public for k in("test_inputs","reference","reference_outputs","hecate_source"))
  request_file=folder/(row["id"]+".json");write(request_file,current)
  value=copy.deepcopy(row)
  value.update(generation_guidance="explicit-v12",request_file=str(request_file),
   request_file_sha256=sha(request_file),request_id=current["request_id"],
   prior_request_id=prior["request_id"],semantic_binding_unchanged=True,
   public_context_bytes=len(canonical(public)))
  for name,want in value["source_files"].items():assert sha(name)==want
  for name,want in value["declared_weight_files"].items():assert sha(name)==want
  assert sha(value["source_manifest"])==value["source_manifest_sha256"]
  cases.append(value)
 identity=current_identity_manifest()
 offline=read(ROOT/"docs/baseline/agent-tensor-axis-guidance-r1332.json")
 assert identity["manifest_sha256"]==offline["identity_sha256"],"qualified_identity_changed"
 Config(model="deepseek-flash",service_provider="deepseek",reasoning_effort="high",
  max_tokens=384000,max_calls=4,timeout_seconds=1200,provider_retries=3,stream=True)
 proposal=dict(format="poseidon-mha-tensor-axis-live-proposal-v1",status="awaiting_user_approval",
  purpose="Qualify opt-in v12 grouping guidance and host transport receipt on the two unresolved unchanged MHA tasks.",
  old_quota_closed=True,prior_proposal_sha256=sha(ROOT/"docs/baseline/agent-mha-feedback-live-proposal-r1330.json"),
  prior_live_report_sha256=sha(ROOT/"docs/baseline/agent-mha-feedback-live-results-r1331.json"),
  offline_report=str(ROOT/"docs/baseline/agent-tensor-axis-guidance-r1332.json"),offline_report_sha256=sha(ROOT/"docs/baseline/agent-tensor-axis-guidance-r1332.json"),
  provider=old["provider"],bounds=old["bounds"],cases=cases,source_stage_identity=identity,
  source_capture_implementation=old["source_capture_implementation"],source_implementation_files=old["source_implementation_files"],
  outbound=old["outbound"],numeric_tolerance=old["numeric_tolerance"],monetary_cap=None,
  retries_may_duplicate_billing=True,paid_calls_executed=0,backend_changed=False,
  security_parameters_changed=False,installs=0,downloads=0,
  excluded=["third already qualified MHA","manual candidate answers","private inputs/reference",
   "bootstrap/CUDA","compiler/runtime/library modifications"],
  review_note="A new approval is required for this exact v12 identity; r1330 is closed. No runner starts from this freeze tool.")
 for name,want in proposal["source_capture_implementation"].items():assert sha(BASE/name)==want
 for name,want in proposal["source_implementation_files"].items():assert sha(ROOT/name)==want
 proposal["binding"]=digest(proposal);write(doc,proposal)
 print(json.dumps(dict(proposal=str(doc),binding=proposal["binding"],cases=len(cases),
  public_context_bytes=[c["public_context_bytes"]for c in cases],max_generations=8,
  max_http_attempts=32,paid_calls=0)))
 return 0
if __name__=="__main__":raise SystemExit(main())
