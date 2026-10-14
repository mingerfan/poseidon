"""Prepare current source-bound requests; no generation or encrypted execution."""
import pathlib,sys,json,shlex
HERE=pathlib.Path(__file__).resolve().parent;B=HERE.parents[2];ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5])
sys.path[:0]=[str(HERE),str(B/"benchmarks/tools/static_broadcast_live_v1"),str(B)]
import qualification_scope as q,run as live
from hecate_python_env import VENV,enter_nix
if sys.argv[1:]!=["--inside"]:
 live.no_active()
 cmd='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join([str(VENV/"bin/python"),"-B",str(pathlib.Path(__file__).resolve()),"--inside"])
 raise SystemExit(enter_nix(cmd,seconds=300,environment="poseidon-cpu"))
from application_backend import PoseidonBackend
from application_planning.layout import plan_model,check_plan
from candidate_bundle import load_model_provenance
from pytorch_model_intake import preparation_options
from operator_arithmetic_graph import preparation_options as graph_options
from component_contract import prepare_task,reconstruct_request
from deepseek_provider import public_request
identity=PoseidonBackend().identity_manifest();q.write(HERE/"identity-snapshot.json",identity)
parent=q.read(B/"benchmarks/tools/logical_view_source_v1/capture-scope.json");q.bound(parent)
report=q.read(pathlib.Path(parent["campaign"])/"report.json");q.bound(report)
q.require(report["passed"]==report["source_intake_accepted"]==3 and report["failed"]==0,"capture")
rows=[];frozen={}
for c in parent["cases"]:
 folder=pathlib.Path(parent["campaign"])/c["id"];model=q.read(folder/"model.json");original=q.read(folder/"original.json")
 provenance=load_model_provenance(folder,model,max_bytes=4*1024**2)
 options=graph_options(original,model,preparation_options(provenance,model,c["options"]))
 decision=plan_model(model,options=options);q.require(decision["status"]=="prepared","layout")
 request=check_plan(decision,model);q.require(prepare_task(model,decision["selection"]["options"])==reconstruct_request(request)==request,"request")
 dest=HERE/"prepared"/c["id"];dest.mkdir(parents=True)
 for name,value in (("plan",decision),("request",request),("public-request",public_request(request))):q.write(dest/(name+".json"),value)
 row=dict(id=c["id"],execution_source_kind="trusted_pytorch",source_manifest=c["manifest"],source_intake_failure=None,
  request_file=str(dest/"request.json"),request_file_sha256=q.sha(dest/"request.json"),request_id=request["request_id"],
  public_request_file=str(dest/"public-request.json"),public_request_file_sha256=q.sha(dest/"public-request.json"),
  layout_plan=str(dest/"plan.json"),layout_plan_sha256=q.sha(dest/"plan.json"),options=decision["selection"]["options"],
  source_format="poseidon-trusted-pytorch-v35",capture_parent_binding=report["binding"])
 rows.append(row)
 for key in ("request_file","public_request_file","layout_plan"):frozen[str(pathlib.Path(row[key]).relative_to(ROOT))]=q.sha(row[key])
 for name in ("original.json","model.json","decomposition.json","reference-report.json"):
  frozen[str(folder/name)]=q.sha(folder/name)
 for rel,want in decision["source_authority"]["files"].items():frozen[str((B/rel).relative_to(ROOT))]=want
 print(json.dumps(dict(case=c["id"],nodes=len(model["nodes"]),period=request["layout"]["input_slot_period"],
  public_request_bytes=pathlib.Path(row["public_request_file"]).stat().st_size,request_id=request["request_id"])),flush=True)
frozen.update(parent["frozen_files"])
for name in ("qualification.py","qualification_scope.py","prepare.py","identity-snapshot.json"):frozen[str((HERE/name).relative_to(ROOT))]=q.sha(HERE/name)
for name,want in identity["files"].items() if "files" in identity else ():
 pass
v=dict(format="poseidon-logical-view-prepared-requests-v1",cases=rows,sdk=identity["manifest_sha256"],
 capture_binding=report["binding"],frozen_files=frozen,provider_calls=0,compiler_calls=0,encrypted_calls=0)
v["binding"]=q.digest(v);q.write(HERE/"prepared-requests.json",v)
print("PREPARED",v["binding"])
