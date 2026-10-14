"""Prepare bounded application requests from independent source captures."""
import sys,json,shlex,copy,math
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5]);BASE=ROOT/"scripts/baseline"
sys.path[:0]=[str(BASE/"benchmarks/tools/static_broadcast_live_v1"),str(BASE),str(HERE)]
import run as live
import qualification_scope as q
if "--inside" not in sys.argv:
 live.no_active()
 from hecate_python_env import VENV,enter_nix
 command='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"])
 raise SystemExit(enter_nix(command,seconds=600,environment="poseidon-cpu"))
from application_backend import PoseidonBackend
from application_planning.layout import plan_model,check_plan
from component_contract import prepare_task,reconstruct_request
from candidate_bundle import load_model_provenance
from pytorch_model_intake import preparation_options
from operator_arithmetic_graph import preparation_options as graph_options
from deepseek_provider import public_request
import numpy as np
identity=PoseidonBackend().identity_manifest()
parent_identity=q.read(BASE/"benchmarks/tools/generic_capacity_intake_v1/identity-snapshot.json")
assert identity["backend"]==parent_identity["backend"]
if (HERE/"identity-snapshot.json").exists():assert q.read(HERE/"identity-snapshot.json")==identity
else:q.write(HERE/"identity-snapshot.json",identity)
original=q.read(BASE/"benchmarks/tools/generic_capacity_capture_v3/capture-plan.json");q.bound(original)
continuation=original
first=q.read(Path(original["campaign"])/"report.json");q.bound(first)
second=q.read(Path(continuation["campaign"])/"report.json");q.bound(second)
assert (first["passed"],first["failed"],first["not_run"])==(13,0,0)
assert second==first
def append_record(path,value):
 if Path(path).exists():assert q.canonical(q.read(path))==q.canonical(value),"existing_partial_record_changed"
 else:q.write(path,value)
rows=[];frozen={};all_cases=q.read(HERE/"cases.json")
for case in all_cases:
 name=case["name"]
 parent=continuation if name in ("wide_tail_linear","weights_linear_10k") else original
 report=second if parent is continuation else first
 status=next(r for r in report["cases"] if r["case"]==name)
 assert status["status"]=="passed" and status["groups"]==16 and status["source_format"]=="poseidon-trusted-pytorch-v34"
 folder=Path(parent["campaign"])/name;model=q.read(folder/"model.json")
 provenance=load_model_provenance(folder,model,max_bytes=4*1024**2)
 base_options=dict(case["options"],public_weight_policy="descriptors-compact-origins-v3")
 if name=="weights_linear_10k":base_options["generation_guidance"]="explicit-v14"
 options=preparation_options(provenance,model,base_options)
 options=graph_options(provenance["original"],model,options)
 if name=="weights_linear_10k":
  old=q.read(Path(case["prior_mathematical_overlap"]["source"]))
  old_weights=[np.asarray(v,dtype=np.float64) for v in old["constants"].values()]
  new_weights=[np.asarray(v,dtype=np.float64) for v in model["constants"].values()]
  for a in old_weights:
   assert any(a.shape==b.shape and np.array_equal(a,b) for b in new_weights)
  assert options["chunk_period"]==128
 decision=plan_model(model,options=options)
 target=HERE/"prepared-requests"/name;target.mkdir(parents=True,exist_ok=True)
 append_record(target/"plan.json",decision)
 if decision["status"]!="prepared":
  print(json.dumps(dict(case=name,status=decision["status"],failure=decision["failure"],candidate_count=len(decision["candidates"]))),flush=True)
  raise ValueError(name+":"+decision["failure"]["code"])
 request=check_plan(decision,model)
 assert request==prepare_task(model,decision["selection"]["options"]) and reconstruct_request(request)==request
 public=public_request(request)
 append_record(target/"request.json",request);append_record(target/"public-request.json",public)
 rr=provenance["reference_report"]["finite_reference"]
 row=dict(id=name,source_manifest=case["manifest"],source_manifest_sha256=q.sha(case["manifest"]),
  source_format=case["source_format"],kind=case["source_kind"],purpose=case["purpose"],
  model_import=str(folder),model_sha256=q.digest(model),provenance_digest=q.digest(provenance),
  layout_plan=str(target/"plan.json"),layout_plan_sha256=q.sha(target/"plan.json"),
  request_file=str(target/"request.json"),request_file_sha256=q.sha(target/"request.json"),request_id=request["request_id"],
  public_request_file=str(target/"public-request.json"),public_request_file_sha256=q.sha(target/"public-request.json"),
  selected_period=request["layout"]["input_slot_period"],options=decision["selection"]["options"],
  plaintext_groups=16,plaintext_values=rr["compared_values"],reference_parent_binding=report["binding"],
  input_ranks=case["input_ranks"],total_input_elements=case["total_input_elements"],
  logical_output_elements=request["layout"]["output_shape"][0],
  physical_inputs=len(request["layout"]["inputs"]),physical_outputs=request["layout"]["output_ciphertexts"],
  nonrepeating_public_weights=case["nonrepeating_public_weights"],prior_mathematical_overlap=case["prior_mathematical_overlap"])
 rows.append(row)
 for rel,want in decision["source_authority"]["files"].items():
  path=BASE/rel;assert q.sha(path)==want;frozen[str(path.relative_to(ROOT))]=want
 for key in ("layout_plan","request_file","public_request_file"):frozen[str(Path(row[key]).relative_to(ROOT))]=q.sha(row[key])
 frozen.update(next(c for c in parent["cases"] if c["name"]==name)["files"])
 print(json.dumps(dict(case=name,period=row["selected_period"],input_elements=row["total_input_elements"],output_elements=row["logical_output_elements"],
  physical_inputs=row["physical_inputs"],physical_outputs=row["physical_outputs"],public_request_bytes=Path(row["public_request_file"]).stat().st_size)),flush=True)
result=dict(format="poseidon-generic-capacity-prepared-requests-v1",cases=rows,sdk=identity["manifest_sha256"],
 independent_reference_groups=sum(r["plaintext_groups"] for r in rows),independent_reference_values=sum(r["plaintext_values"] for r in rows),
 capture_bindings=[first["binding"]],paid_calls=0,compiler_calls=0,encrypted_calls=0,
 unique_new_model_contexts=12,historical_model_reused=1,frozen_files=frozen)
result["binding"]=q.digest(result);q.write(HERE/"prepared-requests.json",result)
print(json.dumps(dict(cases=len(rows),groups=result["independent_reference_groups"],values=result["independent_reference_values"],binding=result["binding"])))
