"""Existing frozen models: generic replicated-scalar rule expressions."""
import copy,hashlib,json,math,shutil,signal,sys,time,traceback
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parents[2]
MHA_STAGE=HERE.parent/"staged_mha_ordered_graph_v1"
sys.path.insert(0,str(BASE));sys.path.insert(0,str(MHA_STAGE/"overlay"));sys.path.insert(0,str(HERE))
import numpy as np
from benchmark_graph import canonical,digest,require
from component_contract import prepare_task
from application_backend import PoseidonBackend
from hecate_python_env import ROOT,WORK
from pytorch_model_intake import preparation_options
from pytorch_ordered_contract import source_identity
from candidate_bundle import load_model_provenance,check_bundle
from unified_graph_lowering import candidate_source
from numerical_results import compare
from replicated_scalar_lowering import lower
sys.path.insert(0,str(MHA_STAGE))
from analyze_schedule_compaction import interpret
from benchmark_math import evaluate
NAMES=["self_post_linear_residual__declared_shallow_v1","linear_into_separate_cross_none__declared_shallow_v1","boolean_padding_residual__declared_shallow_v1"]
LIMITS=dict(wall_seconds=1200,case_seconds=300,new_bytes=4*1024**3,
 all_result_bytes=32*1024**3,minimum_free_bytes=4*1024**3,
 native_cpu_seconds=150,native_cpu_hard_seconds=155,native_address_bytes=4*1024**3)
def inventory(root):return sum(p.stat().st_size for p in root.rglob("*") if p.is_file() and not p.is_symlink())
def main():
 out=Path(sys.argv[1]);capture=Path(sys.argv[2])
 require(not out.exists(),"Preserve prior backend evidence")
 backend=PoseidonBackend();start=backend.identity();stages=backend.identity_manifest()
 source=source_identity()
 helper_path=HERE/"replicated_scalar_lowering.py";helper_sha256=hashlib.sha256(helper_path.read_bytes()).hexdigest()
 require(shutil.disk_usage(WORK).free>=LIMITS["minimum_free_bytes"],"Free disk boundary")
 require(inventory(WORK/"results")<=LIMITS["all_result_bytes"],"Global artifact boundary")
 known_results={p.name for p in (WORK/"results").iterdir()}
 out.mkdir();began=time.monotonic()
 report=dict(format="poseidon-replicated-scalar-rule-backend-v1",status="running",limits=LIMITS,
 backend_identity=start,stage_identity=stages,source_implementation=source,
 tool_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
 capture_report_sha256=hashlib.sha256((capture/"report.json").read_bytes()).hexdigest(),
 source_capture=str(capture),cases=[dict(id=n,status="not_run") for n in NAMES],
 api_calls=0,http_attempts=0,backend_changed=False,reference_changed=False,
 source_kind="deterministic_rule_baseline",production_applied=False,
 layout_strategy="replicated_logical_scalars_existing_period4_abi",helper_sha256=helper_sha256,historical_backend_report=str(WORK/"results/agent-mha-ordered-backend-v1/report.json"),
 numerical_tolerance=dict(atol=1e-5,rtol=1e-4),original_accuracy_certified=False)
 def save():
  report["binding"]=digest({k:v for k,v in report.items() if k!="binding"})
  (out/"report.json").write_bytes(canonical(report))
 def boundary():
  require(time.monotonic()-began<LIMITS["wall_seconds"],"Backend batch time boundary")
  require(hashlib.sha256(Path(__file__).read_bytes()).hexdigest()==report["tool_sha256"] and
   hashlib.sha256((capture/"report.json").read_bytes()).hexdigest()==report["capture_report_sha256"],"Tool or source capture identity drift")
  require(backend.identity()==start and backend.identity_manifest()==stages and source_identity()==source and hashlib.sha256(helper_path.read_bytes()).hexdigest()==helper_sha256,"Source identity drift")
  require(sum(inventory(p) if p.is_dir() else p.stat().st_size for p in (WORK/"results").iterdir() if p.name not in known_results)<=LIMITS["new_bytes"],"New attributable artifact boundary")
  require(inventory(WORK/"results")<=LIMITS["all_result_bytes"],"Global artifact boundary")
  require(shutil.disk_usage(WORK).free>=LIMITS["minimum_free_bytes"],"Free disk boundary")
 def alarm(*_):raise TimeoutError("Case wall boundary")
 previous=signal.signal(signal.SIGALRM,alarm);save()
 try:
  for row in report["cases"]:
   boundary();signal.setitimer(signal.ITIMER_REAL,LIMITS["case_seconds"])
   try:
    name=row["id"];case=out/name;case.mkdir()
    model=json.loads((capture/name/"model.json").read_bytes())
    provenance=load_model_provenance(capture/name,model,max_bytes=4*1024**2)
    options=preparation_options(provenance,model,dict(generation_guidance="explicit-v11",public_weight_policy="descriptors-math-roles-v2",chunk_period=4))
    request=prepare_task(model,options)
    old=json.loads((WORK/"results/agent-mha-ordered-backend-v1"/name/"request.json").read_bytes())
    require(request["model"]==old["model"] and request["compiler_configuration"]==old["compiler_configuration"],"Model or compiler parameters changed")
    require(request["layout"]["input_slot_period"]==4,"Existing period-4 layout")
    require(request["request_id"]==old["request_id"],"Original generation request changed")
    text,lowering=lower(request,source_kind="deterministic_rule_baseline")
    from unified_graph_contract import validate_candidate
    validate_candidate(dict(schema=1,request_id=request["request_id"],hecate_source=text),request)
    from benchmark_graph import samples
    control_values=0;control_maximum=0.
    for sample in samples(request["model"],4):
     values={k:np.asarray(v,dtype=np.float64) for k,v in request["public_constants"].items()}
     for item in request["layout"]["inputs"]:
      value=np.asarray(sample[item["name"]]).reshape(-1)[item["offset"]:item["offset"]+item["elements"]]
      values[item["dsl_name"]]=np.pad(value,(0,request["layout"]["input_slot_period"]-item["elements"]))
     values["zero_ct"]=np.zeros(request["layout"]["input_slot_period"],dtype=np.float64)
     physical=interpret(text,values)
     actual=np.array([physical[cipher][slot] for cipher,slot in request["layout"]["output_selectors"]])
     answer=evaluate(request["model"],sample)
     expected=np.concatenate([answer[item["name"]].reshape(-1) for item in request["model"]["outputs"]])
     delta=np.abs(actual-expected);control_maximum=max(control_maximum,float(delta.max()))
     require(np.all(delta<=1e-12+1e-12*np.abs(expected)),"Independent plaintext DSL/model control")
     control_values+=actual.size
    row["plaintext_program_control"]=dict(status="passed",groups=4,compared_values=control_values,
      maximum_absolute_error=control_maximum,
      scope="Independent control of exact public mathematical model, not FHE")
    candidate=dict(schema=1,request_id=request["request_id"],hecate_source=text)
    from candidate_contract import request_file_bytes
    (case/"request.json").write_bytes(request_file_bytes(request));(case/"candidate.py").write_text(text)
    row.update(status="running",request_id=request["request_id"],candidate_sha256=hashlib.sha256(text.encode()).hexdigest(),
     original_accuracy_certified=False,lowering=lowering);save()
    result=backend.qualify(request,candidate,"numerical")
    (case/"qualification.json").write_bytes(canonical(result))
    row.update(qualification_status=result["status"],evidence=result["evidence"],observed_stages=result["observed_stages"])
    require(result["status"]=="passed","Qualification failed:"+str(result.get("failure")))
    boundary()
    package=case/"program";manifest=backend.export(result,package,"numerical",model_provenance=provenance)
    require(check_bundle(package)==manifest,"Package reload mismatch")
    samples=[]
    for group in range(3):
     sample={}
     for item in request["model"]["inputs"]:
      n=math.prod(item["shape"]);values=(np.arange(n,dtype=np.float64)+1)/(n+1)*.35
      if group==0:values=-values
      if group==2:values=np.cos(np.arange(n,dtype=np.float64))*.27
      sample[item["name"]]=values.reshape(item["shape"]).tolist()
     samples.append(sample)
    execution=backend.execute(package,samples)
    actual=np.array([[v for output in request["model"]["outputs"] for v in np.asarray(sample[output["name"]]).reshape(-1)] for sample in execution["outputs"]],dtype=np.float64)
    expected=np.array([[v for output in request["model"]["outputs"] for v in evaluate(request["model"],{k:np.asarray(v,dtype=np.float64) for k,v in sample.items()})[output["name"]].reshape(-1)] for sample in samples],dtype=np.float64)
    comparison=compare(actual,expected,1e-5,1e-4)
    require(comparison["passed"],"New input frozen tolerance")
    if "cross_none" in name:
     require(all(value["weights"] is None for value in execution["structured_outputs"]),"Static None output restoration")
    row.update(status="passed",package_binding=manifest["binding"],
     qualification_comparison={k:result[k] for k in ["max_absolute_error","compared_values"]},
     new_input_comparison={k:comparison[k] for k in ["passed","mae","max_absolute_error","compared_values"]},
     execution_evidence=execution["evidence"],resource_precheck=execution["resource_precheck"]["binding"],
     compiled=True,encrypted_execution=True,reloaded=True,repeated_execution=True,new_agent_generation=False)
    print(name,"passed",flush=True)
   except Exception as error:
    row.update(status="failed",diagnostic=str(error)[:1500],traceback=traceback.format_exc())
    print(row["id"],"failed",type(error).__name__,str(error)[:900],flush=True)
    if "integrity" in str(error).lower() or "identity" in str(error).lower():
     row["environment_stop"]=True;save();break
   finally:signal.setitimer(signal.ITIMER_REAL,0)
   boundary();save()
  report.update(status="complete",elapsed_seconds=time.monotonic()-began,
   passed=sum(r["status"]=="passed" for r in report["cases"]),failed=sum(r["status"]=="failed" for r in report["cases"]),
   not_run=sum(r["status"]=="not_run" for r in report["cases"]),skipped=0)
  boundary();save()
 finally:
  signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,previous)
 return 0 if report.get("passed")==len(NAMES) else 1
if __name__=="__main__":raise SystemExit(main())
