"""Bounded rule-baseline qualification under the explicit mathematical-role policy. No API."""
import copy,hashlib,json,math,signal,time
from pathlib import Path
import sys
sys.path.insert(0,str((Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5])/"scripts/baseline"))
import numpy as np
from benchmark_graph import canonical,digest
from component_contract import prepare_task,reconstruct_request
from application_backend import PoseidonBackend
from hecate_python_env import ROOT,WORK
from public_weight_context import read_request
from pytorch_model_intake import preparation_options
from candidate_bundle import load_model_provenance,check_bundle
from unified_graph_lowering import candidate_source
from numerical_results import compare
from benchmark_math import evaluate

BASE=ROOT/"scripts/baseline"
OUT=WORK/"results/agent-public-math-roles-offline-v3"
NAMES=["tanh_functional"]
LIMITS=dict(wall_seconds=600,case_seconds=300,new_bytes=4*1024**3,
            all_result_bytes=32*1024**3,minimum_free_bytes=4*1024**3,
            native_cpu_seconds=150,native_address_bytes=4*1024**3)
def inventory(root):
    return sum(p.stat().st_size for p in root.rglob("*") if p.is_file() and not p.is_symlink())
def save(value):
    raw=copy.deepcopy(value);raw["binding"]=digest(raw)
    tmp=OUT/"report.partial";tmp.write_bytes(canonical(raw));tmp.replace(OUT/"report.json")
def require(ok,message):
    if not ok:raise ValueError(message)
def main():
    import shutil
    require(not OUT.exists(),"preserve prior offline execution")
    backend=PoseidonBackend();start=backend.identity();stages=backend.identity_manifest()
    require(shutil.disk_usage(WORK).free>=LIMITS["minimum_free_bytes"],"free disk boundary")
    OUT.mkdir()
    began=time.monotonic()
    report=dict(format="poseidon-public-math-roles-offline-v1",status="running",limits=LIMITS,
        backend_identity=start,stage_identity=stages,tool_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        mathematical_context_policy="descriptors-math-roles-v2",historical_live_binding="612aa37b939ec5f3b628f9334ce3147b3c781e6ba0db0e7c0295623c07476877",
        cases=[dict(id=n,status="not_run") for n in NAMES],api_calls=0,http_attempts=0,
        backend_changed=False,reference_changed=False,numerical_tolerance=dict(atol=1e-5,rtol=1e-4))
    save(report)
    def boundary():
        require(time.monotonic()-began<LIMITS["wall_seconds"],"offline wall boundary")
        require(backend.identity()==start and backend.identity_manifest()==stages,"offline source identity drift")
        require(inventory(OUT)<=LIMITS["new_bytes"],"new artifact boundary")
        require(inventory(WORK/"results")<=LIMITS["all_result_bytes"],"global artifact boundary")
        require(shutil.disk_usage(WORK).free>=LIMITS["minimum_free_bytes"],"free disk boundary")
    def alarm(*args):raise TimeoutError("offline case time boundary")
    previous=signal.signal(signal.SIGALRM,alarm)
    try:
        for name,row in zip(NAMES,report["cases"]):
            boundary();signal.setitimer(signal.ITIMER_REAL,LIMITS["case_seconds"])
            try:
                case=OUT/name;case.mkdir();provenance=None
                if name=="historical_linear":
                    old=next((WORK/"results/agent-v12-source-live-rebound-v2/linear").glob("tasks/*/program"))
                    request=reconstruct_request(read_request((old/"request.json").read_text()))
                    source=(old/"candidate.py").read_text()
                    row.update(source_kind="historical_real_agent_unchanged",
                        historical_source_path=str(old/"candidate.py"),
                        historical_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                        historical_package_manifest_sha256=hashlib.sha256((old/"manifest.json").read_bytes()).hexdigest())
                else:
                    capture=WORK/"results/agent-public-math-roles-capture-v2"/name
                    model=json.loads((capture/"model.json").read_bytes())
                    provenance=load_model_provenance(capture,model,max_bytes=4*1024**2)
                    options=preparation_options(provenance,model,dict(generation_guidance="explicit-v11",public_weight_policy="descriptors-math-roles-v2"))
                    request=prepare_task(model,options)
                    source,lowering=candidate_source(request)
                    row.update(source_kind="deterministic_rule_baseline",
                        source_capture=str(capture),lowering=lowering,
                        source_variant="tanh",original_accuracy_certified=False)
                candidate=dict(schema=1,request_id=request["request_id"],hecate_source=source)
                from candidate_contract import request_file_bytes
                (case/"request.json").write_bytes(request_file_bytes(request))
                (case/"candidate.py").write_text(source)
                row.update(status="running",request_id=request["request_id"],candidate_sha256=hashlib.sha256(source.encode()).hexdigest())
                save(report)
                result=backend.qualify(request,candidate,"numerical")
                (case/"qualification.json").write_bytes(canonical(result))
                row.update(qualification_status=result["status"],evidence=result["evidence"],observed_stages=result["observed_stages"])
                require(result["status"]=="passed","qualification failed:"+str(result.get("failure")))
                package=case/"program"
                manifest=backend.export(result,package,"numerical",model_provenance=provenance)
                require(check_bundle(package)==manifest,"package reload mismatch")
                samples=[]
                for group in range(3):
                    sample={}
                    for item in request["model"]["inputs"]:
                        n=math.prod(item["shape"])
                        values=(np.arange(n,dtype=np.float64)+1)/(n+1)*.35
                        if group==0:values=-values
                        if group==2:values=np.cos(np.arange(n,dtype=np.float64))*.27
                        sample[item["name"]]=values.reshape(item["shape"]).tolist()
                    samples.append(sample)
                execution=backend.execute(package,samples)
                actual=np.array([[v for output in request["model"]["outputs"]
                    for v in np.asarray(sample[output["name"]]).reshape(-1)]
                    for sample in execution["outputs"]],dtype=np.float64)
                expected=np.array([[v for output in request["model"]["outputs"]
                    for v in evaluate(request["model"],{k:np.asarray(v,dtype=np.float64) for k,v in sample.items()})[output["name"]].reshape(-1)]
                    for sample in samples],dtype=np.float64)
                comparison=compare(actual,expected,1e-5,1e-4)
                require(comparison["passed"],"new-input frozen tolerance")
                if name=="tanh_input_keyword":
                    require(execution.get("output_structure",{}).get("kind")=="tuple",
                            "ordered tuple output provenance missing")
                    require(all(type(v) is tuple for v in execution["structured_outputs"]),
                            "tuple output restoration changed")
                row.update(status="passed",package_binding=manifest["binding"],
                    qualification_comparison={k:result[k] for k in ["max_absolute_error","compared_values"]},
                    new_input_comparison={k:comparison[k] for k in ["passed","mae","max_absolute_error","compared_values"]},
                    execution_evidence=execution["evidence"],
                    resource_precheck=execution["resource_precheck"]["binding"],
                    retained_payload_lower_bytes=execution["resource_precheck"]["facts"]["known_retained_payload_lower_bytes"],
                    compiled=True,encrypted_execution=True,reloaded=True,repeated_execution=True,new_agent_generation=False)
                print(name,"passed",row["new_input_comparison"],flush=True)
            except Exception as error:
                row.update(status="failed",diagnostic=str(error)[:1200])
                print(name,"failed",type(error).__name__,str(error)[:800],flush=True)
                # Independent candidate/compiler failures do not erase remaining cases.
                if "integrity" in str(error).lower() or "identity" in str(error).lower():
                    row["environment_stop"]=True;save(report);break
            finally:
                signal.setitimer(signal.ITIMER_REAL,0)
            boundary();save(report)
        report.update(status="complete",elapsed_seconds=time.monotonic()-began,
            passed=sum(r["status"]=="passed" for r in report["cases"]),
            failed=sum(r["status"]=="failed" for r in report["cases"]),
            not_run=sum(r["status"]=="not_run" for r in report["cases"]),skipped=0)
        boundary();save(report)
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,previous)
    return 0 if report.get("passed")==len(NAMES) else 1
if __name__=="__main__":raise SystemExit(main())
