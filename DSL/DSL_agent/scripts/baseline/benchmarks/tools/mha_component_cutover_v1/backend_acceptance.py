"""Formal MHA intake -> unchanged graph -> explicit layout -> Poseidon acceptance."""
import copy,hashlib,json,math,shutil,signal,sys,time,traceback
from pathlib import Path
HERE=Path(__file__).resolve().parent
BASE=HERE.parents[2]
sys.path.insert(0,str(BASE))
import numpy as np
from benchmark_graph import canonical,digest,require
from component_contract import prepare_task
from application_backend import PoseidonBackend
from hecate_python_env import ROOT,WORK
from pytorch_model_intake import run as capture_source,preparation_options
from pytorch_ordered_contract import source_identity
from candidate_bundle import load_model_provenance,check_bundle
from unified_graph_lowering import candidate_source
from unified_layout_strategies import MIXED
from program_execution import package_output_structure
from numerical_results import compare
from benchmark_math import evaluate
NAMES=("self_post_linear_residual__declared_shallow_v1",
       "linear_into_separate_cross_none__declared_shallow_v1",
       "boolean_padding_residual__declared_shallow_v1")
LIMITS=dict(wall_seconds=1200,case_seconds=300,new_bytes=4*1024**3,
            all_result_bytes=32*1024**3,minimum_free_bytes=4*1024**3,
            native_cpu_seconds=150,native_cpu_hard_seconds=155,native_address_bytes=4*1024**3,
            concurrency=1,compiler_jobs=2,link_jobs=1)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def total(root):
    return sum(p.stat().st_size for p in root.rglob("*") if p.is_file() and not p.is_symlink())
def declared(tree,named):
    if tree["kind"]=="none":return None
    if tree["kind"]=="tensor":
        value=named[tree["name"]]
        return np.asarray(value).reshape(()).tolist() if "shape" in tree else value
    if tree["kind"]=="dict":
        return {v["key"]:declared(v["value"],named) for v in tree["items"]}
    require(tree["kind"] in ("tuple","list"),"Declared output kind")
    children=[declared(v,named) for v in tree["items"]]
    return tuple(children) if tree["kind"]=="tuple" else children
def fresh_inputs(model):
    result=[]
    for group in range(3):
        row={}
        for item in model["inputs"]:
            size=math.prod(item["shape"]);value=(np.arange(size,dtype=np.float64)+1)/(size+1)*.35
            if group==0:value=-value
            if group==2:value=np.cos(np.arange(size,dtype=np.float64))*.27
            row[item["name"]]=value.reshape(item["shape"]).tolist()
        result.append(row)
    return result
def compare_call(execution,model,inputs,tree):
    require(execution["status"]=="executed" and execution["encrypted_execution"] and
            execution["executor_secret_key"] is False,"Trusted encrypted execution boundary")
    require(execution["samples"]==len(inputs) and execution["output_structure"]==tree,
            "Actual input/output declarations")
    require(canonical([declared(tree,row) for row in execution["outputs"]])==
            canonical(execution["structured_outputs"]),"Full declared output structure including None")
    observed=np.array([[v for o in model["outputs"] for v in np.asarray(row[o["name"]]).reshape(-1)]
                       for row in execution["outputs"]])
    expected=np.array([[v for o in model["outputs"] for v in
        evaluate(model,{k:np.asarray(v,dtype=np.float64) for k,v in row.items()})[o["name"]].reshape(-1)]
                       for row in inputs])
    r=compare(observed,expected,1e-5,1e-4);require(r["passed"],"Frozen numerical tolerance")
    return {k:r[k] for k in ("passed","mae","max_absolute_error","compared_values")}
def main():
    out=WORK/"results/agent-mha-layout-component-backend-v2"
    require(not out.exists(),"Fresh evidence directory")
    control=WORK/"results/agent-mha-layout-component-controls-v1/report.json"
    controls=json.loads(control.read_bytes())
    require(controls["status"]=="passed" and controls["failed"]==controls["errors"]==controls["skipped"]==0,
            "Component controls required")
    frozen=dict(controls["source_files"])
    frozen.update({str(p.relative_to(ROOT)):sha(p) for p in HERE.rglob("*")
                   if p.is_file() and "__pycache__" not in p.parts})
    fixture=BASE/"benchmarks/tools/staged_mha_ordered_graph_v1/reference_tests/source_fixtures"
    fixture_files={str(p):sha(p) for n in NAMES for p in (fixture/n).rglob("*")
                   if p.is_file() and not p.is_symlink()}
    first=WORK/"results/agent-mha-layout-component-backend-v1"
    fixture_files.update({str(p):sha(p) for p in (first/"capture").rglob("*") if p.is_file()})
    fixture_files[str(first/"report.json")]=sha(first/"report.json")
    previous=WORK/"results/agent-mha-mixed-packed-scalar-backend-v2"
    historic=json.loads((previous/"report.json").read_bytes())
    fixed={r["id"]:r["candidate_sha256"] for r in historic["cases"]}
    old_requests=WORK/"results/agent-mha-ordered-backend-v1"
    for n in NAMES:
        for p in (previous/n/"candidate.py",old_requests/n/"request.json"):
            fixture_files[str(p)]=sha(p)
    require(all(sha(ROOT/n)==s for n,s in frozen.items()),"Frozen component source")
    backend=PoseidonBackend();identity=backend.identity();stages=backend.identity_manifest()
    capture_hashes=source_identity()
    known={p.name for p in (WORK/"results").iterdir()}
    out.mkdir();(out/"capture").mkdir()
    prior_elapsed=json.loads((WORK/"results/agent-mha-layout-component-backend-v1/report.json").read_bytes())["elapsed_seconds"]
    start=time.monotonic();capture_files={}
    report=dict(format="poseidon-mha-layout-component-backend-v1",status="running",limits=LIMITS,
                backend_identity=identity,stage_identity=stages,source_implementation=capture_hashes,
                tool_files=frozen,fixture_files=fixture_files,control_report=dict(path=str(control),sha256=sha(control),binding=controls["binding"]),
                cases=[dict(id=n,status="not_run") for n in NAMES],api_calls=0,http_attempts=0,
                production_mha_active=True,rule_strategy=MIXED,prior_elapsed_seconds=prior_elapsed,production_default_rule_unchanged=True,
                source_kind="deterministic_rule_baseline",new_agent_generation=False,
                backend_changed=False,parameters_changed=False,reference_changed=False,
                original_accuracy_certified=False,numerical_tolerance=dict(atol=1e-5,rtol=1e-4))
    def save():
        report["binding"]=digest({k:v for k,v in report.items() if k!="binding"})
        (out/"report.json").write_bytes(canonical(report))
    def boundary():
        require(time.monotonic()-start+controls["elapsed_seconds"]+
                prior_elapsed<LIMITS["wall_seconds"],"Cumulative batch wall boundary")
        require(all(sha(ROOT/n)==s for n,s in frozen.items()),"Frozen component source drift")
        require(all(sha(p)==s for p,s in {**fixture_files,**capture_files}.items()),"Source/evidence binding drift")
        require(backend.identity()==identity and backend.identity_manifest()==stages and source_identity()==capture_hashes,
                "Backend and capture identity drift")
        require(sum(total(p) if p.is_dir() else p.stat().st_size for p in (WORK/"results").iterdir()
                    if p.name not in known)<=LIMITS["new_bytes"],"Attributable artifact boundary")
        require(total(WORK/"results")<=LIMITS["all_result_bytes"] and
                shutil.disk_usage(WORK).free>=LIMITS["minimum_free_bytes"],"Global artifact or free disk boundary")
    def alarm(*_):raise TimeoutError("Case wall boundary")
    old_alarm=signal.signal(signal.SIGALRM,alarm);save()
    try:
        for row in report["cases"]:
            boundary();signal.setitimer(signal.ITIMER_REAL,LIMITS["case_seconds"])
            try:
                name=row["id"];case=out/name;case.mkdir();captured=out/"capture"/name
                captured=WORK/"results/agent-mha-layout-component-backend-v1/capture"/name
                prior_capture=json.loads((WORK/"results/agent-mha-layout-component-backend-v1/report.json").read_bytes())
                prior_row=next(r for r in prior_capture["cases"] if r["id"]==name)
                require(prior_row["capture"]["status"]=="captured","Formal facade capture passed")
                metadata=json.loads((captured/"reference-report.json").read_bytes())
                require(metadata["source_hashes"]==capture_hashes,"Current exact source capture identity")
                row["capture"]=dict(prior_row["capture"],reused_current_source_capture=True)
                model=json.loads((captured/"model.json").read_bytes())
                provenance=load_model_provenance(captured,model,max_bytes=4*1024**2)
                options=preparation_options(provenance,model,dict(generation_guidance="explicit-v11",
                            public_weight_policy="descriptors-math-roles-v2",chunk_period=4))
                request=prepare_task(model,options)
                old=json.loads((old_requests/name/"request.json").read_bytes())
                from candidate_contract import request_file_bytes
                require(canonical(json.loads(request_file_bytes(request)))==canonical(old),
                        "Original serialized model/weights/layout/request changed")
                source,lowering=candidate_source(request,strategy=MIXED)
                require(hashlib.sha256(source.encode()).hexdigest()==fixed[name] and
                        (previous/name/"candidate.py").read_text()==source,"Fixed rule DSL changed during extraction")
                capture_files.update({str(p):sha(p) for p in captured.rglob("*") if p.is_file()})
                from candidate_contract import request_file_bytes
                (case/"request.json").write_bytes(request_file_bytes(request))
                (case/"candidate.py").write_text(source)
                row.update(status="running",request_id=request["request_id"],candidate_sha256=fixed[name],lowering=lowering,
                           capture_reference_sha256=sha(captured/"reference-report.json"),
                           source_capture_evidence=str(captured),historical_candidate_unchanged=True);save()
                result=backend.qualify(request,dict(schema=1,request_id=request["request_id"],hecate_source=source),"numerical")
                (case/"qualification.json").write_bytes(canonical(result))
                row.update(qualification_status=result["status"],evidence=result["evidence"],
                           observed_stages=result["observed_stages"])
                require(result["status"]=="passed","Qualification failed:"+str(result.get("failure")))
                boundary()
                package=case/"program"
                manifest=backend.export(result,package,"numerical",model_provenance=provenance)
                require(check_bundle(package)==manifest,"Formal package reload")
                tree=package_output_structure(package,manifest,request["model"])
                inputs=fresh_inputs(model)
                call=backend.execute(package,inputs);comparison=compare_call(call,model,inputs,tree)
                (case/"new-input-execution.json").write_bytes(canonical(call))
                repeat=backend.execute(package,inputs[:1]);repeated=compare_call(repeat,model,inputs[:1],tree)
                (case/"repeat-execution.json").write_bytes(canonical(repeat))
                row.update(status="passed",compiled=True,encrypted_execution=True,reloaded=True,repeated_execution=True,
                           qualification_comparison={k:result[k] for k in ("max_absolute_error","compared_values")},
                           new_input_comparison=comparison,repeat_comparison=repeated,
                           new_input_groups=3,repeat_groups=1,package_binding=manifest["binding"],
                           execution_evidence=call["evidence"],repeat_evidence=repeat["evidence"])
                print(name,"passed",flush=True)
            except Exception as error:
                row.update(status="failed",diagnostic=str(error)[:1500],traceback=traceback.format_exc())
                print(row["id"],"failed",type(error).__name__,str(error)[:700],flush=True)
                if any(x in str(error).lower() for x in ("identity","integrity","drift","budget","boundary")):
                    row["batch_stop"]=True;save();break
            finally:signal.setitimer(signal.ITIMER_REAL,0)
            boundary();save()
        report.update(status="complete",elapsed_seconds=time.monotonic()-start,capture_files=capture_files,
                      passed=sum(r["status"]=="passed" for r in report["cases"]),
                      failed=sum(r["status"]=="failed" for r in report["cases"]),
                      not_run=sum(r["status"]=="not_run" for r in report["cases"]),skipped=0)
        boundary();save()
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old_alarm)
    print(json.dumps({k:report[k] for k in ("status","passed","failed","not_run","skipped","elapsed_seconds","binding","api_calls")}))
    return 0 if report["passed"]==len(NAMES) else 1
if __name__=="__main__":raise SystemExit(main())
