"""Approved mixed JSON/PyTorch matrix qualification; fixed rule sources and existing Poseidon backend."""
import copy,hashlib,json,shutil,subprocess,sys,time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path.cwd()/"scripts/baseline"))
from benchmark_graph import canonical,digest,require,samples
from benchmark_math import evaluate as mathematical
from benchmark_torch import evaluate as tensor_reference
from operator_graph import implementation_sources as matrix_identity
from operator_arithmetic_graph import evaluate_reference as high_reference, implementation_sources as arithmetic_identity, FORMAT as ARITHMETIC
from numerical_results import compare
from semantic_benchmark_execution import runtime_sources
from application_execution.context import WorkerContext
from application_execution.backend import application
from component_contract import prepare_task
from candidate_bundle import check_execution_dependencies,replay_bundle,load_model_provenance,check_provenance
from pytorch_model_intake import source_identity
from trusted_pytorch_import import source_identity as prior_source_identity
from unified_graph_lowering import candidate_source
root=Path.cwd();work=Path("/home/lhohy/poseidon-work/platforms/aarch64-linux/results")
pre=work/"agent-pytorch-operator-shape-integration-r985";out=work/"agent-operator-shape-backend-r992"
load=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
proposal_path=root/"docs/baseline/agent-operator-shape-offline-proposal-r991.json"
proposal=load(proposal_path);scope=load(pre/"approved-backend-scope-r992.json")
started=time.monotonic()
require(not out.exists(),"Fresh JSON/PyTorch backend directory")
out.mkdir(mode=0o700);(out/"imports").mkdir();(out/"prepared").mkdir()
initial_results=set(scope["initial_result_paths"])
context=WorkerContext.capture(out/"tasks");app=application(context);context.persist()
sources=runtime_sources();identity=app.backend.identity();capture_identity=source_identity()
require(sources==proposal["source_hashes"],"Approved source binding")
require(identity==proposal["backend_identity"] and capture_identity==proposal["capture_identity"] and
        matrix_identity()==proposal["matrix_identity"] and arithmetic_identity()==proposal["arithmetic_identity"] and prior_source_identity()==proposal["legacy_capture_identity"],"Approved backend/capture/matrix identity")
require(scope["proposal_binding"]==proposal["binding"] and scope["approval_answer"]=="approved",
        "Human approved current offline scope")
files=proposal["case_hashes"].copy()
files[str(proposal_path)]=sha(proposal_path)
protected={}
for name in ("protected-backend.json","protected-references.json","protected-history.json","protected-new-history.json","protected-current-history.json"):
    protected.update(load(pre/name))
tool=Path(__file__);tool_sha=sha(tool)
audit_tool=Path(__file__).with_name("audit.py")
plan=dict(format="poseidon-arithmetic-matrix-offline-plan-v1",cases=proposal["cases"],
    source_hashes=sources,case_hashes=files,capture_identity=capture_identity,matrix_identity=matrix_identity(),backend_identity=identity,
    tool_sha256=tool_sha,audit_tool_sha256=sha(audit_tool),proposal_binding=proposal["binding"],
    worker_binding=context.binding,arithmetic_identity=arithmetic_identity(),legacy_capture_identity=prior_source_identity(),shards=proposal["shards"],scope=scope,tolerance=dict(atol=1e-5,rtol=1e-4),
    candidate_origin="trusted_rule_baseline",paid_calls=0,http_attempts=0,new_agent_generation=False)
plan["binding"]=digest(plan);(out/"plan.json").write_bytes(canonical(plan))
rows=[];prepared={};new_files={}
def measured(path,allocated=False):
    command=["/usr/bin/du","-s","--block-size=1","--one-file-system",str(path)]
    if not allocated:command.insert(1,"--apparent-size")
    return int(subprocess.check_output(command,text=True,timeout=30).split()[0])
def new_bytes():
    return sum(max(measured(p),measured(p,True)) for p in work.iterdir() if p.name not in initial_results)+max(0,max(measured(pre),measured(pre,True))-scope["initial_pre_bytes"])
def remaining():return min(scope["deadline_wall"]-time.time(),scope["wall_seconds"]-(time.monotonic()-started))
def boundary():
    require(remaining()>8,"Offline time budget")
    require(runtime_sources()==sources and app.backend.identity()==identity and source_identity()==capture_identity and matrix_identity()==plan["matrix_identity"] and arithmetic_identity()==proposal["arithmetic_identity"] and prior_source_identity()==proposal["legacy_capture_identity"],"Source/backend drift")
    require(sha(tool)==tool_sha and sha(audit_tool)==plan["audit_tool_sha256"],"Frozen execution and audit tools")
    require(all(sha(p)==h for p,h in proposal["tool_hashes"].items()),"Frozen batch tools")
    require(all(sha(p)==h for p,h in files.items()),"Frozen approved inputs")
    require(all(sha(root/p)==h for p,h in protected.items()),"Protected backend/reference/history")
    require(all(sha(p)==h for p,h in new_files.items()),"Frozen prepared requests/sources")
    require(shutil.disk_usage(work).free>=scope["reserve_bytes"],"Disk reserve")
    require(new_bytes()<=scope["artifact_bytes"],"New artifact budget")
    require(max(measured(work),measured(work,True))<=scope["global_bytes"],"Global artifact budget")
def save(status):
    report=dict(format="poseidon-arithmetic-matrix-offline-report-v1",status=status,binding=plan["binding"],
        planned=len(proposal["cases"]),captured=sum(bool(r.get("captured")) for r in rows),
         input_checked=sum(bool(r.get("input_checked")) for r in rows),
        prepared=len(prepared),passed=sum(r["status"]=="passed" for r in rows),
        failed=sum(r["status"]=="failed" for r in rows),
        not_run=len(proposal["cases"])-sum(r["status"] in ("passed","failed") for r in rows),
        rows=rows,seconds=time.monotonic()-started,paid_calls=0,http_attempts=0,new_agent_generation=False)
    (out/"report.json").write_bytes(canonical(report))
class Source:
    kind="scripted_replay";agent_calls=0
    def __init__(self,name):self.name=name
    def generate(self,request,history):
        require(not history,"No paid generation or rule repair")
        item=prepared[self.name];require(request==item["request"],"Exact frozen mathematical request")
        return json.dumps(dict(schema=1,request_id=request["request_id"],hecate_source=item["source"]))
class NeverGenerate:
    kind="scripted_replay";agent_calls=0
    def generate(self,*args):raise AssertionError("Cached program expected")
def flatten(value,tree):
    if tree["kind"]=="tensor":
        if tree.get("shape")==[]:require(type(value) in (int,float),"Scalar output restoration")
        return [np.asarray(value).reshape(-1)]
    if tree["kind"]=="dict":
        require(type(value) is dict and list(value)==[i["key"] for i in tree["items"]],"Dict output ordering")
        return [v for item in tree["items"] for v in flatten(value[item["key"]],item["value"])]
    require(type(value)==(tuple if tree["kind"]=="tuple" else list) and len(value)==len(tree["items"]),"Output container")
    return [a for v,t in zip(value,tree["items"]) for a in flatten(v,t)]
save("preparing")
try:
    for case in proposal["cases"]:
        require(case["shard"] in (1,2,3),"Approved shard identity")
        boundary();case_started=time.monotonic();name=case["id"];manifest=Path(case["manifest"]);folder=Path(case["imported"])
        row=dict(id=name,shard=case["shard"],status="preparing",manifest=str(manifest),imported=str(folder),origin="original_failed_capacity_rule" if case["historical_status"]=="failed" else ("new_"+case["input_kind"] if case["historical_status"]=="new_input_not_run" else "compatibility_regression"),
            source_origin=case["source_origin"],input_kind=case["input_kind"],new_input_runs=[],input_checked=False,captured=False)
        rows.append(row);save("running")
        try:
            model=load(folder/"model.json");provenance=load_model_provenance(folder,model)
            check_provenance(provenance,model);reference=provenance["reference_report"]
            if case["input_kind"] in ("trusted_pytorch_v7","trusted_pytorch_v8","trusted_pytorch_v9","trusted_pytorch_v10"):
                version=case["input_kind"].removeprefix("trusted_pytorch_")
                require(reference["format"]=="poseidon-trusted-pytorch-reference-"+version and reference["probes"]==16 and
                    reference["original_module_checked"] and reference["pure_forward_checked"],"Real retained capture and original reference")
                require(reference["source_hashes"]==(prior_source_identity() if version=="v7" else source_identity()),
                        "Exact versioned capture identity")
            else:
                version=case["input_kind"].removeprefix("declared_json_")
                require(version in ("v2","v3") and provenance["original"]["format"]=="poseidon-operator-graph-"+version and
                    reference["mode"]=="offline_model_import" and reference["probes"]==16 and reference["reference_status"]=="passed" and
                    provenance["decomposition"]["implementation_sources"]==(matrix_identity() if version=="v2" else arithmetic_identity()),
                    "Independent versioned JSON import binding")
            row["input_checked"]=True
            row["captured"]=case["input_kind"] in ("trusted_pytorch_v7","trusted_pytorch_v8","trusted_pytorch_v9","trusted_pytorch_v10")
            original_manual=None
            groups=0
            for inputs in samples(model,16):
                a,b,h=mathematical(model,inputs),tensor_reference(model,inputs),high_reference(provenance["original"],inputs)
                for key in a:
                    np.testing.assert_allclose(a[key],b[key],atol=2e-12,rtol=2e-12)
                    np.testing.assert_allclose(a[key],h[key],atol=2e-12,rtol=2e-12)
                if original_manual is not None:
                    require(original_manual["inputs"]==model["inputs"],"Original split input signature")
                    manual=mathematical(original_manual,inputs)
                    require(list(manual)==list(a),"Original split named output binding")
                    for key in a:np.testing.assert_allclose(a[key],manual[key],atol=2e-12,rtol=2e-12)
                    groups+=1
            options=case["options"]
            request=load(case["request"]);source=Path(case["candidate"]).read_text()
            require(request==prepare_task(model,options) and candidate_source(request)[0]==source,
                    "Frozen current request and rule expression")
            package=out/"prepared"/name;package.mkdir()
            (package/"request.json").write_bytes(canonical(request));(package/"candidate.py").write_text(source)
            for p in folder.rglob("*"):
                if p.is_file():new_files[str(p)]=sha(p)
            for p in package.iterdir():new_files[str(p)]=sha(p)
            prepared[name]=dict(request=request,source=source,model=model,provenance=provenance,options=options)
            row.update(status="prepared",model_sha256=digest(model),original_sha256=digest(provenance["original"]),
                request_id=request["request_id"],candidate_sha256=hashlib.sha256(source.encode()).hexdigest(),
                input_reference_status="passed",original_module_checked=bool(reference.get("original_module_checked")),
                pure_forward_checked=bool(reference.get("pure_forward_checked")),reference_probes=16,
                original_manual_equivalence_groups=groups,options=options)
        except (ValueError,TypeError,KeyError,IndexError,AssertionError) as exc:
            row.update(status="failed",stage="capture_or_preparation",failure_layer="input_capture_or_rule_preparation",
                error_type=type(exc).__name__,error=str(exc)[:600])
            save("running");boundary()
            print(json.dumps(dict(id=name,status="failed",stage=row["stage"],error=row["error"])),flush=True)
            continue
        boundary()
        seconds=min(scope["candidate_seconds"]-(time.monotonic()-case_started),remaining()-20)
        require(seconds>=1,"Candidate remaining budget")
        task=app.submit_import(folder,options=options,limits=dict(generations=1,http_attempts=1,
            wall_seconds=int(seconds),api_timeout_seconds=1200,max_tokens=384000))
        row.update(task_id=task,status="running");save("running")
        outcome=app.run(task,Source(name));row["result"]=outcome;save("running");boundary()
        if outcome["status"]!="succeeded":
            row["status"]="failed";row["stage"]="qualification";save("running")
            print(json.dumps(dict(id=name,status="failed",failure=outcome["failure"])),flush=True)
            if outcome["status"] in ("service_error","timed_out","cancelled"):
                raise RuntimeError("Environment or budget failure")
            continue
        validation=load(out/"tasks"/task/"validation-0.json");evidence=Path(validation["evidence"]);observed=load(evidence/"report.json")
        require(observed["rule_baseline_status"]=="not_requested_by_component" and
            observed["qualification"]["generation_performed"] is False,"Direct component qualification boundary")
        program=out/"tasks"/task/"program";metadata,bound=check_execution_dependencies(program)
        require(bound==request and not metadata["agent_generated"] and metadata["model_provenance"]=="included","Program provenance and request")
        require(sha(program/"candidate.py")==row["candidate_sha256"],"Unchanged rule source")
        row.update(package=str(program),package_binding=metadata["binding"],status="passed",qualification_session=observed["qualification"])
        save("running")
        print(json.dumps(dict(id=name,status="passed",max_absolute_error=outcome["acceptance"]["maximum_absolute_error"])),flush=True)
    for row in rows:
        if row["status"]=="passed" and row["id"].startswith("scalar_domain_"):
            boundary();item=prepared[row["id"]];before_paths=set(work.iterdir())
            rejected=app.execute(row["task_id"],[{s["name"]:np.full(s["shape"],10.).tolist() for s in item["model"]["inputs"]}])
            require(rejected["status"]=="validation_failed" and not rejected["encrypted_execution"] and
                rejected["failure"]["code"]=="approximation_domain_rejected" and
                rejected["failure"]["location"]["id"]=="activation" and
                rejected["agent_calls"]==rejected["compiler_calls"]==0,"V2 declared domain rejected before execution")
            require(set(work.iterdir())==before_paths,"Rejected domain created no backend evidence")
            row["domain_rejection"]=rejected;save("running")
    (out/"prepared-bindings.json").write_bytes(canonical(dict(files=new_files,rows=[
        {k:r[k] for k in ("id","model_sha256","original_sha256","request_id","candidate_sha256")} for r in rows if r.get("request_id")])))
    for name in proposal["package_reuse_cases"]:
        row=next(r for r in rows if r["id"]==name)
        if row["status"]!="passed":
            row["reuse_status"]="not_run_original_failed";save("running");continue
        boundary();program=Path(row["package"]);before=set(work.iterdir())
        require(replay_bundle(program)==0,"Real bound program reload")
        reloads=[p for p in set(work.iterdir())-before if p.name.startswith("poseidon-package-replay-")]
        require(len(reloads)==1,"Unique reload result")
        row["reload"]=load(reloads[0]/"report.json")
        require(row["reload"]["encrypted_execution"] and row["reload"]["numerically_validated"],"Reload numerical acceptance")
        logical=[];expected=[];item=prepared[name]
        for probe in range(3):
            rng=np.random.default_rng(20261027+probe)
            values={s["name"]:rng.uniform(-.3,.3,s["shape"]) for s in item["model"]["inputs"]}
            a,b=mathematical(item["model"],values),tensor_reference(item["model"],values)
            for key in a:np.testing.assert_allclose(a[key],b[key],atol=2e-12,rtol=2e-12)
            logical.append({k:v.tolist() for k,v in values.items()})
            expected.append(np.concatenate([a[o["name"]].reshape(-1) for o in item["model"]["outputs"]]))
        (out/(name+"-new-inputs.json")).write_bytes(canonical(logical))
        tree=item["provenance"]["reference_report"].get("capture_metadata",{}).get("output_structure")
        for repetition in range(2):
            boundary();executed=app.execute(row["task_id"],logical)
            require(executed["status"]=="executed" and executed["encrypted_execution"] and not executed["executor_secret_key"],"New-input ciphertext execution")
            actual=np.asarray([np.concatenate([np.asarray(s[o["name"]]).reshape(-1) for o in item["model"]["outputs"]]) for s in executed["outputs"]])
            if tree is not None:
                restored=np.asarray([np.concatenate(flatten(s,tree)) for s in executed["structured_outputs"]])
                np.testing.assert_array_equal(restored,actual)
            else:
                require("structured_outputs" not in executed,"JSON named output contract")
                restored=actual
            comparison=compare(actual,np.asarray(expected),1e-5,1e-4);require(comparison["passed"],"New-input numerical gate")
            row["new_input_runs"].append(dict(repetition=repetition,evidence=executed["evidence"],comparison=comparison,
                structured_values=int(restored.size),agent_calls=executed["agent_calls"],compiler_calls=executed["compiler_calls"]))
            save("running")
        cached=app.run(app.submit_import(Path(row["imported"]),options=item["options"]),NeverGenerate())
        require(cached["status"]=="succeeded" and cached["package"]["reused"] and cached["generations"]==cached["http_attempts"]==0,"Verified program cache")
        row["cache_reuse"]=dict(status="succeeded",reused=True,generations=0,http_attempts=0);row["reuse_status"]="passed";save("running")
    boundary();save("complete")
    raise SystemExit(0 if all(r["status"]=="passed" for r in rows) else 1)
except BaseException as exc:
    if not isinstance(exc,SystemExit):save("stopped")
    raise
