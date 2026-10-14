"""Formal export provenance and fixed-DSL acceptance; never generates candidates."""
import hashlib,importlib.util,json,math,shutil,signal,sys,time,traceback,unittest
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parents[2]
MHA=HERE.parent/"staged_mha_ordered_graph_v1"
SELECTED=[str(HERE),str(MHA/"overlay"),str(BASE)]
sys.path[:]=SELECTED+[p for p in sys.path if p not in SELECTED]
import numpy as np
from benchmark_graph import canonical,digest,require
from benchmark_math import evaluate
from application_backend import PoseidonBackend
from candidate_bundle import check_bundle,load_model_provenance
from candidate_contract import request_file_bytes
from hecate_python_env import ROOT,WORK
from numerical_results import compare
from pytorch_ordered_contract import source_identity
NAME="boolean_padding_residual__declared_shallow_v1"
MODULES=("test_candidate_bundle","test_staged_bundle","test_model_source_bundle",
         "test_application_identity","test_identity_syntax_cache",
         "test_program_execution","test_agent_component","test_unified_capacity_tiles")
LIMITS=dict(wall_seconds=1200,stage_seconds=300,new_bytes=4*1024**3,total_bytes=32*1024**3,
            free_bytes=4*1024**3,native_cpu_seconds=150,native_cpu_hard_seconds=155,
            native_address_bytes=4*1024**3,execution_concurrency=1,compile_jobs=2,link_jobs=1)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def total(root):
    return sum(p.stat().st_size for p in root.rglob("*") if p.is_file() and not p.is_symlink())
def main():
    out=Path(sys.argv[1]);require(not out.exists(),"Preserve prior evidence")
    backend=PoseidonBackend();identity=backend.identity();stages=backend.identity_manifest();capture_identity=source_identity()
    known={p.name for p in (WORK/"results").iterdir()}
    require(total(WORK/"results")<=LIMITS["total_bytes"] and shutil.disk_usage(WORK).free>=LIMITS["free_bytes"],"Initial disk boundary")
    protected_file=ROOT/"docs/baseline/agent-v12-live-rebound-proposal-r1304.json"
    protected=json.loads(protected_file.read_bytes())["protected_sources"]
    paths=[Path(__file__),HERE/"test_serialization.py",BASE/"candidate_bundle.py"]+[BASE/(n+".py") for n in MODULES]
    tools={str(p):sha(p) for p in paths}
    capture=WORK/"results/agent-mha-ordered-capture-v2"
    original_capture=json.loads((capture/"report.json").read_bytes())
    require(capture_identity==original_capture["implementation"],"Source capture implementation changed")
    retained=WORK/"results/agent-mha-replicated-scalar-backend-v1"/NAME
    source=(retained/"candidate.py").read_text();request=json.loads((retained/"request.json").read_bytes())
    capture_folder=capture/NAME
    captures={str(p):sha(p) for p in capture_folder.rglob("*") if p.is_file() and not p.is_symlink()}
    prior_files={str(p):sha(p) for p in retained.rglob("*") if p.is_file() and not p.is_symlink()}
    out.mkdir();started=time.monotonic()
    report=dict(format="poseidon-formal-provenance-serialization-acceptance-v1",status="running",
        source_kind="deterministic_rule_baseline",candidate_sha256=hashlib.sha256(source.encode()).hexdigest(),
        request_id=request["request_id"],limits=LIMITS,tool_sha256=tools,backend_identity=identity,
        stage_identity=stages,source_capture_implementation=capture_identity,protected_sources=protected,
        protected_table_sha256=digest(protected),protected_table_file=str(protected_file),
        protected_table_file_sha256=sha(protected_file),capture_files=captures,prior_files=prior_files,
        api_calls=0,http_attempts=0,new_agent_generation=False,backend_modified=False,
        production_export_changed=True,production_mha_enabled=False,models_changed=False,reference_changed=False,
        numerical_tolerance=dict(atol=1e-5,rtol=1e-4),compiler_reexecuted=False)
    def save():
        report["binding"]=digest({k:v for k,v in report.items() if k!="binding"})
        (out/"report.json").write_bytes(canonical(report))
    def boundary():
        require(time.monotonic()-started<LIMITS["wall_seconds"],"Window time boundary")
        require(all(sha(p)==s for p,s in tools.items()),"Frozen tool/source drift")
        require(all(sha(ROOT/p)==s for p,s in protected.items()),"Protected source drift")
        require(all(sha(p)==s for p,s in captures.items()),"Capture evidence drift")
        require(all(sha(p)==s for p,s in prior_files.items()),"Prior evidence drift")
        require(backend.identity()==identity and backend.identity_manifest()==stages and source_identity()==capture_identity,"Frozen dependency identity drift")
        require(total(WORK/"results")<=LIMITS["total_bytes"] and shutil.disk_usage(WORK).free>=LIMITS["free_bytes"],"Global disk boundary")
        require(sum(total(p) if p.is_dir() else p.stat().st_size for p in (WORK/"results").iterdir() if p.name not in known)<=LIMITS["new_bytes"],"New artifact boundary")
    def alarm(*_):raise TimeoutError("Acceptance stage time boundary")
    previous=signal.signal(signal.SIGALRM,alarm)
    save()
    try:
        boundary()
        suite=unittest.TestSuite()
        for name in MODULES:suite.addTests(unittest.defaultTestLoader.loadTestsFromName(name))
        spec=importlib.util.spec_from_file_location("test_formal_serialization",HERE/"test_serialization.py")
        extra=importlib.util.module_from_spec(spec);spec.loader.exec_module(extra)
        suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(extra))
        sys.path[:]=SELECTED+[p for p in sys.path if p not in SELECTED]
        signal.setitimer(signal.ITIMER_REAL,LIMITS["stage_seconds"])
        with (out/"controls.log").open("w") as log:
            controls=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
        signal.setitimer(signal.ITIMER_REAL,0)
        report["controls"]=dict(modules=list(MODULES),ran=controls.testsRun,
            passed=max(0,controls.testsRun-len(controls.failures)-len(controls.errors)-len(controls.skipped)),
            failed=len(controls.failures),errors=len(controls.errors),skipped=len(controls.skipped))
        save();print("controls",report["controls"],flush=True)
        require(controls.wasSuccessful(),"Controls failed; retain log")
        boundary()
        # The retained qualification froze candidate_bundle.py's old identity.
        # Requalify the identical request and DSL rather than edit its history.
        (out/"request.json").write_bytes(request_file_bytes(request));(out/"candidate.py").write_text(source)
        provenance=load_model_provenance(capture_folder,request["model"],max_bytes=4*1024**2)
        signal.setitimer(signal.ITIMER_REAL,LIMITS["stage_seconds"])
        result=backend.qualify(request,dict(schema=1,request_id=request["request_id"],hecate_source=source),"numerical")
        signal.setitimer(signal.ITIMER_REAL,0)
        (out/"qualification.json").write_bytes(canonical(result))
        report["qualification"]=result;report["compiler_reexecuted"]=result["observed_stages"]["compiled"];save()
        require(result["status"]=="passed","Current-source qualification failed")
        require((Path(result["evidence"])/"attempt-00/candidate.py").read_bytes()==(retained/"candidate.py").read_bytes(),"Fixed DSL changed")
        boundary()
        package=out/"program"
        manifest=backend.export(result,package,"numerical",model_provenance=provenance)
        require(check_bundle(package)==manifest,"Formal exported package reload")
        require(not manifest["agent_generated"],"Rule result must remain non-Agent")
        report.update(package_binding=manifest["binding"],package_check_passed=True,
            formal_exporter_used=True,transport_repacker_used=False,package_files=manifest["files"]);save()
        model=request["model"];samples=[]
        for group in range(3):
            sample={}
            for index,item in enumerate(model["inputs"]):
                n=math.prod(item["shape"]);v=(np.arange(n,dtype=np.float64)+1)/(n+1)*.35
                if group==0:v=-v
                if group==2:v=np.cos(np.arange(n,dtype=np.float64)+index*.19)*.27
                sample[item["name"]]=v.reshape(item["shape"]).tolist()
            samples.append(sample)
        executions=[]
        for title,inputs in (("new_input",samples),("repeated_input",[samples[0]])):
            boundary();require(check_bundle(package)==manifest,"Independent package reload")
            signal.setitimer(signal.ITIMER_REAL,LIMITS["stage_seconds"])
            execution=backend.execute(package,inputs)
            signal.setitimer(signal.ITIMER_REAL,0)
            actual=np.array([[v for output in model["outputs"] for v in np.asarray(row[output["name"]]).reshape(-1)] for row in execution["outputs"]])
            expected=np.array([[v for output in model["outputs"] for v in evaluate(model,{k:np.asarray(v,dtype=np.float64) for k,v in row.items()})[output["name"]].reshape(-1)] for row in inputs])
            comparison=compare(actual,expected,1e-5,1e-4)
            require(comparison["passed"],"Frozen numerical gate")
            (out/(title+".json")).write_bytes(canonical(dict(execution=execution,comparison=comparison)))
            executions.append(dict(kind=title,groups=len(inputs),
                comparison={k:comparison[k] for k in ("passed","mae","max_absolute_error","compared_values")},
                evidence=execution["evidence"],resource_precheck_binding=execution["resource_precheck"]["binding"]))
            report["executions"]=executions;save();print(title,"passed",comparison["max_absolute_error"],flush=True)
        boundary();report.update(status="passed",reloaded=True,new_input_passed=True,repeated_execution=True,
            elapsed_seconds=time.monotonic()-started);save();print("formal_export_acceptance_passed",report["binding"],flush=True)
    except Exception as error:
        report.update(status="failed",diagnostic=str(error),traceback=traceback.format_exc(),elapsed_seconds=time.monotonic()-started)
        save();print(report["traceback"],flush=True);return 1
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,previous)
    return 0
if __name__=="__main__":raise SystemExit(main())
