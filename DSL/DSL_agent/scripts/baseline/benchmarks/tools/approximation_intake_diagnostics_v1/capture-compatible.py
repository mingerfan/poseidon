"""Real isolated PyTorch rejection transport with an injected, unused backend."""
import hashlib,json,os,resource,shlex,shutil,signal,sys,time,threading
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5]);BASE=ROOT/"scripts/baseline"
sys.path[:0]=[str(BASE),str(BASE/"benchmarks/tools/static_broadcast_live_v1")]
import run as live
from benchmark_graph import digest
def load(p):return json.loads(Path(p).read_text())
def save(p,v):
    with Path(p).open("x")as f:json.dump(v,f,sort_keys=True,indent=2)
def verify(plan):
    assert plan["binding"]==digest({k:v for k,v in plan.items()if k!="binding"})
    assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest()==plan["tool_sha256"]
    for name,h in plan["sdk_sources"].items():assert hashlib.sha256((BASE/name).read_bytes()).hexdigest()==h,name
    for case in plan["cases"]:
        for name,h in case["files"].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h,name
    return plan
plan=verify(load(HERE/"capture-plan-compatible.json"));budget=plan["budget"];campaign=Path(plan["campaign"])
results=Path(plan["results_root"])
if "--inside"not in sys.argv:
    live.no_active();assert not campaign.exists()
    assert live.measure(results)+budget["new_artifact_bytes"]<=budget["global_result_bytes"]
    assert shutil.disk_usage(results).free>=budget["disk_reserve_bytes"]
    campaign.mkdir(mode=0o700)
    save(campaign/"owner.json",dict(plan_binding=plan["binding"],owner=live.process_identity(os.getpid()),
        started=time.time(),deadline=time.time()+budget["batch_wall_seconds"]))
    from hecate_python_env import enter_nix,VENV
    command='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join(
        [str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"])
    tree=live.OwnedTree().start()
    try:code=enter_nix(command,seconds=budget["batch_wall_seconds"],environment="poseidon-cpu")
    finally:background=tree.finish()
    assert not background,"nix_background_survived"
    raise SystemExit(code)
from process_resource_limits import limits
limits()
assert resource.getrlimit(resource.RLIMIT_AS)==(4294967296,4294967296)
assert resource.getrlimit(resource.RLIMIT_CPU)==(150,155)
assert resource.getrlimit(resource.RLIMIT_FSIZE)==(16777216,16777216)
assert resource.getrlimit(resource.RLIMIT_NOFILE)==(128,128)
assert tuple(sys.version_info[:3])==(3,10,14)
from application_component import Application
from application_model_source import FORMAT
from application_policy import DEFAULT_BUDGET
from test_application_component import Backend as TestBackend
from pytorch_intake_diagnostics import FILENAME,read
from pytorch_ordered_contract import source_identity
class Backend(TestBackend):
    def qualify(self,*args,**kwargs):
        self.calls+=1
        raise AssertionError("Input rejection must precede backend qualification")
    def export(self,*args,**kwargs):raise AssertionError("No program for rejected input")
class Provider:
    def __init__(self):self.calls=0
    def generate(self,*args,**kwargs):
        self.calls+=1
        raise AssertionError("Input rejection must precede generation")
owner=load(campaign/"owner.json");rows=[]
for case in plan["cases"]:
    verify(plan)
    assert time.time()<owner["deadline"]
    assert live.measure(results)+budget["per_case_bytes"]<=budget["global_result_bytes"]
    assert live.measure(campaign)+budget["per_case_bytes"]<=budget["new_artifact_bytes"]
    target=campaign/case["name"];target.mkdir(mode=0o700)
    backend=Backend();app=Application(target/"tasks",backend);provider=Provider()
    stop=None;finished=threading.Event();tree=live.OwnedTree().start();started=time.monotonic()
    def monitor():
        global stop
        while not finished.wait(.2):
            if time.time()>=owner["deadline"]or time.monotonic()-started>=budget["per_capture_seconds"]:stop="time_budget"
            elif live.measure(target)>budget["per_case_bytes"]or live.measure(campaign)>budget["new_artifact_bytes"]:stop="artifact_budget"
            elif live.measure(results)>budget["global_result_bytes"]or shutil.disk_usage(results).free<budget["disk_reserve_bytes"]:stop="global_budget"
            if stop:
                tree.finish();break
    watcher=threading.Thread(target=monitor,daemon=True);watcher.start()
    value={}
    try:
        task=app.submit_source(dict(format=FORMAT,kind="trusted_pytorch",manifest=case["manifest"],trust_local_code=True),
            resource_profile="general-tensors-v1",limits=dict(DEFAULT_BUDGET,wall_seconds=180))
        assert app.status(task)["phase"]=="intake_queued"
        status=app.prepare_source(task)
        assert status["status"]=="unsupported"and status["phase"]=="complete",status
        failure=status["failure"]
        assert failure["code"]=="explicit_approximation_required"and failure["layer"]=="input"
        assert failure["context"]["function"]==case["expected_function"],failure
        assert failure["context"]["operator"]==case["expected_operator"],failure
        imported=app._dir(task)/"model-import"
        diagnostic=read(imported/FILENAME,load(app._dir(task)/"source/manifest.json"),source_identity())
        assert diagnostic==failure
        assert not(imported/"model.json").exists()and not(imported/"reference-report.json").exists()
        assert not(app._dir(task)/"task.json").exists()
        assert status["generations"]==status["http_attempts"]==0
        assert status["package"]is None and status["acceptance"]is None
        assert app.run(task,provider)["failure"]==failure
        assert provider.calls==backend.calls==0
        save(target/"application-status.json",status)
        value=dict(status="expected_rejection_passed",failure=failure,
            model_import_succeeded=False,actual_sandbox_capture=True,actual_application_intake=True,
            backend_identity_injected=True,backend_calls=backend.calls,provider_calls=provider.calls)
    except Exception as error:value=dict(status="failed",error_type=type(error).__name__)
    finally:
        finished.set();watcher.join(timeout=35);assert not watcher.is_alive()
        background=tree.finish()
    value.update(case=case["name"],seconds=time.monotonic()-started,bytes=live.measure(target),
        budget_stop=stop,background_survived=bool(background))
    if stop or background or value["bytes"]>budget["per_case_bytes"]:value["status"]="failed"
    rows.append(value);save(campaign/(case["name"]+"-status.json"),value)
    print(json.dumps({k:value[k]for k in("case","status","seconds","bytes","budget_stop")}),flush=True)
    if value["status"]!="expected_rejection_passed":break
verify(plan)
report=dict(format="poseidon-approximation-input-rejection-results-v1",plan_binding=plan["binding"],
 sdk=plan["sdk"],cases=rows,planned=3,expected_rejections_passed=sum(r["status"]=="expected_rejection_passed"for r in rows),
 failed=sum(r["status"]!="expected_rejection_passed"for r in rows),not_run=3-len(rows),skipped=0,
 source_capture_calls=len(rows),model_import_successes=0,compiler_calls=0,encrypted_calls=0,provider_calls=0,
 results_bytes=live.measure(campaign),global_result_bytes=live.measure(results),
 independent_reference_or_numerical_acceptance=False,backend_identity_injected=True)
report["binding"]=digest(report);save(campaign/"report.json",report)
print(json.dumps({k:report[k]for k in("binding","expected_rejections_passed","failed","not_run","results_bytes","global_result_bytes")}),flush=True)
raise SystemExit(not(report["expected_rejections_passed"]==3 and report["failed"]==report["not_run"]==0))
