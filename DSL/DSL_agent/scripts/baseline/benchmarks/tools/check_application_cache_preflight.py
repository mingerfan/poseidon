"""Actual pinned Poseidon adapter identity/preflight rejection; no backend execution."""
import json,sys,time
from pathlib import Path
ROOT=(Path(__file__).resolve().parents[6] if Path(__file__).resolve().parents[4].name == "DSL_agent" and Path(__file__).resolve().parents[5].name == "DSL" else Path(__file__).resolve().parents[4])
sys.path.insert(0,str(ROOT/"scripts/baseline"))
from benchmark_graph import canonical,digest,require
from benchmark_suite import Builder
from application_component import Application
from application_backend import PoseidonBackend

class CountedPoseidonBackend(PoseidonBackend):
    """Count real package-check calls; no checker or identity behavior is replaced."""
    def __init__(self):self.package_checks=0
    def check_package(self,*args):
        self.package_checks+=1
        return super().check_package(*args)

class ForbiddenProvider:
    kind="scripted_replay";agent_calls=0
    def __init__(self):self.calls=0
    def generate(self,*args):
        self.calls+=1
        raise AssertionError("Preflight must finish before any generation")

def models():
    b=Builder([(2,)]);square=b.finish(b.node("square",["input0"]))
    b=Builder([(2,)]);negative=b.finish(b.node("negate",["input0"]))
    return square,negative

def run(output):
    from hecate_python_env import VENV
    import os
    require(os.environ.get("IN_NIX_SHELL")=="pure" and Path(sys.prefix)==VENV,"Pinned pure environment")
    output=Path(output);require(not output.exists(),"Fresh adapter preflight")
    output.mkdir(mode=0o700);start=time.monotonic();rows=[]
    for mode in ("copied_cache_key","different_delivery_origin","changed_origin_binding","missing_program"):
        backend=CountedPoseidonBackend();app=Application(output/mode,backend)
        first,second=models()
        origin=app.submit(first,validation_level="compiled")
        target=app.submit(second if mode!="missing_program" else first,validation_level="compiled")
        require(app.status(origin)["status"]==app.status(target)["status"]=="queued","Actual Poseidon SDK identity preflight")
        origin_meta=app._checked_task_metadata(origin)
        target_meta=app._checked_task_metadata(target)
        original=app._load(app._dir(origin)/"state.json")
        target_state=app._load(app._dir(target)/"state.json")
        # Explicit fault fixture: no artifact, compilation or prior success is claimed.
        original.update(status="succeeded",phase="complete",acceptance={"fixture_only":True,"level":"compiled"},
                        package=dict(task_id=origin,binding="0"*64,reused=False))
        if mode in ("copied_cache_key","changed_origin_binding","missing_program"):
            original["reuse_key"]=target_state["reuse_key"]
            app._write(app._dir(origin)/"state.json",original)
            if mode=="changed_origin_binding":
                path=app._dir(origin)/"task.json";meta=app._load(path)
                meta["context"]["owned_fault"]="stale";app._write(path,meta)
            provider=ForbiddenProvider()
            result=app.run(target,provider)
        else:
            target_state.update(status="succeeded",phase="complete",package=original["package"],
                                acceptance=original["acceptance"])
            app._write(app._dir(target)/"state.json",target_state)
            provider=ForbiddenProvider()
            result=app.result(target)
        require(result["status"]=="service_error" and result["package"] is None and provider.calls==0,
                "Fault cannot be delivered or cause generation")
        expected_checks=1 if mode=="missing_program" else 0
        require(backend.package_checks==expected_checks,"Request guard must precede package loading")
        state=app._load(app._dir(target)/"state.json")
        require(state["generations"]==state["http_attempts"]==0,"No generation/HTTP reservations")
        rows.append(dict(case=mode,status=result["status"],failure=result["failure"],
            actual_identity_checked=True,real_package_checker_calls=backend.package_checks,
            generator_calls=provider.calls,generations=state["generations"],http_attempts=state["http_attempts"],
            compiler_calls=0,encrypted_execution=0,request_id=target_meta["request"]["request_id"],
            origin_request_id=origin_meta["request"]["request_id"]))
    report=dict(format="poseidon-application-cache-adapter-preflight-v1",planned=4,passed=len(rows),failed=0,
        rows=rows,seconds=time.monotonic()-start,backend="actual PoseidonBackend with forwarded package-check counter",
        lifecycle="Owned injected fault states only; no real compilation or successful program was fabricated as evidence",
        compiled=0,encrypted_execution=0,decrypted=0,paid_calls=0,installs=0,downloads=0)
    report["binding"]=digest(report)
    (output/"report.json").write_bytes(canonical(report))
    print(json.dumps(report))
if __name__=="__main__":run(sys.argv[1])
