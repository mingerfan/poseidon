"""Reproduce cache-key corruption in the actual application controller with a mock backend."""
import copy,json,sys
from pathlib import Path
ROOT=(Path(__file__).resolve().parents[6] if Path(__file__).resolve().parents[4].name == "DSL_agent" and Path(__file__).resolve().parents[5].name == "DSL" else Path(__file__).resolve().parents[4])
sys.path.insert(0,str(ROOT/"scripts/baseline"))
from benchmark_graph import canonical,digest,require
from benchmark_suite import Builder
from application_component import Application
from test_application_component import Backend,Provider

class BoundTestBackend(Backend):
    def qualify(self,request,candidate,level):
        result=super().qualify(request,candidate,level)
        result["mock_request_id"]=request["request_id"]
        return result
    def export(self,result,destination,level):
        destination.mkdir()
        (destination/"manifest.json").write_bytes(canonical(dict(test_only=True,level=level,
                                                   request_id=result["mock_request_id"])))

def main(output):
    output=Path(output);require(not output.exists(),"Fresh control reproduction")
    output.mkdir(mode=0o700);backend=BoundTestBackend()
    app=Application(output/"tasks",backend)
    b=Builder([(2,)]);square=b.finish(b.node("square",["input0"]))
    b=Builder([(2,)]);negative=b.finish(b.node("negate",["input0"]))
    origin=app.submit(square,validation_level="compiled")
    passed=app.run(origin,Provider())
    require(passed["status"]=="succeeded","Harness origin")
    target=app.submit(negative,validation_level="compiled")
    old=app._load(app._dir(origin)/"state.json")
    new=app._load(app._dir(target)/"state.json")
    origin_meta=app._load(app._dir(origin)/"task.json")
    target_meta=app._load(app._dir(target)/"task.json")
    require(origin_meta["request"]!=target_meta["request"],"Distinct valid models")
    # Owned fault injection: a copied cache key in prior state, not a backend alteration.
    old["reuse_key"]=new["reuse_key"]
    app._write(app._dir(origin)/"state.json",old)
    provider=Provider(["-x"])
    outcome=app.run(target,provider)
    delivered=app._load(app._dir(outcome["package"]["task_id"])/"program/manifest.json")
    reproduced=(outcome["status"]=="succeeded" and outcome["package"]["reused"] is True and
                provider.calls==0 and delivered["request_id"]!=target_meta["request"]["request_id"])
    report=dict(format="poseidon-application-cache-request-reproduction-v1",
        reproduced=reproduced,fixture="private task state cache key copied between distinct models",
        origin_request=origin_meta["request"]["request_id"],
        requested_request=target_meta["request"]["request_id"],delivered_request=delivered["request_id"],
        outcome=outcome,scripted_provider_calls=provider.calls,
        backend_kind="injected_test_double",actual_controller=True,
        compile_calls=0,encrypted_execution=0,paid_calls=0,
        scope="Controller cache-selection fault reproduction; not real Poseidon or Agent evidence")
    report["binding"]=digest(report);(output/"report.json").write_bytes(canonical(report))
    print(json.dumps({k:v for k,v in report.items() if k!="outcome"}))
    require(reproduced,"Expected cache fault not reproduced")
if __name__=="__main__":main(sys.argv[1])
