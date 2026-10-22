"""Offline application lifecycle over the real Poseidon backend and a helper."""
import argparse,json
from pathlib import Path
from application_component import Application
from application_backend import PoseidonBackend
from candidate_bundle import replay_bundle,check_bundle
from upstream_helper_directed_cases import cases
from upstream_candidate_helpers import BN_PROFILE
from hecate_python_env import WORK
from benchmark_graph import require
class Replay:
    kind="scripted_replay";agent_calls=0
    def __init__(self,source):self.source=source;self.calls=0
    def generate(self,request,history):
        self.calls+=1
        return json.dumps(dict(schema=1,request_id=request["request_id"],hecate_source=self.source))
def main():
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True);a=p.parse_args()
    require(a.output.resolve().is_relative_to(WORK/"results") and not a.output.exists(),"Fresh results")
    a.output.mkdir();row=next(r for r in cases() if r["name"]=="bn_asymmetric_0")
    options=dict(helper_profile=row.get("profile",BN_PROFILE),helper_exercise=row["required_helpers"])
    app=Application(a.output/"tasks",PoseidonBackend());records=[]
    for level in ("compiled","numerical"):
        task=app.submit(row["model"],validation_level=level,options=options)
        provider=Replay(row["source"]);result=app.run(task,provider)
        record=dict(level=level,task_id=task,result=result,scripted_generations=provider.calls)
        records.append(record)
        (a.output/"report.json").write_text(json.dumps(dict(rows=records,paid_calls=0,agent_generated=False),indent=2))
        require(result["status"]=="succeeded","Application failed: "+str(result))
        package=a.output/"tasks"/result["package"]["task_id"]/"program"
        require(check_bundle(package)["validation"]["level"]==level,"Package level")
        before=set(WORK.joinpath("results").glob("poseidon-package-replay-*"))
        require(replay_bundle(package)==0,"Package reload failed")
        after=set(WORK.joinpath("results").glob("poseidon-package-replay-*"))
        require(len(after-before)==1,"One fresh replay evidence")
        evidence=(after-before).pop()
        record["reload"]=json.loads((evidence/"report.json").read_text());record["reload_evidence"]=str(evidence)
        again=app.submit(row["model"],validation_level=level,options=options);unused=Replay(row["source"])
        reused=app.run(again,unused)
        require(reused["status"]=="succeeded" and reused["package"]["reused"] and unused.calls==0,"Package reuse")
        record["reuse"]=reused;record["unused_provider_calls"]=unused.calls
        (a.output/"report.json").write_text(json.dumps(dict(rows=records,paid_calls=0,agent_generated=False),indent=2))
    print(a.output,flush=True)
    return 0
if __name__=="__main__":raise SystemExit(main())
