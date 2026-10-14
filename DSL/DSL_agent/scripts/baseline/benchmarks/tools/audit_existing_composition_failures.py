"""Read-only re-audit of eight preserved composition compiler failures.

No generation, compilation or encrypted execution. Compiler logs are advisory
input; results prove only the observed old expressions' failed constraints.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import math

from benchmark_graph import digest, require
from earth_failure_diagnostic import diagnose
from poseidon_runtime_paths import PROFILE
from strict_json_files import dump

CASES = ("attention_core_0","attention_core_1","attention_core_2",
         "channel_broadcast_mix_0","channel_broadcast_mix_1","channel_broadcast_mix_2",
         "norm_depth_branches_1","norm_depth_branches_2")


def run(root, destination):
    require(not destination.exists(),"Preserve existing diagnosis evidence")
    raw = (root/"report.json").read_bytes()
    old = json.loads(raw)
    rows = {row["id"]:row for row in old["rows"]}
    hashes = {}
    def bound(path):
        h=hashlib.sha256(path.read_bytes()).hexdigest()
        hashes[str(path)]=h
        return h
    bound(root/"report.json")
    profile_bytes=PROFILE.read_bytes()
    profile_hash=bound(PROFILE)
    output=[]
    for name in CASES:
        row=rows[name]
        task=root/"tasks"/row["task_id"]
        payload=json.loads((task/"task.json").read_text())
        require(digest({k:v for k,v in payload.items() if k!="binding"})==payload["binding"],
                "Historical task metadata binding")
        req=payload["request"]
        candidate=(root/"prepared"/name/"candidate.py").read_bytes()
        require(hashlib.sha256(candidate).hexdigest()==row["candidate_sha256"],
                "Preserved rule candidate binding")
        response=json.loads((task/"response-0.json").read_text())
        response_candidate=json.loads(response["raw"])
        require(response_candidate["hecate_source"].encode()==candidate and
                response_candidate["request_id"]==req["request_id"]==row["request_id"],
                "Original response/source/request identity")
        validation=json.loads((task/"validation-0.json").read_text())
        require(validation["paid_calls"]==0 and validation["new_agent_generation"] is False,
                "Historical rule candidate origin")
        stages=validation["observed_stages"]
        require(stages["traced"] is True and stages["compiled"] is False and
                validation["encrypted_execution"] is False,"Historical failed stage")
        evidence=Path(validation["evidence"])
        log=evidence/"attempt-00/compile.log"
        content=log.read_text()
        require(log.stat().st_size<=4000,"Bounded compiler diagnostic")
        diagnosis=diagnose(content,profile_bytes)
        require(diagnosis==validation["compiler_diagnosis"] and
                diagnosis["failed_conditions"]==["compiler_accumulated_scale_budget"],
                "Reconstructed original compiler diagnosis")
        model=req["model"]
        for p in (task/"task.json",task/"response-0.json",task/"validation-0.json",
                  root/"prepared"/name/"candidate.py",log,evidence/"report.json"):
            bound(p)
        group=name.rsplit("_",1)[0]
        output.append(dict(id=name,group=group,old_task_id=row["task_id"],
            request_id=req["request_id"],candidate_sha256=row["candidate_sha256"],
            original_generation="scripted rule candidate; not a new Agent result",
            logical_input_elements=sum(math.prod(i["shape"]) for i in model["inputs"]),
            expanded_graph_nodes=len(model["nodes"]),operators=dict(Counter(n["op"] for n in model["nodes"])),
            compiler_log_sha256=hashes[str(log)],compiler_diagnosis=diagnosis,
            historical_observed_stages=stages,new_backend_execution=False,
            formal_infeasibility_proved=False,runtime_capacity_checked=False))
    require(all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items()),
            "Historical evidence changed during audit")
    result=dict(format="poseidon-existing-composition-failures-audit-v1",
        rows=output,failure_count=8,mathematical_groups=3,
        groups=dict(Counter(row["group"] for row in output)),
        compiler_profile_sha256=profile_hash,historical_hashes=hashes,
        confirmed="Eight original Earth mul type-inference rejections reconstruct the fixed compiler scale budget",
        inference="Candidate lowering and depth/scale demand need separate investigation; small tensor sizes alone are not the failing check",
        unconfirmed="Equivalent expression feasibility, refreshed runtime execution and bootstrap need new bounded evidence",
        suggested_agent_layer_actions=[
            "Inspect adjacent public factors and layout masks before changing DSL expression",
            "Validate equivalent expression against unchanged graph/reference and packing rules",
            "Recompile only approved affected cases; classify depth limitations separately from input acceptance"],
        new_compile_calls=0,new_encrypted_execution=False,paid_calls=0,
        backend_modified=False,parameters_modified=False,numerical_tolerance_modified=False)
    result["binding"]=digest(result)
    dump(destination,result)
    print(json.dumps(dict(binding=result["binding"],cases=8,groups=result["groups"],new_compile_calls=0,paid_calls=0)))
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--historical-root",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    run(a.historical_root,a.output)
