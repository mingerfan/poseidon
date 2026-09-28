"""Bounded validation-adapter, repaired-answer and capability-composition acceptance."""
import argparse,hashlib,json,os,shlex,subprocess,sys,time
from pathlib import Path
BASE=Path(__file__).resolve().parents[2];ROOT=BASE.parents[1];sys.path.insert(0,str(BASE))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument("--inside",action="store_true");a=p.parse_args()
    from hecate_python_env import enter_nix,VENV
    if not a.inside:return enter_nix('LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"]),seconds=1800)
    from workspace_paths import RESULTS
    from benchmark_graph import digest,signature
    from benchmark_runner import dump
    from component_contract import VERSION,prepare_task,reconstruct_request
    from semantic_benchmark_execution import runtime_sources
    from capability_combination_cases import cases
    output=RESULTS/"validation-adapter-r167/acceptance";output.mkdir()
    before=runtime_sources();guard=json.loads((output.parent/"before.json").read_text())["compiler"]
    def frozen():
        assert runtime_sources()==before,"Sources changed during batch"
        assert all(sha(Path(p))==v for p,v in guard.items()),"Compiler/runtime changed"
    frozen()
    compatibility=RESULTS/"stage2-agent-campaign-v7-r130/index.json";count=0
    for shard in json.loads(compatibility.read_text())["shards"]:
        file=compatibility.parent/shard["file"];assert sha(file)==shard["sha256"]
        for row in json.loads(file.read_text())["cases"]:
            assert reconstruct_request(row["request"])==row["request"];count+=1
    dump(output/"request-compatibility.json",dict(checked=count,changed=0))
    tasks=cases()
    for row in tasks:row["group"]="new_composition"
    # Explicit three-topology requirement for each newly opened composition.
    groups={}
    for row in tasks:groups.setdefault(row["construction"],set()).add(signature(row["model"],topology=True))
    assert all(len(g)>=3 for g in groups.values())
    current=json.loads((output.parent/"static-after-v2.json").read_text())
    prior=json.loads((ROOT/"docs/baseline/stage2-noncompiler-repairs-r160.json").read_text())
    by={r["id"]:r for r in prior["remaining_static_cases"]}
    for row in current["retained_candidate_static_checks"]:
        if row["result"]["status"]!="static_accepted_not_executed":continue
        old=by[row["id"]];raw=Path(old["original_response"]);assert sha(raw)==old["original_response_sha256"]
        request=json.loads((raw.parent.parent/"request.json").read_text());candidate=json.loads(raw.read_text())
        assert hashlib.sha256(candidate["hecate_source"].encode()).hexdigest()==old["source_sha256"]
        tasks.append(dict(name=row["id"],group="unchanged_agent_replay",request=request,candidate=candidate,
                          original_response_sha256=old["original_response_sha256"],new_agent_generation=False))
    from benchmark_suite import Builder
    from unified_public_exercises import SPECS
    assert "unified-public-counter-loop_iterations" in SPECS
    for variant in range(3):
        b=Builder([(6,)]);h=b.node("square",["input0"])
        if variant==1:h=b.node("add",[h,"input0"])
        if variant==2:h=b.node("add",[h,b.const([.125])])
        model=b.finish(h)
        request=prepare_task(model,dict(construction_profile="hecate-unified-public-v1",
            construction="unified-public-counter-loop_iterations",chunk_period=4))
        source='@hc.func("c,c,c")\ndef golden(x,y,zero_ct):\n    def calc(v):\n        acc=zero_ct\n        for i in range(2):\n            acc=acc+v*v*0.5\n        return acc\n    out=[]\n    for cell in [x,y]:\n        h=calc(cell)\n'
        if variant==1:source+='        h=h+cell\n'
        if variant==2:source+='        h=h+0.125\n'
        source+='        out.append(h)\n    return out\n'
        tasks.append(dict(name="public-loop-chunk-"+str(variant),group="existing_composition_regression",
                          request=request,candidate=dict(schema=1,request_id=request["request_id"],hecate_source=source)))
    from unified_graph_contract import validate_candidate
    for row in tasks:validate_candidate(row["candidate"],row["request"])
    assert len(tasks)<=48
    dump(output/"plan.json",dict(tasks=tasks,source_hashes=before,native_concurrency=1,new_paid_calls=0,
        max_tasks=48,per_execution_seconds=300,compiler_build_jobs=0))
    rows=[];start=time.monotonic()
    for task in tasks:
        frozen();folder=output/task["name"];folder.mkdir()
        job=dict(format=VERSION,action="qualify",request=task["request"],candidate=task["candidate"])
        dump(folder/"job.json",job)
        with (folder/"worker.log").open("x") as log:
            proc=subprocess.run([str(VENV/"bin/python"),"-B",str(ROOT/"scripts/agent_component.py"),
                "--job",str(folder/"job.json")],cwd=folder,stdout=subprocess.PIPE,stderr=log,text=True,timeout=300)
        result=json.loads(proc.stdout);dump(folder/"result.json",result)
        rows.append(dict(name=task["name"],group=task["group"],request_id=task["request"]["request_id"],
                         source_sha256=hashlib.sha256(task["candidate"]["hecate_source"].encode()).hexdigest(),
                         result=result))
        dump(output/"progress.json",dict(rows=rows))
        if result.get("failure",{}).get("layer")=="integrity":raise ValueError("Integrity batch stop")
    # Existing CLI schema-2 Linear scripted faults exercise the same extracted adapter.
    with (output/"legacy-self-test.log").open("x") as log:
        old=subprocess.run([str(VENV/"bin/python"),"-B",str(BASE/"run_candidate.py"),"--inside",
            "--case",str(BASE/"cases/linear-example.json"),"--self-test"],cwd=ROOT,
            stdout=log,stderr=subprocess.STDOUT,timeout=300)
    frozen()
    report=dict(format="poseidon-validation-adapter-r167",rows=rows,source_hashes=before,
        new_composition_topology_counts={k:len(v) for k,v in groups.items()},
        unchanged_requests=count,legacy_self_test_exit=old.returncode,compiler_files_unchanged=len(guard),
        new_paid_calls=0,new_agent_generation=False,seconds=time.monotonic()-start,
        passed=sum(r["result"]["status"]=="passed" for r in rows),failed=sum(r["result"]["status"]!="passed" for r in rows))
    report["binding"]=digest(report);dump(output/"report.json",report)
    print(json.dumps({k:v for k,v in report.items() if k not in ("rows","source_hashes")}))
    return bool(report["failed"] or old.returncode)
if __name__=="__main__":raise SystemExit(main())
