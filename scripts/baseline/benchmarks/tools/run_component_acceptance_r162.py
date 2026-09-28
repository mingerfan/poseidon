"""Bounded offline component/bundle acceptance. Manual fixtures, no model API."""
import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time
import unittest

ROOT=Path(__file__).resolve().parents[4]
BASE=ROOT/"scripts/baseline"
sys.path.insert(0,str(BASE))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument("--inside",action="store_true");a=p.parse_args()
    from hecate_python_env import enter_nix,VENV
    if not a.inside:
        cmd=shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"])
        return enter_nix('LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+cmd,seconds=1800)
    from workspace_paths import RESULTS
    from benchmark_graph import digest,canonical
    from benchmark_runner import dump
    from benchmark_suite import Builder
    from component_contract import prepare_task,reconstruct_request,VERSION
    from semantic_benchmark_execution import runtime_sources
    from candidate_bundle import export_bundle,check_bundle
    root=RESULTS/"component-integration-r161"
    output=root/"acceptance-final";output.mkdir()
    before=runtime_sources()
    guard=json.loads((root/"before.json").read_text())["compiler"]
    def frozen():
        assert runtime_sources()==before
        assert all(sha(Path(n))==h for n,h in guard.items())
    frozen()
    modules=["test_agent_component","test_candidate_bundle","test_candidate_pipeline",
             "test_unified_graph","test_unified_chunk_layout","test_typed_witness_repairs",
             "test_portability","test_compiler_configuration","test_unified_public"]
    with (output/"tests.log").open("x") as log:
        tests=unittest.TextTestRunner(stream=log,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromNames(modules))
    summary=dict(run=tests.testsRun,passed=tests.testsRun-len(tests.failures)-len(tests.errors)-len(tests.skipped),
                 failed=len(tests.failures),errors=len(tests.errors),skipped=len(tests.skipped))
    dump(output/"tests.json",summary)
    if not tests.wasSuccessful():raise ValueError("Regression suite failed")
    # Frozen public requests are reconstructed but not executed or sent to a provider.
    index=RESULTS/"stage2-agent-campaign-v7-r130/index.json";count=0
    for part in json.loads(index.read_text())["shards"]:
        file=index.parent/part["file"];assert sha(file)==part["sha256"]
        for row in json.loads(file.read_text())["cases"]:
            assert reconstruct_request(row["request"])==row["request"];count+=1
    dump(output/"request-compatibility.json",dict(checked=count,changed=0,index_sha256=sha(index)))
    cases=[]
    def add(name,model,source,options=None,provenance=None):
        request=prepare_task(model,options or {})
        cases.append(dict(name=name,request=request,candidate=dict(schema=1,request_id=request["request_id"],
                     hecate_source=source),model_import=str(provenance) if provenance else None))
    b=Builder([(2,)]);square=b.finish(b.node("square",["input0"]))
    add("native",square,'@hc.func("c,c")\ndef golden(x,zero_ct):\n    return [x*x]\n')
    add("public-directed",square,'@hc.func("c,c")\ndef golden(x,zero_ct):\n    return list([x*x])\n',
        dict(construction_profile="hecate-unified-public-v1",construction="unified-public-call-list",generation_guidance="explicit-v5"))
    b=Builder([(2,),(3,)]);model=b.finish(b.node("square",["input0"]),b.node("square",["input1"]))
    add("multioutput",model,'@hc.func("c,c,c")\ndef golden(x,y,zero_ct):\n    return [x*x,y*y]\n')
    b=Builder([(6,)]);model=b.finish(b.node("square",["input0"]))
    add("chunk",model,'@hc.func("c,c,c")\ndef golden(x,y,zero_ct):\n    return [x*x,y*y]\n',dict(chunk_period=4))
    from upstream_bn_candidate_cases import cases as bn_cases
    bn=next(v for v in bn_cases() if v["name"]=="bn_multi_residual")
    add("actual-helper",bn["model"],bn["source"],dict(helper_profile="upstream-poly-bn-silu-v2",
        helper_exercise=["HE_BN0","HE_BN1"]))
    imported=output/"mlp-import"
    cmd=[str(VENV/"bin/python"),"-B",str(ROOT/"scripts/prepare_model.py"),"--inside",
         "--python-manifest",str(BASE/"cases/operator-decomposition-v1/mlp-python.json"),"--write","--output",str(imported)]
    with (output/"model-import.log").open("x") as log:
        subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=120)
    model=json.loads((imported/"model.json").read_text())
    from unified_graph_lowering import candidate_source
    req=prepare_task(model);source,_=candidate_source(req)
    add("mlp-provenance",model,source,provenance=imported)
    from unified_graph_contract import prepare, COMPACT
    from compiler_configuration import PROFILE_SHA256, configuration
    compact=prepare(square,PROFILE_SHA256,configuration("seal-cpu-eva-w45-v1"),constant_policy=COMPACT)
    cases.append(dict(name="explicit-compact",request=compact,candidate=dict(schema=1,
        request_id=compact["request_id"],hecate_source='@hc.func("c,c")\ndef golden(x,zero_ct):\n    return [x*x]\n'),model_import=None))
    dump(output/"plan.json",dict(manual_fixtures=True,new_paid_calls=0,cases=cases,max_native_concurrency=1,
         max_executions=16,per_execution_seconds=300,source_hashes=before))
    rows=[]
    for case in cases:
        frozen();folder=output/case["name"];folder.mkdir()
        job=dict(format=VERSION,action="qualify",request=case["request"],candidate=case["candidate"])
        dump(folder/"job.json",job)
        t=time.monotonic()
        with (folder/"worker.log").open("x") as log:
            run=subprocess.run([str(VENV/"bin/python"),"-B",str(ROOT/"scripts/agent_component.py"),
                 "--job",str(folder/"job.json")],cwd=folder,stdout=subprocess.PIPE,stderr=log,text=True,timeout=300)
        try:result=json.loads(run.stdout)
        except ValueError:
            (folder/"worker.stdout").write_text(run.stdout);raise
        dump(folder/"worker-result.json",result)
        row=dict(case=case["name"],first=result,status="qualified_not_exported" if result["status"]=="passed" else result["status"],seconds=time.monotonic()-t)
        rows.append(row);dump(output/"progress.json",dict(rows=rows))
        if run.returncode!=0:continue
        evidence=Path(result["evidence"])
        manifest=export_bundle(evidence,folder/"export",model_import=Path(case["model_import"]) if case["model_import"] else None)
        shutil.copytree(folder/"export",folder/"relocated bundle")
        check_bundle(folder/"relocated bundle")
        with (folder/"replay.log").open("x") as log:
            replay=subprocess.run([str(VENV/"bin/python"),"-B",str(ROOT/"scripts/dsl_bundle.py"),
                "replay","--bundle",str(folder/"relocated bundle"),"--execute"],
                cwd=folder,stdout=log,stderr=subprocess.STDOUT,timeout=300)
        text=(folder/"replay.log").read_text().splitlines()
        terminal=next((json.loads(line) for line in reversed(text) if line.startswith('{"bundle_binding"')),None)
        row.update(bundle_binding=manifest["binding"],replay=terminal,
                   status="passed" if replay.returncode==0 and terminal and terminal["numerically_validated"] else "replay_failed",
                   seconds=time.monotonic()-t)
        dump(output/"progress.json",dict(rows=rows))
    # Exercise the retained v1 native replay path with an explicit old-format manifest.
    native=output/"native";v2=json.loads((native/"export/manifest.json").read_text())
    legacy=output/"legacy-v1";legacy.mkdir()
    for name in ("model.json","candidate.py"):shutil.copyfile(native/"export"/name,legacy/name)
    m={k:v2[k] for k in ("compiler_configuration","origin","agent_generated")}
    m.update(format="poseidon-dsl-bundle-v1",files={n:sha(legacy/n) for n in ("model.json","candidate.py")})
    m["binding"]=digest(m);dump(legacy/"manifest.json",m)
    with (output/"legacy-v1.log").open("x") as log:
        old=subprocess.run([str(VENV/"bin/python"),"-B",str(ROOT/"scripts/dsl_bundle.py"),
            "replay","--bundle",str(legacy),"--execute"],stdout=log,stderr=subprocess.STDOUT,timeout=300)
    assert old.returncode==0
    # SIGTERM during temporary key setup: child is killed, finally performs key cleanup.
    existing=set(RESULTS.glob("candidate-replay-*"));cancel_log=(output/"cancel.log").open("x")
    cancel=subprocess.Popen([str(VENV/"bin/python"),"-B",str(ROOT/"scripts/agent_component.py"),
         "--job",str(native/"job.json")],stdout=subprocess.PIPE,stderr=cancel_log,text=True)
    deadline=time.monotonic()+30;active=None
    while time.monotonic()<deadline and cancel.poll() is None:
        active=next((f for f in set(RESULTS.glob("candidate-replay-*"))-existing if (f/"private-keys").is_dir()),None)
        if active is not None:break
        time.sleep(.01)
    assert active is not None,"Cancellation stage not reached"
    cancel.terminate()
    stdout,_=cancel.communicate(timeout=30);cancel_log.close()
    cancellation=json.loads(stdout)
    assert cancel.returncode==130 and cancellation["status"]=="cancelled"
    cleanup=json.loads((active/"key-cleanup-outcome.json").read_text())
    assert cleanup["complete"]
    dump(output/"cancellation.json",dict(result=cancellation,cleanup=cleanup))
    frozen()
    report=dict(schema=1,tests=summary,unchanged_requests=count,source_hashes=before,rows=rows,
                compiler_files_unchanged=len(guard),legacy_v1_replay="passed",cancellation="passed",new_paid_calls=0,new_agent_generation=False,
                passed=sum(r["status"]=="passed" for r in rows),failed=sum(r["status"]!="passed" for r in rows))
    report["binding"]=digest(report);dump(output/"report.json",report)
    print(json.dumps({k:v for k,v in report.items() if k not in ("rows","source_hashes")}))
    return bool(report["failed"])
if __name__=="__main__":raise SystemExit(main())
