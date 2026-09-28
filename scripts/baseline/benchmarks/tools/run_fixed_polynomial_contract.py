"""Bounded manual v10 candidate acceptance; no provider calls or SDK rebuild."""
import argparse,hashlib,json,os,re,shlex,signal,subprocess,sys,time
from pathlib import Path
BASE=Path(__file__).resolve().parents[2];ROOT=BASE.parents[1];sys.path.insert(0,str(BASE))
from benchmark_graph import digest
from benchmark_runner import dump,strict_file
from fixed_polynomial_cases import cases
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--execute",action="store_true")
    p.add_argument("--inside",action="store_true",help=argparse.SUPPRESS)
    a=p.parse_args();selected=cases()
    if len(selected)>48:raise ValueError("Shard limit")
    if not a.execute:
        print(json.dumps(dict(cases=[r["id"] for r in selected],concurrency=1,max_seconds=1800,max_retained_mib=256,paid_calls=0)));return 0
    from hecate_python_env import enter_nix,VENV
    from workspace_paths import RESULTS
    if a.output.exists() or not a.output.resolve().is_relative_to(RESULTS.resolve()):p.error("New platform result directory required")
    if not a.inside:
        return enter_nix('LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),*sys.argv[1:],"--inside"]),seconds=1860)
    if os.environ.get("IN_NIX_SHELL")!="pure" or Path(sys.prefix)!=VENV:p.error("Pinned pure Python required")
    import numpy as np,torch
    from platform_config import require_python_packages,identity
    require_python_packages(torch,np);torch.set_num_threads(1)
    from semantic_benchmark_execution import runtime_sources
    from upstream_candidate_helpers import verify_sources
    from poly_dependencies import verify
    from compiler_configuration import PROFILE_SHA256,configuration
    from unified_graph_contract import prepare,validate_candidate
    from upstream_helper_coverage import check_exercise
    from benchmark_graph import samples
    from benchmark_math import evaluate
    from benchmark_torch import evaluate as torch_reference
    from audit_unified_candidate import verify_candidate
    sources=runtime_sources();start=time.monotonic();a.output.mkdir()
    fixtures=Path(__file__).with_name("fixed_polynomial_cases.py")
    plan=dict(format="fixed-polynomial-contract-v1",cases=selected,source_hashes=sources,
              helper_sources=verify_sources(),dependencies=verify(),platform=identity(),
              runner_sha256=sha(Path(__file__)),fixtures_sha256=sha(fixtures),
              max_wall_seconds=1800,max_retained_mib=256,concurrency=1,paid_calls=0)
    plan["binding"]=digest(plan);dump(a.output/"plan.json",plan)
    (a.output/"runner.py").write_bytes(Path(__file__).read_bytes())
    (a.output/"fixed_polynomial_cases.py").write_bytes(fixtures.read_bytes())
    rows=[];folders=[]
    def boundary():
        if runtime_sources()!=sources or sha(Path(__file__))!=plan["runner_sha256"] or sha(fixtures)!=plan["fixtures_sha256"]:raise ValueError("Source drift")
        if time.monotonic()-start>=1800:raise TimeoutError("Batch budget")
        size=sum(f.stat().st_size for folder in [a.output,*folders] for f in folder.rglob("*") if f.is_file())
        if size>256*1024**2:raise ValueError("Evidence budget")
    for spec in selected:
        boundary();case=a.output/spec["id"];case.mkdir()
        dump(case/"model.json",spec["model"]);(case/"candidate.py").write_text(spec["source"])
        request=prepare(spec["model"],PROFILE_SHA256,configuration("seal-cpu-eva-w40-v1"),
                        helper_profile=spec["profile"],helper_exercise=[spec["helper"]])
        validate_candidate(dict(schema=1,request_id=request["request_id"],hecate_source=spec["source"]),request)
        witness=check_exercise(spec["source"],request);dump(case/"intervention.json",witness)
        for probe in samples(spec["model"],16):
            expected=evaluate(spec["model"],probe);actual=torch_reference(spec["model"],probe)
            for key in expected:np.testing.assert_allclose(actual[key],expected[key],rtol=1e-12,atol=1e-12)
        command=[str(VENV/"bin/python"),"-B",str(BASE/"run_candidate.py"),"--inside",
                 "--case",str(case/"model.json"),"--golden-file",str(case/"candidate.py"),
                 "--max-repairs","0","--compiler-configuration","seal-cpu-eva-w40-v1",
                 "--unified-helpers",spec["profile"],"--unified-helper-exercise",spec["helper"]]
        with (case/"run.log").open("w") as stream:
            child=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            try:code=child.wait(timeout=min(300,max(1,1800-(time.monotonic()-start))))
            except BaseException:
                if child.poll() is None:
                    os.killpg(child.pid,signal.SIGINT)
                    try:child.wait(timeout=15)
                    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=10)
                raise
        matches=re.findall(r"^Candidate evidence: (.+)$",(case/"run.log").read_text(),re.M)
        if not matches:raise ValueError("No candidate evidence; launcher/environment failure")
        folder=Path(matches[-1]).resolve()
        if folder.parent!=RESULTS.resolve():raise ValueError("Unexpected result path")
        folders.append(folder);result=strict_file(folder/"report.json",8*1024**2)
        if result["agent_calls"]!=0:raise ValueError("Unexpected provider call")
        record=dict(id=spec["id"],helper=spec["helper"],context=spec["context"],evidence=str(folder),
                    status=result["status"],exit_code=code,report_sha256=sha(folder/"report.json"),
                    reference_probes=16,agent_generated=False,
                    failure_layer=result.get("failure_layer") or next((r.get("failure_layer") for r in result.get("attempts",[]) if r.get("failure_layer")),None),
                    failures=[r.get("diagnostic") for r in result.get("attempts",[]) if r.get("status")!="passed"])
        if code==0 and result["status"]=="passed":
            audit=verify_candidate(folder)
            if audit["model_sha256"]!=spec["model_sha256"]:raise ValueError("Executed model changed")
            if audit["upstream_helper_trace"]["actual_upstream_calls"]!=1:raise ValueError("Missing actual call")
            if not audit["helper_coverage"]["finite_return_influence_checked"] or not audit["helper_coverage"]["actual_frontend_checked"]:raise ValueError("Missing return contribution")
            dump(case/"audit.json",audit);record.update(audit_sha256=sha(case/"audit.json"),comparison=audit["comparison"])
        rows.append(record);dump(a.output/"progress.json",rows);boundary()
        if record["failure_layer"] in ("environment","key_setup","seal_runtime"):raise ValueError("Environment failure")
    report=dict(format="poseidon-fixed-polynomial-candidate-acceptance-v1",plan_binding=plan["binding"],rows=rows,
                passed=sum(r["status"]=="passed" for r in rows),failed=sum(r["status"]!="passed" for r in rows),
                skipped=0,planned=len(selected),paid_calls=0,agent_generated=False,
                seconds=time.monotonic()-start,source_sha256=digest(sources))
    report["binding"]=digest(report);dump(a.output/"report.json",report)
    print(json.dumps({k:v for k,v in report.items() if k!="rows"}));return int(report["failed"]!=0)
if __name__=="__main__":raise SystemExit(main())
