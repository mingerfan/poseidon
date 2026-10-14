"""Bounded serial Poseidon acceptance for wide-period layouts and the original frozen ResNet stem; no provider calls."""
import argparse,contextlib,hashlib,json,math,os,shlex,shutil,subprocess,sys,time
from pathlib import Path
ROOT=(Path(__file__).resolve().parents[6] if Path(__file__).resolve().parents[4].name == "DSL_agent" and Path(__file__).resolve().parents[5].name == "DSL" else Path(__file__).resolve().parents[4]);BASE=ROOT/"scripts/baseline";sys.path.insert(0,str(BASE))
from benchmark_graph import canonical,digest,require

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--execute",action="store_true");p.add_argument("--inside",action="store_true",help=argparse.SUPPRESS)
    p.add_argument("--case-id",action="append",default=[],help="Run only these original case IDs; defaults to all eight")
    p.add_argument("--compiler-configuration",default="poseidon-cpu-eva-w45-v1")
    a=p.parse_args()
    from compiler_configuration import configuration
    require(configuration(a.compiler_configuration)["backend"]=="poseidon_cpu","CPU compiler configuration required")
    manifest=BASE/"cases/capacity-wide-v1/manifest.json"
    cases=[(manifest.parent/row["file"],row["period"]) for row in json.loads(manifest.read_text())["cases"]]
    stem=Path("/home/lhohy/poseidon-work/platforms/aarch64-linux/results/poseidon-resnet20-source-r533/stem-model.json")
    if not a.execute:
        print(json.dumps(dict(cases=[dict(id=x.stem,period=period) for x,period in cases]+[dict(id="resnet20_frozen_stem_pre_relu",period=4096)],historical_agent_regressions=["mlp","rmsnorm"],
            selected=a.case_id or "all",compiler_configuration=a.compiler_configuration,
            paid_calls=0,native_workers=1,seconds=1800,artifact_bytes=1024**3)));return 0
    from hecate_python_env import WORK,VENV,enter_nix
    if not a.inside:
        cmd='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join(
            [str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside","--execute","--output",str(a.output.resolve()),"--compiler-configuration",a.compiler_configuration,*[arg for name in a.case_id for arg in ("--case-id",name)]])
        return enter_nix(cmd,seconds=1900,environment="poseidon-cpu")
    require(os.environ.get("IN_NIX_SHELL")=="pure" and Path(sys.prefix)==VENV,"Pinned pure Python")
    require(not a.output.exists() and a.output.parent.resolve().is_relative_to(WORK/"results"),"Fresh results directory")
    from semantic_benchmark_execution import runtime_sources
    from poseidon_backend_identity import identity
    from component_contract import prepare_task
    from component_backend import qualify
    from unified_capacity_constants import CHANNEL_VERSION
    from unified_graph_contract import validate_request
    from unified_graph_lowering import candidate_source
    from candidate_bundle import export_bundle,replay_bundle
    old=json.loads((ROOT/"docs/baseline/poseidon-whole-models-agent-acceptance-r453.json").read_text())
    pending=[]
    require(hashlib.sha256(stem.read_bytes()).hexdigest()=="3b3a5ab74bf5a2aa49b675550b59173fbdd1f7f2bc3ca415a1269aaa14a9cb58","Original stem model changed")
    for f,period in [*cases,(stem,4096)]:
        model=json.loads(f.read_text());r=None
        try:
            r=prepare_task(model,dict(chunk_period=period,capacity_packing_version=CHANNEL_VERSION,generation_guidance="explicit-v10",compiler_configuration=a.compiler_configuration))
            source,strategy=candidate_source(r)
            pending.append(dict(id=model["id"],request=r,source=source,strategy=strategy,origin="rule_baseline"))
        except ValueError as error:
            pending.append(dict(id=model["id"],request=r,source=None,origin="rule_baseline",
                                preflight_diagnostic=str(error)))
    for row in old["rows"]:
        if row["status"]!="passed":continue
        program=Path(row["program"]);r=json.loads((program/"request.json").read_text());validate_request(r)
        source=(program/"candidate.py").read_text()
        require(hashlib.sha256(source.encode()).hexdigest()==row["attempts"][-1]["source_sha256"],"Original Agent source mismatch")
        pending.append(dict(id="historical_"+row["id"],request=r,source=source,
            origin="historical_real_agent_unchanged",original_evidence=row["evidence"],
            original_source_sha256=row["attempts"][-1]["source_sha256"]))
    require(len(pending)==8,"Exact bounded task pool")
    if a.case_id:
        require(len(set(a.case_id))==len(a.case_id) and set(a.case_id)<={x["id"] for x in pending},"Unknown or duplicate selected case")
        pending=[x for x in pending if x["id"] in a.case_id]
    require(1<=len(pending)<=8,"Bounded selected task count")
    a.output.mkdir();frozen=runtime_sources();backend=identity();started=time.monotonic()
    own_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    files={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f,period in [*cases,(stem,4096)]}
    plan=dict(format="poseidon-wide-acceptance-plan-v1",
        tasks=[{k:v for k,v in x.items() if k not in ("request","source")}|
               dict(request_id=x["request"]["request_id"] if x["request"] is not None else None,
                    source_sha256=hashlib.sha256(x["source"].encode()).hexdigest() if x["source"] is not None else None) for x in pending],
        files=files,source_hashes=frozen,backend_identity=backend,developer_sha256=own_sha,
        compiler_configuration=a.compiler_configuration,selected_case_ids=a.case_id,
        limits=dict(seconds=1800,artifact_bytes=1024**3,global_bytes=32*1024**3,free_bytes=4*1024**3,native_workers=1,compile_jobs=2,link_jobs=1),paid_calls=0)
    (a.output/"plan.json").write_bytes(canonical(plan))
    rows=[dict(id=x["id"],status="not_run",origin=x["origin"]) for x in pending];paths={a.output}
    def usage():return sum(f.stat().st_size for d in paths for f in d.rglob("*") if f.is_file())
    def boundary():
        require(runtime_sources()==frozen and identity()==backend,"Backend/source drift")
        require(hashlib.sha256(Path(__file__).read_bytes()).hexdigest()==own_sha,"Developer drift")
        require(all(hashlib.sha256(Path(n).read_bytes()).hexdigest()==h for n,h in files.items()),"Model drift")
        require(time.monotonic()-started<=1800 and shutil.disk_usage(WORK).free>=4*1024**3,"Time/free-space budget")
        require(usage()<=1024**3,"Batch artifact budget")
        total=int(subprocess.check_output(["du","-sb",str(WORK/"results")],text=True,timeout=30).split()[0])
        require(total<=32*1024**3,"Global artifact budget")
    def save():
        report=dict(format="poseidon-wide-acceptance-v1",binding=digest(plan),rows=rows,
            passed=sum(x["status"]=="passed" for x in rows),failed=sum(x["status"]=="failed" for x in rows),
            not_run=sum(x["status"]=="not_run" for x in rows),skipped=0,paid_calls=0,new_agent_generation=False,
            input_groups=sum(x.get("input_groups",4 if x["status"]=="passed" else 0) for x in rows),
            seconds=time.monotonic()-started,artifact_bytes=usage(),
            compared_values=sum(x.get("compared_values",0) for x in rows),
            max_absolute_error=max((x.get("max_absolute_error",0) for x in rows),default=0),
            reloaded_packages=sum("reload_evidence" in x for x in rows),
            reload_values=sum(x.get("reload_comparison",{}).get("compared_values",0) for x in rows),
            reload_max_absolute_error=max((x.get("reload_comparison",{}).get("max_absolute_error",0) for x in rows),default=0))
        (a.output/"report.json").write_bytes(canonical(report))
    save()
    for i,task in enumerate(pending):
        boundary();row=rows[i];folder=a.output/task["id"];folder.mkdir()
        if task["source"] is None:
            row.update(status="failed",layer="static_check",diagnostic=task["preflight_diagnostic"],
                       compiled=False,encrypted_execution=False)
            save();boundary();print("WIDE",task["id"],row["status"],flush=True);continue
        r=task["request"];candidate=dict(schema=1,request_id=r["request_id"],hecate_source=task["source"])
        (folder/"request.json").write_bytes(__import__("candidate_contract").request_file_bytes(r))
        (folder/"candidate.py").write_text(task["source"])
        try:
            result=qualify(r,candidate);row.update(result);row["origin"]=task["origin"];paths.add(Path(result["evidence"]))
            if result["status"]=="passed":
                package=folder/"program";manifest=export_bundle(Path(result["evidence"]),package)
                before=set((WORK/"results").glob("poseidon-package-replay-*"))
                with (folder/"reload.log").open("x") as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
                    require(replay_bundle(package)==0,"Package numerical replay")
                added=set((WORK/"results").glob("poseidon-package-replay-*"));added-=before
                require(len(added)==1,"Unique package replay")
                replay=next(iter(added));paths.add(replay);rr=json.loads((replay/"report.json").read_text())
                require(rr["status"]=="passed" and rr["encrypted_execution"] and rr["agent_calls"]==rr["compiler_calls"]==0,"Verified package replay")
                row.update(program=str(package),package_binding=manifest["binding"],reload_evidence=str(replay),
                    reload_comparison=rr["comparison"],package_generation_calls=0,package_compiler_calls=0)
            else:row["status"]="failed"
        except Exception as e:
            row.update(status="failed",diagnostic=str(e)[:2000])
            if any(s in str(e) for s in ("drift","budget","Pinned")):save();raise
        save();boundary();print("WIDE",task["id"],row["status"],flush=True)
    return 0 if all(r["status"]=="passed" for r in rows) else 1
if __name__=="__main__":sys.exit(main())
