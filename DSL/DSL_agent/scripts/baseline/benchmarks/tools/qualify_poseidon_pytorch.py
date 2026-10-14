"""Serial trusted-PyTorch import, Poseidon qualification and package replay."""
import argparse,contextlib,hashlib,json,math,os,shlex,shutil,subprocess,sys,time
from pathlib import Path
ROOT=(Path(__file__).resolve().parents[6] if Path(__file__).resolve().parents[4].name == "DSL_agent" and Path(__file__).resolve().parents[5].name == "DSL" else Path(__file__).resolve().parents[4])
sys.path[:0]=[str(ROOT/"scripts/baseline"),str(Path(__file__).resolve().parent)]

def load_cases(manifest_path):
    """V1 retains its frozen medium ABI; V2 selects configured capacity layouts."""
    from benchmark_graph import require
    from benchmark_runner import strict_file
    from trusted_pytorch_import import prepare
    manifest_path=Path(manifest_path)
    manifest=strict_file(manifest_path,131072)
    require(set(manifest)=={"format","cases"} and manifest["format"] in (
            "poseidon-pytorch-acceptance-list-v1","poseidon-pytorch-acceptance-list-v2")
            and type(manifest["cases"]) is list and 1<=len(manifest["cases"])<=48,"Case manifest")
    files={str(manifest_path.resolve()):hashlib.sha256(manifest_path.read_bytes()).hexdigest()}
    seen=set();cases=[]
    for item in manifest["cases"]:
        require(set(item)=={"id","file","sha256","resource_profile","chunk_period"},"Case fields")
        label=item["id"];name=item["file"]
        require(type(label) is str and label.isidentifier() and label not in seen,"Case identity")
        require(type(name) is str and Path(name).name==name and name.endswith(".json"),"Case filename")
        if manifest["format"]=="poseidon-pytorch-acceptance-list-v1":
            require(item["resource_profile"]=="medium-v1" and type(item["chunk_period"]) is int
                    and item["chunk_period"]==256,"Frozen medium layout")
        else:
            from model_resources import resolve
            from unified_capacity_layout import physical_limits
            from unified_chunk_layout import PERIODS
            require(item["resource_profile"] in ("medium-v1","resnet20-v1"),"Configured capacity profile")
            resolve(item["resource_profile"]);physical_limits(item["resource_profile"])
            require(type(item["chunk_period"]) is int and item["chunk_period"] in PERIODS,
                    "Configured packing period")
        path=manifest_path.resolve().parent/name
        require(not path.is_symlink() and hashlib.sha256(path.read_bytes()).hexdigest()==item["sha256"],"Case hash")
        spec,paths=prepare(path)
        seen.add(label);cases.append((item,path,spec))
        files[str(path)]=item["sha256"]
        for src in paths.values():files[str(src)]=hashlib.sha256(src.read_bytes()).hexdigest()
    return cases,files

def main():
    from benchmark_graph import require,canonical,digest
    from benchmark_runner import strict_file
    from trusted_pytorch_import import prepare,run as capture
    from hecate_python_env import WORK,VENV,enter_nix
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--execute",action="store_true")
    p.add_argument("--inside",action="store_true",help=argparse.SUPPRESS)
    a=p.parse_args()
    cases,files=load_cases(a.manifest)
    if not a.execute:
        print(json.dumps(dict(mode="plan",cases=[x[0] for x in cases],files=files,paid_calls=0,
                             seconds=1800,artifact_bytes=1024**3,native_workers=1)))
        return 0
    if not a.inside:
        cmd='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join(
            [str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside","--execute",
             "--manifest",str(a.manifest.resolve()),"--output",str(a.output.resolve())])
        return enter_nix(cmd,seconds=1900,environment="poseidon-cpu")
    require(os.environ.get("IN_NIX_SHELL")=="pure" and Path(sys.prefix)==VENV,"Pinned pure Python")
    require(not a.output.exists() and a.output.parent.resolve().is_relative_to(WORK/"results"),"Fresh results directory")
    from semantic_benchmark_execution import runtime_sources
    from poseidon_backend_identity import identity
    from component_contract import prepare_task
    from component_backend import qualify
    from unified_graph_lowering import candidate_source
    from candidate_bundle import export_bundle,replay_bundle
    a.output.mkdir();frozen=runtime_sources();backend=identity();started=time.monotonic()
    own_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    plan=dict(format="poseidon-pytorch-import-plan-v1",cases=[x[0] for x in cases],files=files,
              source_hashes=frozen,backend_identity=backend,developer_sha256=own_sha,
              limits=dict(seconds=1800,artifact_bytes=1024**3,global_bytes=32*1024**3,
                          free_bytes=4*1024**3,native_workers=1,compile_jobs=2,link_jobs=1),paid_calls=0)
    (a.output/"plan.json").write_bytes(canonical(plan))
    rows=[dict(id=x[0]["id"],status="not_run") for x in cases]
    paths={a.output}
    def usage():
        return sum(f.stat().st_size for d in paths for f in d.rglob("*") if f.is_file())
    def boundary():
        require(runtime_sources()==frozen and identity()==backend,"Backend/source drift")
        require(all(hashlib.sha256(Path(n).read_bytes()).hexdigest()==h for n,h in files.items()),"Input source drift")
        require(hashlib.sha256(Path(__file__).read_bytes()).hexdigest()==own_sha,"Developer tool drift")
        require(time.monotonic()-started<=1800 and shutil.disk_usage(WORK).free>=4*1024**3,"Time/free-space budget")
        require(usage()<=1024**3,"Batch artifact budget")
        total=int(subprocess.check_output(["du","-sb",str(WORK/"results")],text=True,timeout=30).split()[0])
        require(total<=32*1024**3,"Global artifact budget")
    def save():
        report=dict(format="poseidon-pytorch-import-acceptance-v1",binding=digest(plan),rows=rows,
                    passed=sum(r["status"]=="passed" for r in rows),failed=sum(r["status"]=="failed" for r in rows),
                    blocked=sum(r["status"]=="blocked" for r in rows),not_run=sum(r["status"]=="not_run" for r in rows),skipped=0,paid_calls=0,new_agent_generation=False,
                    origin="rule_baseline_from_isolated_pytorch_capture",seconds=time.monotonic()-started,
                    artifact_bytes=usage(),
                    compared_values=sum(r.get("compared_values",0) for r in rows),
                    max_absolute_error=max((r.get("max_absolute_error",0) for r in rows),default=0),
                    reload_values=sum(r.get("reload_comparison",{}).get("compared_values",0) for r in rows),
                    reload_max_absolute_error=max((r.get("reload_comparison",{}).get("max_absolute_error",0) for r in rows),default=0))
        (a.output/"report.json").write_bytes(canonical(report))
    save()
    for i,(item,path,spec) in enumerate(cases):
        row=rows[i];boundary();folder=a.output/item["id"];folder.mkdir()
        try:
            imported=folder/"imported"
            row["capture"]=capture(path,imported,resource_profile=item["resource_profile"],trusted=True)
            model=json.loads((imported/"model.json").read_text())
            req=prepare_task(model,dict(chunk_period=item["chunk_period"],generation_guidance="explicit-v10"))
            source,strategy=candidate_source(req)
            (folder/"request.json").write_bytes(__import__("candidate_contract").request_file_bytes(req))
            (folder/"candidate.py").write_text(source)
            result=qualify(req,dict(schema=1,request_id=req["request_id"],hecate_source=source))
            row.update(result);row["origin"]="rule_baseline"
            paths.add(Path(result["evidence"]))
            row.update(request_id=req["request_id"],input_elements=sum(math.prod(x["shape"]) for x in model["inputs"]),
                       output_elements=len(req["layout"]["output_selectors"]),
                       physical_inputs=len(req["layout"]["inputs"]),physical_outputs=req["layout"]["output_ciphertexts"],
                       capture_probes=16,strategy=strategy)
            boundary()
            if result["status"]=="passed":
                package=folder/"program"
                package_manifest=export_bundle(Path(result["evidence"]),package,model_import=imported)
                before=set((WORK/"results").glob("poseidon-package-replay-*"))
                with (folder/"reload.log").open("x") as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
                    require(replay_bundle(package)==0,"Package replay failed")
                added=set((WORK/"results").glob("poseidon-package-replay-*"))-before
                require(len(added)==1,"Unique package replay")
                replay=next(iter(added));paths.add(replay)
                rr=json.loads((replay/"report.json").read_text())
                require(rr["status"]=="passed" and rr["encrypted_execution"] and rr["agent_calls"]==rr["compiler_calls"]==0,
                        "Package numerical execution")
                row.update(program=str(package),package_binding=package_manifest["binding"],reload_evidence=str(replay),
                           reload_comparison=rr["comparison"],package_generation_calls=0,package_compiler_calls=0)
            else:row["status"]="failed"
        except Exception as error:
            if str(error) in ("Capacity public tensor registry budget","Capacity spatial map work budget",
                              "Capacity spatial route budget","Unified constant registry budget"):
                row.update(status="blocked",failure_layer="request_preparation",diagnostic=str(error))
            else:
                row.update(status="failed",diagnostic=str(error)[:2000])
            if any(term in str(error) for term in ["drift","Time/free-space budget","Batch artifact budget",
                                                 "Global artifact budget","Pinned","capture failed"]):
                save();raise
        save();boundary()
        print("PYTORCH",item["id"],row["status"],flush=True)
    save()
    return int(any(row["status"]!="passed" for row in rows))

if __name__=="__main__":raise SystemExit(main())
