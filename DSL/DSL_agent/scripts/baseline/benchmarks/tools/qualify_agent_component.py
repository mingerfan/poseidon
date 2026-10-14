"""Bounded no-API application, package and caller-input acceptance."""
import argparse,contextlib,hashlib,json,math,os,shlex,shutil,subprocess,sys,time
from pathlib import Path
ROOT=(Path(__file__).resolve().parents[6] if Path(__file__).resolve().parents[4].name == "DSL_agent" and Path(__file__).resolve().parents[5].name == "DSL" else Path(__file__).resolve().parents[4])
sys.path.insert(0,str(ROOT/"scripts/baseline"))

def select_fixtures(fixtures, requested):
    if not requested:
        return fixtures
    if len(set(requested)) != len(requested):
        raise ValueError("Duplicate case selection")
    available = {fixture["id"] for fixture in fixtures}
    if any(case not in available for case in requested):
        raise ValueError("Unknown case selection")
    selected = set(requested)
    return [fixture for fixture in fixtures if fixture["id"] in selected]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--execute",action="store_true")
    p.add_argument("--suite",choices=("component-v1","general-tensors-v1","trusted-pytorch-v2","public-weights-v1"),default="component-v1")
    p.add_argument("--case",action="append",default=[],help="Run only the named case; repeat to select several")
    p.add_argument("--source-lines",choices=("original","crlf"),default="original")
    p.add_argument("--inside",action="store_true",help=argparse.SUPPRESS)
    a=p.parse_args()
    if a.source_lines!="original" and a.suite!="trusted-pytorch-v2":p.error("Source lines require the PyTorch suite")
    from hecate_python_env import WORK,VENV,enter_nix
    from benchmark_graph import canonical,digest,require
    if a.execute and not a.inside:
        cmd='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join(
            [str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside","--execute","--suite",a.suite,"--output",str(a.output.resolve()),
             "--source-lines",a.source_lines,*[argument for case in a.case for argument in ("--case",case)]])
        return enter_nix(cmd,seconds=1850,environment="poseidon-cpu")
    from component_contract import request_options
    from benchmark_runner import strict_file
    if a.execute:
        from semantic_benchmark_execution import runtime_sources
        from poseidon_backend_identity import identity
        require(os.environ.get("IN_NIX_SHELL")=="pure" and Path(sys.prefix)==VENV,"Pinned qualification environment")
        require(not a.output.exists() and a.output.resolve().parent==(WORK/"results").resolve(),"Fresh bounded output")
        a.output.mkdir()
        sources=runtime_sources();backend=identity();started=time.monotonic()
        tool_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        paths={a.output}
        fixture_root=ROOT/"scripts/baseline/cases/trusted-pytorch-general-v2"
        fixture_sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in fixture_root.glob("*") if p.is_file()}
        def boundary():
            require(runtime_sources()==sources and identity()==backend,"Source or backend changed")
            require({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in fixture_root.glob("*") if p.is_file()}==fixture_sources,"Source fixture changed")
            require(hashlib.sha256(Path(__file__).read_bytes()).hexdigest()==tool_hash,"Qualification tool changed")
            require(time.monotonic()-started<=1800,"Batch time budget")
            require(shutil.disk_usage(WORK).free>=4*1024**3,"Free-space budget")
            require(sum(p.stat().st_size for d in paths for p in d.rglob("*") if p.is_file())<=1024**3,"Batch artifact budget")
            total=int(subprocess.check_output(["du","-sb",str(WORK/"results")],text=True,timeout=30).split()[0])
            require(total<=32*1024**3,"Global artifact budget")
        (a.output/"initial-binding.json").write_bytes(canonical(dict(source_hashes=sources,backend_identity=backend,fixture_sources=fixture_sources,developer_sha256=tool_hash)))
        boundary()
    fixtures=[]
    if a.suite=="component-v1":
        historical=ROOT/"docs/baseline/poseidon-whole-models-agent-acceptance-r453.json"
        old=json.loads(historical.read_text())
        for row in old["rows"]:
            if row["id"] not in ("mlp","rmsnorm"):continue
            package=Path(row["program"])
            request=strict_file(package/"request.json",131072)
            source=(package/"candidate.py").read_text()
            source_hash=hashlib.sha256(source.encode()).hexdigest()
            report=Path(row["evidence"])/"report.json"
            require(hashlib.sha256(report.read_bytes()).hexdigest()==row["report_sha256"],"Historical report changed")
            original=json.loads(report.read_text())
            passed=next(x for x in original["attempts"] if x["status"]=="passed")
            retained=Path(row["evidence"])/("attempt-%02d"%passed["index"])/"candidate.py"
            require(retained.read_text()==source,"Historical candidate changed")
            fixtures.append(dict(id=row["id"],model=request["model"],options=request_options(request),
                                 source=source,source_sha256=source_hash,origin="historical_real_agent_unchanged",
                                 original_report=str(report),original_report_sha256=row["report_sha256"],
                                 original_source=str(retained)))
        from benchmark_suite import Builder
        from component_contract import prepare_task
        b=Builder([(5,),(5,)])
        add=b.node("add",["input0","input1"])
        diff=b.node("subtract",["input0","input1"])
        graph=b.finish(add,diff)
        req=prepare_task(graph,dict(chunk_period=4))
        source='@hc.func("c,c,c,c,c")\ndef golden(x,y,z,t,zero_ct):\n    return [x+z,y+t,x-z,y-t]\n'
        fixtures.append(dict(id="chunked_multi_io",model=graph,options=dict(chunk_period=4),
                             source=source,source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                             origin="rule_baseline"))
    elif a.suite=="general-tensors-v1":
        from general_tensor_cases import models
        from component_contract import prepare_task
        if a.execute:
            from unified_graph_lowering import candidate_source
        for graph in models():
            options=dict(chunk_period=4,generation_guidance="explicit-v10")
            request=prepare_task(graph,options)
            source=candidate_source(request)[0] if a.execute else None
            fixtures.append(dict(id=graph["id"],model=graph,options=options,
                source=source,source_sha256=hashlib.sha256(source.encode()).hexdigest() if source is not None else None,
                request_id=request["request_id"],origin="rule_baseline"))
    elif a.suite=="public-weights-v1":
        from public_weight_cases import models,period
        from component_contract import prepare_task
        from public_weight_context import POLICY
        from deepseek_provider import public_request
        from candidate_contract import request_file_bytes
        graphs=models()
        probe=graphs[-1]
        probe_request=prepare_task(probe,dict(chunk_period=period(probe),generation_guidance="explicit-v10",public_weight_policy=POLICY))
        probe_context=public_request(probe_request)
        admission_probe=dict(id=probe["id"],model_sha256=digest(probe),request_id=probe_request["request_id"],
            weight_elements=10000,nonrepeating_weights=10000,
            full_model_bytes=len(canonical(probe)),trusted_request_bytes=len(request_file_bytes(probe_request)),
            generation_context_bytes=len(canonical(probe_context)),admission_passed=True,
            compiled=False,encrypted_execution=False,agent_generated=False)
        if a.execute:(a.output/"large-request-probe.json").write_bytes(canonical(admission_probe))
        for graph in graphs[:-1]:
            options=dict(chunk_period=period(graph),generation_guidance="explicit-v10",public_weight_policy=POLICY)
            request=prepare_task(graph,options)
            source=None
            imported=None
            if a.execute:
                import io,numpy as np
                from model_decomposition import FORMAT as OPERATOR
                from model_weight_package import FORMAT as WEIGHT_PACKAGE
                package_folder=a.output/"weight-packages"/graph["id"];package_folder.mkdir(parents=True)
                original=dict(graph,format=OPERATOR);original.pop("resource_profile")
                original["constants"]={}
                records={}
                for name,value in graph["constants"].items():
                    raw=io.BytesIO();np.save(raw,np.asarray(value,dtype="<f8"),allow_pickle=False)
                    content=raw.getvalue();filename=name+".npy";(package_folder/filename).write_bytes(content)
                    records[name]=dict(file=filename,sha256=hashlib.sha256(content).hexdigest(),bytes=len(content),
                                       shape=list(np.shape(value)),dtype="<f8")
                manifest=package_folder/"manifest.json"
                manifest.write_bytes(canonical(dict(format=WEIGHT_PACKAGE,resource_profile="general-tensors-v1",model=original,weights=records)))
                imported=a.output/"imports"/graph["id"];imported.parent.mkdir(exist_ok=True)
                command=[str(VENV/"bin/python"),"-B",str(ROOT/"scripts/prepare_model.py"),
                         "--model-package",str(manifest),"--output",str(imported),"--write","--inside"]
                subprocess.run(command,check=True,timeout=180,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
                graph=json.loads((imported/"model.json").read_text())
                request=prepare_task(graph,options)
                from unified_graph_lowering import candidate_source
                source=candidate_source(request)[0]
            fixtures.append(dict(id=graph["id"],model=graph,options=options,source=source,
                source_sha256=hashlib.sha256(source.encode()).hexdigest() if source is not None else None,
                request_id=request["request_id"],origin="rule_baseline",public_weight_policy=POLICY,
                **(dict(imported=str(imported)) if imported is not None else {})))
    else:
        from trusted_pytorch_import import prepare as prepare_import,run as capture_import
        from candidate_bundle import load_model_provenance
        from component_contract import prepare_task
        inputs=select_fixtures([dict(id="pytorch_"+name,manifest=ROOT/"scripts/baseline/cases/trusted-pytorch-general-v2"/(name+".json"))
            for name in ("mlp","conv_pool","residual","batchnorm","matrix_fusion","shape_linear")],a.case)
        for item in inputs:
            spec,source_paths=prepare_import(item["manifest"])
            manifest=item["manifest"]
            if a.source_lines=="crlf":
                raw_files={name:path.read_bytes().replace(b"\r\n",b"\n").replace(b"\n",b"\r\n")
                           for name,path in source_paths.items()}
                spec["files"]={name:hashlib.sha256(raw).hexdigest() for name,raw in raw_files.items()}
                if a.execute:
                    source_folder=a.output/"source-packages"/item["id"]
                    source_folder.mkdir(parents=True)
                    for name,raw in raw_files.items():(source_folder/name).write_bytes(raw)
                    manifest=source_folder/"manifest.json";manifest.write_bytes(canonical(spec))
            fixture=dict(id=item["id"],origin="rule_baseline",source=None,source_sha256=None,
                         trusted_manifest_sha256=digest(spec),source_hashes=spec["files"],source_lines=a.source_lines)
            if a.execute:
                boundary()
                folder=a.output/"imports"/item["id"];folder.parent.mkdir(exist_ok=True)
                capture_import(manifest,folder,resource_profile="general-tensors-v1",trusted=True)
                model=json.loads((folder/"model.json").read_text())
                provenance=load_model_provenance(folder,model)
                from unified_graph_lowering import candidate_source
                options=dict(chunk_period=4,generation_guidance="explicit-v10")
                request=prepare_task(model,options)
                source=candidate_source(request)[0]
                fixture.update(model=model,options=options,source=source,
                    source_sha256=hashlib.sha256(source.encode()).hexdigest(),request_id=request["request_id"],
                    imported=str(folder),capture_reference=provenance["reference_report"])
                boundary()
            fixtures.append(fixture)
    fixtures=select_fixtures(fixtures,a.case)
    if not a.execute:
        print(json.dumps(dict(mode="plan",cases=[{k:v for k,v in f.items() if k not in ("source","model")} for f in fixtures],
            paid_calls=0,new_agent_generation=False,seconds=1800,artifact_bytes=1024**3,native_tasks=1)))
        return 0
    from application_component import Application
    from application_backend import PoseidonBackend
    from check_poseidon_application import Replay
    from candidate_bundle import check_bundle,replay_bundle
    from program_execution import execute_program
    from numerical_results import compare
    from benchmark_math import evaluate as mathematical
    from benchmark_torch import evaluate as independent
    from semantic_benchmark_execution import runtime_sources
    from poseidon_backend_identity import identity
    import numpy as np
    plan=dict(format="poseidon-component-generalization-plan-v1",
        source_hashes=sources,backend_identity=backend,fixture_sources=fixture_sources,developer_sha256=tool_hash,suite=a.suite,
        cases=[{k:v for k,v in f.items() if k not in ("source","model")} for f in fixtures],
        limits=dict(seconds=1800,artifact_bytes=1024**3,global_bytes=32*1024**3,free_bytes=4*1024**3,
                    native_tasks=1,compile_jobs=2,link_jobs=1),paid_calls=0)
    (a.output/"plan.json").write_bytes(canonical(plan))
    rows=[]
    boundary()
    def save():
        report=dict(format="poseidon-component-generalization-acceptance-v1",binding=digest(plan),
            rows=rows,planned=len(fixtures),passed=sum(x["status"]=="passed" for x in rows),
            failed=sum(x["status"]=="failed" for x in rows),not_run=len(fixtures)-len(rows),skipped=0,
            paid_calls=0,new_agent_generation=False,seconds=time.monotonic()-started)
        (a.output/"report.json").write_bytes(canonical(report))
    save()
    app=Application(a.output/"tasks",PoseidonBackend())
    for f in fixtures:
        boundary()
        row=dict(id=f["id"],origin=f["origin"],source_sha256=f["source_sha256"],status="not_run")
        rows.append(row)
        try:
            provider=Replay(f["source"])
            task=(app.submit_import(f["imported"],options=f["options"]) if "imported" in f
                  else app.submit(f["model"],options=f["options"]))
            if "capture_reference" in f:row["capture_reference"]=f["capture_reference"]
            row["task_id"]=task
            with (a.output/(f["id"]+".log")).open("x") as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
                result=app.run(task,provider)
            row["application_result"]=result
            require(result["status"]=="succeeded","Application task did not succeed")
            package=a.output/"tasks"/result["package"]["task_id"]/"program"
            manifest=check_bundle(package)
            require(manifest["format"]==("poseidon-dsl-bundle-v6" if f.get("public_weight_policy") else "poseidon-dsl-bundle-v5"),
                    "New task must export the matching staged-identity package")
            require(hashlib.sha256((package/"candidate.py").read_bytes()).hexdigest()==f["source_sha256"],"Candidate was rewritten")
            row.update(program=str(package),package_binding=manifest["binding"],scripted_provider_deliveries=provider.calls)
            validations=list((a.output/"tasks"/task).glob("validation-*.json"))
            for path in validations:
                evidence=json.loads(path.read_text()).get("evidence")
                if evidence:paths.add(Path(evidence))
            before=set((WORK/"results").glob("poseidon-package-replay-*"))
            with (a.output/(f["id"]+"-reload.log")).open("x") as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
                require(replay_bundle(package)==0,"Package replay failed")
            added=set((WORK/"results").glob("poseidon-package-replay-*"))-before
            require(len(added)==1,"Unique replay evidence")
            replay=next(iter(added));paths.add(replay)
            row["reload"]=json.loads((replay/"report.json").read_text())
            require(row["reload"]["encrypted_execution"] and row["reload"]["numerically_validated"],"Reload numerical acceptance")
            logical=[]
            for index in range(3):
                sample={}
                for ordinal,spec in enumerate(f["model"]["inputs"]):
                    shape=spec["shape"]
                    values=np.arange(math.prod(shape),dtype=np.float64).reshape(shape)
                    sample[spec["name"]]=((values%7-3)*(index+1)/20+ordinal/40).tolist()
                logical.append(sample)
            (a.output/(f["id"]+"-inputs.json")).write_bytes(canonical(dict(format="poseidon-local-inputs-v1",samples=logical)))
            execution=app.execute(task,logical)
            row["execution"]=execution
            require(execution["status"]=="executed" and execution["encrypted_execution"],"Caller-input execution")
            paths.add(Path(execution["evidence"]))
            expected=[];actual=[]
            for inputs,outputs in zip(logical,execution["outputs"]):
                arrays={k:np.asarray(v,dtype=np.float64) for k,v in inputs.items()}
                ref=mathematical(f["model"],arrays);second=independent(f["model"],arrays)
                for name in ref:np.testing.assert_allclose(second[name],ref[name],atol=1e-12,rtol=1e-12)
                expected.append(np.concatenate([ref[o["name"]].reshape(-1) for o in f["model"]["outputs"]]))
                actual.append(np.concatenate([np.asarray(outputs[o["name"]]).reshape(-1) for o in f["model"]["outputs"]]))
            comparison=compare(np.asarray(actual),np.asarray(expected),1e-5,1e-4)
            require(comparison["passed"],"Caller-input numerical comparison")
            row.update(execution=execution,new_input_comparison=comparison)
            if "capture_reference" in f:
                tree=f["capture_reference"]["capture_metadata"]["output_structure"]
                require(execution.get("output_structure")==tree and len(execution.get("structured_outputs",[]))==3,
                        "Package output structure")
            task2=(app.submit_import(f["imported"],options=f["options"]) if "imported" in f
                   else app.submit(f["model"],options=f["options"]))
            unused=Replay(f["source"]);cached=app.run(task2,unused)
            require(cached["status"]=="succeeded" and cached["package"]["reused"] and unused.calls==0,"Program reuse")
            row["reuse"]=cached
            row["status"]="passed"
        except Exception as error:
            row.update(status="failed",diagnostic=str(error)[:2000])
            save();boundary()
            outcome=row.get("execution") or row.get("application_result") or {}
            failure=outcome.get("failure") or {}
            if (outcome.get("status") in ("service_error","timed_out","cancelled","budget_exhausted")
                    or failure.get("layer") in ("integrity","environment","worker","pipeline")):
                raise
        save();boundary()
        print("COMPONENT",f["id"],row["status"],flush=True)
    save()
    return int(any(row["status"]!="passed" for row in rows))

if __name__=="__main__":raise SystemExit(main())
