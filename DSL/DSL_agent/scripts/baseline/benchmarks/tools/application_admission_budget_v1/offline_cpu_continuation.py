"""Resume only the collector checks and not-yet-run scope of r1382."""
import os,sys,json,time,shlex,hashlib,subprocess,shutil
from pathlib import Path
ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5])
sys.path.insert(0,str(ROOT/"scripts/baseline"))
sys.path.insert(0,str(Path(__file__).resolve().parent))
import offline_cpu as original
from benchmark_graph import digest,require
from hecate_python_env import WORK,VENV,enter_nix
APPENDIX=ROOT/"docs/baseline/agent-application-control-offline-continuation-r1382.json"
def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):return original.write(path,value)
def verify():
    plan=read(original.PROPOSAL);original.verify(plan);v=read(APPENDIX)
    require(v["binding"]==digest({k:x for k,x in v.items() if k!="binding"}),"continuation_binding")
    require(v["original_proposal_binding"]==plan["binding"] and
            read(original.AUTH)["approved"] is True and
            read(original.AUTH)["proposal_binding"]==plan["binding"],"continuation_authority")
    require(sha(__file__)==v["tool_sha256"],"continuation_tool")
    for name,h in v["preserved_files"].items():require(sha(name)==h,"continuation_preserve_history")
    require(v["remaining_compiler_calls"]==2 and v["remaining_encrypted_calls"]==8
            and v["provider_calls"]==0 and v["scope_extended"] is False,"continuation_scope")
    return plan,v
def inside():
    import numpy as np
    from application_execution.context import WorkerContext
    from application_execution.backend import application
    from candidate_bundle import check_bundle
    from model_source_provenance import check_package as check_source,FILE as SOURCE_FILE
    from strict_json_files import strict_file
    from benchmark_graph import samples
    from operator_reference import evaluate
    from numerical_results import compare
    plan,v=verify();output=Path(plan["output"]);scope=plan["scope"]
    require(not (output/"continuation-report.json").exists(),"continuation_preserve_output")
    historical=read(output/"report.json");first=historical["rows"][0]
    started=time.monotonic();rows=[]
    report=dict(format="poseidon-application-control-offline-continuation-report-v1",
       proposal_binding=plan["binding"],continuation_binding=v["binding"],sdk_identity=plan["sdk_identity"],
       preserved_report_sha256=sha(output/"report.json"),rows=rows,provider_calls=0,http_attempts=0,
       agent_generated=False,candidate_origin=plan["candidate_origin"],compiler_reserved=1,encrypted_reserved=1,
       compiler_confirmed=0,encrypted_confirmed=0,status="running",scope_extended=False)
    def save():
        report.update(seconds=time.monotonic()-started,passed=sum(r["status"]=="passed" for r in rows),
             failed=sum(r["status"]=="failed" for r in rows),not_run=len(plan["cases"])-len(rows))
        report["binding"]=digest({k:x for k,x in report.items() if k!="binding"})
        p=output/"continuation-report.tmp";p.write_text(json.dumps(report,sort_keys=True,indent=2,allow_nan=False)+"\n")
        p.replace(output/"continuation-report.json")
    def boundary():
        require(time.time()<v["inherited_deadline_wall"],"continuation_original_wall_budget")
        require(time.monotonic()-started<=scope["wall_seconds"],"continuation_worker_budget")
        current=original.measure(WORK/"results")
        require(current-v["conservative_global_before"]<=scope["new_bytes"],"continuation_cumulative_new_bytes")
        require(current<=scope["global_bytes"] and shutil.disk_usage(WORK).free>=scope["reserve_bytes"],"continuation_disk_budget")
        require(report["compiler_reserved"]<=3 and report["encrypted_reserved"]<=9,"continuation_call_budget")
    save()
    try:
        for i,case in enumerate(plan["cases"]):
            boundary();row=dict(case=case["case"],status="running",compiled=False,encrypted_execution=False,package_reloaded=False)
            rows.append(row);save();folder=output/case["case"]
            if i:folder.mkdir()
            app=application(WorkerContext.capture(folder/"tasks"))
            source=dict(format="poseidon-model-source-request-v1",kind="trusted_pytorch",
                manifest=case["source_manifest"],trust_local_code=True)
            expected=read(case["request"])
            if i==0:
                task=first["task_id"];qualified=read(folder/"qualification.json")
                require(app.result(task)==qualified,"continuation_completed_qualification_changed")
                row["qualification_reused_from_interrupted_collector"]=True
            else:
                task=app.submit_source(source,resource_profile=case["resource_profile"],
                   options=plan["options"],admission_policy="structured-v1",
                   operational_limits=plan["request_operational_limits"],limits=plan["generation_limits"])
                status=app.prepare_source(task)
                require(status["status"]=="queued" and status["intake"]["reference_verified"] is True,"control_source_admission")
                candidate_source=Path(case["candidate"]).read_text()
                class FrozenRule:
                    kind="scripted_replay";agent_calls=0
                    def generate(self,actual,feedback=None):
                        require(actual==expected and feedback==[],"control_one_frozen_candidate")
                        return json.dumps(dict(schema=1,request_id=actual["request_id"],hecate_source=candidate_source))
                report["compiler_reserved"]+=1;report["encrypted_reserved"]+=1;save();boundary()
                qualified=app.run(task,FrozenRule());write(folder/"qualification.json",qualified)
            row["task_id"]=task;meta=app._checked_task_metadata(task);request=meta["request"]
            require(request==expected and request["request_id"]==case["request_id"],"control_unchanged_request")
            require(meta["model_provenance"]["original"]==read(case["historical_task_metadata"])["model_provenance"]["original"],"control_unchanged_reference_model")
            vp=app._dir(task)/"validation-0.json"
            if vp.exists():
                observed=app._load(vp);row["observed_stages"]=observed.get("observed_stages",{})
                row["validation_evidence"]=observed.get("evidence")
                row["compiled"]=row["observed_stages"].get("compiled") is True
                row["encrypted_execution"]=observed.get("encrypted_execution") is True
                report["compiler_confirmed"]+=int(row["compiled"])
                report["encrypted_confirmed"]+=int(row["encrypted_execution"])
            row["application_result"]=qualified
            require(qualified["status"]=="succeeded" and qualified["http_attempts"]==0 and
                    not qualified["generation"]["agent_generated"] and not qualified["package"]["reused"],"control_numerical_qualification")
            require(qualified["acceptance"]["numerically_validated"] is True and
                    qualified["acceptance"]["input_groups"]==4,"control_numerical_scope")
            require(qualified["operational_usage"]["evidence_directories"]==1 and
                    qualified["operational_usage"]["hard_filesystem_quota"] is False,"control_qualification_monitor")
            program=app._dir(task)/"program";manifest=check_bundle(program)
            require(manifest["validation"]["level"]=="numerical" and manifest["model_provenance"]=="included" and
                    manifest["origin"].get("model_source",{}).get("file")==SOURCE_FILE,"control_source_bundle")
            source_checked=check_source(program,manifest,request,meta["model_provenance"])
            require(source_checked is not None,"control_checked_source_bundle")
            frozen={name:sha(program/name) for name in manifest["files"]}
            row.update(package_reloaded=True,package=str(program),package_binding=manifest["binding"])
            probes=[{k:np.asarray(x,dtype=np.float64)*.5+.015 for k,x in batch.items()} for batch in samples(request["model"],4)[:3]]
            row["reuse"]={}
            for name,batches in (("new-inputs",probes),("repeat",probes[:1])):
                boundary();report["encrypted_reserved"]+=1;save();boundary()
                result=app.execute(task,[{k:x.tolist() for k,x in b.items()} for b in batches],
                    operational_limits=plan["execution_operational_limits"])
                write(folder/(name+".json"),result)
                report["encrypted_confirmed"]+=int(result.get("encrypted_execution") is True)
                require(result["status"]=="executed" and result["encrypted_execution"] is True and
                    result["agent_calls"]==result["compiler_calls"]==0 and not result["executor_secret_key"],"control_package_execution")
                require(result["operational_usage"]["evidence_directories"]==1,"control_execution_monitor")
                actual=np.asarray([np.concatenate([np.asarray(x[o["name"]]).reshape(-1) for o in request["model"]["outputs"]]) for x in result["outputs"]])
                reference=np.asarray([np.concatenate([evaluate(meta["model_provenance"]["original"],b)[o["name"]].reshape(-1) for o in request["model"]["outputs"]]) for b in batches])
                comparison=compare(actual,reference,1e-5,1e-4);write(folder/(name+"-comparison.json"),comparison)
                require(comparison["passed"],"control_independent_reference")
                if case["case"]=="compat_v23_named":
                    require(all(list(x)==["primary","saved"] and x["saved"][1] is None for x in result["structured_outputs"]),"control_none_output_structure")
                row["reuse"][name]=dict(groups=len(batches),values=comparison["compared_values"],
                    maximum_absolute_error=comparison["max_absolute_error"])
            require(check_bundle(program)==manifest and all(sha(program/n)==h for n,h in frozen.items()),"control_immutable_program")
            cached=app.submit_source(source,resource_profile=case["resource_profile"],
                options=plan["options"],admission_policy="structured-v1",
                operational_limits=plan["request_operational_limits"],limits=plan["generation_limits"])
            class NoGeneration:
                kind="scripted_replay"
                def generate(self,*a,**k):raise AssertionError("Cache must not generate")
            reused=app.run(cached,NoGeneration());write(folder/"cached.json",reused)
            require(reused["status"]=="succeeded" and reused["package"]["reused"] is True and
                    reused["generations"]==0 and reused["http_attempts"]==0,"control_cache_reuse")
            row.update(status="passed",cached_without_generation=True);boundary();save()
        verify();boundary();report["status"]="complete";save();return 0
    except BaseException as error:
        if rows and rows[-1]["status"]=="running":rows[-1].update(status="failed",error_type=type(error).__name__)
        report.update(status="failed",stop_reason=type(error).__name__);save()
        import traceback
        (output/"continuation-failure-private.txt").write_text(traceback.format_exc());return 1
if __name__=="__main__":
    if "--inside" in sys.argv:sys.exit(inside())
    plan,v=verify();remaining=int(v["inherited_deadline_wall"]-time.time())
    require(remaining>0,"continuation_deadline")
    command="PYTHONPATH="+shlex.quote(str(ROOT/"scripts/baseline"))+' LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '
    command+=shlex.quote(str(VENV/"bin/python"))+" -B "+shlex.quote(__file__)+" --inside"
    sys.exit(enter_nix(command,seconds=min(remaining,630),environment="poseidon-cpu"))
