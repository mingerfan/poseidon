"""Developer-only joint cutover; never an application request dependency."""
import argparse,fcntl,hashlib,json,os,shutil,subprocess,sys,time,uuid
from pathlib import Path
ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5])
HERE=Path(__file__).resolve().parent
FEEDBACK=HERE.parent/"staged_binding_feedback_v1"
ORDERED=HERE.parent/"staged_ordered_intake_v1"
WORK=Path("/home/lhohy/poseidon-work/platforms/aarch64-linux")
sys.path.insert(0,str(FEEDBACK))
import guarded_apply as existing_guard
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def require(condition,message):
    if not condition:raise ValueError(message)
def validate_receipt(report,sandbox,expected):
    require(report.get("status")=="passed" and report.get("tests")==251,"Complete joint regression")
    for field in ("failed","errors","skipped","provider_calls","compilations","encrypted_executions"):
        require(type(report.get(field)) is int and report[field]==0,"No missing or additional execution claims")
    require(report.get("production_applied") is False and sandbox.get("production_applied") is False,"Staging acceptance only")
    require(sandbox.get("exit_code")==0 and sandbox.get("freeze_before")==sandbox.get("freeze_after"),"Sandbox acceptance/freeze")
    require(sandbox.get("clearenv") is True and sandbox.get("unshare_all") is True,"Isolation acceptance")
    for relative,want in expected.items():
        require(sandbox["source_hashes"].get(relative)==want,"Exact tested joint source")
    rows=report.get("application_submissions")
    require(type(rows) is list and len(rows)==6 and
            all(type(r) is dict and r.get("task_status")=="queued"
                and type(r.get("generations")) is int and type(r.get("http_attempts")) is int
                and r["generations"]==r["http_attempts"]==0
                and r.get("encrypted_execution") is False and r.get("public_context_verified") is True
                for r in rows),"Bounded application intake coverage")
    imports=[r for r in rows if "source_api" not in r]
    source=[r for r in rows if r.get("source_api")=="snapshot_then_admit_checked_capture"]
    cases={"linear","norm","split_weight"}
    require(len(imports)==len(source)==3 and {r.get("case") for r in imports}==cases
            and {r.get("case") for r in source}==cases,"Independent import and source admission coverage")
    require(all(r.get("source_format")=="poseidon-trusted-pytorch-v12" and
                r.get("generation_guidance")=="explicit-v11" for r in imports),"Tested import formats")
    require(all(r.get("capture_supervisor_executed") is False for r in source),
            "Source supervisor remains separately pending")
    require(report.get("runtime")==dict(python=[3,10,14],torch="2.0.1",numpy="1.25.2"),"Pinned tested runtime")
def validate_preflight(report,sandbox,expected):
    require(report.get("format")=="poseidon-joint-post-cutover-request-preflight-v1" and
            report.get("status")=="passed" and report.get("models")==3,"Three-case request preflight")
    require(report.get("runtime")==dict(python=[3,10,14],torch="2.0.1",numpy="1.25.2"),"Pinned preflight runtime")
    require(all(type(report.get(k)) is int and report[k]==0 for k in
                ("provider_calls","compilations","encrypted_executions")),"Static preflight execution scope")
    rows=report.get("rows")
    require(type(rows) is list and len(rows)==3 and {r.get("case") for r in rows}=={"linear","norm","split_weight"},
            "Independent preflight contexts")
    for row in rows:
        require(row.get("static_status")=="static_accepted_not_executed" and
                row.get("compiled") is False and row.get("encrypted_execution") is False and
                row.get("generations")==row.get("http_attempts")==0 and
                row.get("public_context_checked") is True,"No numerical preflight claim")
        require(row.get("source_format")=="poseidon-trusted-pytorch-v12" and
                row.get("resource_profile")=="general-tensors-v1" and
                row.get("generation_guidance")=="explicit-v11" and
                row.get("public_weight_policy")=="descriptors-v1","Exact post-cutover options")
        for key in ("request_id","rule_source_sha256"):
            value=row.get(key)
            require(type(value) is str and len(value)==64 and all(c in "0123456789abcdef" for c in value),
                    "Exact preflight program binding")
    require(sandbox.get("exit_code")==0 and sandbox.get("test_scope")=="request_preflight" and
            sandbox.get("freeze_before")==sandbox.get("freeze_after") and
            sandbox.get("clearenv") is True and sandbox.get("unshare_all") is True,"Preflight isolation")
    require(all(sandbox["source_hashes"].get(rel)==want for rel,want in expected.items()),"Exact preflight targets")
def verify():
    plan=read(HERE/"plan.json")
    require(plan["stage"]=="not_applied","Joint sources already applied")
    expected={}
    for name,item in plan["source_overlays"].items():
        path=Path(item["source"])
        require(path.name==name and path.parent in (FEEDBACK/"targets",ORDERED/"targets")
                and path.is_file() and not path.is_symlink(),"Reviewed source path")
        require(sha(path)==item["sha256"],"Staged target changed")
        expected[str(path.relative_to(ROOT))]=item["sha256"]
    require(len(expected)==25,"Exact reviewed target set")
    report=Path(plan["acceptance_report"]);sandbox=Path(plan["acceptance_sandbox"])
    require(sha(report)==plan["acceptance_report_sha256"] and sha(sandbox)==plan["acceptance_sandbox_sha256"],
            "Joint acceptance changed")
    validate_receipt(read(report),read(sandbox),expected)
    preflight=plan["post_cutover_request_preflight"]
    require(sha(preflight["report"])==preflight["report_sha256"] and
            sha(preflight["sandbox"])==preflight["sandbox_sha256"],"Preflight record changed")
    validate_preflight(read(preflight["report"]),read(preflight["sandbox"]),expected)
    for rel,want in plan["original_files"].items():
        target=ROOT/rel;backup=HERE/"recovery-originals"/target.name
        require(target.is_file() and not target.is_symlink() and sha(target)==want,"Preserve current modified source")
        require(backup.is_file() and not backup.is_symlink() and sha(backup)==want,"Recovery originals")
    for name in plan["new_files"]:
        require(not (ROOT/"scripts/baseline"/name).exists(),"New source collision")
    for path in plan["protected_records"].values():
        require(all(sha(ROOT/rel)==want for rel,want in read(path).items()),"Backend/reference changed")
    require(subprocess.check_output(["git","branch","--show-current"],cwd=ROOT,text=True).strip()==plan["guard_expected_branch"],"Development branch")
    require(subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()==plan["guard_expected_head"],"Development HEAD")
    current=read(plan["production_freeze"])
    require(current["binding"]==plan["source_batch_binding"],"Original scope binding changed")
    return plan
def check():
    plan=read(HERE/"plan.json")
    if plan["stage"]!="not_applied":
        path=Path(plan["cutover_record"])
        require(path.is_file() and sha(path)==plan["cutover_record_sha256"],"Joint cutover record")
        record=read(path)
        changed=[rel for rel,want in record["targets"].items() if not (ROOT/rel).is_file() or sha(ROOT/rel)!=want]
        accepted=plan.get("representative_backend_acceptance")
        if plan["stage"]=="applied_representative_backend_verified":
            require(type(accepted) is dict and sha(accepted["report"])==accepted["report_sha256"],
                    "Representative backend receipt")
            receipt=read(accepted["report"])
            require(receipt.get("format")=="poseidon-joint-post-cutover-independent-audit-v1" and
                    receipt.get("status")=="passed" and receipt.get("passed")==3 and
                    set(receipt.get("cases",[]))=={"linear","norm","split_weight"} and
                    receipt.get("observed_compilations")==3 and receipt.get("observed_encrypted_phases")==6 and
                    receipt.get("observed_cache_reuses")==3 and receipt.get("provider_calls")==0,
                    "Complete representative backend scope")
        verified=plan["stage"]=="applied_representative_backend_verified" and not changed
        return dict(status="applied_source_changed" if changed else plan["stage"],applied=True,changed=changed,
                    post_cutover_backend_acceptance_required=not verified,
                    representative_backend_cases=accepted.get("cases",[]) if verified else [])
    verified=verify()
    boundary=existing_guard.check()
    return dict(boundary,targets=len(verified["source_overlays"]),joint_staging_verified=True,
                post_cutover_backend_acceptance_required=True)
def apply():
    result=check()
    if result["status"]!="ready":return result
    with (WORK/"deps/nix-portable-v012/launcher.lock").open("r+") as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        require(check()["status"]=="ready","Cutover boundary changed")
        plan=verify();recovery=HERE/("applied-"+time.strftime("%Y%m%dT%H%M%S",time.gmtime())+"-"+uuid.uuid4().hex[:8])
        recovery.mkdir();shutil.copytree(HERE/"recovery-originals",recovery/"originals")
        shutil.copy2(HERE/"plan.json",recovery/"plan.json")
        (recovery/"git-status-before.txt").write_text(subprocess.check_output(["git","status","--short","--branch"],cwd=ROOT,text=True))
        queue=[]
        for name,item in plan["source_overlays"].items():
            source=Path(item["source"]);target=ROOT/"scripts/baseline"/name
            temporary=target.with_name("."+name+"."+uuid.uuid4().hex+".tmp")
            with temporary.open("xb") as stream:stream.write(source.read_bytes());stream.flush();os.fsync(stream.fileno())
            os.chmod(temporary,target.stat().st_mode&0o777 if target.exists() else 0o644)
            queue.append((temporary,target,item["sha256"]))
        # Entrypoints are installed after their trusted dependencies. No active
        # batch may load any member of the queue during this guarded boundary.
        entrypoints=("pytorch_model_capture.py","pytorch_model_intake.py","candidate_bundle.py",
                     "component_contract.py","unified_graph_contract.py","validation_adapter.py",
                     "sandbox_policy.py","run_candidate.py","application_component.py")
        queue.sort(key=lambda row:(row[1].name in entrypoints,
                                   entrypoints.index(row[1].name) if row[1].name in entrypoints else row[1].name))
        try:
            for temporary,target,want in queue:
                os.replace(temporary,target);require(sha(target)==want,"Installed source integrity")
        except BaseException:
            (recovery/"incomplete.txt").write_text("Exact local recovery originals retained; do not reset the checkout.\n")
            raise
        record=dict(format="poseidon-agent-joint-cutover-v1",source_batch=plan["source_batch_binding"],
            targets={str(target.relative_to(ROOT)):want for _,target,want in queue},recovery=str(recovery),
            acceptance_report_sha256=plan["acceptance_report_sha256"],provider_calls=0,compilations=0,
            encrypted_executions=0,backend_changes=False,post_cutover_backend_acceptance_required=True,
            historical_evidence_unchanged=True)
        record_path=recovery/"record.json";record_path.write_text(json.dumps(record,indent=2,sort_keys=True)+"\n")
        plan.update(stage="applied_pending_backend_acceptance",cutover_record=str(record_path),
                    cutover_record_sha256=sha(record_path))
        (HERE/"plan.json").write_text(json.dumps(plan,indent=2,sort_keys=True)+"\n")
        previous=read(FEEDBACK/"plan.json")
        previous.update(stage="applied_in_joint_cutover_pending_acceptance",cutover_record=str(record_path))
        (FEEDBACK/"plan.json").write_text(json.dumps(previous,indent=2,sort_keys=True)+"\n")
        ordered=read(ORDERED/"integration.json")
        ordered.update(stage="applied_pending_backend_acceptance",production_route_registered=True,
                       cutover_record=str(record_path),actual_fhe_verification="pending",agent_quality_evaluation="pending")
        (ORDERED/"integration.json").write_text(json.dumps(ordered,indent=2,sort_keys=True)+"\n")
        return dict(status=plan["stage"],applied=True,record=str(record_path),
                    post_cutover_backend_acceptance_required=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--apply",action="store_true")
    args=p.parse_args();print(json.dumps(apply() if args.apply else check()))
