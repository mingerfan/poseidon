"""Only bounded current-source capture/reference/request preparation; never live."""
import argparse, fcntl, importlib.abc, json, os, shlex, shutil, subprocess, sys, time
from pathlib import Path
from scope import *

def write(path,value):
    path=Path(path)
    with path.open("xb") as stream:
        os.chmod(path,0o600);stream.write(canonical(value))
def measure(path):
    return max(int(subprocess.check_output(["du",arg,"-s","-B1",str(path)],text=True,timeout=30).split()[0])
        for arg in ("--apparent-size","--count-links"))
def no_active():
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name)==os.getpid():continue
        try:args=(entry/"cmdline").read_bytes().split(b"\0")
        except OSError:continue
        if any(a in (os.fsencode(BASE/"candidate_worker.py"),os.fsencode(BASE/"deepseek_http_worker.py"),
            os.fsencode(BASE/"run_candidate.py")) for a in args):
            raise ValueError("active_project_execution")
def source_check(old,cut):
    rd=Reader()
    for rel,want in {**cut["production_sources"],**cut["protected_sources"]}.items():
        require(rd.hash(ROOT/rel)==want,"preparation_source_changed")
    for c in old["cases"]:
        for rel,want in c["source_files"].items():require(rd.hash(ROOT/rel)==want,"original_fixture_changed")
        require(rd.hash(ROOT/c["request_file"])==c["request_file_sha256"],"original_request_changed")
    rd.finish()
def outer():
    require(not PREPARATION.exists() and not PROPOSAL.exists(),"single_preparation_only")
    old,cut=anchors();source_check(old,cut);no_active()
    require(measure(RESULTS)<=PREP_SCOPE["global_bytes"] and shutil.disk_usage(WORK).free>=PREP_SCOPE["reserve_bytes"],
        "preparation_artifact_or_disk_boundary")
    expected=tools();require(sha(STARTUP_FREEZE)==STARTUP_FREEZE_SHA,"original_window_changed")
    started=read(STARTUP_FREEZE)["started_wall"];require(time.time()<started+600,"original_preparation_window_closed")
    lock=RESULTS/"agent-v12-source-rebind-preparation-v2.lock"
    with owned_lock(lock):
        PREPARATION.mkdir(mode=0o700)
        write(PREPARATION/"freeze.json",rebound(dict(format="poseidon-source-rebind-freeze-v2",scope=PREP_SCOPE,
            tool_hashes=expected,old_proposal_sha256=OLD_SHA,cutover_sha256=CUTOVER_SHA,
            started_wall=started,deadline_wall=started+600)))
        sys.path.insert(0,str(BASE))
        from hecate_python_env import VENV,enter_nix
        command='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join(
            [str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"])
        code=enter_nix(command,seconds=max(1,int(started+600-time.time())),keep_env=(),environment="poseidon-cpu")
        require(code==0,"preparation_worker_failed")
        require(time.time()-started<=600,"preparation_wall_boundary")
        require(expected==tools(),"preparation_tool_changed")
        source_check(old,cut)
        verify()
    return 0
class OfflineImports(importlib.abc.MetaPathFinder):
    BLOCKED={"agent_credentials","deepseek_provider","deepseek_http_worker","component_backend",
        "qualification_session","program_execution","hecate"}
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split(".")[0] in self.BLOCKED:raise RuntimeError("offline_preparation_import:"+fullname)
        return None
def inside():
    freeze=read(PREPARATION/"freeze.json");check_binding(freeze)
    require(freeze["tool_hashes"]==tools() and freeze["scope"]==PREP_SCOPE and
        freeze["old_proposal_sha256"]==OLD_SHA and freeze["cutover_sha256"]==CUTOVER_SHA,"preparation_frozen_scope")
    require(not (PREPARATION/"report.json").exists() and not PROPOSAL.exists(),"single_preparation_only")
    def boundary():
        require(time.time()<freeze["deadline_wall"],"preparation_wall_boundary")
        require(measure(PREPARATION)<=PREP_SCOPE["artifact_bytes"] and
            measure(RESULTS)<=PREP_SCOPE["global_bytes"] and shutil.disk_usage(WORK).free>=PREP_SCOPE["reserve_bytes"],
            "preparation_artifact_or_disk_boundary")
        require(freeze["tool_hashes"]==tools(),"preparation_tool_changed")
    boundary();sys.meta_path.insert(0,OfflineImports());sys.path.insert(0,str(BASE))
    from hecate_python_env import VENV
    require(os.environ.get("IN_NIX_SHELL")=="pure" and Path(sys.prefix)==VENV,"pinned_preparation_worker")
    from application_execution.context import WorkerContext
    from application_execution.backend import ContextBackend
    from application_component import Application
    from application_model_source import FORMAT as SOURCE_FORMAT
    old,cut=anchors();source_check(old,cut)
    class PrepareBackend(ContextBackend):
        def qualify(self,*a,**kw):raise RuntimeError("preparation_never_qualifies")
        def export(self,*a,**kw):raise RuntimeError("preparation_never_exports_program")
        def execute(self,*a,**kw):raise RuntimeError("preparation_never_executes")
    class PrepareApplication(Application):
        def run(self,*a,**kw):raise RuntimeError("preparation_never_generates")
    cases=[];backend_identity=None;stages=None;worker=None
    for prior in old["cases"]:
        boundary();name=prior["id"];folder=CASE_ROOTS[name]
        if name=="linear":
            task_files=list((folder/"tasks").glob("*/task.json"))
            require(len(task_files)==1 and (folder/"request.json").is_file(),"single_retained_linear_capture")
            task_path=task_files[0].parent;meta=read(task_files[0]);q=meta["request"]
            retained_worker=read(folder/"tasks/worker-context.json")
            actual_backend=retained_worker["backend_identity"];actual_stages=meta["identity"]
            require(actual_backend["tools"]==old["backend_identity"],"unchanged_low_level_sdk")
            backend_identity,stages,worker=actual_backend,actual_stages,retained_worker
        else:
            folder.mkdir(mode=0o700)
            ctx=WorkerContext.capture(folder/"tasks");ctx.persist()
            app=PrepareApplication(ctx.task_store,PrepareBackend(ctx))
            actual_backend=app.backend.identity();actual_stages=app.backend.identity_manifest()
            require(actual_backend["tools"]==old["backend_identity"],"unchanged_low_level_sdk")
            require(actual_backend==backend_identity and actual_stages==stages,"same_current_application_identity")
            source=dict(format=SOURCE_FORMAT,kind="trusted_pytorch",manifest=str(ROOT/prior["source_manifest"]),trust_local_code=True)
            task=app.submit_source(source,resource_profile=PROFILE,options=OPTIONS,
                limits=dict(generations=1,http_attempts=1,wall_seconds=min(180,max(1,int(freeze["deadline_wall"]-time.time()))),
                    api_timeout_seconds=1200,max_tokens=384000))
            state=app.prepare_source(task)
            require(state["status"]=="queued" and state["phase"]=="queued","real_source_prepare_failed")
            meta=app._checked_task_metadata(task);q=meta["request"];task_path=app._dir(task)
            write(folder/"request.json",q)
        files={str(p):sha(regular(p)) for p in folder.rglob("*") if p.is_file()}
        c=dict(id=name,source_format=prior["source_format"],source_manifest=prior["source_manifest"],
            source_files=prior["source_files"],free_generation=True,directed_construction=False,
            request_file=str(folder/"request.json"),request_file_sha256=sha(folder/"request.json"),
            request_id=q["request_id"],request_sha256=digest(q),
            historical=dict(request_id=prior["request_id"],request_file=str(ROOT/prior["request_file"]),
                request_file_sha256=prior["request_file_sha256"],qualification=prior["rule_qualification_evidence"],
                qualification_sha256=prior["rule_qualification_evidence_sha256"],current_identity=False),
            request_differences=sorted(k for k in set(q)|set(read(ROOT/prior["request_file"]))
                if q.get(k)!=read(ROOT/prior["request_file"]).get(k)),
            prepared=dict(task_path=str(task_path),evidence_files=files),
            prepared_context=meta["context"],model_provenance=meta["model_provenance"],model_source=meta["model_source"],
            current_validation_sha256=digest(meta["identity"]["validation"]),
            current_validation_sources=meta["identity"]["validation"]["sources"])
        validate_case(c,Reader(),stages)
        cases.append(c)
        report=read(task_path/"model-import/reference-report.json")
        # No model values, input arrays, reference outputs or credentials are printed.
        print(json.dumps(dict(case=name,status="prepared",request_id=q["request_id"],
            reference_probes=report["probes"],model_sha256=digest(q["model"]),
            request_sha256=digest(q),current_validation_sha256=c["current_validation_sha256"],
            changed_request_fields=c["request_differences"],evidence_files=len(files))),flush=True)
        boundary()
    write(PREPARATION/"worker-context.json",worker)
    source_check(old,cut);boundary()
    report=rebound(dict(format="poseidon-source-rebind-preparation-report-v2",status="prepared",
        scope=PREP_SCOPE,tool_hashes=tools(),seconds=time.time()-freeze["started_wall"],
        provider_calls=0,http_attempts=0,compiler_calls=0,encrypted_executions=0,
        new_agent_generation=False,case_bindings={c["id"]:digest(c) for c in cases}))
    write(PREPARATION/"report.json",report)
    proposal=rebound(dict(format=FORMAT,approved=False,approval_reason="New exact source/request/tool binding awaits explicit paid-case approval.",
        source_root=str(ROOT),branch=old["branch"],head=old["head"],
        historical=dict(path=str(OLD),sha256=OLD_SHA,binding=OLD_BINDING,validation_identity_current=False),
        cutover=dict(path=str(CUTOVER),sha256=CUTOVER_SHA,binding=cut["binding"]),
        source_hashes=cut["production_sources"],protected_sources=cut["protected_sources"],provider=PROVIDER,budget=BUDGET,
        options=OPTIONS,resource_profile=PROFILE,cases=cases,tool_hashes=tools(),
        application_backend_identity=backend_identity,backend_identity=backend_identity["tools"],
        application_stage_identity=stages,
        preparation=dict(output=str(PREPARATION),scope=PREP_SCOPE,report_sha256=sha(PREPARATION/"report.json"),
            report_binding=report["binding"],freeze_sha256=sha(PREPARATION/"freeze.json"),
            current_identity_not_qualification=True),
        disclosure=old["disclosure"],original_terminal_scope=old["original_terminal_scope"],
        validation=dict(backend="poseidon_cpu",level="numerical",new_agent_generation=True,
            no_compiler_or_runtime_changes=True,required=["source_capture","numerical_qualification",
            "new_input_execution","package_cache_reuse"],tolerance=dict(atol=1e-5,rtol=1e-4))))
    write(PROPOSAL,proposal);verify()
    print(json.dumps(dict(status="prepared",proposal=str(PROPOSAL),proposal_sha256=sha(PROPOSAL),
        binding=proposal["binding"],cases=3,source_files=706,protected_files=654,
        provider_calls=0,compiler_calls=0,encrypted_executions=0)),flush=True)
    return 0
if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare",action="store_true");mode.add_argument("--inside",action="store_true",help=argparse.SUPPRESS)
    args=parser.parse_args()
    raise SystemExit(inside() if args.inside else outer())
