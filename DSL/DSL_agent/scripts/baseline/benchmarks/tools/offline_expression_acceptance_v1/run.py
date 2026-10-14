"""Single-use offline qualification through the existing component API."""
import fcntl,json,os,shlex,signal,subprocess,sys,time
from pathlib import Path
import control
ROOT,BASE,HERE,WORK,RESULTS,RUN=control.ROOT,control.BASE,control.HERE,control.WORK,control.RESULTS,control.RUN
def env():
    values={k:os.environ[k] for k in ("PATH","IN_NIX_SHELL","POSEIDON_PLATFORM","POSEIDON_WORK_ROOT",
        "HECATE_PYTHON_LIBRARY_PATH","LD_LIBRARY_PATH","LANG") if k in os.environ}
    values.update(HOME="/nonexistent",PYTHONPATH=str(BASE),PYTHONDONTWRITEBYTECODE="1",PYTHONNOUSERSITE="1",
                  OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1",MKL_NUM_THREADS="1")
    return values
def frozen():
    proposal,receipt,request=control.approved_plan()
    sys.path.insert(0,str(BASE))
    from platform_config import require_platform
    control.require(require_platform()["id"]=="aarch64-linux" and
      os.environ.get("POSEIDON_WORK_ROOT")=="/home/lhohy/poseidon-work","Explicit verified ARM work root")
    branch=subprocess.check_output(["/usr/bin/git","-C",str(ROOT),"branch","--show-current"],text=True,timeout=10).strip()
    head=subprocess.check_output(["/usr/bin/git","-C",str(ROOT),"rev-parse","HEAD"],text=True,timeout=10).strip()
    control.require(branch==proposal["branch"] and head==proposal["head"],"Original branch and HEAD")
    from poseidon_backend_identity import identity
    control.check_backend(identity(),control.expected_backend(proposal))
    return proposal,receipt,request
def launch():
    proposal,receipt,request=frozen()
    control.fresh_result()
    sys.path.insert(0,str(HERE.parent/"staged_binding_feedback_v1"))
    import guarded_apply
    control.require(not guarded_apply.active_processes(),"Existing batch must reach a safe boundary")
    with (WORK/"deps/nix-portable-v012/launcher.lock").open("r+") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        fcntl.flock(lock,fcntl.LOCK_UN)
    initial=control.measure(RESULTS)
    import shutil
    control.require(initial<=control.SCOPE["global_bytes"] and
      shutil.disk_usage(WORK).free>=control.SCOPE["reserve_bytes"],"Initial resource boundary")
    entries=sorted(p.name for p in RESULTS.iterdir())
    started=time.time()
    plan=dict(format="poseidon-manual-expression-runtime-plan-v1",proposal_sha256=control.sha(control.PROPOSAL),
      approval_sha256=control.sha(control.APPROVAL),driver_hashes=control.local_driver_hashes(),
      started_wall=started,deadline_wall=started+control.SCOPE["wall_seconds"],
      scope=dict(control.SCOPE),initial_entries=entries,initial_bytes=initial,
      host_network_namespace=str(Path("/proc/self/ns/net").readlink()))
    plan["binding"]=control.digest(plan)
    RUN.mkdir(mode=0o700)
    control.write(RUN/"plan.json",plan)
    control.write(RUN/"approval.json",receipt)
    snapshot=RUN/"tool-snapshot";snapshot.mkdir()
    for name,want in plan["driver_hashes"].items():
        raw=(HERE/name).read_bytes();control.require(control.sha(HERE/name)==want,"Driver snapshot identity")
        (snapshot/name).write_bytes(raw)
    command=[sys.executable,"-B",str(Path(__file__).resolve()),"--nix-worker",str(RUN)]
    with (RUN/"controller.log").open("xb") as log:
        child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                               start_new_session=True,close_fds=True)
        code=None;reason=None
        try:
            while child.poll() is None:
                control.check_plan(plan);control.resource_boundary(plan)
                try:child.wait(timeout=min(2,max(.05,plan["deadline_wall"]-time.time())))
                except subprocess.TimeoutExpired:pass
            code=child.returncode
        except (ValueError,KeyboardInterrupt) as error:
            reason=type(error).__name__+": "+str(error)
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=5)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
            code=124 if "deadline" in str(error).lower() else 1
        finally:
            terminal=dict(format="poseidon-manual-expression-controller-terminal-v1",exit_code=code,
              reason=reason,plan_binding=plan["binding"],seconds=time.time()-started,
              provider_calls=0,http_attempts=0,production_applied=False)
            control.write(RUN/"terminal.json",terminal)
            control.check_sources(proposal)
    return code
def nix_worker(output):
    control.require(Path(output)==RUN,"Original result path")
    frozen();plan=control.read(RUN/"plan.json");control.check_plan(plan)
    from hecate_python_env import enter_nix,VENV
    command='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join(
        [str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--namespace",str(RUN)])
    return enter_nix(command,seconds=max(1,int(plan["deadline_wall"]-time.time())),keep_env=())
def spawn(arguments,log):
    control.require(arguments[0]=="/usr/bin/bwrap","Pinned Linux sandbox")
    separator=arguments.index("--")
    options=[os.fsencode(s) for s in arguments[1:separator]]
    control.require(all(b"\0" not in s for s in options),"No NUL sandbox arguments")
    fd=os.memfd_create("poseidon-expression-bwrap",os.MFD_CLOEXEC)
    try:
        with os.fdopen(os.dup(fd),"wb") as stream:stream.write(b"\0".join(options)+b"\0")
        os.lseek(fd,0,os.SEEK_SET)
        return subprocess.Popen([arguments[0],"--args",str(fd),*arguments[separator:]],
             pass_fds=(fd,),env={"PATH":"/usr/bin:/bin"},stdout=log,stderr=subprocess.STDOUT,
             start_new_session=True)
    finally:os.close(fd)
def mount_page(arguments,plan,start):
    entries=plan["initial_entries"]
    control.require(type(start) is int and 0<=start<=len(entries) and
      len(entries)==len(set(entries)) and all(isinstance(n,str) and n not in ("",".","..") and
        Path(n).name==n for n in entries),"Readonly historical inventory")
    from hecate_python_env import VENV
    tail=["--chdir",str(ROOT),"--",str(VENV/"bin/python"),"-B",str(Path(__file__).resolve())]
    capacity=(8500-len(arguments)-len(tail)-3)//3
    control.require(capacity>0,"Finite mount page")
    end=min(len(entries),start+capacity)
    result=list(arguments)
    for name in entries[start:end]:
        path=str(RESULTS/name);result+=["--ro-bind",path,path]
    result+=tail+(["--inside",str(RUN)] if end==len(entries) else ["--mount-layer",str(RUN),str(end)])
    control.require(len(result)<=8500,"Installed sandbox option bound")
    return result
def wait_namespace(arguments,log,plan):
    child=spawn(arguments,log)
    try:code=child.wait(timeout=max(.1,plan["deadline_wall"]-time.time()))
    except (subprocess.TimeoutExpired,KeyboardInterrupt):
        os.killpg(child.pid,signal.SIGKILL);child.wait();code=124
    control.check_plan(plan)
    return code
def pinned(output):
    control.require(Path(output)==RUN,"Original single-use result")
    control.approved_plan();sys.path.insert(0,str(BASE))
    from hecate_python_env import VENV
    control.require(os.environ.get("IN_NIX_SHELL")=="pure" and Path(sys.prefix)==VENV,"Actual pinned Nix worker")
    return control.read(RUN/"plan.json"),VENV
def namespace(output):
    plan,venv=pinned(output);control.check_plan(plan)
    command=["/usr/bin/bwrap","--unshare-all","--die-with-parent","--new-session","--cap-drop","ALL","--clearenv"]
    for name in ("/usr","/bin","/sbin","/lib","/lib64","/nix/store"):
        if Path(name).exists():command+=["--ro-bind",name,name]
    for name in ("/etc/ld.so.cache","/etc/localtime","/etc/nsswitch.conf","/etc/passwd","/etc/group"):
        if Path(name).exists():command+=["--ro-bind",name,name]
    command+=["--proc","/proc","--dev","/dev","--tmpfs","/tmp",
              "--ro-bind","/sys/devices/system/cpu","/sys/devices/system/cpu","--dir",str(ROOT)]
    for name in ("scripts","src","include","cmake","third_party","docs",".agents",".git"):
        path=ROOT/name
        if path.exists():command+=["--ro-bind",str(path),str(path)]
    for name in ("CMakeLists.txt","README.md","LICENSE","AGENTS.md",".gitmodules",".gitignore"):
        path=ROOT/name
        if path.is_file():command+=["--ro-bind",str(path),str(path)]
    command+=["--ro-bind",str(WORK),str(WORK)]
    locks=WORK/"cache/agent-native-slots"
    control.require(locks.is_dir() and locks==locks.resolve(),"Existing native lock directory")
    command+=["--bind",str(locks),str(locks),"--bind",str(RESULTS),str(RESULTS)]
    for key,value in env().items():command+=["--setenv",key,value]
    command=mount_page(command,plan,0)
    with (RUN/"namespace.log").open("xb") as log:return wait_namespace(command,log,plan)
def mount_layer(output,start):
    plan,_=pinned(output);control.check_plan(plan);control.resource_boundary(plan)
    control.require(not (ROOT/".env").exists(),"Credential path excluded")
    command=["/usr/bin/bwrap","--unshare-all","--die-with-parent","--new-session","--cap-drop","ALL",
             "--clearenv","--bind","/","/","--proc","/proc","--dev","/dev"]
    for key,value in env().items():command+=["--setenv",key,value]
    return wait_namespace(mount_page(command,plan,start),sys.stdout,plan)
def inside(output):
    plan,venv=pinned(output);control.check_plan(plan);control.resource_boundary(plan)
    control.require(not (ROOT/".env").exists() and "DEEPSEEK_API_KEY" not in os.environ,"Credentials excluded")
    control.require(str(Path("/proc/self/ns/net").readlink())!=plan["host_network_namespace"],"Network namespace isolated")
    control.write(RUN/"namespace-receipt.json",dict(format="poseidon-expression-namespace-v1",
      actual_nix_shell=os.environ["IN_NIX_SHELL"],network_namespace=str(Path("/proc/self/ns/net").readlink()),
      credentials_absent=True,provider_calls=0,production_applied=False))
    from process_resource_limits import limits
    with (RUN/"qualification.log").open("xb") as log:
        child=subprocess.Popen([str(venv/"bin/python"),"-B",str(Path(__file__).resolve()),"--qualify",str(RUN)],
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,close_fds=True,start_new_session=True,
            preexec_fn=limits)
        try:code=child.wait(timeout=min(control.SCOPE["qualification_seconds"],
                                      max(.1,plan["deadline_wall"]-time.time())))
        except (subprocess.TimeoutExpired,KeyboardInterrupt):
            os.killpg(child.pid,signal.SIGKILL);child.wait();code=124
    control.check_plan(plan);control.resource_boundary(plan)
    if not (RUN/"qualification.json").exists():
        control.write(RUN/"qualification-terminal.json",dict(status="timed_out" if code==124 else "service_error",
            exit_code=code,compiler_not_run_unproven=True,encrypted_not_run_unproven=True,
            provider_calls=0,production_applied=False))
        return code or 1
    result=control.read(RUN/"qualification.json")
    if result["status"]!="passed":return code or 1
    control.require(code==0 and result["validation_level"]=="numerical" and
        result["encrypted_execution"] is True and result["numerically_validated"] is True,
        "Actual numerical qualification required")
    from candidate_bundle import export_bundle,check_bundle
    from application_backend import PoseidonBackend
    evidence=Path(result["evidence"]);report=control.read(evidence/"report.json")
    control.require(report["agent_calls"]==0 and len(report["attempts"])==1 and
       report["provider"]=="scripted_replay","One offline candidate; no provider call")
    import numpy as np
    with np.load(evidence/"arrays.npz",allow_pickle=False) as arrays:
        control.require(arrays["reference"].shape[0]==4,"Original four reference groups")
    bundle=RUN/"program"
    export_bundle(evidence,bundle,validation_level="numerical",
                  application_identities=PoseidonBackend().identity_manifest())
    manifest=check_bundle(bundle)
    control.require(manifest["agent_generated"] is False and manifest["validation"]["level"]=="numerical",
                    "Manual candidate stays non-Agent numerical package")
    control.write(RUN/"package-reload.json",dict(binding=manifest["binding"],readonly_check=True,
        validation_level="numerical",agent_generated=False,new_input_execution=False,cache_execution=False))
    control.write(RUN/"report.json",dict(format="poseidon-manual-expression-offline-report-v1",
        status="passed",plan_binding=plan["binding"],case="ordered_common_boolean",
        qualification=result,package_binding=manifest["binding"],fixed_groups=4,
        candidate_origin="manual_selected_rule_variant",original_agent_candidate_unchanged=True,
        original_failure_retained=True,provider_calls=0,http_attempts=0,production_applied=False))
    return 0
def qualification(output):
    plan,_=pinned(output);control.check_plan(plan);control.resource_boundary(plan)
    control.require((RUN/"namespace-receipt.json").is_file(),"Qualified isolated parent")
    # Exclusive file is the consumed compiler/native attempt reservation.
    control.write(RUN/"qualification-started.json",dict(plan_binding=plan["binding"],started_wall=time.time(),
                  compiler_attempt_limit=1,encrypted_phase_limit=1))
    proposal,_,request=control.approved_plan()
    candidate=dict(schema=1,request_id=request["request_id"],hecate_source=Path(proposal["entry"]["candidate"]).read_text())
    from component_backend import qualify
    result=qualify(request,candidate,validation_level="numerical")
    control.write(RUN/"qualification.json",result)
    return result["exit_code"]
def main():
    args=sys.argv[1:]
    if args==["--plan"]:
        proposal=control.read(control.PROPOSAL);control.check_sources(proposal)
        approved=control.APPROVAL.is_file()
        if approved:control.approved_plan()
        print(json.dumps(dict(status="ready" if approved else "approval_required",proposal_binding=control.BINDING,
             approval=str(control.APPROVAL),result=str(RUN),scope=control.SCOPE,production_applied=False,
             compiler_calls=0,encrypted_executions=0,provider_calls=0)))
        return 0
    if args==["--run"]:return launch()
    if len(args)==2 and args[0] in ("--nix-worker","--namespace","--inside","--qualify"):
        return {"--nix-worker":nix_worker,"--namespace":namespace,"--inside":inside,"--qualify":qualification}[args[0]](args[1])
    if len(args)==3 and args[0]=="--mount-layer":return mount_layer(args[1],int(args[2]))
    raise SystemExit("Usage: run.py --plan | --run")
if __name__=="__main__":
    try:raise SystemExit(main())
    except (ValueError,FileExistsError,BlockingIOError) as error:
        print(json.dumps(dict(status="control_rejected",diagnostic=str(error),provider_calls=0)),file=sys.stderr)
        raise SystemExit(2)
