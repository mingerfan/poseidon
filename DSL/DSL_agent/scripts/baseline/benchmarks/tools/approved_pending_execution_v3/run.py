"""Bounded continuation of the exact 15 + 60 + 1 approved live cases.

The application owns generation, repair, task ledgers and validation. This
developer driver only preserves group deadlines, prior consumption and evidence.
"""
import copy,hashlib,importlib.util,json,os,shutil,signal,socket,subprocess,sys,time,traceback
from pathlib import Path
ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5])
sys.path.insert(0,str(ROOT/"scripts/baseline"))
from benchmark_graph import canonical,digest,require
from semantic_benchmark_execution import runtime_sources
from application_backend import PoseidonBackend
from application_component import Application
from component_contract import prepare_task
from deepseek_provider import Config,DeepSeekProvider,HTTPSTransport,public_request
from candidate_bundle import check_bundle,replay_bundle,check_execution_dependencies
from hecate_python_env import VENV
from budget import remaining,limits,exact_certificate_stop
WORK=Path("/home/lhohy/poseidon-work/platforms/aarch64-linux/results")
PRE=WORK/"agent-pending-live-preparation-r1084"
OUT=WORK/"agent-approved-pending-continuation-r1099"
OLD=WORK/"agent-approved-pending-execution-r1088"
FIRST=WORK/"agent-approved-functional15-live-r1088-1/tasks/25fcd44a9e224a5fbd1d26e378c1d3d0"
load=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def evaluation():
    spec=importlib.util.spec_from_file_location("approved_original_evaluation",ROOT/"scripts/baseline/benchmarks/tools/run_application_matrix.py")
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def verify_parent(plan):
    require(digest({k:v for k,v in plan.items() if k!="binding"})==plan["binding"],"Continuation plan binding")
    require(runtime_sources()==plan["source_hashes"],"Original frozen production sources")
    require(all(sha(p)==h for p,h in plan["tool_hashes"].items()),"Continuation driver freeze")
    require(all(sha(p)==h for p,h in plan["retained_hashes"].items()),"Retained history unchanged")
    require(PoseidonBackend().identity()==plan["backend_identity"],"Original execution backend")
    require(int(subprocess.check_output(["du","-sb",str(WORK)],text=True,timeout=30).split()[0])<=32*1024**3,"Global artifact boundary")
    require(shutil.disk_usage(WORK).free>=4*1024**3,"Disk reserve")
def tls_check():
    code='''import importlib.util,json,ssl,socket,hashlib
from pathlib import Path
p=Path("/home/lhohy/Code/Poseidon/scripts/baseline/deepseek_http_worker.py")
s=importlib.util.spec_from_file_location("trusted_transport",p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
c=m.tls_context()
assert c.check_hostname and c.verify_mode==ssl.CERT_REQUIRED
with socket.create_connection(("api.deepseek.com",443),timeout=8) as sock:
 with c.wrap_socket(sock,server_hostname="api.deepseek.com") as con:
  cert=con.getpeercert()
  print(json.dumps(dict(status="verified",host="api.deepseek.com",issuer=cert.get("issuer"),subject=cert.get("subject"),protocol=con.version(),certificate_sha256=hashlib.sha256(con.getpeercert(binary_form=True)).hexdigest())))
'''
    env={k:os.environ[k] for k in ("LD_LIBRARY_PATH",) if k in os.environ}
    r=subprocess.run([sys.executable,"-I","-B","-c",code],env=env,text=True,capture_output=True,timeout=15)
    require(r.returncode==0,"Verified credential-free TLS required")
    return json.loads(r.stdout)
def prepare():
    require(not OUT.exists(),"Fresh continuation; never repeat uncertain outcomes")
    require(os.environ.get("IN_NIX_SHELL")=="pure" and Path(sys.prefix)==VENV,"Pinned preparation")
    authority=load(PRE/"approval.json")
    require(authority["approval_text"]=="批准当前计划内积累的所有待批准执行","Original human approval")
    require(digest({k:v for k,v in authority.items() if k!="binding"})==authority["binding"],"Original authorization binding")
    require(all(sha(p)==h for p,h in authority["tool_hashes"].items()),"Prior execution tools")
    state=load(FIRST/"state.json");require(exact_certificate_stop(state),"Only confirmed certificate stop may be continued")
    require(not (FIRST/"program").exists() and not list(FIRST.glob("candidate*")),"Original call produced no candidate")
    e=evaluation();groups={};retained={}
    for group,s in authority["groups"].items():
        retained[s["parent"]["path"]]=s["parent"]["sha256"]
        records=[]
        for p in s["plans"]:
            require(sha(p["path"])==p["sha256"],"Original plan retained")
            old=load(p["path"]);retained[p["path"]]=p["sha256"]
            if group!="attention1":
                require(e.freeze(e.items_from_plan(old))==old,"Original model, context, dependencies and backend unchanged")
            else:
                require(PoseidonBackend().identity()==old["backend_identity"],"Attention backend identity")
                item=old["cases"][0];model=load(item["model_file"]);request=prepare_task(model,item["options"])
                require(sha(item["model_file"])==item["model_file_sha256"] and digest(model)==item["model_sha256"],"Attention model identity")
                require(request["request_id"]==item["request_id"] and digest(public_request(request))==item["public_context_sha256"],"Attention original public request")
            require(1<=len(old["cases"])<=48,"Original shard limit")
            records.extend(dict(case=c,original_plan=str(p["path"]),original_binding=p["binding"],shard=Path(p["path"]).stem,case_position=n) for n,c in enumerate(old["cases"],1))
        require(len(records)==s["parent"]["cases"],"All original cases retained")
        groups[group]=dict(cases=records,max_generations=s["max_generations"],max_http_attempts=s["max_http_attempts"],
                          shared_batch_seconds=43200,deadline_wall=None)
    require([len(groups[g]["cases"]) for g in ("functional15","matrix_instance60","attention1")]==[15,60,1],"Exact approved 76 cases")
    all_ids=[r["case"]["id"] for g in groups.values() for r in g["cases"]]
    require(len(set(all_ids))==76,"Unique original contexts")
    groups["functional15"]["deadline_wall"]=load(OLD/"report.json")["groups"][0]["deadline_wall"]
    for folder in (FIRST,OLD):
        for p in folder.rglob("*"):
            if p.is_file():retained[str(p)]=sha(p)
    for p in (PRE/"approval.json",PRE/"report.json",WORK/"agent-common-pytorch-reuse-continuation-r1091/audit.json",
              WORK/"agent-current-output-package-read-only-r1090/report.json"):
        retained[str(p)]=sha(p)
    for label in ("backend","references","history","new-history","current-history"):
        p=WORK/"agent-common-pytorch-offline-prepared-r1040"/("protected-"+label+".json")
        retained[str(p)]=sha(p)
        for name,h in load(p).items():
            require(sha(ROOT/name)==h,"Protected library/reference/history")
            retained[str(ROOT/name)]=h
    peer=tls_check()
    plan=dict(format="poseidon-approved-live-continuation-v1",original_approval=authority["binding"],
        human_approval="批准当前计划内积累的所有待批准执行",resume_message="已恢复有效连接",
        groups=groups,source_hashes=authority["source_hashes"],backend_identity=PoseidonBackend().identity(),
        provider=e.PROVIDER,limits=e.BOUNDS,used_by_case={"pytorch_state_mlp":{"generations":1,"http_attempts":1}},
        max_total_generations=304,max_total_http_attempts=1216,prior_generation_reservations=1,prior_http_attempts=1,
        max_new_generation_reservations=303,max_new_http_attempts=1215,retained_hashes=retained,
        tool_hashes={str(p):sha(p) for p in Path(__file__).parent.glob("*.py")},tls=peer,
        no_monetary_ceiling=True,automatic_uncertain_retry=False,models_changed=False,references_changed=False,
        numeric_threshold=dict(atol=1e-5,rtol=1e-4))
    plan["binding"]=digest(plan);verify_parent(plan)
    OUT.mkdir(mode=0o700);(OUT/"plan.json").write_bytes(canonical(plan))
    print(json.dumps(dict(status="prepared",cases=76,binding=plan["binding"],tls="verified",
        functional_deadline=groups["functional15"]["deadline_wall"],new_generations=303,new_http_attempts=1215)),flush=True)
def new_inputs(app,task,package,model):
    import numpy as np
    from benchmark_math import evaluate as math_ref
    from benchmark_torch import evaluate as torch_ref
    from numerical_results import compare
    inputs=[];reference=[]
    for i in range(3):
        rng=np.random.default_rng(20261007+i)
        values={s["name"]:rng.uniform(-.2,.2,s["shape"]) for s in model["inputs"]}
        a,b=math_ref(model,values),torch_ref(model,values)
        for key in a:np.testing.assert_allclose(a[key],b[key],atol=2e-12,rtol=2e-12)
        inputs.append({k:v.tolist() for k,v in values.items()})
        reference.append(np.concatenate([a[o["name"]].reshape(-1) for o in model["outputs"]]))
    result=app.execute(task,inputs)
    require(result["status"]=="executed" and result["encrypted_execution"] and not result["executor_secret_key"],"Real new input ciphertext execution")
    require(result["agent_calls"]==result["compiler_calls"]==0,"Reuse without generation or compilation")
    actual=np.asarray([np.concatenate([np.asarray(s[o["name"]]).reshape(-1) for o in model["outputs"]]) for s in result["outputs"]])
    comparison=compare(actual,np.asarray(reference),1e-5,1e-4)
    return dict(status="passed" if comparison["passed"] else "failed",execution=result,comparison=comparison,
                input_groups=3,compiler_calls=0,agent_calls=0)
def run_group(group,deadline):
    plan=load(OUT/"plan.json");verify_parent(plan)
    target=OUT/group;require(not target.exists(),"Fresh group execution")
    target.mkdir(mode=0o700);rows=[];started=time.monotonic();e=evaluation()
    def save(status):
        gen=sum(r.get("result",{}).get("generations",0) for r in rows);http=sum(r.get("result",{}).get("http_attempts",0) for r in rows)
        old_g=old_h=1 if group=="functional15" else 0
        data=dict(format="poseidon-approved-live-group-continuation-v1",group=group,status=status,binding=plan["binding"],
          planned=len(plan["groups"][group]["cases"]),rows=rows,passed=sum(r["status"]=="passed" for r in rows),
          failed=sum(r["status"]=="failed" for r in rows),not_run=sum(r["status"] in ("not_run","running") for r in rows)+len(plan["groups"][group]["cases"])-len(rows),
          generations=gen,http_attempts=http,prior_generations=old_g,prior_http_attempts=old_h,
          cumulative_generations=gen+old_g,cumulative_http_attempts=http+old_h,deadline_wall=deadline,
          seconds=time.monotonic()-started,cost_usd=None)
        require(data["cumulative_generations"]<=plan["groups"][group]["max_generations"] and data["cumulative_http_attempts"]<=plan["groups"][group]["max_http_attempts"],"Original group call ceilings")
        (target/"report.json").write_bytes(canonical(data))
    save("running")
    from agent_credentials import load_api_key
    key=""
    try:
        for entry in plan["groups"][group]["cases"]:
            item=entry["case"];name=item["id"]
            used=plan["used_by_case"].get(name,dict(generations=0,http_attempts=0))
            bound=limits(deadline,time.time(),used["generations"],used["http_attempts"])
            row=dict(id=name,status="not_run",original_plan=entry["original_plan"],request_id=item["request_id"],prior_consumption=used,remaining_limits=bound)
            rows.append(row);save("running")
            if bound is None:
                row["reason"]="original_group_or_call_budget_exhausted";save("running");continue
            verify_parent(plan)
            case_dir=target/name;case_dir.mkdir()
            paths={case_dir}
            def boundary():
                verify_parent(plan)
                require(time.time()<deadline,"Original shared group deadline")
                require(sum(p.stat().st_size for d in paths for p in d.rglob("*") if p.is_file())<=4*1024**3,"Original per-case artifact limit")
            class BoundedBackend(PoseidonBackend):
                def qualify(self,request,candidate,level):
                    boundary();before=set(WORK.iterdir())
                    value=super().qualify(request,candidate,level)
                    paths.update(set(WORK.iterdir())-before);boundary();return value
                def export(self,*args,**kwargs):
                    boundary();super().export(*args,**kwargs);boundary()
            app=Application(case_dir/"tasks",BoundedBackend())
            if group=="attention1":
                model=load(item["model_file"]);task=app.submit(model,options=item["options"],limits=bound)
            else:
                fresh,request=e.snapshot_case({k:item[k] for k in ("id","imported_directory","options")})
                require(fresh==item,"Original approved case unchanged at call boundary")
                model=load(Path(item["imported_directory"])/"model.json")
                task=app.submit_import(item["imported_directory"],options=item["options"],limits=bound)
            row.update(task_id=task,status="running");save("running")
            if not key:key=load_api_key(ROOT,provider="deepseek")
            os.environ.pop("DEEPSEEK_API_KEY",None)
            config=Config(model="deepseek-flash",reasoning_effort="high",max_tokens=384000,max_calls=bound["generations"],
                          timeout_seconds=1200,service_provider="deepseek",stream=True,provider_retries=3,loopback_proxy_port=0)
            transport=HTTPSTransport(api_key=key,approved_request_id=item["request_id"],approved_config=config,enabled=True)
            provider=DeepSeekProvider(config,transport=transport)
            before=set(WORK.iterdir());result=app.run(task,provider);paths.update(set(WORK.iterdir())-before)
            row.update(result=result,status="failed");save("running")
            if result["status"]=="succeeded":
                require(result["acceptance"]["level"]=="numerical" and result["acceptance"]["encrypted_execution"] and result["acceptance"]["numerically_validated"],"Numerical acceptance retained")
                package=case_dir/"tasks"/task/"program";manifest=check_bundle(package)
                require(manifest["agent_generated"] and result["http_attempts"]>0 and not result["package"]["reused"],"Current real Agent generation")
                boundary();before=set(WORK.iterdir());require(replay_bundle(package)==0,"Real package reload");paths.update(set(WORK.iterdir())-before)
                row.update(status="passed",package=str(package),package_binding=manifest["binding"],reloaded=True)
                save("running")
                if group!="attention1":
                    _,request=check_execution_dependencies(package);require(request["request_id"]==item["request_id"],"Current package request")
                    boundary();before=set(WORK.iterdir());check=new_inputs(app,task,package,model);paths.update(set(WORK.iterdir())-before)
                    row["new_input_acceptance"]=check
                    (case_dir/"new-input-acceptance.json").write_bytes(canonical(check))
                    if check["status"]!="passed":row.update(status="failed",failure_layer="new_input_numerical")
            boundary();save("running");print(json.dumps(dict(group=group,id=name,status=row["status"],generations=result["generations"],http_attempts=result["http_attempts"])),flush=True)
            if result["status"] in ("service_error","cancelled","timed_out"):
                save("stopped");return 1
        save("complete");return 0
    except BaseException:
        save("stopped");raise
    finally:key=""
def run_all():
    plan=load(OUT/"plan.json");verify_parent(plan)
    claim=OUT/"execution-claim.json";require(not claim.exists(),"No automatic restart of execution")
    with claim.open("x") as f:json.dump(dict(pid=os.getpid(),started=time.time(),binding=plan["binding"]),f)
    groups=[];started=time.monotonic()
    def save(status):
        (OUT/"report.json").write_bytes(canonical(dict(format="poseidon-approved-live-continuation-status-v1",status=status,binding=plan["binding"],
          groups=groups,planned=76,prior_generations=1,prior_http_attempts=1,automatic_restart=False,seconds=time.monotonic()-started)))
    save("running")
    try:
        for name in ("functional15","matrix_instance60","attention1"):
            verify_parent(plan);deadline=plan["groups"][name]["deadline_wall"]
            if deadline is None:deadline=time.time()+43200
            group=dict(id=name,status="running",deadline_wall=deadline);groups.append(group);save("running")
            log=OUT/(name+".log")
            with log.open("x") as f:
                child=subprocess.Popen([sys.executable,"-B",str(Path(__file__).resolve()),"--group",name,str(deadline)],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
                try:code=child.wait(timeout=max(1,int(deadline-time.time())+3))
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid,signal.SIGTERM)
                    try:child.wait(timeout=3)
                    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                    group.update(status="stopped",exit_code=124);save("stopped");return 124
            group["exit_code"]=code
            path=OUT/name/"report.json";require(path.exists(),"Group terminal report")
            report=load(path);group.update(status=report["status"],passed=report["passed"],failed=report["failed"],not_run=report["not_run"],
                 cumulative_generations=report["cumulative_generations"],cumulative_http_attempts=report["cumulative_http_attempts"])
            save("running")
            if report["status"]!="complete" or code!=0:save("stopped");return code or 1
        save("complete");return 0
    except BaseException:save("stopped");raise
if __name__=="__main__":
    require(os.environ.get("IN_NIX_SHELL")=="pure" and Path(sys.prefix)==VENV,"Pinned live environment")
    if sys.argv[1:]==["--prepare"]:prepare()
    elif len(sys.argv)==4 and sys.argv[1]=="--group":raise SystemExit(run_group(sys.argv[2],float(sys.argv[3])))
    elif sys.argv[1:]==["--run"]:raise SystemExit(run_all())
    else:raise SystemExit("Expected --prepare or --run")
