"""Frozen offline scope for the reviewed common/fused Agent overlay."""
import hashlib,importlib.util,json,math,os,re,shutil,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
COMMON=HERE.parent/"staged_ordered_common_v1"
ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5]);BASE=ROOT/"scripts/baseline"
WORK=Path("/home/lhohy/poseidon-work/platforms/aarch64-linux");RESULTS=WORK/"results"
PROPOSAL=ROOT/"docs/baseline/agent-common-fused-offline-proposal-r1296.json"
APPROVAL=ROOT/"docs/baseline/agent-common-fused-offline-approval-r1296.json"
RUN=RESULTS/"agent-common-fused-runtime-r1296"
VERSION="poseidon-common-fused-offline-plan-v1"
CANDIDATE_ORIGIN="manual_selected_generic_balanced_scalar"
ALLOW_CONTINUATION=False
UNIQUE_PHASES=True
COMMON_CASES=("alpha","dropout","boolean")
FUSED_CASES=tuple(op+"_"+kind for op in ("addmm","baddbmm","addbmm")
                  for kind in ("cipher","public_right","public_bias"))
CASES=tuple("common_"+name for name in COMMON_CASES)+FUSED_CASES
PHASES=("qualification","new_inputs","cache")
SCOPE=dict(wall_seconds=3600,phase_seconds=180,artifact_bytes=4*1024**3,
 global_bytes=32*1024**3,reserve_bytes=4*1024**3,provider_calls=0,http_attempts=0,
 native_workers=1,compile_jobs=2,link_jobs=1,compiler_attempts_max=12,
 qualification_encrypted_phases_max=12,new_input_encrypted_phases_max=12,
 installations=0,downloads=0,backend_changes=0,parameter_changes=0)
def require(value,message):
    if not value:raise ValueError(message)
def canonical(value):return json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
def digest(value):return hashlib.sha256(canonical(value)).hexdigest()
def sha(path):
    p=Path(path);require(p.is_file() and not p.is_symlink(),"Regular bound file")
    return hashlib.sha256(p.read_bytes()).hexdigest()
def read(path):
    p=Path(path);require(p.is_file() and not p.is_symlink() and p.stat().st_size<=4*1024**2,"Bounded regular control JSON")
    def unique(items):
        result={}
        for k,v in items:
            require(k not in result,"Duplicate control field");result[k]=v
        return result
    return json.loads(p.read_text(),object_pairs_hook=unique,
        parse_constant=lambda _:(_ for _ in ()).throw(ValueError("Finite control JSON")))
def write(path,value):
    with Path(path).open("xb") as stream:stream.write(canonical(value))
def measure(folder):
    return max(int(subprocess.check_output(["du",arg,"-s","-B1",str(folder)],text=True,timeout=30).split()[0])
               for arg in ("--apparent-size","--count-links"))
def guard():
    spec=importlib.util.spec_from_file_location("_joint_reviewed_guard",HERE/"check.py")
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module
def fixture_for(name):
    require(name in CASES,"Reviewed case")
    if name.startswith("common_"):return COMMON/"fixtures"/name[len("common_"):]
    return RESULTS/"agent-fused-matrix-intake-r1268/source-fixtures"/name
def case_source(name):
    from application_model_source import FORMAT
    return dict(format=FORMAT,kind="trusted_pytorch",manifest=str(fixture_for(name)/"manifest.json"),trust_local_code=True)
def retained_request(name):
    if name.startswith("common_"):
        return RESULTS/"agent-ordered-common-runtime-r1283"/name[len("common_"):]/"qualification-binding.json"
    return RESULTS/"agent-fused-matrix-requests-r1274"/name/"request.json"
def request_value(name):
    value=read(retained_request(name))
    return value["request"] if name.startswith("common_") else value
def options_for_case(name):
    if name.startswith("common_"):
        return dict(chunk_period=4,generation_guidance="explicit-v11",
         capacity_packing_version="capacity-public-tensors-v9",public_weight_policy="descriptors-v1")
    rows=read(RESULTS/"agent-fused-matrix-requests-r1274/report.json")["request_rows"]
    selected=next(row for row in rows if row["case"]==name)
    return selected["preparation_options"]
def driver_hashes():
    paths=(HERE/"runtime_scope.py",HERE/"runtime_acceptance.py",COMMON/"runtime_driver.py",COMMON/"test_runtime_driver.py",HERE/"test_runtime_scope.py")
    return {str(p.relative_to(ROOT)):sha(p) for p in paths}
def fixture_files(name):
    folder=fixture_for(name);manifest=read(folder/"manifest.json")
    fmt="poseidon-trusted-pytorch-v13" if name.startswith("common_") else "poseidon-trusted-pytorch-v14"
    require(manifest["format"]==fmt,"Exact source contract")
    selected={"manifest.json",*manifest["files"],*(v["file"] for v in manifest["weights"].values())}
    files={}
    for relative in selected:
        rel=Path(relative)
        require(not rel.is_absolute() and ".." not in rel.parts and rel.parts,"Safe declared fixture path")
        path=folder/rel
        require(path.resolve().is_relative_to(folder.resolve()) and not path.is_symlink(),"Bound fixture remains inside source package")
        files[str(path)]=sha(path)
    for name,want in manifest["files"].items():require(sha(folder/name)==want,"Source byte identity")
    for value in manifest["weights"].values():
        path=folder/value["file"]
        require(path.stat().st_size==value["bytes"] and sha(path)==value["sha256"],"Original public weight identity")
    return manifest,files
def source_binding():
    check=guard();joint=read(HERE/"plan.json")
    check.check()
    protected={}
    files={str(HERE/"plan.json"):sha(HERE/"plan.json"),str(HERE/"check.py"):sha(HERE/"check.py")}
    for item in joint["protected_sources"].values():
        protected.update(read(item["path"]));files[item["path"]]=sha(item["path"])
    reports={
      "common":RESULTS/"agent-ordered-common-tests-r1248/report.json",
      "fused":RESULTS/"agent-fused-matrix-intake-r1268/report.json"}
    common={row["case"]:row for row in read(reports["common"])["source_capture_rows"]}
    fused={row["case"]:row for row in read(reports["fused"])["rows"]}
    references={};requests={}
    for name in CASES:
        manifest,selected=fixture_files(name);files.update(selected)
        request=request_value(name);path=retained_request(name);files[str(path)]=sha(path)
        old=common[name[len("common_"):]] if name.startswith("common_") else fused[name]
        require(digest(manifest)==old["source_package_sha256"],"Original source/reference package")
        require(digest(request["model"])==old["model_sha256"],"Original mathematical graph")
        require(request["compiler_configuration"]["name"]=="poseidon-cpu-eva-w45-v1","Original backend profile")
        references[name]={k:old[k] for k in ("model_sha256","original_sha256","source_package_sha256")}
        requests[name]=dict(request_id=request["request_id"],request_canonical_sha256=digest(request),
            model_sha256=digest(request["model"]),public_constants_sha256=digest(request["public_constants"]),
            preparation_options=options_for_case(name),fixture=str(fixture_for(name)),
            source_contract=manifest["format"])
    for path in reports.values():files[str(path)]=sha(path)
    files[str(RESULTS/"agent-fused-matrix-requests-r1274/report.json")]=sha(RESULTS/"agent-fused-matrix-requests-r1274/report.json")
    files.update({str(ROOT/p):h for p,h in driver_hashes().items()})
    backend_path=RESULTS/"candidate-replay-rsh9wadc/report.json"
    files[str(backend_path)]=sha(backend_path)
    backend=read(backend_path)["backend_identity"]
    sys.path.insert(0,str(BASE))
    from poseidon_backend_identity import identity
    require(identity()==backend,"Exact current SDK/source/parameter identity")
    return dict(original_production=joint["original_production_hashes"],
        inside_sources=joint["inside_source_hashes"],protected_sources=protected,
        files=files,source_reference=references,requests=requests,backend_identity=backend,
        combined_plan_binding=joint["binding"])
def plan():
    bindings=source_binding()
    value=dict(format=VERSION,cases=list(CASES),phases=list(PHASES),scope=dict(SCOPE),
      bindings=bindings,candidate_origin=CANDIDATE_ORIGIN,candidate_strategy="lower-balanced-scalar-v1",
      candidate_bytes_bound_at_qualification=True,new_agent_generation=False,production_applied=False,
      bootstrap=False,cuda=False,resource_profile="general-tensors-v1",
      generation_guidance="explicit-v11",capacity_packing_version="capacity-public-tensors-v9",
      public_weight_policy="descriptors-v1",approval_required=True)
    value["binding"]=digest(value)
    if APPROVAL.exists():
        before_launch()
        require(canonical(value)==canonical(read(PROPOSAL)),"Actual inputs match approved frozen plan")
    return value
def check_plan(value):
    require(value["binding"]==digest({k:v for k,v in value.items() if k!="binding"}),"Runtime plan identity")
    require(value["cases"]==list(CASES) and value["phases"]==list(PHASES),"Exact reviewed cases and phases")
    require(type(value["scope"]) is dict and set(value["scope"])==set(SCOPE) and
      all(type(value["scope"][k]) is type(v) and value["scope"][k]==v for k,v in SCOPE.items()),"Exact finite scope")
    if "deadline_wall" in value:
        before_launch()
        proposal=read(PROPOSAL)
        runtime_fields={"binding","deadline_wall","started_wall","initial_result_directories","readonly_result_entries"}
        require(canonical({k:v for k,v in value.items() if k not in runtime_fields})==
                canonical({k:v for k,v in proposal.items() if k!="binding"}),"Runtime preserves approved plan")
        require(type(value.get("started_wall")) in (float,int) and math.isfinite(value["started_wall"]) and
                type(value["deadline_wall"]) in (float,int) and math.isfinite(value["deadline_wall"]) and
                value["deadline_wall"]-value["started_wall"]==SCOPE["wall_seconds"],"Original finite deadline")
        for key in ("initial_result_directories","readonly_result_entries"):
            require(type(value[key]) is list and len(value[key])==len(set(value[key])) and
                    all(type(n) is str and n not in ("",".","..") and Path(n).name==n for n in value[key]),"Safe historical result inventory")
        require(value["readonly_result_entries"]==value["initial_result_directories"],"Complete read-only history protection")
    for path,want in value["bindings"]["files"].items():require(sha(path)==want,"Frozen tool/fixture/reference changed")
    require(len(value["bindings"]["inside_sources"])==706,"Joint implementation identity")
    if "deadline_wall" in value:
        require(type(value["deadline_wall"]) in (float,int) and value["deadline_wall"]>0,"Finite runtime deadline")
def expected_namespace(value):
    check_plan(value);expected=value["bindings"]["inside_sources"]
    require(all(sha(ROOT/p)==want for p,want in expected.items()),"Exact read-only joint overlay")
    require(all(sha(ROOT/p)==want for p,want in value["bindings"]["protected_sources"].items()),"Backend/reference unchanged")
    sys.path.insert(0,str(BASE))
    from poseidon_backend_identity import identity
    require(identity()==value["bindings"]["backend_identity"],"Exact namespace backend identity")
    return expected
def candidate_source(request):
    from unified_graph_lowering import lower
    return lower(request,balanced_power=True)
def validate_candidate(name,request,program,value):
    frozen=value["bindings"]["requests"][name]
    require(type(program) is str and program and len(program.encode())<=131072,"Bounded rule candidate")
    require(digest(request)==frozen["request_canonical_sha256"] and request["request_id"]==frozen["request_id"],
            "Original frozen generation request")
    require(digest(request["model"])==frozen["model_sha256"] and
            digest(request["public_constants"])==frozen["public_constants_sha256"],"Original graph/weight semantics")
    selected=candidate_source(request)
    require(program==(selected[0] if isinstance(selected,tuple) else selected),"Exact deterministic selected expression")
def check_approval(proposal,receipt,proposal_hash,tool_hashes):
    require(type(proposal) is dict and proposal.get("format")==VERSION and
            proposal.get("binding")==digest({k:v for k,v in proposal.items() if k!="binding"}),"Proposal binding")
    require(proposal["cases"]==list(CASES) and proposal["phases"]==list(PHASES),"Exact approved case list")
    require(proposal["scope"]==SCOPE and all(type(proposal["scope"][k]) is type(v) for k,v in SCOPE.items()),"Exact approved budget")
    fields={"format","approved","proposal_binding","proposal_sha256","driver_hashes","human_message"}
    require(type(receipt) is dict and set(receipt)==fields and receipt["format"]=="poseidon-common-fused-offline-approval-v1"
      and receipt["approved"] is True,"Separate human approval")
    require(receipt["proposal_binding"]==proposal["binding"] and receipt["proposal_sha256"]==proposal_hash,"Exact approved proposal")
    require(receipt["driver_hashes"]==tool_hashes and type(receipt["human_message"]) is str and
      1<=len(receipt["human_message"])<=1024,"Approved tools and human message")
    require(proposal.get("new_agent_generation") is False and proposal.get("production_applied") is False and
      proposal.get("candidate_strategy")=="lower-balanced-scalar-v1","Offline rule scope only")
def before_launch():
    require(APPROVAL.is_file() and not APPROVAL.is_symlink(),"Pending independent r1296 offline approval")
    proposal,receipt=read(PROPOSAL),read(APPROVAL)
    check_approval(proposal,receipt,sha(PROPOSAL),driver_hashes())
def resource_boundary(output,value):
    require(time.time()<value["deadline_wall"],"Original offline deadline")
    require(Path(output)==RUN,"Single approved result directory")
    require(shutil.disk_usage(WORK).free>=SCOPE["reserve_bytes"],"Disk reserve")
    entries=set(value["initial_result_directories"])
    directories=[p for p in RESULTS.iterdir() if p.name not in entries or p==Path(output)]
    require(sum(measure(p) for p in directories)<=SCOPE["artifact_bytes"],"New artifact bound")
    require(measure(RESULTS)<=SCOPE["global_bytes"],"Global artifact bound")
def validate_output(output):
    p=Path(output)
    require(p==RUN and p==p.resolve() and not RUN.exists() and not RUN.is_symlink(),"Single-use approved batch")
def validate_runtime_output(output):
    p=Path(output)
    require(p==RUN and p==p.resolve() and p.is_dir() and not p.is_symlink(),"Original single-use runtime path")
