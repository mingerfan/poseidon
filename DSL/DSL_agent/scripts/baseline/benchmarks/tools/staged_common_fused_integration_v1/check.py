"""Read-only developer integration guard; never part of an application request."""
import hashlib, importlib.util, json, os, re, subprocess
from pathlib import Path
ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5])
HERE=Path(__file__).resolve().parent
WORK=Path("/home/lhohy/poseidon-work/platforms/aarch64-linux")
VERSION="poseidon-common-fused-integration-v1"
NAMES={
    "application_capabilities.py","candidate_bundle.py",
    "pytorch_arithmetic_normalization.py","pytorch_common_constant_provenance.py",
    "pytorch_model_intake.py","pytorch_ordered_capture.py","pytorch_ordered_contract.py",
    "pytorch_ordered_intake.py","pytorch_static_trace.py","pytorch_fused_matrix_normalization.py"}
def require(value,message):
    if not value:raise ValueError(message)
def sha(path):
    path=Path(path)
    require(path.is_file() and not path.is_symlink(),"Regular source or evidence")
    return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):
    path=Path(path)
    require(path.is_file() and not path.is_symlink() and path.stat().st_size<=4*1024**2,
            "Regular bounded control record")
    def unique(pairs):
        result={}
        for key,value in pairs:
            require(key not in result,"Duplicate control record key")
            result[key]=value
        return result
    return json.loads(path.read_text(),object_pairs_hook=unique,
                      parse_constant=lambda value:(_ for _ in ()).throw(ValueError("Finite control record")))
def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
def within(root,relative):
    path=Path(relative)
    require(not path.is_absolute() and ".." not in path.parts,"Checkout-relative source")
    value=root/path
    require(value.resolve().is_relative_to(root.resolve()),"Source remains inside checkout")
    return value

def validate(plan,root=ROOT,here=HERE):
    root,here=Path(root),Path(here)
    require(plan.get("format")==VERSION and plan.get("stage")=="prepared_pending_backend_qualification"
            and plan.get("production_applied") is False,"Prepared integration stage")
    require(plan.get("binding")==digest({k:v for k,v in plan.items() if k!="binding"}),"Integration plan identity")
    require(type(plan.get("targets")) is dict and set(plan["targets"])==NAMES,"Exact Agent target set")
    original=plan.get("original_production_hashes")
    require(type(original) is dict and len(original)==704,"Original production freeze")
    for relative,want in original.items():
        require(sha(within(root,relative))==want,"Production freeze changed")
    expected=dict(original)
    originals,new=[],[]
    for name,item in plan["targets"].items():
        require(type(item) is dict and set(item)=={"original_sha256","target_sha256","source","overlay_history","new_file","original_mode","target_mode"},
                "Integration target fields")
        source=within(root,item["source"])
        require(source.parent.name=="targets" and source.name==name,"Selected reviewed source")
        require(sha(source)==item["target_sha256"] and sha(here/"targets"/name)==item["target_sha256"],
                "Integration source changed")
        require(type(item["target_mode"]) is int and 0<=item["target_mode"]<=0o777 and
                (here/"targets"/name).stat().st_mode&0o777==item["target_mode"],"Target permission changed")
        history=item["overlay_history"]
        require(type(history) is list and history and history[-1]["source"]==item["source"]
                and history[-1]["sha256"]==item["target_sha256"],"Overlay precedence")
        for overlay in history:
            require(set(overlay)=={"source","sha256"} and
                    sha(within(root,overlay["source"]))==overlay["sha256"],"Prerequisite layer changed")
        current=root/"scripts/baseline"/name
        if item["original_sha256"] is None:
            require(item["original_mode"] is None and item["target_mode"]==0o644,"New module permission")
            require(item["new_file"] is True and not os.path.lexists(current),"New module collision")
            require(not os.path.lexists(here/"recovery-originals"/name),"New module has no original")
            new.append(name)
        else:
            require(item["new_file"] is False and sha(current)==item["original_sha256"],
                    "Original modified source changed")
            require(sha(here/"recovery-originals"/name)==item["original_sha256"],"Recovery original changed")
            require(type(item["original_mode"]) is int and 0<=item["original_mode"]<=0o777 and
                    item["target_mode"]==item["original_mode"] and
                    current.stat().st_mode&0o777==item["original_mode"],"Original permission changed")
            require((here/"recovery-originals"/name).stat().st_mode&0o777==item["original_mode"],
                    "Recovery permission changed")
            originals.append(name)
        expected["scripts/baseline/"+name]=item["target_sha256"]
    require(len(originals)==8 and len(new)==2,"Reviewed recovery inventory")
    require(plan["inside_source_hashes"]==expected and len(expected)==706,"Combined source identity")
    for category in ("plans","control_helpers"):
        for relative,want in plan["dependencies"][category].items():
            require(sha(within(root,relative))==want,"Prerequisite plan or control helper changed")
    for label,item in plan["source_acceptance"].items():
        path=within(root,item["path"])
        require(sha(path)==item["sha256"],"Accepted evidence changed")
        accepted=read(path)
        require(accepted.get("binding")==item["binding"] and accepted.get("binding")==
                digest({k:v for k,v in accepted.items() if k!="binding"}),"Accepted evidence identity")
        require(accepted.get("production_applied") is False and
                accepted.get("compiler_calls")==accepted.get("encrypted_executions")==accepted.get("provider_calls")==0,
                "Source-only acceptance scope")
        if label=="fused_matrix_source_draft_v1":
            require(accepted.get("stage")=="staged_source_package_and_submission_verified" and
                    accepted["current_method_successes"]["count"]==44,"Source/package acceptance")
        elif label=="fused_matrix_request_acceptance_v1":
            require(accepted.get("stage")=="staged_generation_context_verified" and
                    accepted.get("source_models")==9 and accepted["current_methods"]["successful"]==6,
                    "Generation-context acceptance")
        else:raise ValueError("Unknown accepted evidence type")
    require(set(plan["source_acceptance"])=={"fused_matrix_source_draft_v1","fused_matrix_request_acceptance_v1"},
            "Complete source/request acceptance")
    qualification=plan["backend_qualification"]
    require(qualification==dict(status="not_run",backend="poseidon_cpu",validation_level="numerical",
        compiler_configuration="poseidon-cpu-eva-w45-v1",capability_scopes=["ordered_common","fused_matrix"],
        package_reload_required=True,new_inputs_required=True,exact_combined_identity_required=True),
        "Backend acceptance remains unverified")
    return dict(plan_binding=plan["binding"],targets=10,originals_retained=8,new_files=sorted(new),
                combined_sources=706,source_acceptance=True,request_acceptance=True,
                backend_acceptance="not_run",production_applied=False)

def observe_boundary(active):
    require(type(active) is list and all(type(pid) is int and pid>0 for pid in active),"Observed process IDs")
    return dict(status="waiting_for_process_exit" if active else "awaiting_backend_qualification",
                active_processes=sorted(set(active)),activation_supported=False,production_applied=False)

def active_processes():
    # Reuse the established process inspector, without calling its old batch
    # eligibility or cutover functions. Only process IDs leave this boundary.
    path=HERE.parent/"staged_binding_feedback_v1/guarded_apply.py"
    spec=importlib.util.spec_from_file_location("_poseidon_existing_source_guard",path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    found=set(module.active_processes())
    markers=("component_worker.py","runtime_acceptance.py","source_live_acceptance_v1/run.py",
             "/tests/inside_tests.py","poseidon_ckks_runner","hecate-opt","/frontend")
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name)==os.getpid():continue
        try:args=(entry/"cmdline").read_bytes().split(bytes([0]))
        except OSError:continue
        if any(any(os.fsdecode(argument).endswith(marker) for marker in markers) for argument in args if argument):
            found.add(int(entry.name))
    return sorted(found)

def check():
    plan=read(HERE/"plan.json")
    state=validate(plan)
    for label,item in plan["protected_sources"].items():
        require(label in ("protected-backend.json","protected-references.json") and
                sha(item["path"])==item["sha256"],"Protected inventory")
        sources=read(item["path"])
        require(all(sha(within(ROOT,p))==want for p,want in sources.items()),"Backend/reference changed")
    branch=subprocess.check_output(["git","branch","--show-current"],cwd=ROOT,text=True,timeout=10).strip()
    head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True,timeout=10).strip()
    require(branch==plan["branch"] and head==plan["head"],"Guest branch or HEAD changed")
    return dict(state,**observe_boundary(active_processes()),branch=branch,head=head,
                compiler_calls=0,encrypted_executions=0,provider_calls=0)
if __name__=="__main__":
    print(json.dumps(check(),sort_keys=True))
