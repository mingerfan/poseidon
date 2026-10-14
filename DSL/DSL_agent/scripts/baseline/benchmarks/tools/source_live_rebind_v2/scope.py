"""Fixed three-case 706 preparation and separate live proposal authority."""
import copy, hashlib, json, math, subprocess, fcntl
from contextlib import contextmanager
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = (Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5])
BASE = ROOT / "scripts/baseline"
WORK = Path("/home/lhohy/poseidon-work/platforms/aarch64-linux")
RESULTS = WORK / "results"
STARTUP_FREEZE = RESULTS / "agent-v12-source-rebind-preparation-v2/freeze.json"
STARTUP_FREEZE_SHA = '328d4cf7c28618abc85c441f559b997e6fabd2cfb46c12a88e859df00dbccd81'
PREPARATION = RESULTS / "agent-v12-source-rebind-preparation-v2-attempt3"
CASE_ROOTS = {n:(RESULTS / "agent-v12-source-rebind-preparation-v2-attempt2/linear" if n=="linear" else PREPARATION/n) for n in ("linear","norm","split_weight")}
INITIAL_SHA = "2a4474456d6518fb85815a37085f6a363b3981b15a4f823d6292780d3304f487"
PREP_REPORT_SHA = "dc25b84856f95af7b10ebbe33adf9c5a38cdcc1bb06e42640987c200287b3eb9"
PREP_REPORT_BINDING = "9ba89319909d90090bfef6bbc95e7e595b7f1039a6b4a4df7fb8db509f893bc5"
ORIGIN_FREEZE_SHA = {"linear":"582ab078092a05462048586fa721391f24f7bf695ad0f6961cfc3c8395e85f74",
    "norm":"22b9d9dbc7e713e5dae255da41e7b57204da64ce0b47fcb4e44ccc4b83631811",
    "split_weight":"22b9d9dbc7e713e5dae255da41e7b57204da64ce0b47fcb4e44ccc4b83631811"}
RUN = RESULTS / "agent-v12-source-live-rebound-v2"
PROPOSAL = HERE / "proposal-ready.json"
OLD = ROOT / "docs/baseline/agent-v12-live-proposal-r1232.json"
OLD_SHA = "e59482462d92a206130dc7e24881e63d2cd2cf0dc5793140c031a6bded8f9ad4"
OLD_BINDING = "1120ab0063c801e2d4199e956bb55dec12f0a68a167c17325ef7bb2a77a21481"
CUTOVER = ROOT / "scripts/baseline/benchmarks/tools/staged_common_fused_integration_v1/cutover-records/20261008T072609-37b52747/record.json"
CUTOVER_SHA = "153af39089d0293b3608728ff3f47bdd423a41b47a40756161cdd67a7fab0a50"
HELPER = ROOT / "scripts/baseline/benchmarks/tools/staged_component_cutover_v1/post_cutover_acceptance.py"
FORMAT = "poseidon-source-live-rebound-proposal-v2"
CASES = ("linear", "norm", "split_weight")
OPTIONS = dict(chunk_period=4, generation_guidance="explicit-v11",
    capacity_packing_version="capacity-public-tensors-v9", public_weight_policy="descriptors-v1")
PROFILE = "general-tensors-v1"
PROVIDER = dict(name="deepseek", model="deepseek-flash", reasoning_effort="high",
    max_tokens=384000, max_repairs=3, provider_retries=3, stream=True, timeout_seconds=1200)
BUDGET = dict(api_concurrency=1, batch_wall_seconds=43200, case_wall_seconds=14400,
    compile_jobs=2, currency_limit=None, global_result_bytes=32*1024**3, link_jobs=1,
    max_generations=12, max_http_attempts=48, native_concurrency=1,
    per_case_bytes=4*1024**3, reserve_bytes=4*1024**3)
PREP_SCOPE = dict(wall_seconds=600, case_seconds=180, artifact_bytes=4*1024**3,
    global_bytes=32*1024**3, reserve_bytes=4*1024**3,
    provider_calls=0, http_attempts=0, compiler_calls=0, encrypted_executions=0,
    installations=0, downloads=0)
TOP_KEYS = {"format","approved","approval_reason","source_root","branch","head","historical",
    "cutover","source_hashes","protected_sources","provider","budget","options","resource_profile",
    "cases","preparation","tool_hashes","application_backend_identity","backend_identity",
    "application_stage_identity","disclosure","validation","original_terminal_scope","binding"}

def require(ok, code):
    if not ok: raise ValueError(code)
def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()
def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1024**2),b""): h.update(block)
    return h.hexdigest()
def decode(raw):
    def pairs(items):
        value={}
        for k,v in items:
            require(k not in value,"duplicate_json_field");value[k]=v
        return value
    return json.loads(raw,object_pairs_hook=pairs,
        parse_constant=lambda _:(_ for _ in ()).throw(ValueError("nonfinite_json")))
def read(path): return decode(Path(path).read_bytes())
def rebound(value):
    value=copy.deepcopy(value);value.pop("binding",None);value["binding"]=digest(value);return value
def check_binding(value, key="binding"):
    require(type(value) is dict and value.get(key)==digest({k:v for k,v in value.items() if k!=key}),
        "self_binding")
def regular(path):
    p=Path(path)
    require(p.is_absolute() and p==p.resolve() and p.is_file() and not p.is_symlink(),"regular_frozen_path")
    require(p.is_relative_to(ROOT) or p.is_relative_to(RESULTS),"frozen_path_scope")
    return p
@contextmanager
def owned_lock(path):
    with Path(path).open("a+") as stream:
        fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        yield stream

class Reader:
    def __init__(self): self.files={}
    def hash(self,path):
        p=regular(path);key=str(p);value=sha(p)
        require(key not in self.files or self.files[key]==value,"file_changed_during_read")
        require(len(self.files)<4096 or key in self.files,"frozen_inventory_bound")
        self.files[key]=value;return value
    def json(self,path):
        p=regular(path);require(p.stat().st_size<=8*1024**2,"json_capacity")
        raw=p.read_bytes();h=hashlib.sha256(raw).hexdigest();key=str(p)
        require(key not in self.files or self.files[key]==h,"file_changed_during_read")
        self.files[key]=h;return decode(raw)
    def finish(self):
        for key,want in self.files.items():require(sha(regular(key))==want,"file_changed_during_read")
        return dict(self.files)
def tools():
    paths=[HERE/n for n in ("scope.py","prepare.py","run.py","test_scope.py","test_run.py","finalize.py")]+[HELPER]
    return {str(p.relative_to(ROOT)):sha(regular(p)) for p in paths}
def anchors(reader=None):
    rd=Reader() if reader is None else reader
    require(rd.hash(OLD)==OLD_SHA and rd.hash(CUTOVER)==CUTOVER_SHA,"historical_anchor_changed")
    old,cut=rd.json(OLD),rd.json(CUTOVER)
    check_binding(old);check_binding(cut)
    require(old["binding"]==OLD_BINDING and old["approved"] is False,"historical_proposal")
    require(tuple(c["id"] for c in old["cases"])==CASES and old["provider"]==PROVIDER and old["budget"]==BUDGET,
        "historical_fixed_scope")
    require(cut["state"]=="applied" and cut["production_applied"] is True and
        len(cut["production_sources"])==706 and len(cut["protected_sources"])==654,"applied_706_required")
    return old,cut
def validate(value, old, cut):
    require(type(value) is dict and set(value)==TOP_KEYS,"proposal_fields")
    check_binding(value)
    require(value["format"]==FORMAT and value["approved"] is False,"unapproved_proposal")
    require(value["source_root"]==str(ROOT) and value["branch"]==old["branch"] and value["head"]==old["head"],
        "source_checkout")
    require(value["provider"]==PROVIDER and value["budget"]==BUDGET and
        all(type(value["budget"][k]) is int for k in BUDGET if k!="currency_limit"),
        "reviewed_provider_or_budget")
    require(value["resource_profile"]==PROFILE and value["options"]==OPTIONS,"reviewed_preparation_options")
    require(value["source_hashes"]==cut["production_sources"] and
        value["protected_sources"]==cut["protected_sources"],"current_source_namespace")
    require(value["historical"]==dict(path=str(OLD),sha256=OLD_SHA,binding=OLD_BINDING,
        validation_identity_current=False),"historical_separate")
    require(value["cutover"]==dict(path=str(CUTOVER),sha256=CUTOVER_SHA,binding=cut["binding"]),
        "cutover_binding")
    require(tuple(c["id"] for c in value["cases"])==CASES,"reviewed_cases")
    require(type(value["application_backend_identity"]) is dict and
        set(value["application_backend_identity"])=={"backend","tools","implementation_digest"} and
        value["application_backend_identity"]["backend"]=="poseidon_cpu" and
        value["backend_identity"]==value["application_backend_identity"]["tools"]==old["backend_identity"] and
        value["backend_identity"].get("format")=="poseidon-backend-identity-v2","backend_identity_types")
    require(value["application_stage_identity"].get("format")=="poseidon-application-identities-v1",
        "staged_application_identity")
    require(value["tool_hashes"]==tools(),"frozen_tools")
    require(value["preparation"]["output"]==str(PREPARATION) and
        value["preparation"]["scope"]==PREP_SCOPE and
        value["preparation"]["current_identity_not_qualification"] is True,"preparation_scope")
    require(value["validation"]==dict(backend="poseidon_cpu",level="numerical",
        new_agent_generation=True,no_compiler_or_runtime_changes=True,
        required=["source_capture","numerical_qualification","new_input_execution","package_cache_reuse"],
        tolerance=dict(atol=1e-5,rtol=1e-4)),"reviewed_live_validation")
    require(value["disclosure"]==old["disclosure"] and
        value["original_terminal_scope"]==old["original_terminal_scope"],"historical_accounting_or_disclosure")
    for c,prior in zip(value["cases"],old["cases"]):
        require(c["source_format"]==prior["source_format"]=="poseidon-trusted-pytorch-v12" and
            c["source_manifest"]==prior["source_manifest"] and c["source_files"]==prior["source_files"] and
            c["free_generation"] is True and c["directed_construction"] is False,"reviewed_case_source")
        require(c["historical"]==dict(request_id=prior["request_id"],
            request_file=str(ROOT/prior["request_file"]),request_file_sha256=prior["request_file_sha256"],
            qualification=prior["rule_qualification_evidence"],
            qualification_sha256=prior["rule_qualification_evidence_sha256"],
            current_identity=False),"historical_case_reference")
        require(c["request_file"]==str(CASE_ROOTS[c["id"]]/"request.json") and
            c["prepared"]["task_path"].startswith(str(CASE_ROOTS[c["id"]]/"tasks")+"/"),"prepared_case_path")
    return value
def validate_case(c,rd,stage_identity):
    q=rd.json(c["request_file"])
    require(rd.hash(c["request_file"])==c["request_file_sha256"] and q["request_id"]==c["request_id"] and
        digest(q)==c["request_sha256"],"complete_request_binding")
    old_q=rd.json(c["historical"]["request_file"])
    require(rd.hash(c["historical"]["request_file"])==c["historical"]["request_file_sha256"],
        "historical_request_changed")
    for key in ("model","layout","compiler_configuration"):
        require(q.get(key)==old_q.get(key),"original_mathematical_request_changed")
    require(c["request_differences"]==sorted(k for k in set(q)|set(old_q) if q.get(k)!=old_q.get(k)),
        "request_difference_disclosure")
    prepared=c["prepared"];folder=Path(prepared["task_path"])
    require(folder.parent==CASE_ROOTS[c["id"]]/"tasks" and len(folder.name)==32 and
        all(x in "0123456789abcdef" for x in folder.name),"prepared_task_scope")
    require(prepared["evidence_files"] and all(Path(p).is_relative_to(CASE_ROOTS[c["id"]]) for p in prepared["evidence_files"]),
        "prepared_evidence_scope")
    # Exact tree capture: replacing or adding evidence is not a new valid preparation.
    actual={str(p) for p in CASE_ROOTS[c["id"]].rglob("*") if p.is_file()}
    require(actual==set(prepared["evidence_files"]),"prepared_evidence_inventory")
    for path,want in prepared["evidence_files"].items():require(rd.hash(path)==want,"prepared_evidence_changed")
    meta=rd.json(folder/"task.json");state=rd.json(folder/"state.json");source=rd.json(folder/"source.json")
    check_binding(meta);check_binding(source)
    require(meta["request"]==q and meta["identity_policy"]=="staged-v1" and
        meta["identity"]==stage_identity and meta["validation_level"]=="numerical" and
        meta["context"]==c["prepared_context"] and meta["model_provenance"]==c["model_provenance"] and
        meta["model_source"]==c["model_source"],"current_prepared_identity")
    require(state["status"]=="queued" and state["phase"]=="queued" and
        type(state["generations"]) is int and state["generations"]==0 and
        type(state["http_attempts"]) is int and state["http_attempts"]==0 and state["ledger"]==[] and
        state["intake"]["status"]=="prepared" and state["intake"]["reference_verified"] is True and
        state["intake"]["encrypted_execution"] is False,"preparation_only")
    require(source["requirements"]["options"]==OPTIONS and source["resource_profile"]==PROFILE and
        source["identity"]==stage_identity,"prepared_source_identity")
    report=rd.json(folder/"model-import/reference-report.json")
    original=rd.json(folder/"model-import/original.json")
    model=rd.json(folder/"model-import/model.json")
    manifest=rd.json(folder/"source/manifest.json")
    capture_input=rd.json(folder/"model-import/capture-input.json")
    capture_reference=rd.json(folder/"model-import/capture-reference.json")
    retained_manifest=rd.json(folder/"model-import/trusted-python/manifest.json")
    require(original==meta["model_provenance"]["original"] and
        rd.json(folder/"model-import/decomposition.json")==meta["model_provenance"]["decomposition"] and
        report==meta["model_provenance"]["reference_report"]==capture_reference and
        manifest==capture_input["manifest"]==retained_manifest and
        model==q["model"] and report["original_sha256"]==digest(original) and
        report["model_sha256"]==digest(model) and report["manifest_sha256"]==digest(manifest) and
        report["source_hashes"]==capture_input["source_hashes"] and
        report["runtime"]==capture_input["runtime"],"reference_capture_cross_binding")
    require(report["format"]=="poseidon-trusted-pytorch-reference-v12" and
        report["mode"]=="isolated_trusted_pytorch_reference" and
        report["reference_status"]=="passed" and report["probes"]==16 and
        report["reference_dtype"]=="float64" and
        report["input_mutation_policy"]=="ordered-nonoverlapping-registered-effects" and
        report["source_precision_execution_compared"] is False and
        report["approximation_accuracy_certified"] is False and
        report["encrypted_execution"] is False and report["agent_generated"] is False and
        report["paid_calls"]==0 and report["runtime_layout_validated"] is False,
        "independent_plaintext_reference")
    finite=report["finite_reference"];check_binding(finite)
    require(finite["format"]=="poseidon-ordered-capture-reference-staging-v2" and finite["status"]=="passed" and finite["groups"]==16 and
        finite["graph_sha256"]==digest(original) and finite["lowered_sha256"]==digest(model) and
        finite["capture_binding_sha256"]==rd.json(folder/"model-import/decomposition.json")["binding_sha256"] and
        finite["reference_implementation_sha256"]==report["source_hashes"]["pytorch_ordered_reference.py"] and
        finite["comparison_tolerance"]==dict(atol=1e-12,rtol=1e-12) and
        finite["input_storage_isolated"] is True and finite["public_state_immutable"] is True and
        finite["physical_layout_validated"] is False and finite["compiler_validated"] is False and
        finite["encrypted_execution_validated"] is False and finite["paid_calls"]==0 and
        finite["compared_values"]==16*sum(math.prod(o["shape"]) for o in q["layout"]["outputs"]),
        "finite_reference_binding")
    for key in ("max_original_torch_vs_independent_ideal","max_original_vs_normalized_torch",
        "max_decomposition_error","max_approximation_error","max_reference_tolerance_ratio","max_normalization_tolerance_ratio"):
        metric=finite[key]
        require(type(metric) in (int,float) and math.isfinite(metric) and metric>=0,"finite_reference_metric")
    require(finite["max_reference_tolerance_ratio"]<=1 and finite["max_normalization_tolerance_ratio"]<=1,
        "finite_reference_tolerance")
    for rel,want in report["source_hashes"].items():
        require(rd.hash(BASE/rel)==want,"reference_implementation_changed")
    require(digest(meta["identity"]["validation"])==c["current_validation_sha256"] and
        meta["identity"]["validation"]["sources"]==c["current_validation_sources"],"current_validation_binding")
    require(rd.hash(c["historical"]["qualification"])==c["historical"]["qualification_sha256"],
        "historical_qualification_changed")
    for rel,want in c["source_files"].items():require(rd.hash(ROOT/rel)==want,"fixture_changed")
    for rel,want in c["current_validation_sources"].items():
        require(rd.hash(BASE/rel)==want,"current_validation_source_changed")
    return q
def verify_origins(value,rd):
    origins=value["preparation"]["capture_origins"]
    require(set(origins)==set(CASES),"capture_origin_inventory")
    for name in CASES:
        origin=origins[name]
        expected=(RESULTS/"agent-v12-source-rebind-preparation-v2-attempt2") if name=="linear" else PREPARATION
        require(origin["type"]==("reuse_completed_capture" if name=="linear" else "new_capture") and
            origin["case_root"]==str(CASE_ROOTS[name]) and origin["attempt_root"]==str(expected),
            "capture_origin_scope")
        require(rd.hash(expected/"freeze.json")==origin["freeze_sha256"]==ORIGIN_FREEZE_SHA[name],"capture_origin_freeze")
        frozen=rd.json(expected/"freeze.json");check_binding(frozen)
        require(origin["producer_tool_hashes"]==frozen["tool_hashes"],"capture_origin_toolchain")
        wanted={}
        for rel,want in frozen["tool_hashes"].items():
            original=ROOT/rel
            source=(expected/"tool-snapshot"/original.name) if original.parent==HERE else original
            wanted[str(source)]=want
        require(origin["tool_snapshot_hashes"]==wanted,"capture_origin_snapshot_inventory")
        for path,want in wanted.items():require(rd.hash(path)==want,"capture_origin_snapshot_changed")
        require(frozen["started_wall"]==rd.json(STARTUP_FREEZE)["started_wall"] and
            frozen["deadline_wall"]==rd.json(STARTUP_FREEZE)["deadline_wall"],"capture_original_window")
    require(origins["norm"]["producer_tool_hashes"]==value["preparation"]["producer_tool_hashes"]==
        origins["split_weight"]["producer_tool_hashes"],"producer_report_toolchain")

def verify(value=None):
    rd=Reader();old,cut=anchors(rd)
    if value is None:value=rd.json(PROPOSAL)
    validate(value,old,cut)
    for rel,want in {**value["source_hashes"],**value["protected_sources"],**value["tool_hashes"]}.items():
        require(rd.hash(ROOT/rel)==want,"frozen_source_changed")
    require(rd.hash(PREPARATION/"proposal-initial.json")==INITIAL_SHA and
        rd.hash(PREPARATION/"report.json")==PREP_REPORT_SHA,"fixed_preparation_bytes")
    prep=rd.json(PREPARATION/"report.json")
    require(rd.hash(PREPARATION/"report.json")==value["preparation"]["report_sha256"] and
        prep["binding"]==value["preparation"]["report_binding"]==PREP_REPORT_BINDING,"preparation_report_binding")
    check_binding(prep)
    require(prep["status"]=="prepared" and prep["scope"]==PREP_SCOPE and
        prep["tool_hashes"]==value["preparation"]["producer_tool_hashes"] and
        type(prep["seconds"]) in (float,int) and math.isfinite(prep["seconds"]) and 0<=prep["seconds"]<=600 and
        prep["provider_calls"]==prep["http_attempts"]==prep["compiler_calls"]==prep["encrypted_executions"]==0,
        "preparation_report_scope")
    require(prep["case_bindings"]=={c["id"]:digest(c) for c in value["cases"]},"preparation_case_bindings")
    verify_origins(value,rd)
    for c in value["cases"]:validate_case(c,rd,value["application_stage_identity"])
    worker=rd.json(PREPARATION/"worker-context.json")
    require(worker["backend_identity"]==value["application_backend_identity"] and
        worker["stage_identity_sha256"]==value["application_stage_identity"]["manifest_sha256"],
        "preparation_worker_identity")
    freeze=rd.json(PREPARATION/"freeze.json");origin=rd.json(STARTUP_FREEZE)
    check_binding(freeze);check_binding(origin)
    require(rd.hash(STARTUP_FREEZE)==STARTUP_FREEZE_SHA and
        freeze["started_wall"]==origin["started_wall"] and freeze["deadline_wall"]==origin["deadline_wall"] and
        freeze["deadline_wall"]-freeze["started_wall"]==600 and
        rd.hash(PREPARATION/"freeze.json")==value["preparation"]["freeze_sha256"],"preparation_freeze")
    rd.finish()
    for args,want in ((["git","branch","--show-current"],value["branch"]),(["git","rev-parse","HEAD"],value["head"])):
        require(subprocess.check_output(args,cwd=ROOT,text=True,timeout=10).strip()==want,"checkout_changed")
    return value
def authority(receipt,proposal,proposal_sha):
    require(type(receipt) is dict and set(receipt)=={"format","approved","proposal_binding","proposal_sha256","tool_hashes","authorization"},
        "approval_record_fields")
    require(receipt["format"]=="poseidon-source-live-rebound-approval-v2" and receipt["approved"] is True and
        receipt["proposal_binding"]==proposal["binding"] and receipt["proposal_sha256"]==proposal_sha and
        receipt["tool_hashes"]==proposal["tool_hashes"] and
        receipt["authorization"]=="human_explicit_case_budget_approval","approval_required")
def totals(states):
    g=h=0;resolved=True
    for s in states:
        require(type(s["generations"]) is int and type(s["http_attempts"]) is int and
            0<=s["generations"]<=4 and 0<=s["http_attempts"]<=16,"case_call_ceiling")
        reserved=[e for e in s["ledger"] if e.get("event")=="http_reserved"]
        require([e["index"] for e in reserved]==list(range(s["http_attempts"])),"reservation_ledger")
        checkpoints=[e for e in s["ledger"] if e.get("event")=="provider_checkpoint"]
        outcomes=checkpoints[-1].get("outcomes",[]) if checkpoints else []
        resolved=resolved and len(outcomes)==s["http_attempts"] and all(
            o.get("index")==i and o.get("status") in ("response_received","failed") for i,o in enumerate(outcomes))
        g+=s["generations"];h+=s["http_attempts"]
    require(g<=12 and h<=48,"campaign_call_ceiling")
    return dict(generations=g,http_attempts=h,calls_resolved=resolved)
def terminal(rows,states):
    require(len(rows)<=3 and len({r["case"] for r in rows})==len(rows) and
        all(r["case"] in CASES for r in rows),"result_case_scope")
    data=totals(states);passed=sum(r["status"]=="passed" for r in rows)
    return dict(status="complete" if len(rows)==3 and data["calls_resolved"] else "stopped",
        passed=passed,failed=len(rows)-passed,not_run=3-len(rows),**data)
