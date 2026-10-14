"""Refresh changed capture identities and freeze offline scope; no backend or API call."""
import copy,hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path
ROOT=(Path(__file__).resolve().parents[6] if Path(__file__).resolve().parents[4].name == "DSL_agent" and Path(__file__).resolve().parents[5].name == "DSL" else Path(__file__).resolve().parents[4])
sys.path.insert(0,str(ROOT/"scripts/baseline"))
from benchmark_graph import canonical,digest,require
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def run(*,resume=False):
    from hecate_python_env import VENV,WORK
    from semantic_benchmark_execution import runtime_sources
    from application_backend import PoseidonBackend
    from candidate_bundle import load_model_provenance,check_provenance
    from component_contract import prepare_task
    from unified_graph_lowering import candidate_source
    from pytorch_model_intake import source_identity
    from trusted_pytorch_import import source_identity as legacy_identity
    from operator_graph import implementation_sources as matrix_identity
    from operator_arithmetic_graph import implementation_sources as arithmetic_identity
    require(os.environ.get("IN_NIX_SHELL")=="pure" and Path(sys.prefix)==VENV,"Pinned pure Python")
    pre=WORK/"results/agent-pytorch-shape-integration-r969"
    results=WORK/"results"
    original=ROOT/"docs/baseline/agent-arithmetic-matrix-offline-proposal-r950.json"
    old=json.loads(original.read_text());old_sha=sha(original)
    require(old["binding"]==digest({k:v for k,v in old.items() if k!="binding"}) and len(old["cases"])==94,
            "Retained unexecuted original scope")
    audit_path=results/"agent-pytorch-shape-public-intake-r971/report.json"
    intake=json.loads(audit_path.read_text())
    require(intake["passed"]==33 and intake["not_run"]==0 and intake["legacy_compatibility_completed"]==15,
            "Current V9 intake acceptance")
    freeze=pre/"prepared-backend-r975"
    if resume:
        require(freeze.is_dir() and len(list((freeze/"prepared").iterdir()))==94 and
                len(list((freeze/"imports").iterdir()))==37,"Exact checked interrupted prefix")
    else:
        require(not freeze.exists(),"Fresh preparation")
        freeze.mkdir();(freeze/"imports").mkdir();(freeze/"prepared").mkdir()
    source_hashes=runtime_sources();capture_id=source_identity();prior_id=legacy_identity()
    backend=PoseidonBackend().identity()
    require(backend["tools"]==old["backend_identity"]["tools"],"Existing SDK identity")
    matrices=matrix_identity();arithmetic=arithmetic_identity();started=time.monotonic()
    rows=[];refreshed=0
    def preservation():
        require(runtime_sources()==source_hashes and source_identity()==capture_id and
                legacy_identity()==prior_id and PoseidonBackend().identity()==backend,"Frozen code/SDK drift")
        require(sha(original)==old_sha,"Historical proposal changed")
        require(all(sha(p)==h for p,h in old["case_hashes"].items()),"Historical source/evidence changed")
        require(all(sha(p)==h for p,h in old["tool_hashes"].items()),"Historical drivers changed")
    def prepare(row,folder,old_row=None):
        model=json.loads((folder/"model.json").read_text())
        provenance=load_model_provenance(folder,model);check_provenance(provenance,model)
        require(provenance["reference_report"]["probes"]==16,"Independent reference probes")
        if old_row is not None:
            old_model=json.loads((Path(old_row["imported"])/"model.json").read_text())
            old_high=json.loads((Path(old_row["imported"])/"original.json").read_text())
            require(canonical(model)==canonical(old_model) and canonical(provenance["original"])==canonical(old_high),
                    "Refreshing implementation cannot change model/reference graph")
        request=prepare_task(model,row["options"]);candidate=candidate_source(request)[0]
        if old_row is not None:
            require(request==json.loads(Path(old_row["request"]).read_text()) and
                    candidate==Path(old_row["candidate"]).read_text(),"Frozen old request/candidate changed")
        destination=freeze/"prepared"/row["id"]
        if destination.exists():
            require(resume and canonical(json.loads((destination/"request.json").read_text()))==canonical(request) and
                    (destination/"candidate.py").read_text()==candidate,"Retained prepared request/source differs")
        else:
            destination.mkdir()
            (destination/"request.json").write_bytes(canonical(request))
            (destination/"candidate.py").write_text(candidate)
        files={str(p):sha(p) for p in folder.rglob("*") if p.is_file()}
        files.update({str(p):sha(p) for p in destination.iterdir()})
        row.update(imported=str(folder),request=str(destination/"request.json"),candidate=str(destination/"candidate.py"),
            request_id=request["request_id"],model_sha256=digest(model),candidate_sha256=sha(destination/"candidate.py"),
            capture_identity=prior_id if row["input_kind"]=="trusted_pytorch_v7" else capture_id,
            files=files,fresh_reference_probes=16,current_binding_reference_checked=True)
        rows.append(row)
    for old_row in old["cases"]:
        preservation()
        require(time.monotonic()-started<270,"Plaintext preparation 270-second bound")
        row=copy.deepcopy(old_row)
        if row["input_kind"] in ("trusted_pytorch_v7","trusted_pytorch_v8"):
            folder=freeze/"imports"/row["id"]
            command=[str(VENV/"bin/python"),"-B",str(ROOT/"scripts/prepare_model.py"),
                "--trusted-pytorch-manifest",row["manifest"],"--trust-local-code",
                "--resource-profile","general-tensors-v1","--write","--output",str(folder),"--inside"]
            if not resume:
                proc=subprocess.run(command,text=True,capture_output=True,timeout=min(185,270-(time.monotonic()-started)))
                (freeze/(row["id"]+"-cli.log")).write_text(proc.stdout+proc.stderr)
                require(proc.returncode==0,"Affected source capture failed; preserve log")
            # Actual current source binding, model, reference and request are rechecked below.
            refreshed+=1
            row["prior_imported"]=row["imported"];row["capture_identity_refreshed"]=True
        else:
            folder=Path(row["imported"]);row["capture_identity_refreshed"]=False
        prepare(row,folder,old_row)
    require(refreshed==37,"Only the 37 changed PyTorch bindings require recapture")
    from unified_capacity_constants import INDEX_VERSION
    for current in intake["rows"]:
        if current["kind"]!="positive" or current["style"]!="method" or current["query"] not in ("result_shape","result_size","result_numel"):continue
        name=current["id"];folder=results/"agent-pytorch-shape-public-intake-r971/captures"/name
        row=dict(id="v9_"+name,input_kind="trusted_pytorch_v9",
            manifest=str(ROOT/"scripts/baseline/cases/trusted-pytorch-shape-composition-v9"/name/"manifest.json"),
            options=dict(generation_guidance="explicit-v10",capacity_packing_version=INDEX_VERSION),
            historical_status="new_input_not_run",source_origin="trusted_pytorch_rule_program",
            historical_batch="retained V8 shape failure; explicit V9 intake passed",
            original_failed_case_preserved=True,model_and_reference_preserved=True,
            initial_source_generation="trusted rule, not Agent")
        prepare(row,folder)
    require(len(rows)==103,"94 original tasks plus nine repaired V9 sources")
    for i,row in enumerate(rows):row["shard"]=1+i//48
    preserved=preservation()
    case_hashes=old["case_hashes"].copy()
    for row in rows:case_hashes.update(row["files"])
    case_hashes[str(original)]=old_sha;case_hashes[str(audit_path)]=sha(audit_path)
    proposal=dict(format=old["format"],revision="r976",status="awaiting_human_approval",
        cases=rows,shards=[dict(id=1,count=48),dict(id=2,count=48),dict(id=3,count=7)],
        source_hashes=source_hashes,case_hashes=case_hashes,capture_identity=capture_id,
        legacy_capture_identity=prior_id,backend_identity=backend,matrix_identity=matrices,
        arithmetic_identity=arithmetic,package_reuse_cases=old["package_reuse_cases"]+
            ["v9_shape_add_method_result_shape","v9_shape_subtract_method_result_size","v9_shape_multiply_method_result_numel"],
        explicit_domain_rejection_cases=old["explicit_domain_rejection_cases"],
        scope=dict(wall_seconds=7200,candidate_seconds=180,artifact_bytes=4*1024**3,
            global_bytes=32*1024**3,reserve_bytes=4*1024**3,max_shard_cases=48,shards_sequential=True,
            encrypted_concurrency=1,compile_jobs=2,link_jobs=1,api_calls=0,downloads=0,installs=0,
            backend_changes=0,parameter_changes=0),
        tolerance=dict(atol=1e-5,rtol=1e-4),candidate_origin="trusted_rule_baseline",
        new_agent_generation=False,paid_calls=0,http_attempts=0,
        previous_proposal=dict(binding=old["binding"],preserved=True,executed=False,source_stale=True),
        prior_approval="r870 24-task scope executed and closed; does not authorize this new batch",
        affected_plaintext_capture_count=refreshed,
        current_acceptance_binding=intake["binding"],
        outcome_policy=old["outcome_policy"],exclusions=old["exclusions"])
    old_tools=ROOT/"scripts/baseline/benchmarks/tools/frozen_arithmetic_backend"
    tools=ROOT/"scripts/baseline/benchmarks/tools/frozen_arithmetic_shape_backend"
    require(not tools.exists(),"Keep historical drivers unchanged");tools.mkdir()
    for name in ("backend.py","audit.py","job.py","launch.py"):
        source=(old_tools/name).read_text()
        source=source.replace("agent-public-arithmetic-preflight-r935","agent-pytorch-shape-integration-r969")
        source=source.replace("agent-arithmetic-matrix-backend-r951","agent-arithmetic-shape-backend-r977")
        source=source.replace("agent-arithmetic-matrix-offline-proposal-r950.json","agent-arithmetic-shape-offline-proposal-r976.json")
        source=source.replace("r951","r977").replace("r952","r978")
        source=source.replace('("trusted_pytorch_v7","trusted_pytorch_v8")','("trusted_pytorch_v7","trusted_pytorch_v8","trusted_pytorch_v9")')
        source=source.replace('case["shard"] in (1,2)','case["shard"] in (1,2,3)')
        source=source.replace('==94,','==103,').replace('min(5400,','min(7200,')
        compile(source,str(tools/name),"exec");(tools/name).write_text(source)
    # Copy immutable protection snapshots into the new approval root.
    old_pre=results/"agent-public-arithmetic-preflight-r935"
    for label in ("backend","references","history","new-history","current-history"):
        name="protected-"+label+".json"
        target=pre/name;require(not target.exists(),"Protection snapshot collision")
        target.write_bytes((old_pre/name).read_bytes())
    proposal["tool_hashes"]={str(p):sha(p) for p in tools.iterdir()}
    proposal["binding"]=digest(proposal)
    target=ROOT/"docs/baseline/agent-arithmetic-shape-offline-proposal-r976.json"
    require(not target.exists(),"Fresh proposal");target.write_bytes(canonical(proposal))
    require(not (pre/"approved-backend-scope-r977.json").exists() and
            not (results/"agent-arithmetic-shape-backend-r977").exists(),"No approved scope or execution")
    preparation=dict(format="poseidon-shape-backend-preparation-v1",planned=103,
        refreshed_affected_sources=37,models_requests_candidates_unchanged=94,
        repaired_v9_sources=9,proposal_binding=proposal["binding"],source_count=len(source_hashes),
        source_identity=capture_id,tools_sdk_unchanged=True,seconds=time.monotonic()-started,
        resumed_checked_prefix=resume,completed_source_captures_rerun=0,
        compiled=0,encrypted_execution=0,agent_generated=0,paid_calls=0,approved=False)
    preparation["binding"]=digest(preparation)
    (pre/"prepared-backend-r975.json").write_bytes(canonical(preparation))
    print(json.dumps(preparation))
    return 0
if __name__=="__main__":
    require(sys.argv[1:] in ([],["--resume-prepared-prefix"]),"Preparation arguments")
    raise SystemExit(run(resume=bool(sys.argv[1:])))
