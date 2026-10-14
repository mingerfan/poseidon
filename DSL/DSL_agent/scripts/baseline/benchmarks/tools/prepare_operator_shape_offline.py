"""Refresh original 103 identities and add nine v10 sources without backend execution."""
import copy,json,os,subprocess,sys,time
from pathlib import Path
ROOT=(Path(__file__).resolve().parents[6] if Path(__file__).resolve().parents[4].name == "DSL_agent" and Path(__file__).resolve().parents[5].name == "DSL" else Path(__file__).resolve().parents[4])
sys.path.insert(0,str(ROOT/"scripts/baseline"))
from benchmark_graph import canonical,digest,require
from benchmarks.tools.audit_pytorch_operator_shape_contexts import checked,sha
def run():
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
    from unified_capacity_constants import INDEX_VERSION
    require(os.environ.get("IN_NIX_SHELL")=="pure" and Path(sys.prefix)==VENV,"Pinned pure Python")
    pre=WORK/"results/agent-pytorch-operator-shape-integration-r985"
    freeze=pre/"prepared-backend-r990";require(not freeze.exists(),"Fresh checked preparation")
    old_path=ROOT/"docs/baseline/agent-arithmetic-shape-offline-proposal-r976.json"
    old=checked(json.loads(old_path.read_text()));old_sha=sha(old_path)
    require(len(old["cases"])==103,"Original 103 tasks retained")
    live=checked(json.loads((WORK/"results/agent-operator-shape-public-v10-r988/report.json").read_text()))
    application=checked(json.loads((WORK/"results/agent-operator-shape-application-v10-r989/report.json").read_text()))
    require(live["passed"]==39 and application["application_accepted"]==39 and application["public_rejection_checks"]==7,"Current public/app acceptance")
    sources=runtime_sources();capture_id=source_identity();legacy_id=legacy_identity()
    backend=PoseidonBackend().identity();require(backend["tools"]==old["backend_identity"]["tools"],"Unchanged compiled SDK")
    matrices=matrix_identity();arithmetic=arithmetic_identity()
    frozen_old={**old["case_hashes"],**old["tool_hashes"],str(old_path):old_sha}
    require(all(sha(Path(p))==h for p,h in frozen_old.items()),"Original evidence/cases/drivers preserved")
    freeze.mkdir();(freeze/"imports").mkdir();(freeze/"prepared").mkdir()
    started=time.monotonic();rows=[];captured=0
    def preservation():
        require(runtime_sources()==sources and source_identity()==capture_id and legacy_identity()==legacy_id,"Current source binding drift")
        require(all(sha(Path(p))==h for p,h in frozen_old.items()),"Historical artifacts changed")
    def save_preparation(status):
        body=dict(format="poseidon-operator-shape-backend-preparation-v1",status=status,planned=112,prepared=len(rows),
            recaptured_sources=captured,original_scope=103,new_v10_tasks=9,source_files=len(sources),
            original_proposal_binding=old["binding"],current_intake_binding=live["binding"],seconds=time.monotonic()-started,
            compiled=0,encrypted_execution=0,paid_calls=0,approved=False,rows=rows)
        body["binding"]=digest(body);(pre/"preparation-progress-r990.json").write_bytes(canonical(body));return body
    save_preparation("preparing")
    def prepare(row,folder,prior=None):
        model=json.loads((folder/"model.json").read_text());provenance=load_model_provenance(folder,model);check_provenance(provenance,model)
        require(provenance["reference_report"]["probes"]==16,"Finite independent reference scope")
        request=prepare_task(model,row["options"]);candidate=candidate_source(request)[0]
        if prior is not None:
            require(canonical(model)==canonical(json.loads((Path(prior["imported"])/"model.json").read_text()))
                and canonical(provenance["original"])==canonical(json.loads((Path(prior["imported"])/"original.json").read_text())),
                "Original model and reference graph unchanged")
            require(request==json.loads(Path(prior["request"]).read_text()) and candidate==Path(prior["candidate"]).read_text(),
                "Original request and candidate must be byte-equivalent")
        target=freeze/"prepared"/row["id"];target.mkdir()
        (target/"request.json").write_bytes(canonical(request));(target/"candidate.py").write_text(candidate)
        files={str(p):sha(p) for p in folder.rglob("*") if p.is_file()}
        files.update({str(p):sha(p) for p in target.iterdir()})
        row.update(imported=str(folder),request=str(target/"request.json"),candidate=str(target/"candidate.py"),
            request_id=request["request_id"],model_sha256=digest(model),candidate_sha256=sha(target/"candidate.py"),
            capture_identity=legacy_id if row["input_kind"]=="trusted_pytorch_v7" else capture_id,
            files=files,current_binding_reference_checked=True,fresh_reference_probes=16)
        rows.append(row);save_preparation("preparing")
    for original in old["cases"]:
        preservation();left=270-(time.monotonic()-started);require(left>0,"Plaintext preparation 270-second bound")
        row=copy.deepcopy(original)
        if row["input_kind"] in ("trusted_pytorch_v7","trusted_pytorch_v8","trusted_pytorch_v9"):
            destination=freeze/"imports"/row["id"]
            command=[str(VENV/"bin/python"),"-B",str(ROOT/"scripts/prepare_model.py"),"--trusted-pytorch-manifest",row["manifest"],
                "--trust-local-code","--resource-profile","general-tensors-v1","--write","--output",str(destination),"--inside"]
            child=subprocess.run(command,text=True,capture_output=True,timeout=min(185,left))
            (freeze/(row["id"]+"-cli.log")).write_text(child.stdout+child.stderr)
            require(child.returncode==0,"Current source capture failed; original and failure preserved")
            row["prior_imported"]=row["imported"];row["capture_identity_refreshed"]=True;captured+=1
        else:destination=Path(row["imported"]);row["capture_identity_refreshed"]=False
        prepare(row,destination,original)
    require(captured==46 and len(rows)==103,"Only 46 affected PyTorch bindings recaptured")
    chosen=["adaptive_pool_0_functional_query","adaptive_pool_1_functional_query","adaptive_pool_2_functional_query",
        "instance_norm_0_module_query","instance_norm_1_functional_query","instance_norm_2_module_query",
        "padding_0_module_query","padding_1_module_query","padding_2_module_query"]
    for name in chosen:
        row=dict(id="v10_"+name,input_kind="trusted_pytorch_v10",
            manifest=str(ROOT/"scripts/baseline/cases/trusted-pytorch-operator-shape-v10"/name/"manifest.json"),
            options=dict(generation_guidance="explicit-v10",capacity_packing_version=INDEX_VERSION),
            historical_status="new_input_not_run",source_origin="trusted_pytorch_rule_program",
            historical_batch="Original v9 shape/mapping rejection retained; explicit v10 acceptance",
            original_failed_case_preserved=True,model_and_reference_preserved=True,
            initial_source_generation="trusted rule, not Agent")
        prepare(row,WORK/"results/agent-operator-shape-public-v10-r988/captures"/name)
    require(len(rows)==112,"Complete retained scope plus v10 contexts")
    for i,row in enumerate(rows):row["shard"]=1+i//48
    preservation();require(PoseidonBackend().identity()==backend,"Compiled SDK identity drift")
    hashes=dict(frozen_old)
    for row in rows:hashes.update(row["files"])
    for name in ["agent-operator-shape-public-v10-r988/report.json","agent-operator-shape-application-v10-r989/report.json"]:
        path=WORK/"results"/name;hashes[str(path)]=sha(path)
    proposal=dict(format=old["format"],revision="r991",status="awaiting_human_approval",cases=rows,
        shards=[dict(id=1,count=48),dict(id=2,count=48),dict(id=3,count=16)],
        source_hashes=sources,case_hashes=hashes,capture_identity=capture_id,legacy_capture_identity=legacy_id,
        backend_identity=backend,matrix_identity=matrices,arithmetic_identity=arithmetic,
        package_reuse_cases=old["package_reuse_cases"]+["v10_adaptive_pool_0_functional_query","v10_instance_norm_1_functional_query","v10_padding_1_module_query"],
        explicit_domain_rejection_cases=old["explicit_domain_rejection_cases"],scope=old["scope"],
        tolerance=old["tolerance"],candidate_origin=old["candidate_origin"],new_agent_generation=False,paid_calls=0,http_attempts=0,
        previous_proposal=dict(binding=old["binding"],preserved=True,executed=False,source_stale=True),
        prior_approval="r870 24-task scope closed; r976 unexecuted proposal now stale",
        affected_plaintext_capture_count=captured,current_acceptance_binding=live["binding"],
        application_acceptance_binding=application["binding"],outcome_policy=old["outcome_policy"],exclusions=old["exclusions"])
    old_tools=ROOT/"scripts/baseline/benchmarks/tools/frozen_arithmetic_shape_backend"
    tools=ROOT/"scripts/baseline/benchmarks/tools/frozen_operator_shape_backend"
    require(not tools.exists(),"Fresh frozen backend drivers");tools.mkdir()
    for name in ("backend.py","audit.py","job.py","launch.py"):
        code=(old_tools/name).read_text()
        code=code.replace("agent-pytorch-shape-integration-r969","agent-pytorch-operator-shape-integration-r985")
        code=code.replace("agent-arithmetic-shape-backend-r977","agent-operator-shape-backend-r992")
        code=code.replace("agent-arithmetic-shape-offline-proposal-r976.json","agent-operator-shape-offline-proposal-r991.json")
        code=code.replace("r977","r992").replace("r978","r993")
        code=code.replace('("trusted_pytorch_v7","trusted_pytorch_v8","trusted_pytorch_v9")',
            '("trusted_pytorch_v7","trusted_pytorch_v8","trusted_pytorch_v9","trusted_pytorch_v10")')
        code=code.replace("==103,","==112,")
        compile(code,str(tools/name),"exec");(tools/name).write_text(code)
    for label in ("backend","references","history","new-history","current-history"):
        name="protected-"+label+".json";target=pre/name
        require(not target.exists(),"Immutable protection snapshot collision")
        target.write_bytes((WORK/"results/agent-pytorch-shape-integration-r969"/name).read_bytes())
    proposal["tool_hashes"]={str(path):sha(path) for path in tools.iterdir()}
    proposal["binding"]=digest(proposal);target=ROOT/"docs/baseline/agent-operator-shape-offline-proposal-r991.json"
    require(not target.exists(),"Fresh bound proposal");target.write_bytes(canonical(proposal))
    result=save_preparation("prepared_not_approved");result["proposal_binding"]=proposal["binding"]
    result["binding"]=digest({k:v for k,v in result.items() if k!="binding"});(pre/"prepared-backend-r990.json").write_bytes(canonical(result))
    print(json.dumps({k:result[k] for k in ("binding","planned","prepared","recaptured_sources","source_files","proposal_binding","seconds")}))
    return 0
if __name__=="__main__":raise SystemExit(run())
