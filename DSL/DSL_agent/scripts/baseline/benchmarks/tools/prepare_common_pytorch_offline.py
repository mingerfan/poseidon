"""Refresh source-only evidence and freeze offline scope; never compile or execute ciphertexts."""
import argparse,copy,hashlib,json,os,sys,time
from pathlib import Path
from benchmark_graph import canonical,digest,require,graph_resource_profile
from hecate_python_env import ROOT,VENV,WORK
from pytorch_model_intake import run as capture,source_identity,preparation_options
from trusted_pytorch_import import source_identity as legacy_identity
from candidate_bundle import load_model_provenance
from component_contract import prepare_task
from application_policy import prepare
from application_backend import PoseidonBackend
from application_execution.context import WorkerContext
from application_execution.backend import application
from model_source_provenance import reuse_identity,verify
from semantic_benchmark_execution import runtime_sources
from unified_graph_lowering import candidate_source
from operator_graph import implementation_sources as matrix_identity
from operator_arithmetic_graph import implementation_sources as arithmetic_identity
from strict_json_files import dump

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def bound(path):
    value=json.loads(Path(path).read_text())
    require(value["binding"]==digest({k:v for k,v in value.items() if k!="binding"}),"Historical input binding")
    return value
def files(folder):return {str(p):sha(p) for p in sorted(folder.rglob("*")) if p.is_file()}
def run(output):
    require(os.environ.get("IN_NIX_SHELL")=="pure" and Path(sys.prefix)==VENV,"Pinned pure preparation")
    require(not output.exists() and output.parent.resolve().is_relative_to(WORK/"results"),"Fresh preparation evidence")
    previous=ROOT/"docs/baseline/agent-application-source-offline-proposal-r1025.json"
    old=bound(previous)
    require(all(sha(Path(p) if Path(p).is_absolute() else ROOT/p)==h for p,h in old["case_hashes"].items()),
            "Original source/model/candidate/history preservation")
    public_dir=WORK/"results/agent-common-pytorch-v11-public-r1038"
    public=bound(public_dir/"report.json")
    require(public["passed"]==33 and public["failed"]==0,"Actual public V11 acceptance")
    capture_id=source_identity();sources=runtime_sources();backend=PoseidonBackend().identity()
    require(public["source_identity"]==capture_id and backend["tools"]==old["backend_identity"]["tools"],
            "Current capture; existing backend unchanged")
    output.mkdir(mode=0o700);(output/"imports").mkdir();(output/"prepared").mkdir()
    app_context=WorkerContext.capture(output/"source-tasks");app_context.persist();app=application(app_context)
    app_rows=[];rows=[];refreshed=0;started=time.monotonic()
    def finalize(case,folder,source_meta=None):
        model=json.loads((folder/"model.json").read_text())
        provenance=load_model_provenance(folder,model,max_bytes=4*1024**2)
        options=case["options"]
        request=prepare(model,"poseidon_cpu","numerical",options)[0] if source_meta else prepare_task(model,options)
        code=candidate_source(request)[0]
        destination=output/"prepared"/case["id"];destination.mkdir()
        dump(destination/"request.json",request);(destination/"candidate.py").write_text(code)
        item=copy.deepcopy(case)
        item.update(imported=str(folder),request=str(destination/"request.json"),candidate=str(destination/"candidate.py"),
                    request_id=request["request_id"],candidate_sha256=sha(destination/"candidate.py"),
                    model_sha256=digest(model),current_binding_reference_checked=True,fresh_reference_probes=16,
                    capture_identity_refreshed=True,files=files(folder)|files(destination))
        if source_meta:
            origin=source_meta["model_source"]
            verify(origin,request,provenance)
            item.update(source_receipt=origin,source_reuse_identity=reuse_identity(origin))
        return item
    for case in old["cases"]:
        require(time.monotonic()-started<300 and runtime_sources()==sources and source_identity()==capture_id,
                "Source preparation freeze/time")
        old_folder=Path(case["imported"]);old_model=json.loads((old_folder/"model.json").read_text())
        source_meta=None
        if case.get("source_api") is not None:
            task=app.submit_source(case["source_api"],validation_level="numerical",**case["source_requirements"])
            status=app.prepare_source(task)
            require(status["status"]=="queued","Original source application preparation: "+case["id"])
            source_meta=app._checked_task_metadata(task);folder=app._dir(task)/"model-import"
            require(app.cancel(task)["status"]=="cancelled","Cancel before generation")
            app_rows.append(dict(id=case["id"],task_id=task,reference_groups=16,status="intake_passed"))
        elif case["input_kind"].startswith("trusted_pytorch_"):
            folder=output/"imports"/case["id"]
            capture(case["manifest"],folder,resource_profile=graph_resource_profile(old_model),trusted=True);refreshed+=1
        else:folder=old_folder
        require(canonical(json.loads((folder/"model.json").read_text()))==canonical(old_model),
                "Original mathematical model changed: "+case["id"])
        item=finalize(case,folder,source_meta)
        require(Path(item["candidate"]).read_bytes()==Path(case["candidate"]).read_bytes(),
                "Original fixed DSL expression changed: "+case["id"])
        require(json.loads(Path(item["request"]).read_text())==json.loads(Path(case["request"]).read_text()),
                "Original frozen mathematical request changed: "+case["id"])
        item.update(prior_imported=str(old_folder),prior_request=str(case["request"]),prior_candidate=str(case["candidate"]),
                    capture_identity=legacy_identity() if case["input_kind"]=="trusted_pytorch_v7" else capture_id,
                    model_and_reference_preserved=True,request_hash_changed=False)
        rows.append(item)
        if len(rows)%12==0:print(json.dumps(dict(original_cases_prepared=len(rows),fresh_captures=refreshed)),flush=True)
    index=bound(ROOT/"scripts/baseline/cases/trusted-pytorch-common-v11/index.json")
    by_id={r["id"]:r for r in index["rows"]}
    for observed in public["rows"]:
        fixture=by_id[observed["id"]]
        item=dict(id="v11_"+observed["id"],input_kind="trusted_pytorch_v11",source_origin="trusted_pytorch_rule_program",
            initial_source_generation="trusted_rule_baseline",historical_status="new_input_not_run",
            historical_batch=None,historical_report=None,historical_candidate_sha256=None,
            manifest=fixture["manifest"],capture_identity=capture_id,
            options=dict(capacity_packing_version="capacity-public-tensors-v9",chunk_period=4,
                         generation_guidance="explicit-v10",public_weight_policy="descriptors-v1"),
            family=fixture["family"],context=fixture["context"],model_and_reference_preserved=True)
        rows.append(finalize(item,public_dir/"captures"/observed["id"]))
    for i,item in enumerate(rows):item["shard"]=i//48+1
    require(len(rows)==151 and len({row["id"] for row in rows})==151,"Finite full scope")
    require(runtime_sources()==sources and source_identity()==capture_id and PoseidonBackend().identity()==backend,
            "Preparation code/backend drift")
    require(all(sha(Path(p) if Path(p).is_absolute() else ROOT/p)==h for p,h in old["case_hashes"].items()),
            "Original evidence changed")
    historical_hashes={**old["case_hashes"],str(previous):sha(previous)}
    current_hashes={}
    for item in rows:
        current_hashes.update(item["files"]);current_hashes[item["manifest"]]=sha(item["manifest"])
    current_hashes.update(files(ROOT/"scripts/baseline/cases/trusted-pytorch-common-v11"))
    prepared=dict(format="poseidon-common-pytorch-offline-prepared-v1",previous_proposal=str(previous),
        previous_binding=old["binding"],previous_source_binding_outdated=True,cases=rows,
        original_cases=118,new_v11_contexts=33,fresh_old_pytorch_captures=refreshed,
        fresh_old_source_api_contexts=6,app_rows=app_rows,new_public_reference_binding=public["binding"],
        original_models_requests_and_candidates_preserved=True,source_hashes=sources,capture_identity=capture_id,
        legacy_capture_identity=legacy_identity(),matrix_identity=matrix_identity(),arithmetic_identity=arithmetic_identity(),
        backend_identity=backend,case_hashes=historical_hashes|current_hashes,
        scope=old["scope"],package_reuse_cases=old["package_reuse_cases"]+[
            "v11_residual_functional","v11_dropout_spatial","v11_public_alpha_none_0",
            "v11_projected_attention_none_0","v11_projected_attention_boolean_0",
            "v11_projected_attention_additive_2","v11_projected_attention_causal_1"],
        extra_source_hashes={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/"scripts/baseline/application_execution").rglob("*.py")},
        seconds=time.monotonic()-started,compiler_calls=0,encrypted_execution=False,paid_calls=0,new_agent_generation=False,
        status="prepared_only_no_backend_authorization",tool_sha256=sha(Path(__file__)))
    prepared["binding"]=digest(prepared);dump(output/"prepared.json",prepared)
    print(json.dumps({k:prepared[k] for k in ("binding","original_cases","new_v11_contexts","fresh_old_pytorch_captures","seconds")}))
    return 0
if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",required=True,type=Path)
    raise SystemExit(run(p.parse_args().output))
