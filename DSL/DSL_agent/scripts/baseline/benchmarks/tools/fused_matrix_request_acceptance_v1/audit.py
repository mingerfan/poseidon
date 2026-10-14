"""Audit original request artifacts and narrow rerun without changing evidence."""
import ast, hashlib, json, math, re, sys
from pathlib import Path
ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5])
HERE=Path(__file__).resolve().parent
RESULTS=Path("/home/lhohy/poseidon-work/platforms/aarch64-linux/results")
SOURCE=HERE.parent/"fused_matrix_source_draft_v1"
RUNS=("agent-fused-matrix-requests-r1274","agent-fused-matrix-requests-r1275")
def require(value,message):
    if not value:raise ValueError(message)
def read(path):
    path=Path(path)
    require(path.is_file() and not path.is_symlink(),"Regular audit evidence")
    return json.loads(path.read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def canonical(value):return json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
def digest(value):return hashlib.sha256(canonical(value)).hexdigest()
def methods(path):
    result={}
    for node in ast.parse(Path(path).read_text()).body:
        if isinstance(node,ast.ClassDef) and node.name=="RequestTests":
            for fn in node.body:
                if isinstance(fn,ast.FunctionDef) and fn.name.startswith("test_"):
                    result[fn.name]=digest(ast.dump(fn,include_attributes=False))
    return result
if len(sys.argv)!=2:raise SystemExit("Usage: audit.py FRESH_AUDIT_RESULT")
name=sys.argv[1];prefix="agent-fused-matrix-request-audit-r"
require(name.startswith(prefix) and name[len(prefix):].isdigit(),"Fresh audit name")
out=RESULTS/name;require(not out.exists(),"Fresh audit result")
plan=read(HERE/"plan.json")
parent=read(SOURCE/"plan.json")
require(all(sha(ROOT/p)==want for p,want in parent["production_hashes"].items()),"Production freeze changed")
require(all(sha(SOURCE/"targets"/p)==want for p,want in plan["target_hashes"].items()),"Source implementation changed")
for protected in ("protected-backend.json","protected-references.json"):
    records=read(RESULTS/"agent-common-pytorch-offline-prepared-r1040"/protected)
    require(all(sha(ROOT/p)==want for p,want in records.items()),"Protected backend/reference changed")
expected=methods(HERE/"inside_tests.py");require(len(expected)==6,"Current method inventory")
ok={};evidence_files={};runs=[];reports={}
for name in RUNS:
    folder=RESULTS/name;report=read(folder/"report.json");receipt=read(folder/"sandbox.json");reports[name]=report
    require(receipt["staged_implementation_hashes"]==plan["target_hashes"] and
            receipt["source_acceptance_binding"]==plan["acceptance_binding"],"Same implementation boundary")
    require(report["source_models"]==report["planned_source_models"]==9 and report["preparation_failures"]==[],
            "Nine prepared source cases")
    require(report["runtime"]==dict(python=[3,10,14],numpy="1.25.2",torch="2.0.1",cuda=None),"Pinned runtime")
    require(report["namespace"]!=receipt["host_network_namespace"] and
            receipt["clearenv"] and receipt["unshare_all"] and report["secret_probe_removed"],"Actual isolation")
    require(report["production_applied"] is False and report["source_module_imports"]==
            report["compiler_calls"]==report["encrypted_executions"]==report["provider_calls"]==0,
            "Preparation-only scope")
    require(report["errors"]==report["skipped"]==0,"Unexpected error/skip")
    require((report["status"]=="passed")==(receipt["exit_code"]==0),"Raw status/exit")
    for rel,want in receipt["source_hashes"].items():
        require(sha(folder/"source-snapshot"/rel)==want,"Original helper snapshot changed")
    snapshot=folder/"source-snapshot"/(HERE/"inside_tests.py").relative_to(ROOT)
    oldmethods=methods(snapshot)
    for method in re.findall(r"(test_[a-zA-Z0-9_]+) \(__main__\.RequestTests\) \.\.\. ok",
                             (folder/"tests.log").read_text()):
        require(oldmethods[method]==expected[method],"Successful test body changed")
        ok.setdefault(method,[]).append(name)
    for label in ("report.json","sandbox.json","tests.log","runner.log"):
        evidence_files[str(folder/label)]=sha(folder/label)
    runs.append(dict(run=name,status=report["status"],tests=report["tests"],failed=report["failed"],
                     errors=report["errors"],skipped=report["skipped"],seconds=receipt["seconds"]))
require(set(ok)==set(expected),"Current test method missing successful evidence")
source_report=reports[RUNS[0]]
patched=reports[RUNS[1]]
require(source_report["failed"]==9 and patched["failed"]==0 and patched["suite"]=="privacy","Known affected rerun")
require("AssertionError: {'hecate_source'} is not false" in (RESULTS/RUNS[0]/"tests.log").read_text(),
        "Retained schema-name false positive")
contexts=[]
for row in source_report["request_rows"]:
    folder=RESULTS/RUNS[0]/row["case"]
    model=read(folder/"model.json")
    provenance=read(folder/"model-provenance.json")
    req=read(folder/"request.json")
    ctx=read(folder/"generation-context.json")
    preflight=read(folder/"preflight.json")
    require(digest(model)==row["model_sha256"] and digest(provenance)==row["provenance_sha256"],
            "Original source/model identity")
    require(ctx["request_id"]==row["request_id"]==req["request_id"] and
            digest({k:v for k,v in ctx.items() if k!="request_id"})==ctx["request_id"],"Projected request identity")
    require(ctx["response_schema"]==dict(type="object",additionalProperties=False,
        properties=dict(schema=dict(type="integer",enum=[1]),request_id=dict(type="string"),
                        hecate_source=dict(type="string")),required=["schema","request_id","hecate_source"]),
            "Typed response declaration only")
    require(req["model"]==model and len(canonical(ctx))==row["generation_context_bytes"]<=131072,
            "Unchanged model and bounded context")
    require(preflight["admission"]=="accepted_pending_validation" and
            preflight["execution_readiness"]=="unknown","Honest readiness")
    for stage in ("compiler","encrypted","agent"):
        require(preflight["stages"][stage]["status"]=="unverified","No executable claim")
    require(set(ctx["public_constants"])==set(req["public_constants"]),"Same public symbol binding")
    for constant,value in req["public_constants"].items():
        descriptor=ctx["public_constants"][constant]
        require(set(descriptor)=={"kind","shape","elements","content_sha256"} and
                descriptor["content_sha256"]==digest(value) and
                descriptor["elements"]==math.prod(descriptor["shape"]), "Immutable descriptors")
    require(not set(ctx).intersection({"reference_report","test_inputs","hecate_source","capture_metadata",
                                      "model_provenance","decomposition"}),"Private top-level payload")
    for filename in ("model.json","model-provenance.json","request.json","generation-context.json","preflight.json"):
        evidence_files[str(folder/filename)]=sha(folder/filename)
    contexts.append(dict(case=row["case"],request_id=row["request_id"],
        generation_context_bytes=row["generation_context_bytes"],trusted_request_bytes=row["trusted_request_bytes"],
        period=row["period"],input_ciphertexts=row["input_ciphertexts"],output_ciphertexts=row["output_ciphertexts"],
        named_outputs=[o["name"] for o in model["outputs"]],model_sha256=digest(model),
        provenance_sha256=digest(provenance),admission=preflight["admission"],execution_readiness="unknown"))
# Check every retained input recorded by the narrow rerun, including the raw failed report.
for path,want in read(RESULTS/RUNS[1]/"sandbox.json")["prepared_inputs"].items():
    require(sha(path)==want,"Prepared original evidence changed")
report=dict(format="poseidon-fused-matrix-request-audit-v1",stage="staged_generation_context_verified",
    source_models=9,production_applied=False,source_acceptance_binding=plan["acceptance_binding"],
    current_methods=dict(successful=len(ok),methods=ok,method_body_sha256=expected,single_green_full_run=False),
    original_runs=runs,contexts=contexts,
    context_bytes=dict(min=min(r["generation_context_bytes"] for r in contexts),
                      max=max(r["generation_context_bytes"] for r in contexts),limit=131072),
    retained_failure=dict(run=RUNS[0],failed_subtests=9,layer="test_assertion",
        reason="The trusted response_schema properties declare the name hecate_source, not candidate source content",
        correction="Verify the exact trusted response schema, exclude only that schema from payload checks",
        affected_rerun=RUNS[1],production_code_changed=False,source_adapter_changed=False),
    implementation_hashes=plan["target_hashes"],evidence_files=evidence_files,
    compiler_calls=0,encrypted_executions=0,provider_calls=0,new_dependencies=0,backend_changes=0,
    output_scope="source/request preparation; no compiled program bundle",
    pending=dict(physical_layout_execution="unverified",compiler="not_run",poseidon_execution="not_run",
                 new_agent_generation="not_run"),audit_sha256=sha(__file__))
report["binding"]=digest(report)
out.mkdir(mode=0o700);(out/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
print(json.dumps(dict(result=str(out),binding=report["binding"],successful_current_methods=len(ok),
                     sources=9,context_bytes=report["context_bytes"])))
