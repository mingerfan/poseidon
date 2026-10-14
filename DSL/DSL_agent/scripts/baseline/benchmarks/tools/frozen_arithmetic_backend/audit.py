"""Independent bounded audit; no compile, inference generation or API."""
import copy,hashlib,json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path.cwd()/"scripts/baseline"))
from benchmark_graph import canonical,digest,require,samples
from benchmark_math import evaluate as mathematical
from benchmark_torch import evaluate as tensor_reference
from operator_graph import implementation_sources as matrix_identity
from operator_arithmetic_graph import evaluate_reference as high_reference, implementation_sources as arithmetic_identity
from numerical_results import compare
from semantic_benchmark_execution import runtime_sources
from application_backend import PoseidonBackend
from candidate_bundle import check_execution_dependencies,load_model_provenance,check_provenance
from component_contract import reconstruct_request
from pytorch_model_intake import source_identity
from trusted_pytorch_import import source_identity as prior_source_identity
root=Path.cwd();work=Path("/home/lhohy/poseidon-work/platforms/aarch64-linux/results")
pre=work/"agent-public-arithmetic-preflight-r935";out=work/"agent-arithmetic-matrix-backend-r951"
load=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
plan=load(out/"plan.json");report=load(out/"report.json")
require(report["status"]=="complete" and report["not_run"]==0 and len(report["rows"])==len(plan["cases"])==94,"Complete bounded original task set")
require(digest({k:v for k,v in plan.items() if k!="binding"})==plan["binding"]==report["binding"],"Plan/report binding")
require(runtime_sources()==plan["source_hashes"] and source_identity()==plan["capture_identity"] and matrix_identity()==plan["matrix_identity"] and arithmetic_identity()==plan["arithmetic_identity"] and prior_source_identity()==plan["legacy_capture_identity"],"Current code and trusted capture identity")
require(PoseidonBackend().identity()==plan["backend_identity"],"Backend identity")
require(sha(Path(__file__).with_name("backend.py"))==plan["tool_sha256"] and sha(Path(__file__))==plan["audit_tool_sha256"],"Frozen execution and audit tools")
for p,h in plan["case_hashes"].items():require(sha(p)==h,"Frozen source and original-equivalent graph")
prepared=load(out/"prepared-bindings.json")
for p,h in prepared["files"].items():require(sha(p)==h,"Frozen imported model/reference/request/rule source")
protected={}
for label in ("backend","references","history","new-history","current-history"):
    hashes=load(pre/("protected-"+label+".json"))
    require(all(sha(root/p)==h for p,h in hashes.items()),"Protected "+label)
    protected[label]=len(hashes)
expected_limits=dict(AS=[4294967296,4294967296],CPU=[150,155],FSIZE=[16777216,16777216],NOFILE=[128,128],CORE=[0,0])
def expected(model,inputs):
    a,b=mathematical(model,inputs),tensor_reference(model,inputs)
    for key in a:np.testing.assert_allclose(a[key],b[key],atol=2e-12,rtol=2e-12)
    return np.concatenate([a[o["name"]].reshape(-1) for o in model["outputs"]])
def checked(c,reference):
    require(c["atol"]==1e-5 and c["rtol"]==1e-4,"Frozen numerical threshold")
    np.testing.assert_allclose(c["reference"],reference,atol=2e-12,rtol=2e-12)
    value=compare(np.asarray(c["actual"]),np.asarray(reference),1e-5,1e-4)
    require(value["passed"] and c["passed"] and value["compared_values"]==c["compared_values"] and
        value["max_absolute_error"]==c["max_absolute_error"],"Independent pointwise encrypted check")
    return {k:value[k] for k in ("passed","compared_values","mae","max_absolute_error","max_nonzero_reference_relative_error","cosine_similarity")}
def execution(e,gate):
    require(e["backend"]=="poseidon_cpu" and e["encrypted_execution"] and not e["executor_secret_key"] and not e["bootstrap"],"Ciphertext execution and evaluation-key boundary")
    require(e["security"]=="tc128" and e["degree"]==32768 and e["data_moduli"]==13 and e["special_moduli"]==1 and e["q"]==gate["q"] and e["p"]==gate["p"],"Exact approved parameters")
rows=[];groups=values=package_files=artifact_files=references=rejections=equivalence=0;maximum=0.
for row in report["rows"]:
    name=row["id"]
    if not row["input_checked"]:
        require(row["status"]=="failed" and row["stage"]=="capture_or_preparation","Original capture failure preserved")
        rows.append(dict(id=name,status="failed",layer="input_capture_or_reference",error=row["error"],captured=False,
            source_origin=row["source_origin"],compiled=False,encrypted_execution=False))
        continue
    imported=Path(row["imported"]);model=load(imported/"model.json");provenance=load_model_provenance(imported,model)
    check_provenance(provenance,model);ref=provenance["reference_report"]
    if row["input_kind"] in ("trusted_pytorch_v7","trusted_pytorch_v8"):
        expected_sources=prior_source_identity() if row["input_kind"]=="trusted_pytorch_v7" else source_identity()
        require(ref["probes"]==16 and ref["original_module_checked"] and ref["pure_forward_checked"] and
                ref["source_hashes"]==expected_sources,"Exact retained trusted reference")
    else:
        expected_sources=matrix_identity() if row["input_kind"]=="declared_json_v2" else arithmetic_identity()
        require(row["input_kind"] in ("declared_json_v2","declared_json_v3") and ref["mode"]=="offline_model_import" and
            ref["probes"]==16 and provenance["decomposition"]["implementation_sources"]==expected_sources,
            "Independent JSON version reference binding")
        # Retained imports record an older CLI dispatcher. Its math/decomposition
        # sources remain exact, and current references are recomputed below.
        changed={p:(h,sha(root/"scripts/baseline"/p)) for p,h in ref["source_hashes"].items()
                 if sha(root/"scripts/baseline"/p)!=h}
        require(set(changed)<={"model_import_cli.py"},"Retained reference computation source drift")
        row["retained_reference_dispatcher_drift"]=changed
    original_manual=load(work/"agent-pytorch-split-equivalence-r858"/(name+"-import/model.json")) if row["origin"]=="original_failed_pytorch_source" else None
    for inputs in samples(model,16):
        a=expected(model,inputs);high=high_reference(provenance["original"],inputs)
        np.testing.assert_allclose(a,np.concatenate([high[o["name"]].reshape(-1) for o in model["outputs"]]),atol=2e-12,rtol=2e-12);references+=1
        if original_manual is not None:
            np.testing.assert_allclose(a,expected(original_manual,inputs),atol=2e-12,rtol=2e-12);equivalence+=1
    for edit in ("missing","drift"):
        bad=copy.deepcopy(provenance)
        if row["input_kind"] in ("trusted_pytorch_v7","trusted_pytorch_v8"):
            if edit=="missing":bad["reference_report"]["source_hashes"].pop("pytorch_split_plan.py")
            else:bad["reference_report"]["source_hashes"]["pytorch_split_plan.py"]="0"*64
        else:
            if edit=="missing":bad["decomposition"]["implementation_sources"].pop("operator_graph.py")
            else:bad["decomposition"]["implementation_sources"]["operator_graph.py"]="0"*64
        try:check_provenance(bad,model)
        except ValueError:rejections+=1
        else:raise ValueError("Invalid input implementation identity accepted")
    if not row.get("request_id"):
        require(row["status"]=="failed","Preparation failure retained")
        rows.append(dict(id=name,status="failed",layer="rule_preparation",error=row["error"],captured=True,
            source_origin=row["source_origin"],compiled=False,encrypted_execution=False));continue
    request=reconstruct_request(load(out/"prepared"/name/"request.json"))
    require(request["request_id"]==row["request_id"] and digest(model)==row["model_sha256"] and digest(provenance["original"])==row["original_sha256"] and request["model"]==model,"Source mathematical binding")
    require(sha(out/"prepared"/name/"candidate.py")==row["candidate_sha256"] and row["source_origin"]==("trusted_pytorch_rule_program" if row["input_kind"] in ("trusted_pytorch_v7","trusted_pytorch_v8") else "declared_json_rule_program"),"Fixed rule provenance")
    if row["status"]!="passed":
        require(row["status"]=="failed" and row["result"]["status"]!="succeeded","Failure excluded from pass")
        validation=load(out/"tasks"/row["task_id"]/"validation-0.json");evidence=Path(validation["evidence"])
        failed=load(evidence/"attempt-00/report.json");log=evidence/"attempt-00/compile.log"
        rows.append(dict(id=name,status="failed",failure=row["result"]["failure"],source_origin=row["source_origin"],
            captured=row["captured"],input_checked=True,checked=failed["checked"],compiled=failed["compiled"],encrypted_execution=failed["executed"],
            failure_layer=failed["failure_layer"],compiler_log_sha256=sha(log) if log.exists() else None))
        continue
    package=Path(row["package"]);manifest,bound=check_execution_dependencies(package)
    require(bound==request and not manifest["agent_generated"] and manifest["model_provenance"]=="included" and
        manifest["validation"]==dict(level="numerical",encrypted_execution=True,numerically_validated=True),"Qualified reusable package")
    require(sha(package/"candidate.py")==row["candidate_sha256"],"Fixed rule source")
    for name,h in manifest["files"].items():require(sha(package/name)==h,"Package hash");package_files+=1
    validation=load(package.parent/"validation-0.json");evidence=Path(validation["evidence"])
    observed=load(evidence/"report.json");attempt=load(evidence/"attempt-00/report.json");probe=observed["sandbox_probe"]
    require(all(probe[k] is True for k in ("frontend_loaded","private_sentinel_hidden","workspace_hidden","keys_hidden","environment_cleared","network_namespace_changed","pid_namespace_changed","only_loopback_interface","request_read_only","resource_limits_applied")) and probe["resource_limits"]==expected_limits,"Real resource/network sandbox")
    require(all(attempt[k] for k in ("parsed","checked","compiled","executed","numerically_correct")),"True compilation/execution/numerical stages")
    require(attempt["trace"]["frontend"]=="real_Hecate" and attempt["trace"]["candidate_python_executed"] is False,"Checked AST and real frontend")
    require(observed["qualification"]["generation_performed"] is False and row["result"]["http_attempts"]==0,"No hidden generation/API")
    for name,h in attempt["artifact_hashes"].items():require(sha(evidence/"attempt-00/output"/name)==h,"Artifact hash");artifact_files+=1
    for name in ("candidate_trace.mlir","lowered.earth.mlir","lowered.ckks.mlir"):
        require((package/"artifacts"/name).stat().st_size>0,"Nonempty real IR")
    gate=attempt["artifact_gate"]
    require(gate["raw"]["plan_sha256"]=="sha256:"+sha(package/"artifacts/lowered._hecate_golden.runtime-plan.json") and
        gate["bound"]["plan_sha256"]=="sha256:"+sha(package/"artifacts/bound/plan.json") and
        gate["raw"]["spec_sha256"]==gate["bound"]["spec_sha256"]=="sha256:"+sha(package/"artifacts/operator-spec.json"),"RuntimePlan/OperatorSpec integrity")
    execution(attempt["execution"],gate["bound"])
    c=checked(attempt["comparison"],np.asarray([expected(model,p) for p in samples(model,4)]))
    groups+=4;values+=c["compared_values"];maximum=max(maximum,c["max_absolute_error"])
    data=dict(id=row["id"],status="passed",captured=row["captured"],input_checked=True,request_id=row["request_id"],package=str(package),comparison=c,
        original_function_max_approximation_error=ref["max_original_function_approximation_error"],
        max_decomposition_error=ref["max_decomposition_error"],max_original_module_error=ref.get("max_original_module_error"),
        approximation_accuracy_certified=ref["approximation_accuracy_certified"],source_origin=row["source_origin"],
        compiled=True,encrypted_execution=True,numerically_validated=True,original_manual_equivalence_groups=row["original_manual_equivalence_groups"])
    if row["id"].startswith("scalar_domain_"):
        rejection=row["domain_rejection"]
        require(rejection["status"]=="validation_failed" and not rejection["encrypted_execution"] and
            rejection["failure"]["code"]=="approximation_domain_rejected" and
            rejection["failure"]["location"]["id"]=="activation" and
            rejection["agent_calls"]==rejection["compiler_calls"]==0,"Pre-encryption explicit-domain rejection")
        data["domain_rejection"]=rejection
    if "reload" in row:
        reload=row["reload"]
        require(reload["compiled_artifacts_reused"] and reload["agent_calls"]==reload["compiler_calls"]==0 and reload["encrypted_execution"] and reload["numerically_validated"],"Reload without regeneration/recompile")
        execution(reload["execution"],gate["bound"])
        rc=checked(reload["comparison"],np.asarray([expected(model,p) for p in samples(model,4)]))
        groups+=4;values+=rc["compared_values"];maximum=max(maximum,rc["max_absolute_error"]);data["reload_comparison"]=rc
        supplied=load(out/(row["id"]+"-new-inputs.json"));new_reference=np.asarray([expected(model,{k:np.asarray(v) for k,v in sample.items()}) for sample in supplied])
        require(len(row["new_input_runs"])==2 and len(supplied)==3,"Two runs of three supplied inputs")
        for run in row["new_input_runs"]:
            raw=load(Path(run["evidence"])/"report.json")
            require(raw["encrypted_execution"] and raw["executor_secret_key"] is False and raw["agent_calls"]==raw["compiler_calls"]==0 and not raw["numerically_validated_on_supplied_inputs"],"Supplied inputs separately reference-audited")
            execution(raw["execution"],gate["bound"])
            actual=np.asarray([np.concatenate([np.asarray(sample[o["name"]]).reshape(-1) for o in model["outputs"]]) for sample in raw["outputs"]])
            require(np.array_equal(actual,np.asarray(run["comparison"]["actual"])),"Raw output versus compared value")
            nc=checked(run["comparison"],new_reference);groups+=3;values+=nc["compared_values"];maximum=max(maximum,nc["max_absolute_error"])
        require(row["cache_reuse"]==dict(status="succeeded",reused=True,generations=0,http_attempts=0),"Verified cache reuse")
        data["new_input_runs"]=[dict(repetition=run["repetition"],evidence=run["evidence"],comparison={k:v for k,v in run["comparison"].items() if k not in ("actual","reference","absolute_error","relative_error","elementwise_pass")},agent_calls=0,compiler_calls=0) for run in row["new_input_runs"]]
        data["cache_reused"]=True
    rows.append(data)
audit=dict(format="poseidon-arithmetic-matrix-offline-audit-v1",status="passed",binding=plan["binding"],
    planned=len(plan["cases"]),passed=sum(r["status"]=="passed" for r in rows),failed=sum(r["status"]=="failed" for r in rows),not_run=0,
    captured=report["captured"],input_checked=report["input_checked"],rows=rows,positive_input_groups=groups,compared_values=values,max_absolute_error=maximum,
    independent_reference_groups=references,original_manual_equivalence_reference_groups=equivalence,
    capture_identity_rejections=rejections,package_file_hashes_checked=package_files,compiled_artifact_hashes_checked=artifact_files,
    source_hashes_checked=len(plan["source_hashes"]),protected=protected,
    audit_tool_sha256=sha(Path(__file__)),paid_calls=0,http_attempts=0,new_agent_generation=False,bootstrap=False,cuda=False)
target=pre/"audit-r952.json";require(not target.exists(),"Fresh audit snapshot")
target.write_bytes(canonical(audit))
print(json.dumps({k:v for k,v in audit.items() if k!="rows"},ensure_ascii=False),flush=True)
