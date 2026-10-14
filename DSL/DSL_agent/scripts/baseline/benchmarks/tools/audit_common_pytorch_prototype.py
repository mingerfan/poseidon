"""Verify staged common-call normalization without changing the public intake."""
import argparse,copy,hashlib,json,math,os,sys,time
from pathlib import Path
from benchmark_graph import canonical,digest,require
from hecate_python_env import ROOT,VENV,WORK
from strict_json_files import dump
from benchmarks.tools.audit_common_pytorch_idioms import bound,frozen,corpus_frozen
from benchmarks.tools.audit_operator_shape_capture_prototype import sandbox
from pytorch_model_intake import prepare,source_identity,check_reference,shim
from pytorch_weight_package import read
from model_decomposition import decompose
from trusted_pytorch_import import runtime_identity
from pytorch_arithmetic_normalization import check_record
from pytorch_static_trace import policy_record,OPERATOR_POLICY

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def prototype_sources(directory):
    value=bound(json.loads((directory/"prototype.json").read_text()))
    current=source_identity()
    require(value["production_integrated"] is False and value["backend_changes"]==value["reference_changes"]==0,
            "Staged adapter scope")
    require(set(value["staged_hashes"])==set(value["original_hashes"])==
            {"pytorch_model_intake.py","pytorch_arithmetic_normalization.py"},"Exact staged files")
    for name,h in value["original_hashes"].items():require(current[name]==h,"Production prototype base changed")
    for section in ("staged_hashes","patch_hashes"):
        for name,h in value[section].items():require(sha(directory/name)==h,"Staged artifact changed")
    staged=dict(current);staged.update(value["staged_hashes"])
    return value,current,staged

def verify_reference(report,original,lowered,spec,expected):
    import trusted_pytorch_import as legacy
    from model_resources import identity
    require(report["source_hashes"]==expected and report["manifest_sha256"]==digest(spec) and
            report["original_sha256"]==digest(original) and report["model_sha256"]==digest(lowered),
            "Staged capture identity")
    require(report["runtime"]==runtime_identity() and report["resource_profile"]==identity("general-tensors-v1")
            and report["probes"]==16,"Staged runtime and scope")
    require(report["format"]=="poseidon-trusted-pytorch-reference-v10" and
            report["reference_status"]=="passed" and report["comparison_tolerance"]==dict(atol=1e-12,rtol=1e-12),
            "Staged finite reference")
    for key in ("original_module_checked","normalized_module_checked","arithmetic_semantics_preserved","pure_forward_checked"):
        require(report[key] is True,"Staged reference check missing")
    for key in ("encrypted_execution","agent_generated","runtime_layout_validated","approximation_accuracy_certified"):
        require(report[key] is False,"Staged scope cannot claim backend acceptance")
    require(report["paid_calls"]==0,"No generation")
    for key in ("max_reference_tolerance_ratio","max_arithmetic_normalization_ratio"):
        require(type(report[key]) in (float,int) and math.isfinite(report[key]) and 0<=report[key]<=1,
                "Staged reference comparison")
    metadata=copy.deepcopy(report["capture_metadata"])
    check_record(metadata.pop("arithmetic_capture"),original)
    require(metadata.pop("source_shape_capture")==policy_record(spec["inputs"],"general-tensors-v1",policy=OPERATOR_POLICY),
            "Unchanged static capture policy")
    legacy.check_capture_metadata(metadata,original,shim(spec),resource_profile="general-tensors-v1")
    projected=copy.deepcopy(report)
    projected.update(format=legacy.REFERENCE_V7,manifest_sha256=digest(shim(spec)),
                     source_hashes=legacy.source_identity(),capture_metadata=metadata)
    legacy.check_reference(projected,original,lowered,shim(spec),resource_profile="general-tensors-v1")
    try:check_reference(report,original,lowered,spec,resource_profile="general-tensors-v1")
    except ValueError as error:require(str(error)=="Arithmetic reference source identity","Unexpected production rejection")
    else:raise ValueError("Staged result accepted as production evidence")

def run(previous,prototype,output):
    require(os.environ.get("IN_NIX_SHELL")=="pure" and Path(sys.prefix)==VENV,"Existing pinned environment")
    require(not output.exists() and output.parent.resolve().is_relative_to(WORK/"results"),"Fresh staged evidence")
    proposal=bound(json.loads((ROOT/"docs/baseline/agent-application-source-offline-proposal-r1025.json").read_text()))
    index=bound(json.loads((ROOT/"scripts/baseline/cases/trusted-pytorch-common-idioms-v1/index.json").read_text()))
    old=bound(json.loads(previous.read_text()))
    require(old["passed"]==13 and old["failed"]==5 and old["corpus_binding"]==index["binding"],"Exact retained gap audit")
    stage,current,expected=prototype_sources(prototype)
    require(old["source_identity"]==current and stage["corpus_binding"]==index["binding"],"Unchanged source and models")
    freeze=frozen(proposal);corpus_frozen(index)
    output.mkdir(mode=0o700);(output/"captures").mkdir()
    rows=[];tick=time.monotonic()
    failed={r["id"] for r in old["rows"] if r["status"]=="failed"}
    for item in index["rows"]:
        require(time.monotonic()-tick<120,"Staged plaintext time boundary")
        require(prototype_sources(prototype)[2]==expected,"Staged source changed")
        manifest=Path(item["manifest"]);spec,paths=prepare(manifest)
        _,weights=read(manifest.parent,spec["weights"],resource_profile="general-tensors-v1")
        target=output/"captures"/item["id"]
        code=sandbox(spec,paths,weights,target,expected,prototype,30)
        row=dict(id=item["id"],family=item["family"],source_gap=item["id"] in failed,
                 exit_code=code,log_sha256=sha(target/"capture.log"),production_integrated=False,
                 compiler_calls=0,encrypted_execution=False,paid_calls=0)
        if code:
            row.update(status="failed",diagnostic=(target/"capture.log").read_text()[-4500:],reference_groups=0)
        else:
            original=json.loads((target/"original.json").read_text());lowered,_=decompose(original,resource_profile="general-tensors-v1")
            report=json.loads((target/"capture-reference.json").read_text())
            verify_reference(report,original,lowered,spec,expected)
            normalization=bound(json.loads((target/"common-normalization.json").read_text()))
            require(normalization["production_integrated"] is False and normalization["source_preserved"] is True and
                    normalization["public_state_preserved"] is True and normalization["input_signature"]==spec["inputs"],
                    "Explicit common source transformation record")
            row.update(status="passed",reference_groups=16,original_sha256=digest(original),model_sha256=digest(lowered),
                       production_evidence_rejected=True,normalization_rules=[e["rule"] for e in normalization["entries"]],
                       max_normalization_error=report["max_arithmetic_normalization_error"],
                       max_decomposition_error=report["max_decomposition_error"],
                       max_approximation_error=report["max_original_function_approximation_error"],
                       reference_sha256=sha(target/"capture-reference.json"),
                       normalization_sha256=sha(target/"common-normalization.json"))
        rows.append(row);dump(output/"progress.json",dict(rows=rows))
        print(json.dumps({k:row[k] for k in ("id","status")}),flush=True)
    require(frozen(proposal)==freeze and prototype_sources(prototype)[2]==expected,"Final production freeze")
    corpus_frozen(index)
    report=dict(format="poseidon-common-pytorch-prototype-audit-v1",prototype_binding=stage["binding"],
                corpus_binding=index["binding"],original_audit_binding=old["binding"],proposal_binding=proposal["binding"],
                production_integrated=False,current_source_hashes=current,staged_source_hashes=expected,
                passed=sum(r["status"]=="passed" for r in rows),failed=sum(r["status"]=="failed" for r in rows),
                skipped=0,original_gaps_passed=sum(r["source_gap"] and r["status"]=="passed" for r in rows),
                controls_passed=sum(not r["source_gap"] and r["status"]=="passed" for r in rows),
                production_reference_rejections=sum(r.get("production_evidence_rejected",False) for r in rows),
                reference_groups=sum(r["reference_groups"] for r in rows),rows=rows,seconds=time.monotonic()-tick,
                tool_sha256=sha(__file__),sandbox_tool_sha256=sha(ROOT/"scripts/baseline/benchmarks/tools/audit_operator_shape_capture_prototype.py"),
                production_freeze_preserved=True,compiler_calls=0,encrypted_execution=False,agent_generated=False,paid_calls=0)
    report["binding"]=digest(report);dump(output/"report.json",report)
    print(json.dumps({k:report[k] for k in ("binding","passed","failed","original_gaps_passed","controls_passed","reference_groups")}))
    return 0 if report["passed"]==18 else 1

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for key in ("previous","prototype","output"):p.add_argument("--"+key,type=Path,required=True)
    a=p.parse_args();raise SystemExit(run(a.previous,a.prototype,a.output))
