"""Audit completed frozen execution evidence; no generation, compilation or re-execution."""
import copy,hashlib,json,math
from pathlib import Path
import numpy as np
from benchmark_graph import canonical,digest
from benchmark_math import evaluate
from numerical_results import compare
from public_weight_context import read_request
from component_contract import reconstruct_request
from candidate_bundle import check_execution_dependencies,check_bundle
from application_backend import PoseidonBackend
from program_execution import pack_samples,named_outputs,package_execution_metadata
from poseidon_validation import input_payload
from audit_unified_candidate import verify_candidate
from hecate_python_env import WORK,ROOT,digest as file_digest

ROOT_EVIDENCE=WORK/"results/agent-resource-gelu-offline-v2"
REPORT=ROOT_EVIDENCE/"report.json"
OUT=WORK/"results/agent-resource-gelu-post-reference-audit-v1"
def require(ok,why):
    if not ok:raise ValueError(why)
def main():
    require(not OUT.exists(),"preserve prior post-reference audit")
    baseline=json.loads(REPORT.read_bytes())
    require(baseline["status"]=="complete" and len(baseline["cases"])==4,"closed original batch")
    authority=PoseidonBackend()
    require(authority.identity()==baseline["backend_identity"] and authority.identity_manifest()==baseline["stage_identity"],
            "original execution sources changed")
    OUT.mkdir();rows=[];numerical=[]
    for previous in baseline["cases"]:
        name=previous["id"];case=ROOT_EVIDENCE/name
        require(previous["status"]=="failed" and previous["diagnostic"]=="'list' object has no attribute 'dtype'",
                "expected developer reference invocation failure")
        original=hashlib.sha256(REPORT.read_bytes()).hexdigest()
        package=case/"program";manifest,request=check_execution_dependencies(package)
        qualification=json.loads((case/"qualification.json").read_bytes())
        q=verify_candidate(Path(qualification["evidence"]))
        q_report=json.loads((Path(qualification["evidence"])/"report.json").read_bytes())
        q_comparison=next(a["comparison"] for a in q_report["attempts"] if a["status"]=="passed")
        matches=[]
        for p in (WORK/"results").glob("poseidon-program-execution-*/report.json"):
            row=json.loads(p.read_bytes())
            if row.get("bundle_binding")==manifest["binding"] and row.get("request_id")==request["request_id"]:
                matches.append((p,row))
        require(len(matches)==1,"one actual native execution under this package binding")
        file,execution=matches[0];out=file.parent/"output"
        require(execution["status"]=="executed" and execution["encrypted_execution"] is True and
                execution["executor_secret_key"] is False and execution["agent_calls"]==0 and
                execution["compiler_calls"]==0,"actual execution facts")
        samples=[]
        for group in range(3):
            sample={}
            for item in request["model"]["inputs"]:
                n=math.prod(item["shape"]);values=(np.arange(n,dtype=np.float64)+1)/(n+1)*.35
                if group==0:values=-values
                if group==2:values=np.cos(np.arange(n,dtype=np.float64))*.27
                sample[item["name"]]=values.reshape(item["shape"]).tolist()
            samples.append(sample)
        expected_payload=input_payload(pack_samples(request,samples),request["layout"])
        require(json.loads((out/"inputs.json").read_bytes())==expected_payload,"exact new-input frozen formulas")
        actual=np.load(out/"decrypted.npy",allow_pickle=False)
        named=named_outputs(request,actual,len(samples))
        require(named==execution["outputs"],"named output binding")
        expected=[]
        for sample in samples:
            arrays={k:np.asarray(v,dtype=np.float64) for k,v in sample.items()}
            calculated=evaluate(request["model"],arrays)
            expected.append(np.concatenate([calculated[o["name"]].reshape(-1) for o in request["model"]["outputs"]]))
        comparison=compare(actual,np.asarray(expected,dtype=np.float64),1e-5,1e-4)
        require(comparison["passed"],"frozen new-input numerical tolerance")
        from application_planning.compiled_resources import execution_storage_precheck
        resource=execution_storage_precheck(out,request)
        require(resource==execution["resource_precheck"],"actual execution resource identity")
        if name=="tanh_input_keyword":
            metadata=package_execution_metadata(package,manifest,request["model"])
            from pytorch_source_contract import restore_outputs
            tree=metadata["output_structure"]
            require(tree["kind"]=="tuple" and execution["output_structure"]==tree,
                    "v15 original tuple structure")
            restored=[restore_outputs(x,tree) for x in named]
            require(all(type(x) is tuple and len(x)==2 for x in restored),"reloaded tuple output protocol")
        require(check_bundle(package)==manifest and hashlib.sha256(REPORT.read_bytes()).hexdigest()==original,
                "original evidence untouched")
        (OUT/(name+"-comparison.json")).write_bytes(canonical(comparison))
        phases=[dict(kind="qualification",groups=4,comparison=q_comparison),
                dict(kind="new_input",groups=3,comparison=comparison)]
        numerical.extend(phases)
        rows.append(dict(id=name,status="passed",source_kind=previous["source_kind"],
            historical_source_sha256=previous.get("historical_source_sha256"),
            candidate_sha256=file_digest(case/"candidate.py"),request_id=request["request_id"],
            qualification_evidence=qualification["evidence"],execution_evidence=str(file.parent),
            compilation_performed_in_original_batch=True,encrypted_execution_performed_in_original_batch=True,
            package_reloaded=True,new_input_numeric_verified=True,repeat_execution_verified=True,
            new_agent_generation=False,phases=[dict(kind=x["kind"],groups=x["groups"],
                values=x["comparison"]["compared_values"],max_absolute_error=x["comparison"]["max_absolute_error"])
                for x in phases],
            resource_binding=resource["binding"],resource_lower_bytes=resource["facts"]["known_retained_payload_lower_bytes"],
            resource_budget_bytes=resource["limits"]["memory_budget_bytes"],
            peak_memory_certified=False,original_application_accuracy_certified=False,
            package_binding=manifest["binding"],program_files=len(manifest["files"])))
    require(authority.identity()==baseline["backend_identity"] and authority.identity_manifest()==baseline["stage_identity"],
            "source identity at final audit")
    protected=json.loads((ROOT/"scripts/baseline/benchmarks/tools/source_live_rebind_v2/proposal-ready.json").read_bytes())["protected_sources"]
    require(all(file_digest(ROOT/p)==h for p,h in protected.items()),"protected backend/reference/x86 sources changed")
    count=sum(x["comparison"]["compared_values"] for x in numerical)
    report=dict(format="poseidon-resource-gelu-composite-acceptance-v1",status="complete",
        passed=4,failed=0,skipped=0,not_run=0,cases=rows,
        source_batch_report=str(REPORT),source_batch_sha256=file_digest(REPORT),source_batch_binding=baseline["binding"],
        original_failed_report_preserved=True,correction_scope="developer reference API input converted to float64 arrays; no model/reference/tolerance change",
        source_identity=baseline["stage_identity"],backend_identity=baseline["backend_identity"],
        groups=sum(x["groups"] for x in numerical),values=count,
        max_absolute_error=max(x["comparison"]["max_absolute_error"] for x in numerical),
        mae=sum(x["comparison"]["mae"]*x["comparison"]["compared_values"] for x in numerical)/count,
        compiled_programs=4,encrypted_phases=8,
        this_post_audit=dict(generation=0,http=0,compilation=0,encrypted_execution=0),
        protected_sources_checked=len(protected),paid_api_calls=0,downloads=0,installs=0,
        backend_edits=0,tolerance=dict(atol=1e-5,rtol=1e-4))
    report["binding"]=digest(report)
    (OUT/"report.json").write_bytes(canonical(report))
    print(json.dumps({k:report[k] for k in ["status","passed","groups","values","max_absolute_error","mae","compiled_programs","encrypted_phases","binding"]}),flush=True)
    return 0
if __name__=="__main__":raise SystemExit(main())
