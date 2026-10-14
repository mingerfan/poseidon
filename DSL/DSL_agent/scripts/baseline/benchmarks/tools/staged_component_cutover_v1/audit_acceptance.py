"""Read-only audit of completed joint-source Poseidon acceptance receipts."""
import argparse,hashlib,json,math
from pathlib import Path
ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5])
RESULTS=Path("/home/lhohy/poseidon-work/platforms/aarch64-linux/results")
CASES=("linear","norm","split_weight")
def require(value,message):
    if not value:raise ValueError(message)
def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def canonical(value):return json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
def comparison(data):
    require(data["atol"]==1e-5 and data["rtol"]==1e-4,"Frozen tolerance")
    a,b=data["actual"],data["reference"]
    require(type(a) is list and type(b) is list and len(a)==len(b)>0,"Compared input groups")
    count=0;maximum=0.;total=0.
    for left,right in zip(a,b):
        require(type(left) is list and type(right) is list and len(left)==len(right)>0,"Output rows")
        for x,y in zip(left,right):
            require(type(x) in (int,float) and type(y) in (int,float) and math.isfinite(x) and math.isfinite(y),"Finite values")
            error=abs(x-y)
            require(error<=1e-5+1e-4*abs(y),"Elementwise numerical rejection")
            count+=1;total+=error;maximum=max(maximum,error)
    require(data["passed"] is True and data["compared_values"]==count,"Compared value accounting")
    return dict(input_groups=len(a),values=count,maximum_absolute_error=maximum,mae=total/count)
def inspect(root,output):
    root=Path(root);output=Path(output)
    require(root==root.resolve() and root.parent==RESULTS and root.is_dir(),"Platform acceptance root")
    require(output==output.resolve() and output.is_relative_to(RESULTS) and not output.exists(),"Fresh audit output")
    plan=read(root/"plan.json");report=read(root/"report.json")
    require(plan["binding"]==hashlib.sha256(canonical({k:v for k,v in plan.items() if k!="binding"})).hexdigest(),"Plan binding")
    require(report["binding"]==plan["binding"] and report["status"]=="passed" and len(report["rows"])==9,"Complete three-case receipt")
    require(report["provider_calls"]==report["http_attempts"]==0 and report["new_agent_generation"] is False,"Rule-only provenance")
    require(all(sha(path)==want for path,want in plan["files"].items()),"Reviewed fixture/driver bytes")
    production=read(root/"production-sources.json")
    require(all(sha(ROOT/path)==want for path,want in production.items()),"Current production source binding")
    hashes={str(root/"plan.json"):sha(root/"plan.json"),str(root/"report.json"):sha(root/"report.json")}
    rows=[];file_checks=0;groups=values=0;maximum=0.
    preflight=read(plan["request_preflight"]["report"])
    for name in CASES:
        q,n,c=[read(root/name/(phase+".json")) for phase in ("qualification","new_inputs","cache")]
        require(all(row["case"]==name and row["status"]=="passed" for row in (q,n,c)),"Completed independent phases")
        require(q["source_capture_supervisor_executed"] is True and q["compiled"] is True and q["encrypted_execution"] is True,"Real source/compiler/backend")
        require(q["result"]["intake"]["reference_verified"] is True and q["result"]["generation"]["agent_generated"] is False,"Checked source/reference provenance")
        package=Path(q["package"])
        require(package==package.resolve() and package.is_relative_to(root/name) and not package.is_symlink(),"Owned package")
        manifest=read(package/"manifest.json")
        require(manifest["binding"]==q["package_binding"]==c["result"]["package"]["binding"] and manifest["agent_generated"] is False,"Exact package provenance")
        require(manifest["compiler_configuration"]["backend"]=="poseidon_cpu" and
                manifest["validation"]==dict(encrypted_execution=True,level="numerical",numerically_validated=True),"Explicit Poseidon numerical grade")
        matched=[item for item in preflight["rows"] if item["case"]==name]
        require(len(matched)==1 and q["request_id"]==matched[0]["request_id"] and
                q["candidate_sha256"]==matched[0]["rule_source_sha256"]==sha(package/"candidate.py"),"Frozen mathematical request/source")
        for relative,want in manifest["files"].items():
            path=package/relative
            require(path.resolve().is_relative_to(package) and path.is_file() and not path.is_symlink(),"Regular package file")
            require(sha(path)==want,"Package content hash")
            hashes[str(path)]=want;file_checks+=1
        require({"artifacts/bound/plan.json","artifacts/operator-spec.json",
                 "artifacts/lowered.ckks.mlir","artifacts/lowered.earth.mlir"}.issubset(manifest["files"]),"Real compiler artifacts")
        task=package.parent;validation=read(task/"validation-0.json")
        evidence=Path(validation["evidence"]);backend=read(evidence/"report.json")
        require(backend["backend"]=="poseidon_cpu" and backend["agent_calls"]==0,"Poseidon backend receipt")
        raw=read(evidence/"attempt-00/report.json")
        require(raw["executed"] is True,"Real FHE receipt")
        base=comparison(raw["comparison"]);new=comparison(n["comparison"])
        require(base["input_groups"]==q["input_groups"]==4 and new["input_groups"]==n["input_groups"]==3,"Four plus three actual samples")
        require(n["execution"]["backend"]=="poseidon_cpu" and n["execution"]["encrypted_execution"] is True and
                n["execution"]["executor_secret_key"] is False and
                n["execution"]["agent_calls"]==n["execution"]["compiler_calls"]==0 and n["package_reloaded"] is True,"Package execution boundary")
        require(c["package_reused"] is True and c["result"]["generations"]==c["result"]["http_attempts"]==0,"Reuse without generation")
        for path in (package/"manifest.json",task/"validation-0.json",evidence/"report.json",evidence/"attempt-00/report.json",
                     root/name/"qualification.json",root/name/"new_inputs.json",root/name/"cache.json"):
            hashes[str(path)]=sha(path)
        groups+=base["input_groups"]+new["input_groups"];values+=base["values"]+new["values"]
        maximum=max(maximum,base["maximum_absolute_error"],new["maximum_absolute_error"])
        rows.append(dict(case=name,request_id=q["request_id"],candidate_sha256=q["candidate_sha256"],
            package=str(package),package_binding=q["package_binding"],qualification=base,new_inputs=new,
            actual_source_capture=True,compiled=True,encrypted_execution=True,package_reloaded=True,cache_reused=True,
            candidate_origin="rule_baseline",new_agent_generation=False))
    require(all(sha(path)==want for path,want in hashes.items()),"Receipt changed during audit")
    result=dict(format="poseidon-joint-post-cutover-independent-audit-v1",status="passed",binding=plan["binding"],
        root=str(root),cases=list(CASES),passed=3,failed=0,skipped=0,rows=rows,input_groups=groups,
        compared_values=values,maximum_absolute_error=maximum,package_file_hash_checks=file_checks,
        evidence_hashes=hashes,observed_compilations=3,observed_encrypted_phases=6,
        observed_source_capture_supervisors=3,observed_package_reloads=3,observed_cache_reuses=3,
        provider_calls=0,new_compilations_by_audit=0,new_encrypted_executions_by_audit=0,
        new_agent_generation=False,source_scope="Three representative V12 source cases with explicit-v11 guidance only",
        audit_implementation_sha256=sha(__file__))
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    return result
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("root",type=Path);p.add_argument("output",type=Path)
    result=inspect(**vars(p.parse_args()))
    print(json.dumps({k:result[k] for k in ("status","passed","input_groups","compared_values","maximum_absolute_error","package_file_hash_checks")}))
