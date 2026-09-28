"""Bind three real, previously executed topology contexts per compiler partition.

Read-only evidence verification; no compiler/runtime calls, keys, downloads or API.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import struct
from collections import Counter
BASE=Path(__file__).resolve().parents[2];ROOT=BASE.parents[1]
sys.path.insert(0,str(BASE))
from benchmark_graph import digest,signature
from semantic_benchmark_execution import runtime_sources

PARTITIONS=("rescale","modswitch","relinearization","rotation_keys","security_parameters")

def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def verified_example(record,requirement,results):
    evidence=Path(record["evidence"]).resolve()
    if not evidence.is_relative_to(results.resolve()):raise ValueError("Evidence outside results")
    if sha(evidence/"report.json")!=record["report_sha256"]:raise ValueError("Candidate report drift")
    report=read(evidence/"report.json")
    if report.get("status")!="passed" or not record["numerical_passed"]:raise ValueError("Positive example is not a pass")
    attempt=next((a for a in reversed(report["attempts"]) if a.get("status")=="passed"),None)
    if attempt is None or not all(attempt.get(k) for k in ("compiled","executed","numerically_correct")):
        raise ValueError("Missing compiler/encrypted evidence")
    if attempt["trace"]["frontend"]!="real_Hecate" or not attempt["execution"]["encrypted_execution"]:
        raise ValueError("Not real frontend/SEAL execution")
    if attempt["execution"]["input_batches"]!=4:raise ValueError("Expected four frozen input groups")
    out=evidence/("attempt-"+str(attempt["index"]).zfill(2))/"output"
    for name,h in attempt["artifact_hashes"].items():
        if Path(name).name!=name or sha(out/name)!=h:raise ValueError("Artifact drift")
    if sha(out/"lowered._hecate_golden.hevm")!=record["lineage"]["hevm_sha256"]:
        raise ValueError("HEVM lineage binding")
    if sha(out/"_hecate_golden.cst")!=record["lineage"]["cst_sha256"]:
        raise ValueError("CST lineage binding")
    if record["lineage"]["gate"]!=attempt["artifact_gate"]:raise ValueError("Stored artifact gate drift")
    if read(out/"execution.json")!=attempt["execution"]:raise ValueError("Execution record drift")
    if read(evidence/"request.json")["request_id"]!=record["request_id"]:raise ValueError("Request identity mismatch")
    raw=(out/"lowered._hecate_golden.hevm").read_bytes()
    offset=24+struct.unpack_from("<Q",raw,24)[0]
    counts=Counter(op[0] for op in struct.iter_unpack("<4H",raw[offset:]))
    if {str(k):v for k,v in counts.items()}!=attempt["artifact_gate"]["opcode_counts"]:
        raise ValueError("Actual opcode counts differ")
    comparison=record["comparison"]
    if comparison!=attempt["comparison"]:raise ValueError("Comparison record drift")
    if comparison["atol"]!=1e-5 or comparison["rtol"]!=1e-4:raise ValueError("Changed tolerance")
    actual,reference=comparison["actual"],comparison["reference"]
    if len(actual)!=4 or len(reference)!=4:raise ValueError("Wrong comparison batch count")
    count=0
    for av,rv in zip(actual,reference):
        if len(av)!=len(rv):raise ValueError("Comparison shape mismatch")
        for a,r in zip(av,rv):
            if not math.isfinite(a) or not math.isfinite(r) or abs(a-r)>1e-5+1e-4*abs(r):
                raise ValueError("Numerical comparison fails")
            count+=1
    if count!=comparison["compared_values"]:raise ValueError("Comparison count mismatch")
    name=requirement.split(".",1)[1];proof=record["compiler_evidence"][name]
    if name in ("rescale","modswitch"):
        opcode=3 if name=="rescale" else 4
        if proof["artifact_operations"]<1 or proof["artifact_operations"]!=counts[opcode]:
            raise ValueError("Requested operation count mismatch")
    if name=="relinearization":
        if proof["fused_mulcc_operations"]<1 or proof["fused_mulcc_operations"]!=counts[8] or proof["per_operation_runtime_observed"]:
            raise ValueError("Missing fused operation or overclaimed runtime observation")
        if proof["runtime_binary_sha256"]!=report["runtime_sha256"]:raise ValueError("Runtime identity mismatch")
        runtime=ROOT/"third_party/dacapo/lib/Runtime/SEAL_HEVM.cpp"
        if sha(runtime)!=proof["pinned_runtime_source_sha256"]:raise ValueError("Runtime source drift")
    if name=="rotation_keys":
        if not proof["actual_key_file_verified"] or not proof["required_steps"] or proof!=attempt["execution"]["rotation_key_check"]:
            raise ValueError("No matching verified rotation requirement")
    if name=="security_parameters":
        p=proof["actual_generated_parameters"]
        if p!=report["parameters"]:raise ValueError("Security parameter record drift")
        if p["seal_version"]!="4.0.0" or p["security_check"]!="tc128" or not p["parameters_set"]:
            raise ValueError("Unexpected actual security profile")
    model=read(evidence/"model.json")
    return dict(requirement=requirement,model_id=model["id"],model_sha256=digest(model),
                topology=signature(model,True),evidence_relative_to_results=str(evidence.relative_to(results)),
                report_sha256=record["report_sha256"],request_id=record["request_id"],
                artifact_hashes=attempt["artifact_hashes"],compiler_evidence=proof,
                positive_numerical_values=count,actual_encrypted_execution_recorded=True,
                new_encrypted_execution=False,all_input_proof=False)

def build(audit_path,results):
    original=read(audit_path)
    if original["contract"]!="hevm-runtime-evidence-v1":raise ValueError("Unsupported artifact evidence contract")
    for name,h in original["checker_sources"].items():
        if sha(ROOT/name)!=h:raise ValueError("Evidence checker drift")
    records={r["evidence"]:r for r in original["records"]}
    examples={}
    for name in PARTITIONS:
        req="compiler."+name;chosen=[];groups=set()
        for path in original["observed_requirements"][req]:
            record=records[path]
            if not record["numerical_passed"]:continue
            value=verified_example(record,req,results)
            if value["topology"] in groups:continue
            chosen.append(value);groups.add(value["topology"])
            if len(chosen)==3:break
        if len(chosen)!=3:raise ValueError("Fewer than three verified topologies: "+req)
        examples[req]=chosen
    return dict(schema=1,kind="verified_compiler_context_bindings",authoritative_ledger=False,
                source_audit_sha256=sha(audit_path),source_audit_relative_to_results=str(audit_path.resolve().relative_to(results.resolve())),
                runtime_source_sha256=digest(runtime_sources()),examples=examples,
                requirements=5,contexts_per_requirement=3,
                distinct_execution_records=len({e["evidence_relative_to_results"] for v in examples.values() for e in v}),
                new_encrypted_executions=0,paid_calls=0,
                scope="Previously executed artifacts, runtime-source inference and observed final ciphertext metadata; not per-instruction tracing")

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--audit",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True);a=p.parse_args()
    if a.output.exists():p.error("Preserve existing binding")
    from workspace_paths import RESULTS
    value=build(a.audit,RESULTS)
    value["runner_sha256"]=sha(Path(__file__))
    value["binding_sha256"]=digest(value)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(value,indent=2,sort_keys=True)+"\n")
    print(json.dumps({k:v for k,v in value.items() if k!="examples"}))
    return 0
if __name__=="__main__":raise SystemExit(main())
