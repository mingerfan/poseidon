"""Supplement a frozen terminal-case audit with bounded helper arithmetic diagnostics."""
import hashlib,json,sys
from pathlib import Path
from interpreter import evaluate,difference,Unsupported
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def audit(base):
    old=json.loads(base.read_text())
    for path,h in old["evidence_hashes"].items():
        if sha(path)!=h:raise ValueError("Original audited evidence changed")
    root=Path("/home/lhohy/poseidon-work/platforms/aarch64-linux/results/agent-approved-unstarted-continuation-r1113")
    closed=json.loads((root/"matrix_instance60/report.json").read_text())
    rows={r["id"]:r for r in closed["rows"]};out=[];snapshots={str(base):sha(base)}
    for item in old["failed_candidate_math"]:
        name=item["id"];index=item["generation"];row=rows[name]
        if row["status"]!="failed" or row["result"]["phase"]!="complete":raise ValueError("Closed original task required")
        task=root/"matrix_instance60"/name/"tasks"/row["task_id"]
        v=json.loads((task/("validation-"+str(index)+".json")).read_text())
        evidence=Path(v["evidence"]);report=json.loads((evidence/"attempt-00/report.json").read_text())
        if not report["compiled"] or not report["executed"]:raise ValueError("Retained encrypted comparison required")
        request=json.loads((evidence/"request.json").read_text());layout=request["layout"]
        samples=json.loads((evidence/"attempt-00/output/inputs.json").read_text())
        source=(evidence/"attempt-00/candidate.py").read_text()
        values=[]
        for inputs in samples:
            ports=evaluate(source,inputs,request["public_constants"],layout["input_slot_period"])
            values.append([ports[c][s] for c,s in layout["output_selectors"]])
        compared=report["comparison"]
        out.append(dict(id=name,generation=index,
            candidate_source_sha256=sha(evidence/"attempt-00/candidate.py"),
            plaintext_candidate_vs_reference=difference(values,compared["reference"]),
            ciphertext_vs_plaintext_candidate=difference(compared["actual"],values),
            original_agent_result_unchanged=True,not_an_execution_backend=True))
        for p in (evidence/"request.json",evidence/"attempt-00/report.json",evidence/"attempt-00/candidate.py",
                  evidence/"attempt-00/output/inputs.json",task/("validation-"+str(index)+".json")):
            snapshots[str(p)]=sha(p)
    for path,h in snapshots.items():
        if sha(path)!=h:raise ValueError("Evidence changed during diagnosis")
    return dict(format="poseidon-closed-helper-source-diagnostic-v2",source_audit=str(base),
        rows=out,evidence_hashes=snapshots,implementation={str(p):sha(p) for p in
            (Path(__file__),Path(__file__).with_name("interpreter.py"))},
        source_executed=False,provider_calls=0,compilations=0,encrypted_executions=0,
        threshold=dict(atol=1e-5,rtol=1e-4),
        scope="Interpret retained candidate arithmetic only. Independent references and ciphertext outputs remain their original bound evidence; no new backend or Agent pass.")
if __name__=="__main__":
    base=Path(sys.argv[1]);output=Path(sys.argv[2])
    if output.exists():raise ValueError("Fresh diagnosis required")
    result=audit(base);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(json.dumps(dict(status="audited",rows=result["rows"],provider_calls=0,compilations=0,encrypted_executions=0)))
