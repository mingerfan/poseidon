"""Read-only terminal numerical-failure diagnosis with helper-aware arithmetic."""
import argparse,hashlib,json,sys
from pathlib import Path
ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5])
sys.path.insert(0,str(ROOT/"scripts/baseline"))
sys.path.insert(0,str(Path(__file__).resolve().parent))
sys.path.insert(0,str(ROOT/"scripts/baseline/benchmarks/tools/closed_live_source_audit_v1"))
from helper_math import evaluate,Unsupported
from audit import difference,sha
from candidate_contract import strict_json
from unified_graph_contract import validate_candidate
def require(value,reason):
    if not value:raise ValueError(reason)
def inspect(root,group,case):
    root=Path(root);plan=json.loads((root/"plan.json").read_text())
    before={}
    for field,base in (("source_hashes",ROOT),("tool_hashes",Path("/")),("retained_hashes",Path("/"))):
        for rel,h in plan[field].items():require(sha(base/rel)==h,"Frozen source/evidence")
        before[field]=len(plan[field])
    rows=json.loads((root/group/"report.json").read_text())["rows"]
    selected=[r for r in rows if r["id"]==case]
    require(len(selected)==1 and selected[0]["status"]=="failed","Explicit closed failure only")
    row=selected[0];task=root/group/case/"tasks"/row["task_id"]
    state=json.loads((task/"state.json").read_text())
    require(state["phase"]=="complete" and state["generations"]==row["result"]["generations"],"Terminal task ledger")
    checkpoints=[e for e in state["ledger"] if e.get("event")=="provider_checkpoint"]
    outcomes=checkpoints[-1]["outcomes"] if checkpoints else []
    require(len(outcomes)==state["http_attempts"] and all(c["status"] in ("failed","response_received") for c in outcomes),
            "Resolved Provider outcomes")
    hashes={str(task/"state.json"):sha(task/"state.json"),str(root/"plan.json"):sha(root/"plan.json")}
    diagnosed=[]
    for index in range(state["generations"]):
        path=task/("validation-%d.json"%index)
        if not path.exists():
            diagnosed.append(dict(generation=index,status="no_candidate_validation"));continue
        hashes[str(path)]=sha(path);validation=json.loads(path.read_text())
        if not validation.get("evidence"):
            diagnosed.append(dict(generation=index,status="not_executed",failure_layer=validation.get("failure",{}).get("layer")));continue
        evidence=Path(validation["evidence"])
        report_path=evidence/"attempt-00/report.json"
        if not report_path.exists():
            diagnosed.append(dict(generation=index,status="no_backend_report"));continue
        hashes[str(report_path)]=sha(report_path);report=json.loads(report_path.read_text())
        if not report.get("executed"):
            diagnosed.append(dict(generation=index,status="not_executed",failure_layer=report.get("failure_layer")));continue
        source_file=evidence/"attempt-00/candidate.py";input_file=evidence/"attempt-00/output/inputs.json"
        request_file=evidence/"request.json";response_file=task/("response-%d.json"%index)
        for p in (source_file,input_file,request_file,response_file):hashes[str(p)]=sha(p)
        request=json.loads(request_file.read_text());source=source_file.read_text()
        candidate=strict_json(json.loads(response_file.read_text())["raw"])
        require(candidate["hecate_source"]==source,"Exact original Agent source")
        validate_candidate(candidate,request)
        try:
            decoded=[]
            for inputs in json.loads(input_file.read_text()):
                slots=evaluate(source,inputs,request["public_constants"],request["layout"]["input_slot_period"])
                decoded.append([slots[c][s] for c,s in request["layout"]["output_selectors"]])
            backend=difference(report["comparison"]["actual"],decoded)
            semantics=difference(decoded,report["comparison"]["reference"])
            classification=("candidate_math_mismatch" if backend["matches_frozen_tolerance"] and
                            not semantics["matches_frozen_tolerance"] else "not_established")
            diagnosed.append(dict(generation=index,status="mathematically_replayed",classification=classification,
                input_groups=len(decoded),ciphertext_vs_candidate=backend,candidate_vs_model=semantics,
                original_encrypted_execution=True,new_encrypted_execution=False,source_sha256=sha(source_file)))
        except Unsupported as error:
            diagnosed.append(dict(generation=index,status="not_established",reason=str(error)))
    for p,h in hashes.items():require(sha(p)==h,"Evidence changed during diagnosis")
    for field,base in (("source_hashes",ROOT),("tool_hashes",Path("/")),("retained_hashes",Path("/"))):
        require(all(sha(base/k)==v for k,v in plan[field].items()),"Frozen source changed")
    return dict(format="poseidon-helper-source-diagnostic-v2",case=case,task_id=row["task_id"],
        original_status=row["status"],binding=plan["binding"],generations=diagnosed,evidence_hashes=hashes,
        frozen=before,diagnostic_implementation=dict(helper=sha(Path(__file__).with_name("helper_math.py")),
        audit=sha(__file__)),provider_calls=0,compilations=0,encrypted_executions=0,
        threshold=dict(atol=1e-5,rtol=1e-4),
        scope="Arithmetic replay only. Original FHE evidence is retained; no new validation or changed pass status.")
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("root");p.add_argument("group");p.add_argument("case");p.add_argument("output")
    args=p.parse_args();output=Path(args.output);require(not output.exists(),"Fresh diagnostic output")
    result=inspect(args.root,args.group,args.case);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n")
    print(json.dumps({k:result[k] for k in ("case","original_status","generations","frozen")}))
