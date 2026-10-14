"""Audit retained, closed application results without new backend execution."""
import json
import sys
from pathlib import Path
from audit import sha, difference

ROOT = (Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5])
sys.path.insert(0,str(ROOT/"scripts/baseline"))
from benchmark_graph import digest
from candidate_contract import strict_json
from unified_graph_contract import validate_candidate
from generation_receipt import verify

def require(condition,reason):
    if not condition:
        raise ValueError(reason)

def inspect(root,source_report):
    root=Path(root)
    plan=json.loads((root/"plan.json").read_text())
    group=json.loads((root/"functional15/report.json").read_text())
    source=json.loads(Path(source_report).read_text())
    require(source["binding"]==plan["binding"] and source["group"]=="functional15","source audit binding")
    hashes={str(Path(source_report)):sha(source_report),str(root/"functional15/report.json"):sha(root/"functional15/report.json")}
    passed=[]
    rejected=[]
    values=0
    samples=0
    package_files=0
    artifact_files=0
    maximum=0
    for item in group["rows"]:
        task=root/"functional15"/item["id"]/"tasks"/item["task_id"]
        envelope=json.loads((task/"task.json").read_text())
        request=envelope["request"]
        state=json.loads((task/"state.json").read_text())
        for index in range(state["generations"]):
            validation=json.loads((task/("validation-%d.json"%index)).read_text())
            if validation.get("failure",{}).get("layer")=="static_check":
                response=task/("response-%d.json"%index)
                raw=json.loads(response.read_text())["raw"]
                try:
                    validate_candidate(strict_json(raw),request)
                except ValueError as error:
                    reason=str(error)
                else:
                    raise ValueError("previous static rejection no longer reproduces")
                rejected.append(dict(id=item["id"],round=index,reason=reason,response_sha256=sha(response)))
            path=Path(validation["evidence"]) if validation.get("evidence") else None
            if path and (path/"attempt-00/report.json").is_file():
                report=json.loads((path/"attempt-00/report.json").read_text())
                for relative,expected in report.get("artifact_hashes",{}).items():
                    file=path/"attempt-00/output"/relative
                    require(sha(file)==expected,"retained artifact hash")
                    hashes[str(file)]=expected
                    artifact_files+=1
        if item["status"]!="passed":
            continue
        folder=Path(item["package"])
        manifest=json.loads((folder/"manifest.json").read_text())
        require(digest({k:v for k,v in manifest.items() if k!="binding"})==manifest["binding"],"package binding")
        require(manifest["binding"]==item["package_binding"] and manifest["agent_generated"] is True,"new Agent package")
        for relative,expected in manifest["files"].items():
            path=folder/relative
            require(path.resolve().is_relative_to(folder.resolve()) and not path.is_symlink(),"package relative path")
            require(sha(path)==expected,"program package hash")
            hashes[str(path)]=expected
            package_files+=1
        candidate=(folder/"candidate.py").read_text()
        receipt=json.loads((folder/"artifacts/generation-provenance.json").read_text())
        generation=verify(receipt,request,candidate)
        require(generation["generations"]==state["generations"] and generation["http_attempts"]==state["http_attempts"],"generation provenance counts")
        require(receipt["candidate_sha256"]==sha(folder/"candidate.py"),"exact returned source")
        last=json.loads((task/("validation-%d.json"%(state["generations"]-1))).read_text())
        path=Path(last["evidence"])
        report=json.loads((path/"attempt-00/report.json").read_text())
        require(report["compiled"] is True and report["executed"] is True and report["numerically_correct"] is True,"backend acceptance flags")
        acceptance=difference(report["comparison"]["actual"],report["comparison"]["reference"])
        require(acceptance["matches_frozen_tolerance"],"recomputed original pointwise comparison")
        require(item["reloaded"] is True and item["new_input_acceptance"]["status"]=="passed","successful package reuse")
        reuse=item["new_input_acceptance"]
        require(reuse["agent_calls"]==0 and reuse["compiler_calls"]==0 and reuse["input_groups"]==3,"reuse without regeneration or compilation")
        require(reuse["execution"]["backend"]=="poseidon_cpu" and reuse["execution"]["encrypted_execution"] is True,"new inputs actual Poseidon CPU")
        new_comparison=difference(reuse["comparison"]["actual"],reuse["comparison"]["reference"])
        require(new_comparison["matches_frozen_tolerance"],"recomputed supplied-input pointwise comparison")
        values+=acceptance["values"]+new_comparison["values"]
        samples+=len(report["comparison"]["actual"])+reuse["input_groups"]
        maximum=max(maximum,acceptance["maximum_absolute_error"],new_comparison["maximum_absolute_error"])
        passed.append(dict(id=item["id"],package_binding=manifest["binding"],
                           first_attempt_passed=generation["first_attempt_passed"],
                           generations=generation["generations"],http_attempts=generation["http_attempts"],
                           usage=generation["usage"],original=acceptance,new_inputs=new_comparison,
                           package_reloaded=True,agent_calls_on_reuse=0,compiler_calls_on_reuse=0))
    require(len(passed)==10 and len(rejected)==5,"closed group acceptance accounting")
    for path,h in hashes.items():
        require(sha(path)==h,"evidence changed during audit")
    return dict(format="poseidon-closed-agent-delivery-audit-v1",binding=plan["binding"],
                passed=len(passed),failed=group["failed"],not_run=0,
                first_attempt_passed=sum(r["first_attempt_passed"] for r in passed),
                success_after_repair=sum(not r["first_attempt_passed"] for r in passed),
                rejection_checks=len(rejected),reproduced_rejections=rejected,packages=passed,
                original_plus_new_input_groups=samples,compared_values=values,
                maximum_absolute_error=maximum,package_hash_checks=package_files,
                artifact_hash_checks=artifact_files,evidence_hashes=hashes,
                readonly_audit=dict(provider_calls=0,compilations=0,encrypted_executions=0),
                source_diagnostic_report=str(source_report),source_diagnostic_sha256=sha(source_report),
                coverage="10 succeeded Agent packages, original four inputs and three new inputs each. Bootstrap, CUDA and application accuracy not certified.")

if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("root")
    parser.add_argument("source_report")
    parser.add_argument("output")
    args=parser.parse_args()
    output=Path(args.output)
    require(not output.exists(),"fresh report required")
    result=inspect(args.root,args.source_report)
    result["audit_tool_hashes"]={str(Path(__file__)):sha(__file__),str(Path(__file__).with_name("audit.py")):sha(Path(__file__).with_name("audit.py"))}
    output.write_text(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+"\n")
    print(json.dumps({k:result[k] for k in ("passed","failed","first_attempt_passed","success_after_repair","rejection_checks","original_plus_new_input_groups","compared_values","maximum_absolute_error","package_hash_checks","artifact_hash_checks")}))
