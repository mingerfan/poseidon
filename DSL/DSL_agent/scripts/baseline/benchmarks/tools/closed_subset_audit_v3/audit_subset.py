"""Read-only terminal subset audit; the containing group may remain active.

The report keeps its original group counters and hash. Selected closed tasks
are counted separately; no old report, provider or backend execution is changed.
"""
import json,sys
from pathlib import Path
ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5])
sys.path.insert(0,str(ROOT/"scripts/baseline"))
sys.path.insert(0,str(ROOT/"scripts/baseline/benchmarks/tools/closed_live_source_audit_v1"))
sys.path.insert(0,"/home/lhohy/poseidon-work/platforms/aarch64-linux/venvs/hecate-2.0.1-cpu/lib/python3.10/site-packages")
from audit import sha,difference,evaluate,Unsupported
from benchmark_graph import digest
from candidate_contract import strict_json
from unified_graph_contract import validate_candidate
from generation_receipt import verify
def require(value,reason):
    if not value:raise ValueError(reason)
def inspect(root,group_name,selected_ids):
    root=Path(root)
    plan=json.loads((root/"plan.json").read_text())
    path=root/group_name/"report.json"
    group=json.loads(path.read_text())
    require(group["status"] in ("running","complete","stopped"),"Known group status")
    require(type(selected_ids) is list and selected_ids and
            all(type(i) is str for i in selected_ids) and len(set(selected_ids))==len(selected_ids),
            "Explicit unique subset")
    original_group={k:group.get(k) for k in ("status","planned","passed","failed","not_run","generations","http_attempts")}
    all_rows={r["id"]:r for r in group["rows"]}
    require(len(all_rows)==len(group["rows"]) and all(i in all_rows for i in selected_ids),"Known subset rows")
    rows=[all_rows[i] for i in selected_ids]
    require(all(r["status"] in ("passed","failed") for r in rows),"No active selected task")
    group=dict(group,status="closed_subset",rows=rows,planned=len(rows),
               passed=sum(r["status"]=="passed" for r in rows),
               failed=sum(r["status"]=="failed" for r in rows),not_run=0,
               generations=sum(r["result"]["generations"] for r in rows),
               http_attempts=sum(r["result"]["http_attempts"] for r in rows))
    require(group["binding"]==plan["binding"],"Group binding")
    frozen={}
    for label,base in (("source_hashes",ROOT),("tool_hashes",Path("/")),("retained_hashes",Path("/"))):
        for p,h in plan[label].items():require(sha(base/p)==h,"Original freeze")
        frozen[label]=len(plan[label])
    hashes={str(path):sha(path),str(root/"plan.json"):sha(root/"plan.json")}
    out=[];rejections=[];provider_only=[];failed_math=[]
    package_n=artifact_n=samples=values=generations=http=0;maximum=0
    for row in group["rows"]:
        task=root/group_name/row["id"]/"tasks"/row["task_id"]
        envelope=json.loads((task/"task.json").read_text())
        request=envelope["request"]
        state=json.loads((task/"state.json").read_text())
        loop=json.loads((task/"loop.json").read_text())
        require(state["phase"]=="complete" and state["status"]==row["result"]["status"],"Closed task state")
        require(state["generations"]==row["result"]["generations"] and
                state["http_attempts"]==row["result"]["http_attempts"],"Call accounting")
        require(state["generations"]<=envelope["budget"]["generations"] and
                state["http_attempts"]<=envelope["budget"]["http_attempts"],"Per-case bounds")
        checkpoints=[e for e in state["ledger"] if e.get("event")=="provider_checkpoint"]
        calls=checkpoints[-1].get("outcomes",[]) if checkpoints else []
        require(len(calls)==state["http_attempts"] and
                sorted(x["index"] for x in calls)==list(range(state["http_attempts"])) and
                all(x["status"] in ("response_received","failed") for x in calls),"Closed provider ledger")
        generations+=state["generations"];http+=state["http_attempts"]
        for p in task.glob("*.json"):hashes[str(p)]=sha(p)
        for index in range(state["generations"]):
            p=task/("validation-%d.json"%index)
            if not p.exists():
                metrics=json.loads((task/"provider-metrics.json").read_text())
                gen_calls=[c for c in metrics["calls"] if c["generation_index"]==index]
                require(gen_calls and all(c["status"]=="failed" for c in gen_calls),"No validation only for failed generation")
                provider_only.append(dict(id=row["id"],generation=index,
                    errors=[c.get("error") for c in gen_calls],candidate_received=False))
                continue
            v=json.loads(p.read_text())
            if v.get("failure",{}).get("layer")=="static_check":
                p=task/("response-%d.json"%index)
                try:validate_candidate(strict_json(json.loads(p.read_text())["raw"]),request)
                except ValueError as error:reason=str(error)
                else:raise ValueError("Static rejection not reproduced")
                rejections.append(dict(id=row["id"],generation=index,reason=reason))
            if v.get("evidence"):
                evidence=Path(v["evidence"]);report_path=evidence/"attempt-00/report.json"
                if not report_path.exists():continue
                report=json.loads(report_path.read_text());hashes[str(report_path)]=sha(report_path)
                for relative,h in report.get("artifact_hashes",{}).items():
                    p=evidence/"attempt-00/output"/relative
                    require(p.resolve().is_relative_to((evidence/"attempt-00/output").resolve()) and not p.is_symlink(),"Artifact path")
                    require(sha(p)==h,"Artifact hash")
                    hashes[str(p)]=h;artifact_n+=1
                if row["status"]!="passed" and report.get("executed"):
                    candidate=evidence/"attempt-00/candidate.py"
                    input_file=evidence/"attempt-00/output/inputs.json"
                    hashes[str(candidate)]=sha(candidate);hashes[str(input_file)]=sha(input_file)
                    original=json.loads((evidence/"request.json").read_text())
                    try:
                        expected=[]
                        for inputs in json.loads(input_file.read_text()):
                            slots=evaluate(candidate.read_text(),inputs,original["public_constants"],
                                           original["layout"]["input_slot_period"])
                            expected.append([slots[c][s] for c,s in original["layout"]["output_selectors"]])
                        failed_math.append(dict(id=row["id"],generation=index,
                            candidate_vs_model=difference(expected,report["comparison"]["reference"]),
                            ciphertext_vs_candidate=difference(report["comparison"]["actual"],expected),
                            diagnostic_interpreter="straight-line arithmetic only; not FHE validation"))
                    except (Unsupported,ValueError,KeyError,SyntaxError) as error:
                        failed_math.append(dict(id=row["id"],generation=index,not_established=str(error)))
        item=dict(id=row["id"],task_id=row["task_id"],status=row["status"],
                  task_status=state["status"],loop_status=loop["status"],
                  terminal_public_failure=state["failure"],
                  generations=state["generations"],http_attempts=state["http_attempts"])
        if loop["status"] in ("provider_failed","provider_exhausted"):
            item["audited_terminal_layer"]="provider"
            item["historical_public_failure_may_be_previous_validation"]=True
        if row["status"]=="passed":
            folder=Path(row["package"])
            manifest=json.loads((folder/"manifest.json").read_text())
            require(digest({k:v for k,v in manifest.items() if k!="binding"})==manifest["binding"]==row["package_binding"],"Package binding")
            require(manifest["agent_generated"] is True,"Real Agent provenance")
            for relative,h in manifest["files"].items():
                p=folder/relative
                require(p.resolve().is_relative_to(folder.resolve()) and not p.is_symlink(),"Package path")
                require(sha(p)==h,"Package hash")
                hashes[str(p)]=h;package_n+=1
            source=(folder/"candidate.py").read_text()
            receipt=json.loads((folder/"artifacts/generation-provenance.json").read_text())
            gen=verify(receipt,request,source)
            require(gen["generations"]==state["generations"] and gen["http_attempts"]==state["http_attempts"],"Provenance counts")
            require(receipt["candidate_sha256"]==sha(folder/"candidate.py"),"Exact generated source")
            v=json.loads((task/("validation-%d.json"%(state["generations"]-1))).read_text())
            evidence=Path(v["evidence"])
            report=json.loads((evidence/"attempt-00/report.json").read_text())
            require(report["compiled"] is True and report["executed"] is True and report["numerically_correct"] is True,"Original backend acceptance")
            require(state["acceptance"]["level"]=="numerical" and state["acceptance"]["encrypted_execution"] is True,"Explicit numerical level")
            original=difference(report["comparison"]["actual"],report["comparison"]["reference"])
            require(original["matches_frozen_tolerance"],"Original pointwise tolerance")
            require(row["reloaded"] is True,"Real package reload")
            reuse=row.get("new_input_acceptance")
            if reuse is not None:
                require(reuse["status"]=="passed" and reuse["agent_calls"]==reuse["compiler_calls"]==0,"Reuse without regeneration/compilation")
                require(reuse["input_groups"]==3 and reuse["execution"]["backend"]=="poseidon_cpu" and
                        reuse["execution"]["encrypted_execution"] is True and
                        reuse["execution"]["executor_secret_key"] is False,"New-input Poseidon CPU")
                new=difference(reuse["comparison"]["actual"],reuse["comparison"]["reference"])
                require(new["matches_frozen_tolerance"],"New input pointwise tolerance")
                item["new_inputs"]=new
                samples+=reuse["input_groups"];values+=new["values"]
                maximum=max(maximum,new["maximum_absolute_error"])
            values+=original["values"];samples+=len(report["comparison"]["actual"])
            maximum=max(maximum,original["maximum_absolute_error"])
            item.update(first_attempt_passed=gen["first_attempt_passed"],original=original,
                        package_binding=manifest["binding"],reloaded=True,usage=gen["usage"])
        out.append(item)
    require(generations==group["generations"] and http==group["http_attempts"],"Group ledger totals")
    require(sum(r["status"]=="passed" for r in out)==group["passed"] and
            sum(r["status"]=="failed" for r in out)==group["failed"] and
            group["planned"]==len(out)+group["not_run"],"Coverage denominator")
    for p,h in hashes.items():require(sha(p)==h,"Evidence changed during audit")
    return dict(format="poseidon-terminal-subset-independent-audit-v3",group=group_name,binding=plan["binding"],
       status=group["status"],planned=group["planned"],passed=group["passed"],failed=group["failed"],
       not_run=group["not_run"],first_attempt_passed=sum(r.get("first_attempt_passed",False) for r in out),
       success_after_repair=sum(r["status"]=="passed" and not r["first_attempt_passed"] for r in out),
       generations=generations,http_attempts=http,provider_only_failed_generations=provider_only,
       failed_candidate_math=failed_math,reproduced_static_rejections=rejections,rows=out,
       original_plus_new_input_groups=samples,compared_values=values,maximum_absolute_error=maximum,
       package_hash_checks=package_n,artifact_hash_checks=artifact_n,frozen=frozen,
       readonly_audit=dict(provider_calls=0,compilations=0,encrypted_executions=0),
       evidence_hashes=hashes,threshold=dict(atol=1e-5,rtol=1e-4),
       scope="Retained current Agent generation receipts, original numerical results and real reuse. Audit makes no new FHE/Agent execution claim; no bootstrap/CUDA/application-accuracy coverage.")
if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser();p.add_argument("root");p.add_argument("group");p.add_argument("output");p.add_argument("ids",nargs="+")
    a=p.parse_args();output=Path(a.output)
    require(not output.exists(),"Fresh audit output")
    report=inspect(a.root,a.group,a.ids);report["audit_tool_sha256"]=sha(__file__)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+"\n")
    print(json.dumps({k:report[k] for k in ("group","status","planned","passed","failed","not_run",
        "first_attempt_passed","success_after_repair","generations","http_attempts",
        "original_plus_new_input_groups","compared_values","maximum_absolute_error","package_hash_checks","artifact_hash_checks")}))
