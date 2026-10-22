"""Read-only evidence audit; no historical run is silently rebound to new code."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from benchmark_graph import digest
from benchmark_runner import DEFAULT,load,strict_file,dump
from semantic_benchmark_execution import runtime_sources,summary

def verify_directed_coverage(coverage,task,expected):
    """Every child of a composite must pass; empty children stay structural."""
    from unified_graph_exercises import STRUCTURAL
    from unified_public_exercises import STRUCTURAL as PUBLIC_STRUCTURAL
    required=expected['required_features']
    if task.get('required_features',[task['requirement']])!=required:
        raise ValueError('Directed subrequirement metadata mismatch')
    structural=[f for f in required if f in STRUCTURAL|PUBLIC_STRUCTURAL]
    numeric=[f for f in required if f not in structural]
    kind='mixed' if structural and numeric else 'structure_only' if structural else 'numeric_influence'
    if task['acceptance_kind']!=kind or task.get('structural_features',structural)!=structural:
        raise ValueError('Directed scoring kind mismatch')
    if not coverage['actual_frontend_checked']:raise ValueError('No real directed trace')
    if coverage.get('numeric_features')!=numeric or coverage.get('structural_features')!=structural:
        raise ValueError('Directed pass missing conjunctive child witnesses')
    if coverage.get('finite_influence_checked') is not bool(numeric) or coverage.get('structural_only') is not (not bool(numeric)):
        raise ValueError('Structural/numerical evidence misclassified')
    from unified_public_exercises import EVIDENCE_SCOPES
    for feature in required:
        if feature in EVIDENCE_SCOPES and coverage.get('evidence_scopes',{}).get(feature)!=EVIDENCE_SCOPES[feature]:
            raise ValueError('Typed child evidence scope mismatch')
    expected_scope=task.get('evidence_scope','executed_operation')
    if expected_scope!='executed_operation' and coverage.get('evidence_scopes',{}).get(task['requirement'])!=expected_scope:
        raise ValueError('Typed context/result evidence scope mismatch')


def audit(suite,plaintext,batches):
    rows,index=load(suite);current=runtime_sources()
    by_id={r["model"]["id"]:r for r in rows}
    run=strict_file(plaintext/"run.json",8*1024**2)
    if run["frozen_index"]!=index:raise ValueError("Plaintext task belongs to a different frozen index")
    plain=strict_file(plaintext/"report.json",4*1024**2)
    if plain["binding"]!=run["binding"]:raise ValueError("Plaintext binding")
    states={}
    for row in rows:
        name=row["model"]["id"];path=plaintext/(name+".json")
        if hashlib.sha256(path.read_bytes()).hexdigest()!=plain["result_hashes"][name]:raise ValueError("Plaintext row changed")
        record=strict_file(path,1024**2)
        if record["model_sha256"]!=row["model_sha256"]:raise ValueError("Plaintext model changed")
        states[name]=dict(id=name,model_sha256=row["model_sha256"],split=row["split"],category=row["category"],
                         plaintext=record["status"],baseline="not_run",free_agent="not_run",evidence=[])
    task_specs={t["id"]:t for t in strict_file(suite/"coverage.json",4*1024**2)["directed_tasks"]}
    directed={t["id"]:dict(id=t["id"],model_id=t["model_id"],task_sha256=t["task_sha256"],
                          requirement=t["requirement"],coverage_kind=t.get("acceptance_kind","numeric_influence"),state="not_run" if t.get("exercise") else "blocked_contract")
              for t in strict_file(suite/"coverage.json",4*1024**2)["directed_tasks"]}
    seen=set();reports=[];values=0;maximum=0.
    for folder in batches:
        plan=strict_file(folder/"plan.json",8*1024**2)
        if plan["source_hashes"]!=current:raise ValueError("Historical runtime binding; audit in its frozen checkout: "+str(folder))
        check=summary(folder);reports.append(dict(path=str(folder),binding=plan["binding"],summary=check))
        items={x["id"]:x for x in plan["preflight"]}
        for item in items.values():
            if plan["mode"]=="baseline" and item["status"]!="ready":
                states[item["id"]]["baseline"]="blocked_preflight"
            elif plan["mode"]=="directed" and item["status"] not in ("ready","other_profile"):
                directed[item["id"]].update(state=item["status"],reason=item.get("reason"))
        for name in plan["selected_ids"]:
            path=folder/(name+".result.json")
            if not path.exists():continue
            result=strict_file(path,1024**2);item=items[name]
            key=(plan["mode"],plan.get("live",False),name)
            if key in seen:raise ValueError("Overlapping tasks must be selected explicitly, not summed")
            seen.add(key)
            model=item.get("model_id",name);model_data=strict_file(folder/(name+".model.json"),131072)
            if digest(model_data)!=by_id[model]["model_sha256"]:raise ValueError("Executed model differs")
            if plan["mode"]=="directed":
                if item["task_sha256"]!=directed[name]["task_sha256"]:raise ValueError("Directed task changed")
                target=directed[name];field="agent_state" if plan.get("live") else "state"
            else:
                target=states[model];field="free_agent" if plan.get("live") else "baseline"
            target[field]=result["status"]
            if result.get("evidence"):
                ev=Path(result["evidence"]);report=strict_file(ev/"report.json",8*1024**2)
                if plan["mode"]=="directed":
                    if plan.get("construction_profile")=="public-v1":
                        from unified_public_exercises import spec
                    else:
                        from unified_graph_exercises import spec
                    task=task_specs[name];expected=spec(task["exercise"])
                    request=strict_file(ev/"request.json",131072)
                    if task["instruction"]!=expected["instruction"] or request.get("construction_exercise")!=expected:
                        raise ValueError("Directed ledger/request instruction mismatch")
                if result["status"]=="passed":
                    successes=[a for a in report["attempts"] if a.get("status")=="passed"]
                    if not successes:raise ValueError("Pass without a successful attempt")
                    a=successes[-1]
                    if not all(a.get(k) for k in ("compiled","executed","numerically_correct")):raise ValueError("Pass without FHE evidence")
                    if a["trace"]["frontend"]!="real_Hecate" or not a["execution"]["encrypted_execution"]:raise ValueError("Backend evidence mismatch")
                    if plan.get("construction_profile")=="public-v1":
                        public=a.get("public_construction_trace",{})
                        if not public.get("real_frontend_checked") or not public.get("source_and_normalization_bound"):
                            raise ValueError("Missing checked real public construction trace")
                    if a["comparison"]["atol"]!=1e-5 or a["comparison"]["rtol"]!=1e-4:raise ValueError("Changed tolerance")
                    if plan["mode"]=="directed":
                        coverage=a["public_expression_coverage" if plan.get("construction_profile")=="public-v1" else "packed_native_coverage"]
                        verify_directed_coverage(coverage,task_specs[name],expected)
                    values+=a["comparison"]["compared_values"]
                    maximum=max(maximum,a["comparison"]["max_absolute_error"])
                target.setdefault("evidence",[]).append(str(ev))
    return dict(schema=1,model_set_sha256=index["model_set_sha256"],models=list(states.values()),
                directed_tasks=list(directed.values()),batches=reports,
                counts={kind:dict(Counter(s[kind] for s in states.values())) for kind in ("plaintext","baseline","free_agent")},
                directed_counts=dict(Counter(t["state"] for t in directed.values())),
                directed_numeric_counts=dict(Counter(t["state"] for t in directed.values() if t["coverage_kind"]=="numeric_influence")),
                directed_mixed_counts=dict(Counter(t["state"] for t in directed.values() if t["coverage_kind"]=="mixed")),
                directed_structural_counts=dict(Counter(t["state"] for t in directed.values() if t["coverage_kind"]=="structure_only")),
                compared_values_across_disjoint_tasks=values,max_absolute_error=maximum,
                agent_calls=sum(r["summary"]["agent_calls"] for r in reports),
                all_dsl_semantics_covered=False,upstream_helper_coverage="separate manual helper reports",
                formal_equivalence_proven=False)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite",type=Path,default=DEFAULT);p.add_argument("--plaintext",type=Path,required=True)
    p.add_argument("--batch",type=Path,action="append",default=[]);p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error("Preserve audit; use a new output")
    result=audit(a.suite,a.plaintext,a.batch);dump(a.output,result)
    print(json.dumps({k:result[k] for k in ("counts","directed_counts","agent_calls","max_absolute_error")},indent=2))
    return 0
if __name__=="__main__":raise SystemExit(main())
