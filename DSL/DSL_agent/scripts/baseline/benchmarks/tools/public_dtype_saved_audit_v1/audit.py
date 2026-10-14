# Stdlib-only, read-only r1376 saved-result aggregation.
import collections, hashlib, json, math, re, sys
from pathlib import Path

def audit(plan_file, report_file):
    files, issues, cases = {}, [], []
    def must(ok, code):
        if not ok: raise ValueError(code)
    def read(p):
        p=Path(p).absolute(); must(p==p.resolve() and p.is_file(),"regular_file")
        raw=p.read_bytes()
        files[str(p)]={"sha256":hashlib.sha256(raw).hexdigest(),"bytes":len(raw)}
        return json.loads(raw)
    def digest(x):
        return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
    def binding(x):
        must(x["binding"]==digest({k:v for k,v in x.items() if k!="binding"}),"binding")
    def metric(x):
        must(type(x) in (int,float) and math.isfinite(x) and x>=0,"metric")
        return x
    def within(p,d):
        p=Path(p).absolute(); must(p==p.resolve() and p.is_relative_to(d),"case_path")
        return p
    plan_file, report_file=Path(plan_file).absolute(),Path(report_file).absolute()
    plan, report=read(plan_file),read(report_file); binding(plan); binding(report)
    snapshot_sha=files[str(report_file)]["sha256"]
    must(report["proposal_binding"]==plan["binding"] and plan["backend"]=="poseidon_cpu"
         and plan["numeric_tolerance"]=={"atol":1e-5,"rtol":1e-4},"proposal")
    items, rows=plan["cases"],report["rows"]
    must(len(items)==len(rows)==36 and len({i["case"] for i in items})==36
         and [i["case"] for i in items]==[r["case"] for r in rows],"36_cases")
    must(len({i["request_id"] for i in items})==plan["unique_requests"]==12,"12_requests")
    must(plan["reuse_cases"]==["f64_to_f32_c0","f64_roundtrip_c2","compat_v23_named"],"3_reuse")
    must(report["provider_calls"]==report["http_attempts"]==0
         and report["agent_generated"] is False and report["candidate_origin"]=="deterministic_rule","origin")
    numerical={k:{"groups":0,"values":0,"max_absolute_error":0}
               for k in ("initial","new_inputs","repeat")}
    calls=collections.Counter({k:0 for k in ("compiler_reserved","compiler_confirmed_successful",
        "encrypted_reserved","encrypted_confirmed_executed","encrypted_numerically_passed")})
    failure_classes=collections.Counter(); statuses=collections.Counter()
    def failure_class(reason,layer=None):
        return ("budget_stop" if str(reason).startswith(("dtype_wall","dtype_disk","dtype_call")) or layer=="resources"
                else "environment_integrity_stop" if layer in ("environment","integrity","worker")
                or str(reason).startswith(("dtype_current","dtype_tool","dtype_frozen","dtype_task"))
                else "candidate_validation_failure")
    def add(key,groups,values,error):
        n=numerical[key]; n["groups"]+=groups; n["values"]+=values
        n["max_absolute_error"]=max(n["max_absolute_error"],metric(error))
    planned_values=0; out=report_file.parent
    for item,row in zip(items,rows):
        name=item["case"]; statuses[row["status"]]+=1
        c={"case":name,"status":row["status"],"initial_verified":False,"reuse_verified":[]}; cases.append(c)
        try:
            must(re.fullmatch(r"[A-Za-z0-9_-]+",name) is not None,"case_name")
            directory=within(out/name,out)
            task_dir=within(Path(item["task_store"])/item["task_id"],directory)
            req_path=within(item["request"],directory); req=read(req_path)
            task=read(task_dir/"task.json"); binding(task)
            must(files[str(req_path)]["sha256"]==item["files"][str(req_path)]
                 and files[str(task_dir/"task.json")]["sha256"]==plan["task_metadata"][name],"frozen_files")
            must(req["request_id"]==item["request_id"] and task["request"]==req
                 and task["identity"]["manifest_sha256"]==plan["sdk_identity"]
                 and {k:task["identity"][k]["sha256"] for k in plan["stage_identities"]}==plan["stage_identities"],"identity")
            width=sum(math.prod(x["shape"]) for x in req["layout"]["outputs"])
            must(type(width) is int and width>0,"output_width")
            planned_values+=4*width+(4*width if name in plan["reuse_cases"] else 0)
            c["request_id"]=item["request_id"]; c["output_values_per_group"]=width
            must(all(type(row[k]) is int and row[k]>=0 for k in
                     ("compiler_attempts_reserved","encrypted_attempts_reserved")),"reserved_counts")
            calls["compiler_reserved"]+=row["compiler_attempts_reserved"]
            calls["encrypted_reserved"]+=row["encrypted_attempts_reserved"]
            q=read(directory/"qualification.json") if (directory/"qualification.json").exists() else None
            v=read(task_dir/"validation-0.json") if (task_dir/"validation-0.json").exists() else None
            if v:
                must(v["request_id"]==item["request_id"] and v["paid_calls"]==0
                     and v["new_agent_generation"] is False,"validation_identity")
                calls["compiler_confirmed_successful"]+=int(v["observed_stages"].get("compiled") is True)
                calls["encrypted_confirmed_executed"]+=int(v["encrypted_execution"] is True)
            if q and q["status"]=="succeeded":
                a=q["acceptance"]
                must(q["task_id"]==item["task_id"] and q["http_attempts"]==0
                     and q["generation"]["agent_generated"] is False and q["package"]["reused"] is False
                     and a["level"]=="numerical" and a["input_groups"]==4
                     and all(a[k] is True for k in ("compiled","artifact_checked","encrypted_execution","numerically_validated"))
                     and v and v["status"]=="passed" and v["compiled_validated"] is True
                     and v["encrypted_execution"] is True and v["numerically_validated"] is True
                     and a["compared_values"]==v["compared_values"]==4*width
                     and a["maximum_absolute_error"]==v["max_absolute_error"]
                     and (row["status"]!="passed" or row["acceptance"]==a),"initial_acceptance")
                add("initial",4,4*width,a["maximum_absolute_error"])
                calls["encrypted_numerically_passed"]+=1; c["initial_verified"]=True
            if name in plan["reuse_cases"]:
                for suffix,key,groups in (("new-inputs","new_inputs",3),("repeat","repeat",1)):
                    ep,cp=directory/(suffix+".json"),directory/(suffix+"-comparison.json")
                    e=read(ep) if ep.exists() else None; comparison=read(cp) if cp.exists() else None
                    if e:
                        must(e["request_id"]==item["request_id"] and e["backend"]=="poseidon_cpu"
                             and e["agent_calls"]==e["compiler_calls"]==0
                             and e["executor_secret_key"] is False,"reuse_identity")
                        calls["encrypted_confirmed_executed"]+=int(e["encrypted_execution"] is True)
                    if comparison:
                        must(e and e["status"]=="executed" and e["encrypted_execution"] is True
                             and e["samples"]==groups and len(e["outputs"])==groups
                             and q and e["bundle_binding"]==q["package"]["binding"]
                             and comparison["passed"] is True and comparison["atol"]==1e-5
                             and comparison["rtol"]==1e-4
                             and comparison["compared_values"]==groups*width,"reuse_comparison")
                        add(key,groups,groups*width,comparison["max_absolute_error"])
                        calls["encrypted_numerically_passed"]+=1; c["reuse_verified"].append(suffix)
            if row["status"]=="passed":
                must(c["initial_verified"] and row["package_reloaded"] is True
                     and row["package_binding"]==q["package"]["binding"]
                     and (name not in plan["reuse_cases"] or
                         c["reuse_verified"]==["new-inputs","repeat"] and
                         row["new_inputs_executed"] is True and row["repeated_input_executed"] is True),"passed_scope")
            if row["status"]=="failed":
                category=failure_class(row.get("reason",""),row.get("failure_layer"))
                failure_classes[category]+=1; c["failure_class"]=category
        except (ValueError,TypeError,KeyError,OSError,json.JSONDecodeError) as e:
            code=str(e) if type(e) is ValueError and re.fullmatch(r"[a-z0-9_]+",str(e)) else type(e).__name__
            issues.append({"case":name,"code":code})
    final_raw=report_file.read_bytes()
    final_read={"sha256":hashlib.sha256(final_raw).hexdigest(),"bytes":len(final_raw)}
    stable=final_read["sha256"]==snapshot_sha
    must(plan["planned_input_groups"]==156 and planned_values==plan["planned_output_values"]==1272,"planned_totals")
    must(calls["compiler_reserved"]<=36 and calls["encrypted_reserved"]<=42,"reserved_budget")
    counts_match=all(report[k]==statuses[k] for k in ("passed","failed","not_run"))
    confirmed_match=(report["compiler_calls_confirmed"]==calls["compiler_confirmed_successful"]
                     and report["encrypted_calls_confirmed"]==calls["encrypted_confirmed_executed"])
    complete=stable and counts_match and confirmed_match and not issues and report["status"]=="complete" and statuses["passed"]==36 and (
        numerical["initial"]["groups"],numerical["new_inputs"]["groups"],numerical["repeat"]["groups"])==(144,9,3)
    complete=complete and tuple(numerical[k]["values"] for k in numerical)==(1152,90,30)
    complete=complete and tuple(calls[k] for k in ("compiler_reserved","compiler_confirmed_successful",
        "encrypted_reserved","encrypted_confirmed_executed","encrypted_numerically_passed"))==(36,36,42,42,42)
    stop=report.get("stop_reason") or {}
    result={"format":"poseidon-public-dtype-saved-count-audit-v1",
        "status":"complete_verified" if complete else "observed_only",
        "proposal_binding":plan["binding"],"report_binding":report["binding"],"sdk_identity":plan["sdk_identity"],
        "report_stable":stable,"report_final_read":final_read,"report_counts_match":counts_match,
        "confirmed_calls_match":confirmed_match,"source_report_status":report["status"],"planned_cases":36,"unique_requests":12,
        "reported_status_counts":dict(statuses),"calls":dict(calls),"numerical":numerical,
        "failure_classes":dict(failure_classes),
        "global_stop_class":failure_class(stop.get("reason","")) if stop else None,
        "issues":issues,"cases":cases,"files":files,
        "scope":"Saved-result counts only. No reference, model, compiler, executor or provider execution."}
    result["binding"]=digest(result); return result

if __name__=="__main__":
    print(json.dumps(audit(sys.argv[1],sys.argv[2]),sort_keys=True,indent=2,allow_nan=False))
