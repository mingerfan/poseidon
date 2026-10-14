"""Developer source cutover after exact joint numerical/package qualification.

Application code never imports this module or the experimental run identifiers.
Read-only check is default. Historical plans, approvals and reports stay intact.
"""
import argparse,contextlib,fcntl,hashlib,json,math,os,re,subprocess,sys,time,uuid
from pathlib import Path
import importlib.util
import runtime_scope as scope
import source_transaction as transaction

HERE=Path(__file__).resolve().parent
ROOT=scope.ROOT
JOURNALS=HERE/"cutover-records"
TARGET_ORDER=("pytorch_common_constant_provenance.py","pytorch_fused_matrix_normalization.py",
  "pytorch_arithmetic_normalization.py","pytorch_static_trace.py","pytorch_ordered_contract.py",
  "pytorch_ordered_capture.py","pytorch_ordered_intake.py","pytorch_model_intake.py",
  "candidate_bundle.py","application_capabilities.py")
QUIESCENCE_CLAIM="reviewed entrants observed idle; not a global Python lock"

def integer(value,wanted):return type(value) is int and value==wanted

def self_binding(value):
    require(type(value) is dict and value.get("binding")==digest({k:v for k,v in value.items() if k!="binding"}),"Complete receipt self binding")

def reviewed_targets(plan):
    require(plan.get("root")==str(ROOT) and plan.get("format")=="poseidon-common-fused-integration-v1"
      and plan.get("stage")=="prepared_pending_backend_qualification" and plan.get("production_applied") is False,
      "Immutable reviewed checkout and plan")
    self_binding(plan)
    require(set(plan["targets"])==set(TARGET_ORDER) and len(plan["original_production_hashes"])==704
      and len(plan["inside_source_hashes"])==706,"Exact reviewed targets and complete inventories")
    targets={}
    for name in TARGET_ORDER:
        item=plan["targets"][name];relative=str(Path("scripts/baseline")/name)
        require(item.get("new_file") is (item["original_sha256"] is None),"Reviewed new file semantics")
        require(plan["inside_source_hashes"].get(relative)==item["target_sha256"],"Reviewed target inventory binding")
        if item["original_sha256"] is None:
            require(relative not in plan["original_production_hashes"] and item["original_mode"] is None
              and integer(item["target_mode"],0o644),"Reviewed new target")
        else:
            require(plan["original_production_hashes"].get(relative)==item["original_sha256"] and
              type(item["original_mode"]) is int and 0<=item["original_mode"]<=0o777 and
              integer(item["target_mode"],item["original_mode"]),"Reviewed original target")
        source=Path(item["source"])
        require(not source.is_absolute() and source.parts and ".." not in source.parts
          and (ROOT/source)==(ROOT/source).resolve(),"Reviewed overlay source path")
        targets[relative]={k:item[k] for k in ("source","target_sha256","original_sha256","original_mode","target_mode")}
        targets[relative]["source"]=str(ROOT/source)
    require(sum(t["original_sha256"] is None for t in targets.values())==2,"Two new and eight existing targets")
    return targets

def reviewed_context():
    scope.before_launch()
    proposal=scope.read(scope.PROPOSAL);runtime=scope.read(scope.RUN/"plan.json");plan=scope.read(HERE/"plan.json")
    self_binding(proposal);self_binding(runtime);scope.check_plan(runtime)
    targets=reviewed_targets(plan)
    require(proposal["bindings"]["files"].get(str(HERE/"plan.json"))==scope.sha(HERE/"plan.json") and
      proposal["bindings"]["combined_plan_binding"]==runtime["bindings"]["combined_plan_binding"]==plan["binding"],
      "Immutable approved integration plan")
    for value in (proposal,runtime):
        require(value["bindings"]["inside_sources"]==plan["inside_source_hashes"] and
          value["bindings"]["original_production"]==plan["original_production_hashes"],"Approved complete source inventories")
    protected={}
    for item in plan["protected_sources"].values():
        require(scope.sha(item["path"])==item["sha256"],"Immutable protected source manifest")
        merge_hashes(protected,scope.read(item["path"]))
    require(runtime["bindings"]["protected_sources"]==proposal["bindings"]["protected_sources"]==protected,
      "Complete approved protected source inventory")
    return plan,runtime,proposal,targets,protected

def merge_hashes(destination,entries):
    require(type(entries) is dict and entries,"Nonempty evidence inventory")
    for path,h in entries.items():
        require(type(path) is str and type(h) is str and re.fullmatch("[0-9a-f]{64}",h),"Evidence hash entry")
        require(path not in destination or destination[path]==h,"Conflicting evidence hashes")
        destination[path]=h

def protect_evidence(evidence,targets):
    """Protect all returned native/AST evidence, excluding only installed targets."""
    from evidence_audit import Reader
    excluded={str(ROOT/relative) for relative in targets}
    require(excluded=={str(ROOT/"scripts/baseline"/name) for name in TARGET_ORDER},"Only reviewed source targets excluded")
    reader=Reader()
    for path,want in evidence["evidence_files"].items():
        if path in excluded:continue
        raw=reader.bytes(Path(path))
        require(hashlib.sha256(raw).hexdigest()==want,"Qualification evidence changed at source boundary")

def evidence_mode(mode):
    require(type(mode) is str and mode in ("single_run","composite"),"Explicit reviewed evidence mode")
    return mode

def capacity_scope_for_evidence():
    """Load the frozen supplement scope without entering its execution path."""
    path=HERE/"candidate_strategy_v2/capacity_scope.py"
    spec=importlib.util.spec_from_file_location("_poseidon_cutover_capacity_scope_r1301",path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

def validate_qualification(value,runtime,proposal,plan):
    self_binding(value)
    if value.get("format")=="poseidon-joint-composite-cutover-evidence-v1":
        runs=value.get("source_runs")
        require(type(runs) is dict and type(runs.get("original")) is dict,
          "Complete original composite source run")
        original=runs["original"]
        require(original.get("runtime_binding")==runtime["binding"] and
          original.get("proposal_binding")==proposal["binding"] and
          value.get("combined_binding")==plan["binding"],"Exact original composite qualification context")
        from composite_evidence import validate_evidence
        validate_evidence(value,plan)
        return value
    require(set(value)=={"format","runtime_binding","proposal_binding","combined_binding","passed","cases",
      "evidence_files","provider_calls","compiler_calls","encrypted_executions","evidence_observed_only",
      "agent_generated","production_applied","binding"} and value["format"]=="poseidon-joint-cutover-evidence-v1",
      "Complete qualification schema")
    require(value["runtime_binding"]==runtime["binding"] and value["proposal_binding"]==proposal["binding"]
      and value["combined_binding"]==plan["binding"] and integer(value["passed"],12)
      and all(integer(value[k],0) for k in ("provider_calls","compiler_calls","encrypted_executions"))
      and value["evidence_observed_only"] is True and value["agent_generated"] is False
      and value["production_applied"] is False,"Exact qualification context")
    require(type(value["cases"]) is dict and set(value["cases"])==set(scope.CASES),"Complete twelve qualification cases")
    merged={str(scope.RUN/f):scope.sha(scope.RUN/f) for f in
      ("plan.json","staged-sources.json","namespace-receipt.json","report.json")}
    for name,item in value["cases"].items():
        expected=runtime["bindings"]["requests"][name]
        require(type(item) is dict and set(item)=={"case","status","candidate_origin","runtime_binding","source_count",
          "request_id","candidate_sha256","package_binding","application_identity_sha256","backend_identity_sha256",
          "qualification","new_inputs","compilations","qualification_encrypted_phases","new_input_encrypted_phases",
          "cache_generations","provider_calls","http_attempts","reference_semantics","evidence_files"},"Complete case qualification summary")
        require(item.get("case")==name and item.get("status")=="passed" and
          item.get("runtime_binding")==runtime["binding"] and integer(item.get("source_count"),706)
          and item.get("request_id")==expected["request_id"] and item.get("candidate_origin")==scope.CANDIDATE_ORIGIN
          and item.get("reference_semantics")=="frozen_reviewed_runner","Qualified case identity")
        require(all(integer(item.get(k),v) for k,v in dict(compilations=1,qualification_encrypted_phases=1,
          new_input_encrypted_phases=1,cache_generations=0,provider_calls=0,http_attempts=0).items()),"Qualified case phase accounting")
        for field in ("candidate_sha256","package_binding","application_identity_sha256","backend_identity_sha256"):
            require(type(item.get(field)) is str and re.fullmatch("[0-9a-f]{64}",item[field]),"Qualified case identity hash")
        require(item["backend_identity_sha256"]==digest(runtime["bindings"]["backend_identity"]),"Qualified exact SDK identity")
        counts=[]
        for phase,groups in (("qualification",4),("new_inputs",3)):
            summary=item[phase]
            require(set(summary)=={"compared_values","max_absolute_error","atol","rtol"} and
              type(summary["compared_values"]) is int and summary["compared_values"]>0 and
              summary["compared_values"]%groups==0 and summary["atol"]==1e-5 and summary["rtol"]==1e-4
              and type(summary["max_absolute_error"]) in (int,float) and math.isfinite(summary["max_absolute_error"])
              and summary["max_absolute_error"]>=0,"Qualified case numerical summary")
            counts.append(summary["compared_values"]//groups)
        require(counts[0]==counts[1],"Qualification and new input output width")
        case_files=item["evidence_files"]
        require(all(str(scope.RUN/name/(phase+".json")) in case_files for phase in scope.PHASES),"Case phase evidence hash chain")
        merge_hashes(merged,case_files)
    require(merged==value["evidence_files"],"Complete qualification evidence hash chain")
    return value

def validate_context(context,plan,evidence):
    require(type(context) is dict and set(context)=={"combined_plan_binding","qualification_binding","tool_hashes",
      "historical_records_unchanged","compiler_calls","encrypted_executions","provider_calls","quiescence_claim"},"Exact cutover context fields")
    require(context["combined_plan_binding"]==plan["binding"] and context["qualification_binding"]==evidence["binding"]
      and context["tool_hashes"]==tool_hashes() and context["historical_records_unchanged"] is True
      and all(integer(context[k],0) for k in ("compiler_calls","encrypted_executions","provider_calls"))
      and context["quiescence_claim"]==QUIESCENCE_CLAIM,"Exact cutover context semantics")


def require(value,message):
    if not value:raise ValueError(message)

def canonical(value):return scope.canonical(value)
def digest(value):return scope.digest(value)

def active_processes():
    """Conservative observations, not a claim of a global Application lock."""
    found=set(scope.guard().active_processes())
    markers=("poseidon_agent_cpu","poseidon-agent-cpu","poseidon_ckks_runner",
      "model_import_cli.py","pytorch_ordered_intake.py","pytorch_model_capture.py",
      "check_poseidon_application.py","check_context.py","component_worker.py",
      "candidate_worker.py","hecate-opt","frontend")
    for p in Path("/proc").iterdir():
        if not p.name.isdigit() or int(p.name)==os.getpid():continue
        try:
            args=[os.fsdecode(x) for x in (p/"cmdline").read_bytes().split(b"\0") if x]
            if not args:continue
            if any(Path(a).name in markers for a in args):
                found.add(int(p.name));continue
            executable=(p/"exe").resolve().name
            cwd=(p/"cwd").resolve()
            if executable.startswith("python") and cwd.is_relative_to(ROOT):
                found.add(int(p.name))
        except (OSError,ValueError):continue
    return sorted(found)

def validate_report(report,plan):
    require(report.get("format")=="poseidon-ordered-common-offline-report-v1"
      and report.get("binding")==plan["binding"] and report.get("status")=="passed","Exact completed joint run")
    require(report.get("provider_calls")==report.get("http_attempts")==0 and
      type(report.get("provider_calls")) is int and type(report.get("http_attempts")) is int and
      report.get("new_agent_generation") is False and report.get("production_applied") is False and
      report.get("candidate_origin")==scope.CANDIDATE_ORIGIN,"Offline provenance only")
    rows=report.get("rows")
    expected={(n,p) for n in scope.CASES for p in scope.PHASES}
    require(type(rows) is list and len(rows)==len(expected) and
      {(r.get("case"),r.get("phase")) for r in rows}==expected and report.get("pending")==[],"All twelve cases and thirty-six phases")
    require(all(r.get("status")=="passed" and type(r.get("exit_code")) is int and r["exit_code"]==0 for r in rows),"No failed skipped or unknown phases")
    require(type(report.get("completed_phases")) is int and report["completed_phases"]==36 and
      type(report.get("compilations")) is int and report["compilations"]==12,"Exact phase accounting")
    return rows

def qualification(mode="single_run"):
    evidence_mode(mode)
    integration,plan,proposal,targets,protected=reviewed_context()
    if mode=="composite":
        from composite_evidence import qualification as qualify_composite
        value=qualify_composite()
        validate_qualification(value,plan,proposal,integration)
        return value
    staged=scope.read(scope.RUN/"staged-sources.json")
    require(staged==plan["bindings"]["inside_sources"] and len(staged)==706,"Exact actually loaded joint source")
    receipt=scope.read(scope.RUN/"namespace-receipt.json")
    require(receipt.get("actual_nix_shell")=="pure" and receipt.get("staged_modules")==10 and
      receipt.get("project_credential_path_absent") is True and receipt.get("production_applied") is False and
      receipt.get("provider_calls")==0,"Actual namespace receipt")
    report=scope.read(scope.RUN/"report.json");validate_report(report,plan)
    from evidence_audit import validate_case
    cases={name:validate_case(scope.RUN/name,name,plan) for name in scope.CASES}
    evidence={str(scope.RUN/f):scope.sha(scope.RUN/f) for f in
      ("plan.json","staged-sources.json","namespace-receipt.json","report.json")}
    for case in cases.values():merge_hashes(evidence,case["evidence_files"])
    data=dict(format="poseidon-joint-cutover-evidence-v1",runtime_binding=plan["binding"],
      proposal_binding=proposal["binding"],combined_binding=plan["bindings"]["combined_plan_binding"],
      passed=12,cases=cases,evidence_files=evidence,provider_calls=0,compiler_calls=0,encrypted_executions=0,
      evidence_observed_only=True,agent_generated=False,production_applied=False)
    data["binding"]=digest(data)
    validate_qualification(data,plan,proposal,integration)
    return data

def validate_journal(journal,folder,targets,context):
    require(set(journal)=={"format","state","context","root","targets","installed","restored",
      "rollback_conflicts","rollback_intents","retained_files","created_wall","completed_wall",
      "install_intents","install_conflicts"} and journal["format"]=="poseidon-agent-source-transaction-v1"
      and journal["state"]=="applied" and journal["root"]==str(ROOT) and journal["context"]==context
      and journal["targets"]==targets and journal["installed"]==list(targets)
      and journal["restored"]==[] and journal["rollback_conflicts"]==[] and journal["rollback_intents"]=={}
      and journal["install_conflicts"]==[],"Complete successful source journal semantics")
    require(all(type(journal[k]) in (int,float) and math.isfinite(journal[k]) for k in ("created_wall","completed_wall"))
      and 0<journal["created_wall"]<=journal["completed_wall"],"Successful journal timing")
    require(type(journal["install_intents"]) is dict and set(journal["install_intents"])==set(targets),"All installation intents committed")
    retained=[]
    for relative,item in targets.items():
        target=ROOT/relative;intent=journal["install_intents"][relative]
        require(type(intent) is dict and set(intent)=={"state","retained","prepared"}
          and intent["state"]=="installed" and type(intent["prepared"]) is str,"Successful installation intent")
        prepared=Path(intent["prepared"])
        require(prepared.parent==target.parent and re.fullmatch(r"\."+re.escape(target.name)+r"\.[0-9a-f]{32}\.tmp",prepared.name)
          and not os.path.lexists(prepared),"Consumed reviewed prepared file")
        if item["original_sha256"] is None:
            require(intent["retained"] is None,"New target has no displaced original")
        else:
            original=folder/"install-displaced"/target.name/"current"
            require(intent["retained"]==str(original) and original==original.resolve() and
              scope.sha(original)==item["original_sha256"] and original.stat().st_mode&0o777==item["original_mode"],
              "Exact retained original and permission")
            retained.append(str(original))
    require(journal["retained_files"]==retained,"Complete retained original inventory")

def applied_state():
    if not JOURNALS.exists():return None
    require(JOURNALS==JOURNALS.resolve() and not JOURNALS.is_symlink(),"Regular journal directory")
    for folder in sorted(JOURNALS.iterdir(),reverse=True):
        if not folder.is_dir() or folder.is_symlink():continue
        if not (folder/"record.json").exists():
            if (folder/"journal.json").exists():
                journal=scope.read(folder/"journal.json")
                require(journal.get("state")=="rolled_back","Incomplete source transaction requires recovery review")
            continue
        record=scope.read(folder/"record.json");self_binding(record)
        require(set(record)=={"format","state","recovery","journal_sha256","qualification_sha256","context",
          "production_sources","protected_sources","production_applied","new_agent_generation","provider_calls",
          "compiler_calls","encrypted_executions","binding"} and record["format"]=="poseidon-common-fused-cutover-v1"
          and record["state"]=="applied" and record["recovery"]==str(folder) and record["production_applied"] is True
          and record["new_agent_generation"] is False and all(integer(record[k],0) for k in
          ("provider_calls","compiler_calls","encrypted_executions")),"Independent applied cutover record")
        require(scope.sha(folder/"journal.json")==record["journal_sha256"] and
          scope.sha(folder/"qualification.json")==record["qualification_sha256"],"Cutover evidence files")
        plan,runtime,proposal,targets,protected=reviewed_context()
        evidence=scope.read(folder/"qualification.json")
        validate_qualification(evidence,runtime,proposal,plan)
        validate_context(record["context"],plan,evidence)
        journal=scope.read(folder/"journal.json")
        validate_journal(journal,folder,targets,record["context"])
        require(record["production_sources"]==plan["inside_source_hashes"] and record["protected_sources"]==protected,
          "Applied record matches immutable complete inventories")
        protect_evidence(evidence,targets)
        mode="composite" if evidence["format"]=="poseidon-joint-composite-cutover-evidence-v1" else "single_run"
        require(qualification(mode=mode)==evidence,"Independent qualification semantic and evidence chain")
        require(current_inventory()==record["production_sources"] and
          all(scope.sha(ROOT/p)==h for p,h in protected.items()),"Applied source or protected file changed")
        require(all(scope.sha(ROOT/p)==item["target_sha256"] and (ROOT/p).stat().st_mode&0o777==item["target_mode"]
          for p,item in targets.items()),"Installed target bytes and permissions")
        return dict(status="applied_with_staged_joint_backend_evidence",record=str(folder/"record.json"),
          production_applied=True,new_agent_generation=False)
    return None

def check(mode="single_run"):
    evidence_mode(mode)
    applied=applied_state()
    if applied is not None:return applied
    scope.guard().check()
    processes=active_processes()
    if processes:return dict(status="waiting_for_process_exit",active_processes=processes,production_applied=False)
    if not scope.APPROVAL.exists():return dict(status="waiting_for_offline_approval",production_applied=False)
    if not (scope.RUN/"report.json").exists():return dict(status="waiting_for_joint_qualification",production_applied=False)
    if mode=="composite":
        capacity=capacity_scope_for_evidence()
        if not capacity.APPROVAL.exists():
            return dict(status="waiting_for_capacity_offline_approval",production_applied=False)
        if not (capacity.RUN/"report.json").exists():
            return dict(status="waiting_for_capacity_qualification",production_applied=False)
    evidence=qualification(mode=mode)
    return dict(status="ready",evidence_binding=evidence["binding"],cases=12,phases=36,production_applied=False)

def tool_hashes():
    names=("cutover.py","source_transaction.py","evidence_audit.py","composite_evidence.py",
           "test_source_transaction.py","test_evidence_audit.py","test_cutover.py","test_composite_evidence.py")
    return {name:scope.sha(HERE/name) for name in names}

@contextlib.contextmanager
def execution_locks():
    # Existing launcher and two native locks are not a global Python entry lock.
    paths=[scope.WORK/"deps/nix-portable-v012/launcher.lock"]
    paths += [scope.WORK/"cache/agent-native-slots"/("slot-"+str(i)+".lock") for i in range(2)]
    with contextlib.ExitStack() as stack:
        for path in paths:
            require(path.is_file() and not path.is_symlink(),"Existing regular execution lock")
            stream=stack.enter_context(path.open("r+"))
            fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        yield

def current_inventory():
    sys.path.insert(0,str(scope.BASE))
    from semantic_benchmark_execution import runtime_sources
    return runtime_sources()

def apply(mode="single_run"):
    evidence_mode(mode)
    state=check(mode=mode)
    if state["status"]!="ready":return state
    with execution_locks():
        require(check(mode=mode)["status"]=="ready","Reviewed cutover boundary changed")
        plan,runtime,proposal,targets,protected=reviewed_context();evidence=qualification(mode=mode)
        validate_qualification(evidence,runtime,proposal,plan)
        require(current_inventory()==plan["original_production_hashes"],"Complete unchanged original inventory")
        original=plan["original_production_hashes"]
        unchanged={p:h for p,h in original.items() if p not in targets}
        def observe():
            protect_evidence(evidence,targets)
            require(subprocess.check_output(["git","branch","--show-current"],cwd=ROOT,text=True,timeout=5).strip()==plan["branch"] and
              subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True,timeout=5).strip()==plan["head"],"Branch or HEAD changed during cutover")
            require(not active_processes(),"New project process at source boundary")
            require(all(scope.sha(ROOT/p)==h for p,h in protected.items()),"Backend reference or tolerance changed")
            require(all(scope.sha(ROOT/p)==h for p,h in unchanged.items()),"Unrelated user source changed")
        def finalize():
            protect_evidence(evidence,targets)
            require(current_inventory()==plan["inside_source_hashes"],"Complete installed joint source inventory")
            require(all(scope.sha(ROOT/p)==h for p,h in protected.items()),"Protected files after cutover")
            transaction.atomic_json(recovery/"qualification.json",evidence)
        observe()
        JOURNALS.mkdir(exist_ok=True)
        require(JOURNALS==JOURNALS.resolve() and not JOURNALS.is_symlink(),"Regular checkout journal directory")
        recovery=JOURNALS/(time.strftime("%Y%m%dT%H%M%S",time.gmtime())+"-"+uuid.uuid4().hex[:8])
        context=dict(combined_plan_binding=plan["binding"],qualification_binding=evidence["binding"],
          tool_hashes=tool_hashes(),historical_records_unchanged=True,compiler_calls=0,
          encrypted_executions=0,provider_calls=0,quiescence_claim=QUIESCENCE_CLAIM)
        journal=transaction.apply_transaction(ROOT,targets,recovery,observe,context,finalize)
        # The full source receipt is attached to the independent transaction.
        record=dict(format="poseidon-common-fused-cutover-v1",state="applied",recovery=str(recovery),
          journal_sha256=scope.sha(recovery/"journal.json"),qualification_sha256=scope.sha(recovery/"qualification.json"),
          context=context,production_sources=plan["inside_source_hashes"],protected_sources=protected,
          production_applied=True,new_agent_generation=False,provider_calls=0,compiler_calls=0,encrypted_executions=0)
        record["binding"]=digest(record)
        transaction.atomic_json(recovery/"record.json",record)
        return dict(status="applied_with_staged_joint_backend_evidence",record=str(recovery/"record.json"),
          production_applied=True,new_agent_generation=False)

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--apply",action="store_true")
    parser.add_argument("--evidence-mode",choices=("single_run","composite"),default="single_run")
    args=parser.parse_args(argv)
    print(json.dumps(apply(mode=args.evidence_mode) if args.apply else check(mode=args.evidence_mode),sort_keys=True))
if __name__=="__main__":main()
