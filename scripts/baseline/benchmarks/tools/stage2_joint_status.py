"""Read-only joint stage-2 status. Writes only to a new explicit report path."""
import argparse,hashlib,json,sys
from pathlib import Path
BASE=Path(__file__).resolve().parents[2];ROOT=BASE.parents[1]
sys.path.insert(0,str(BASE))
from benchmark_graph import digest
from benchmark_runner import strict_file,dump
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def bound(p):
 r=strict_file(p,8*1024**2)
 if digest({k:v for k,v in r.items() if k!="binding"})!=r["binding"]:raise ValueError("Report binding: "+str(p))
 return r
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",type=Path);a=p.parse_args()
 from workspace_paths import RESULTS
 docs=ROOT/"docs/baseline"
 paths=dict(closure=RESULTS/"benchmark-r43-stage2-closure/report.json",
            context=docs/"stage2-context-supplement-r48.json",helpers=docs/"stage2-helper-mapping-r49.json",
            original_gap_plan=docs/"stage2-gap-plan-r43.json",
            regression=RESULTS/"benchmark-r50-revision-regression/report.json",
            gates=RESULTS/"stage2-gates-r51/report.json",
            public_utilities=RESULTS/"stage2-public-utilities-r52/report.json",
            public_constants=RESULTS/"stage2-public-constants-r53/report.json",
            return_paths=docs/"stage2-return-paths-r54.json",
            api_retrace=RESULTS/"stage2-api-retrace-r56/report.json",
            public_stride2=RESULTS/"stage2-public-constants-stride2-r57/report.json",
            expr_frontend=RESULTS/"stage2-expr-frontend-r62/report.json",
            retry0=RESULTS/"stage2-original-blockers-r58-0/report.json",
            retry1=RESULTS/"stage2-original-blockers-r58-1/report.json",
            retry2=RESULTS/"stage2-original-blockers-r58-2/report.json",
            retry_helpers_w45=RESULTS/"stage2-original-blockers-r59-2/report.json",
            retry_helpers_w40=RESULTS/"stage2-original-blockers-r60-w40/report.json",
            native_api=RESULTS/"stage2-native-api-r63/report.json",
            sumslots=RESULTS/"stage2-sumslots-r65/report.json",
            public_exports=docs/"stage2-public-exports-r66.json",
            dependency_acceptance=docs/"stage2-dependency-acceptance-r86.json",
            polynomial_capacity=docs/"stage2-polynomial-capacity-r79.json",
            source_impact=docs/"stage2-source-impact-r81.json",
            original_queue=docs/"stage2-original-queue-r83.json",
            contract_linkage=docs/"stage2-contract-linkage-r90.json",
            polynomial_candidates=RESULTS/"stage2-polynomial-candidates-r89/report.json",
            polynomial_infrastructure=RESULTS/"stage2-polynomial-candidates-r88/report.json",
            v10_regression=RESULTS/"stage2-v10-regression-r91/report.json")
 reports={k:bound(v) for k,v in paths.items()};c=reports["context"];h=reports["helpers"];old=reports["closure"]
 utilities=reports["public_utilities"];gates=reports["gates"]
 if utilities["status"]!="passed" or gates["status"]!="passed":raise ValueError("Incomplete joint audit")
 for name,hsh in gates["files"].items():
  if Path(name).name!=name or sha(paths["gates"].parent/name)!=hsh:raise ValueError("Gate report integrity")
 if sha(paths["public_utilities"].parent/"records.json")!=utilities["records_sha256"]:raise ValueError("Utility records changed")
 constants=reports["public_constants"];returns=reports["return_paths"];retrace=reports["api_retrace"]
 if constants["status"]!="passed" or returns["status"]!="passed":raise ValueError("Incomplete dependency audit")
 for filename,key in (("plan.json","plan_sha256"),("records.json","records_sha256")):
  if sha(paths["public_constants"].parent/filename)!=constants[key]:raise ValueError("Constant evidence changed")
 if returns["parent_audit_sha256"]!=sha(paths["helpers"]):raise ValueError("Return-path parent mismatch")
 if len(returns["mappings"])!=returns["mapped_internal_symbols"]:raise ValueError("Return mapping count")
 mappings={r["symbol"]:r for r in returns["mappings"]}
 # Retracing records actual calls and exact old artifacts, never per-API contribution.
 observed={}
 if any(r["status"]!="identical_artifacts" for r in retrace["rows"]):raise ValueError("Incomplete retracing diagnostic")
 for r in retrace["rows"]:
  out=paths["api_retrace"].parent/r["id"]/"output"
  for name,hsh in r["files"].items():
   if Path(name).name!=name or sha(out/name)!=hsh:raise ValueError("Retrace artifact changed")
  if sha(Path(r["historical_evidence"])/"report.json")!=r["historical_report_sha256"]:raise ValueError("Retrace parent changed")
  for event in r["calls"]:
   observed.setdefault(event["symbol"],set()).add(r["topology"])
 stride=reports["public_stride2"];frontend=reports["expr_frontend"]
 if stride["status"]!="passed" or frontend["status"]!="passed":raise ValueError("Incomplete parameter/frontend check")
 for filename,key in (("plan.json","plan_sha256"),("records.json","records_sha256")):
  if sha(paths["public_stride2"].parent/filename)!=stride[key]:raise ValueError("Stride evidence changed")
 for row in frontend["rows"]:
  for name,hsh in row["files"].items():
   path=paths["expr_frontend"].parent/str(row["context"])/name
   if not path.resolve().is_relative_to(paths["expr_frontend"].parent) or sha(path)!=hsh:raise ValueError("Frontend evidence changed")
  checks={r["check"]:r for r in row["checks"]["records"]}
  for name in ("copy.Expr","copy.Plain","copy.Func","deepcopy.Expr","deepcopy.Plain","deepcopy.Func","copy.installed_hook"):
   if checks[name]["status"]!="rejected":raise ValueError("Copy rejection not verified")
 if len({r["context"] for r in frontend["rows"]})!=3:raise ValueError("Missing frontend contexts")
 retry_rows=[];execution_records=[]
 for key in ("retry0","retry1","retry2","retry_helpers_w45","retry_helpers_w40"):
  retry=reports[key];folder=paths[key].parent
  plan=bound(folder/"plan.json")
  if retry["plan_binding"]!=plan["binding"]:raise ValueError("Retry plan drift")
  planned={r["model"]["id"]:r["model_sha256"] for r in plan["models"]}
  if key in ("retry0","retry1","retry2"):retry_rows.extend(retry["rows"])
  for row in retry["rows"]:
   if row["model_sha256"]!=planned[row["id"]]:raise ValueError("Retry model drift")
   if row.get("evidence"):
    if sha(Path(row["evidence"])/"report.json")!=row["report_sha256"]:raise ValueError("Retry report changed")
    if sha(folder/(row["id"]+".log"))!=row["log_sha256"]:raise ValueError("Retry log changed")
    if row["execution_status"]=="passed":
     audit=folder/(row["id"]+".audit.json")
     if sha(audit)!=row["audit_sha256"]:raise ValueError("Retry audit changed")
     proof=strict_file(audit,8*1024**2)
     evidence=Path(row["evidence"])
     for name,hsh in proof["files"].items():
      relative=Path(name);path=evidence/relative
      if (relative.is_absolute() or ".." in relative.parts or path.is_symlink() or
          not path.resolve().is_relative_to(evidence.resolve()) or sha(path)!=hsh):
       raise ValueError("Retry evidence file integrity")
    execution_records.append(dict(id=row["id"],model_sha256=row["model_sha256"],
       state=row["execution_status"],compiler_configuration=plan["compiler_configuration"],
       evidence=row["evidence"],report_sha256=row["report_sha256"],
       max_absolute_error=row.get("comparison",{}).get("max_absolute_error"),
       compared_values=row.get("comparison",{}).get("compared_values")))
 if len(retry_rows)!=112 or len({r["id"] for r in retry_rows})!=112:raise ValueError("Original retry denominator")
 exports=reports["public_exports"]
 for name,hsh in exports["source_hashes"].items():
  if sha(ROOT/name)!=hsh:raise ValueError("Public export audit source changed")
 if exports["unknown_executable_helper_definitions"]:raise ValueError("Unclassified public helper definition")
 native=reports["native_api"];summation=reports["sumslots"]
 if native["status"]!="passed" or summation["status"]!="passed":raise ValueError("Incomplete native/dependency audit")
 native_plan=bound(paths["native_api"].parent/"plan.json")
 sum_plan=bound(paths["sumslots"].parent/"plan.json")
 if native["plan_binding"]!=native_plan["binding"] or summation["plan_binding"]!=sum_plan["binding"]:
  raise ValueError("Dependency plan identity")
 def verify_retained(folder,hashes):
  for name,hsh in hashes.items():
   relative=Path(name);path=folder/relative
   if (relative.is_absolute() or ".." in relative.parts or path.is_symlink() or
       not path.resolve().is_relative_to(folder.resolve()) or sha(path)!=hsh):
    raise ValueError("Dependency retained evidence changed")
 for row in native["rejections"]:
  verify_retained(paths["native_api"].parent/(row["kind"]+"_"+str(row["context"])),row["files"])
  if row["status"]!="passed":raise ValueError("Native rejection failed")
 native_contexts={}
 for row in native["historical_contexts"]:
  native_contexts.setdefault(row["partition"],set()).add(row["topology"])
  if sha(Path(row["evidence"])/"report.json")!=row["report_sha256"]:raise ValueError("Native historical report changed")
  audit_path=RESULTS/"benchmark-r43-directed-independent"/(row["task"]+".json")
  if sha(audit_path)!=row["audit_sha256"]:raise ValueError("Native historical audit changed")
  verify_retained(Path(row["evidence"]),strict_file(audit_path,8*1024**2)["files"])
 if any(len(v)<3 for v in native_contexts.values()):raise ValueError("Native context coverage")
 dependency_contexts={};dependency_values=0;dependency_maximum=0.
 specs={r["id"]:r for r in sum_plan["models"]}
 from benchmark_graph import signature
 for row in summation["rows"]:
  folder=Path(row["folder"]);verify_retained(folder,row["files"])
  result=strict_file(folder/"report.json",8*1024**2);spec=specs[row["id"]]
  if result["status"]!="passed" or not result["compiled"] or not result["encrypted_execution"]:
   raise ValueError("Dependency missing actual FHE")
  if digest(strict_file(folder/"model.json",131072))!=spec["model_sha256"]:raise ValueError("Dependency model identity")
  if not strict_file(folder/"key-cleanup-outcome.json",65536)["complete"]:raise ValueError("Dependency private keys not cleaned")
  if result["helper_zero_intervention_delta"]<=1e-8 or (spec["m"]>1 and result["rotation_intervention_delta"]<=1e-8):
   raise ValueError("Dependency does not distinguish helper/rotation result")
  comparison=result["comparison"]
  if not comparison["passed"] or (comparison["atol"],comparison["rtol"])!=(1e-5,1e-4):
   raise ValueError("Dependency numerical threshold")
  dependency_values+=comparison["compared_values"]
  dependency_maximum=max(dependency_maximum,comparison["max_absolute_error"])
  dependency_contexts.setdefault(row["regime"],set()).add(signature(spec["model"],True))
 if set(dependency_contexts)!={"identity","power_of_two","remainder"} or any(len(v)<3 for v in dependency_contexts.values()):
  raise ValueError("Dependency regime coverage")
 dependencies=reports["dependency_acceptance"]
 if not dependencies["all_recorded_outcomes_verified"]:raise ValueError("Unverified dependency outcome")
 for batch,source in dependencies["sources"].items():
  report_path=Path(source["report"]);folder=report_path.parent
  for name,key in (("report.json","report_sha256"),("plan.json","plan_sha256"),("runner.py","runner_sha256"),("worker.py","worker_sha256")):
   if sha(folder/name)!=source[key]:raise ValueError("Dependency source snapshot drift")
 from audit_dependency_acceptance import trace_check
 dependency_rows=dependencies["rows"]
 for record in dependency_rows:
  folder=Path(record["folder"]);verify_retained(folder,record["files"])
  result=strict_file(folder/"report.json",8*1024**2)
  if sha(folder/"report.json")!=record["report_sha256"] or result["status"]!=record["status"]:
   raise ValueError("Dependency audit/report identity")
  trace_check(result,record["regime"],"expr" in record["batch"])
 dependency_api={
  "expr.hecateMetaBinary.__new__.binaryFactory.binaryMethod":["normal_public","augmented_public","cipher_pair"],
  "expr.hecateMetaBinary.__new__.binaryFactory.binaryReverseMethod":["reverse_public"],
  "expr.hecateMetaBinary.__new__.innerFactory.innerMethod":["negate_rotate"],
  "expr.hecateMetaBinary.__new__.rotate":["negate_rotate","cipher_pair"],
  "expr.resolveType":["public_conversions","empty_protocol"],
  "expr.Plain":["public_conversions"],"expr.Plain.__init__":["public_conversions"],
  "expr.Empty":["empty_protocol","empty_operators"],"expr.Empty.__init__":["empty_protocol","empty_operators"],
  "MPCB.GenPoly":["leaf","tree15_first","tree15_second","tree27"],
  "MPCB.GenPoly.polynomial":["leaf","tree15_first","tree15_second","tree27"],
  "Poly.GenPoly":["default"]}
 for name in ("__add__","__radd__","__iadd__","__sub__","__rsub__","__isub__"):
  dependency_api["expr.Empty."+name]=["empty_protocol"]
 for symbol,regimes in dependency_api.items():
  for regime in regimes:
   proofs=[r for r in dependency_rows if r["regime"]==regime and r["status"]=="passed"]
   if len({r["topology"] for r in proofs})<3:raise ValueError("Incomplete API numerical contexts: "+symbol)
 capacity=reports["polynomial_capacity"];impact=reports["source_impact"];queue=reports["original_queue"]
 for name,hsh in capacity["sources"].items():
  if sha(ROOT/name)!=hsh:raise ValueError("Capacity source/profile changed")
 for parent in capacity["parents"].values():
  folder=Path(parent["path"])
  if sha(folder/"plan.json")!=parent["plan_sha256"] or sha(folder/"report.json")!=parent["report_sha256"]:
   raise ValueError("Capacity execution evidence changed")
 for mapping in capacity["mappings"]:
  if mapping["encrypted_passes"]!=0 or mapping["contexts"]!=3:raise ValueError("Capacity blocker miscounted")
  for record in mapping["records"]:
   folder=Path(record["folder"])
   for name,key in (("report.json","report_sha256"),("compile.log","log_sha256"),("output/dependency-calls.json","calls_sha256")):
    if sha(folder/name)!=record[key]:raise ValueError("Capacity diagnostic changed")
 from semantic_benchmark_execution import runtime_sources
 linkage=reports["contract_linkage"];polycandidates=reports["polynomial_candidates"]
 if digest(runtime_sources())!=linkage["current_source_sha256"]:raise ValueError("Contract linkage stale")
 before_path=RESULTS/"stage2-polynomial-contract-r87/before-sources.json"
 if digest(strict_file(before_path,8*1024**2))!=impact["current_runtime_sha256"]:raise ValueError("Historical source bridge mismatch")
 if not linkage["all_legacy_contracts_accepted"] or linkage["legacy_counts"]!={"construction:accepted":627,"helper:accepted":57}:raise ValueError("Legacy contract regression")
 for name,hsh in linkage["parents"].items():
  if sha(Path(name))!=hsh:raise ValueError("Contract parent changed")
 polyplan=bound(paths["polynomial_candidates"].parent/"plan.json")
 if polycandidates["plan_binding"]!=polyplan["binding"] or polycandidates["passed"]!=21 or polycandidates["failed"] or polycandidates["skipped"]:raise ValueError("Polynomial candidate coverage")
 if polyplan["source_hashes"]!=runtime_sources():raise ValueError("Polynomial candidate source drift")
 if sha(paths["polynomial_candidates"].parent/"runner.py")!=polyplan["runner_sha256"] or sha(paths["polynomial_candidates"].parent/"fixed_polynomial_cases.py")!=polyplan["fixtures_sha256"]:raise ValueError("Polynomial runner drift")
 polycontexts={};polyvalues=0;polymaximum=0.
 for record in polycandidates["rows"]:
  folder=Path(record["evidence"])
  if sha(folder/"report.json")!=record["report_sha256"]:raise ValueError("Polynomial report drift")
  audit=paths["polynomial_candidates"].parent/record["id"]/"audit.json"
  if sha(audit)!=record["audit_sha256"]:raise ValueError("Polynomial audit drift")
  proof=strict_file(audit,8*1024**2);verify_retained(folder,proof["files"])
  if not proof["helper_coverage"]["finite_return_influence_checked"] or not proof["helper_coverage"]["actual_frontend_checked"]:raise ValueError("Polynomial contribution missing")
  if (proof["comparison"]["atol"],proof["comparison"]["rtol"])!=(1e-5,1e-4):raise ValueError("Polynomial threshold changed")
  polycontexts.setdefault(record["helper"],set()).add(record["context"])
  polyvalues+=proof["comparison"]["compared_values"];polymaximum=max(polymaximum,proof["comparison"]["max_absolute_error"])
 if len(polycontexts)!=7 or any(v!={0,1,2} for v in polycontexts.values()):raise ValueError("Polynomial contexts missing")
 regression=reports["v10_regression"]
 if regression["failed"] or regression["runtime_sha256"]!=digest(runtime_sources()):raise ValueError("v10 regression failed or stale")
 if sha(paths["v10_regression"].parent/"tests.log")!=regression["log_sha256"]:raise ValueError("Regression log drift")
 if not impact["old_passes_remain_historical"]:raise ValueError("Historical evidence rebound")
 for parent in queue["sources"].values():
  folder=Path(parent["path"])
  if sha(folder/"plan.json")!=parent["plan_sha256"] or sha(folder/"report.json")!=parent["report_sha256"]:
   raise ValueError("Original queue evidence changed")
 for model in queue["rows"]:
  for attempt in model["attempts"]:verify_retained(Path(attempt["evidence"]),attempt["files"])
 if not queue["all_planned_cases_accounted_for"] or queue["ready_unrun"]!=0:raise ValueError("Original queue incomplete")
 capacity_by_symbol={r["symbol"]:r for r in capacity["mappings"]}
 api=[dict(r) for r in h["api"]]
 for row in api:
  symbol=Path(row["source"]).stem+"."+row["symbol"]
  if symbol in utilities["symbols"]:
   if utilities["positive_contexts"][symbol]<3:raise ValueError("Insufficient public contexts")
   row.update(state="public_preprocessing_verified",public_contexts=utilities["positive_contexts"][symbol],
      reason="Actual pinned utility with independent public reference; no ciphertext coverage or direct candidate imports added")
  if symbol in constants["positive_contexts"]:
   count=constants["positive_contexts"][symbol]
   if count<3:raise ValueError("Insufficient constant contexts")
   row.update(state="public_preprocessing_verified",public_contexts=count,
     reason="Independent coordinate references for pinned public constants; bounded geometry, no FHE coverage claim")
  if symbol in mappings:
   mapping=mappings[symbol]
   if len({c["topology"] for c in mapping["contexts"]})<3:raise ValueError("Insufficient internal return contexts")
   row.update(state="internal_direct_return_mapped",contexts=len(mapping["contexts"]),
     branches_exhausted=False,
     reason="Callee result equals witnessed parent result in bounded adapter contexts; not arbitrary parameter coverage")
  if symbol in stride["positive_contexts"]:
   row["additional_stride2_checks"]=stride["positive_contexts"][symbol]
   row["stride2_checks_are_not_new_models"]=True
  if symbol=="expr.hecateMetaBase.__new__.raiser":
   row.update(state="intentional_rejection_verified",contexts=3,
      reason="Actual pinned copy/deepcopy rejection for Expr, Plain and Func in fresh frontend contexts; no FHE claim")
  if symbol in {"expr."+name for name in native["symbols"]}:
   row.update(state="native_function_semantics_mapped",partitions=len(native_contexts),
    contexts=len(native["historical_contexts"]),rejection_contexts=len(native["rejections"]),
    reason="Historical native call/return partitions with three contexts each plus actual current frontend rejections; empty returns remain structural")
  if symbol in ("MPCB.roll","MPCB.shapeClosure.SumSlots"):
   row.update(state="cipher_dependency_verified",regimes={k:len(v) for k,v in dependency_contexts.items()},
    reason="Actual unchanged SumSlots, observed roll, direct returned handle, independent coordinate reference and real SEAL; bounded period-16 adapter contexts")
  if symbol=="MPCB.shapeClosure":
   row.update(state="trusted_closure_factory_verified",
    reason="Actual closure factory with geometry/constant/parent-return evidence and SumSlots execution; candidate cannot choose arbitrary closure fields")
  if symbol in dependency_api:
   row.update(state="bounded_dependency_execution_verified",regimes=dependency_api[symbol],
     candidate_direct_api_access_added=False,
     reason="Actual fixed upstream protocols plus three numerical contexts per named regime; trusted/implicit construction is distinguished from direct candidate API access")
   if symbol.startswith("expr.Empty"):
    row["dispatch_scope"]="Six direct protocol methods; public Empty-left/augmented operators. Expr+Empty and Expr-Empty reject; direct reverse methods do not imply Python reverse dispatch."
   if symbol.startswith("MPCB.GenPoly"):
    row["polynomial_scope"]="Fixed odd leaf/tree15/tree27 partitions; default tree and deep compositions have separate unresolved failures"
  if symbol in ("Poly.GenPoly","Poly.sign","Poly.relua","Poly.genRelu6"):
   regime={"Poly.GenPoly":"default","Poly.sign":"sign","Poly.relua":"relua","Poly.genRelu6":"genRelu6"}[symbol]
   row["current_attempts"]=[dict(id=r["id"],batch=r["batch"],status=r["status"],failure_layer=r["failure_layer"],
     actual_ciphertext_execution=r["actual_ciphertext_execution"]) for r in dependency_rows if r["regime"]==regime]
   row["reason"]="Concrete compiler, artifact-gate or numerical failures retained; three-context execution acceptance remains incomplete, no bootstrap substitution"
  if symbol=="Poly.GenPoly":
   proof=[r for r in dependency_rows if r["batch"]=="stage2-polynomial-parent-w40-r75" and r["status"]=="passed"]
   if len({r["topology"] for r in proof})!=3:raise ValueError("Default polynomial parent contexts")
   row.update(reason="Three actual w40 contexts inside x*(fixed_poly(x)+0.5); direct-return scaled/compiler, w45/gate, and w40/rotated numerical failures are retained separately",
      scope="Fixed upstream default tree and coefficients in witnessed parent composition; not arbitrary GenPoly degree or return placement")
  if symbol in ("Poly.GenPoly","MPCB.GenPoly","MPCB.GenPoly.polynomial"):
   row.update(state="candidate_fixed_polynomial_verified",
      candidate_profile="upstream-poly-fixed-polynomials-v10",
      reason="Actual source-locked polynomial calls now exposed via inert candidate AST; seven fixed functions with three contexts each, traced return influence and SEAL numerics",
      scope="Default fixed tree, tree15/tree27, and fixed odd-leaf scale choices; no arbitrary degree/tree/coefficients or automatic schedule guarantee",
      old_failed_spellings_preserved=True)
  if symbol in capacity_by_symbol:
   obstruction=capacity_by_symbol[symbol]
   row.update(state=obstruction["state"],tested_configurations=obstruction["tested_configurations"],
    blocked_contexts=obstruction["contexts"],blocked_attempts=obstruction["attempts"],
    reason=obstruction["reason"],encrypted_passes=0,
    alternative_schedule_impossibility_proven=False,source_contains_bootstrap=False)
  row["observed_trace_contexts"]=len(observed.get(symbol,set()))
  row["trace_presence_is_not_contribution"]=True
 from collections import Counter
 counts=dict(Counter(r["state"] for r in api))

 result=dict(format="poseidon-stage2-status-v1",
   evidence_sources={k:dict(path=str(v),sha256=sha(v),binding=reports[k]["binding"]) for k,v in paths.items()},
   coverage=dict(frozen_requirements=401,mathematical=dict(denominator=c["mathematical_partitions"],
       three_contexts=c["three_context_partitions"],scope="Historical bindings plus explicit 30-model diagnostic supplement"),
       construction=dict(denominator=209,three_contexts=old["directed_three_context_partitions"],
          tasks=old["directed_verified"],numeric=old["directed_numeric"],mixed=old["directed_mixed"],
          structural=old["directed_structural"],scope="Historical r29; structural tasks score separately"),
       helper=dict(denominator=18,three_contexts=h["verified_helper_partitions"],
          blocked=h["blocked_helper_partitions"],tasks=h["verified_helper_tasks"]),
       compiler=dict(denominator=5,three_contexts=gates["compiler_partitions"],scope=gates["compiler_scope"]),
       rejection=dict(denominator=16,three_contexts=gates["static_rejection_partitions"],
          tasks=gates["static_rejection_tasks"],passed=gates["static_rejection_passed"],scope="Static gates only"),
       upstream_api=dict(denominator=107,dispositions=counts,
         exhaustive_behavior_coverage=False,
         bounded_return_mappings=returns["mapped_internal_symbols"],
         retraced_tasks=len(retrace["rows"]),retracing_new_encrypted_executions=0,
         dependency_new_encrypted_executions=len(summation["rows"]))),api_dispositions=api,
   original_baseline=dict(planned=1200,passed=1088,compile_failed=2,preflight_blocked=110,
      source_binding=old["historical_runtime_sha256"],new_contexts_do_not_replace_original_failures=True),
   original_supplement=dict(planned=92,passed=90,preflight_blocked=2,
      blockers=old["supplemental_blocked"]),
   new_math_contexts=dict(models=len(c["new_contexts"]),compared_values=c["compared_values"],
      max_absolute_error=c["max_absolute_error"],initial_preflight_failed=9,retried_passed=9),
   remaining_api=[dict(source=r["source"],symbol=r["symbol"],role=r["role"],reason=r["reason"])
      for r in api if r["state"]=="execution_partition_mapping_pending"],
   original_math_deficits_at_r43=reports["original_gap_plan"]["mathematical_deficits"],
   pending_joint_checks=[
      "Final requirement-by-requirement closure must include current contract linkage and explicitly scoped API partitions",
      "Bounded helper and internal-return mappings do not certify every parameter or branch",
      "Original ready queue is fully attempted; preserve all original compiler and rule-budget failures without calling them Agent failures",
      "Original direct polynomial pre-rotation failure preserved; same model has a verified post-rotation implementation. Deep sign/relua/genRelu6 remain configured-capacity blocked",
      "Manual protocol/API evidence does not by itself prove every spelling is enabled in every candidate contract",
      "Paid Agent free/directed milestone requires separate authorization after offline closure"],
   historical_regression_r50={k:reports["regression"][k] for k in
       ("tests","passed","failed","skipped","runtime_sha256","log_sha256","actual_compile","actual_ciphertext_execution")},
   current_regression={k:regression[k] for k in ("tests","passed","failed","skipped","runtime_sha256","log_sha256","actual_compile","actual_ciphertext_execution")},
   contract_linkage={k:linkage[k] for k in ("corpus","free_request_preflight","legacy_counts","all_legacy_contracts_accepted","current_source_sha256")},
   new_polynomial_candidates=dict(profile="upstream-poly-fixed-polynomials-v10",cases=21,passed=21,failed=0,skipped=0,
      contexts={k:len(v) for k,v in polycontexts.items()},compared_values=polyvalues,max_absolute_error=polymaximum,
      actual_compile=True,actual_ciphertext_execution=True,finite_return_contribution=True,agent_generated=False,
      earlier_infrastructure_failures=reports["polynomial_infrastructure"]["failed"]),
   same_model_polynomial_repair=dependencies["same_model_repairs"],
   original_blocker_recheck_at_r58=dict(denominator=112,reference_passed=sum(r["reference_status"]=="passed" for r in retry_rows),
      ready=sum(r["preflight_status"]=="ready" for r in retry_rows),
      still_preflight_blocked=sum(r["preflight_status"]!="ready" for r in retry_rows),
      encrypted_passed_model_ids=sorted({r["id"] for r in execution_records if r["state"]=="passed"}),
      compile_failed_model_ids=sorted({r["id"] for r in execution_records if r["state"]=="failed"}),
      executions=execution_records,unselected_ready_are_not_encrypted_passes=True),
   original_queue_latest={k:queue[k] for k in ("original_queue","current_primary_counts","current_supplement_blockers","execution_attempts","rechecked_passes","ready_unrun","all_planned_cases_accounted_for","all_models_passed")},
   historical_source_impact={k:impact[k] for k in ("unchanged_runtime_files","reviewed_changes","added_files","old_passes_remain_historical","current_runtime_fhe_reexecution_claimed")},
   package_surface_review=dict(python_files=exports["reviewed_package_python_files"],
      additional_trusted_runtime_interfaces=len(exports["runtime_interfaces"]),
      reference_model_definitions=len(exports["reference_models"]),
      declared_without_definition=[r["symbol"] for r in exports["hecate_declared_exports"] if r["state"]=="declared_without_definition"],
      source_classification_only=True,graph_operators_are_not_identical_to_hecate_exports=True),
   new_sumslots_execution=dict(cases=len(summation["rows"]),regimes={k:len(v) for k,v in dependency_contexts.items()},
      compared_values=dependency_values,max_absolute_error=dependency_maximum,
      actual_compile=True,actual_ciphertext_execution=True,agent_generated=False),
   dependency_acceptance={k:dependencies[k] for k in ("execution_attempts","unique_model_ids","actual_ciphertext_executions",
      "encrypted_passed","failed","skipped","negative_trace_controls","compared_values","max_absolute_error_including_failures","passed_max_absolute_error")},
   native_api_mapping=dict(partitions=len(native_contexts),historical_contexts=len(native["historical_contexts"]),
      real_frontend_rejection_contexts=len(native["rejections"]),new_encrypted_executions=0),
   frontend_contract_checks=dict(contexts=3,scope=frontend["scope"],
      actual_frontend=True,actual_compile=False,actual_ciphertext_execution=False),
   stage3=dict(registry="initial_8_operators",python="bounded_static_multifile_subset",
      small_mlp="encrypted_and_export_replay_passed_at_recorded_versions",
      rmsnorm="compiler_failed",attention_decode="compiler_failed",
      approximation_accuracy_certified=False),
   source_policy="Never rebind old passes to current runtime; exact historical report hashes retained",
   offline_stage_complete=False,agent_stage_complete=False,paid_calls=0,
   installed_dependencies=False,reconstructed_sdk=False,committed=False,pushed=False,
   runner_sha256=sha(Path(__file__)))
 result["binding"]=digest(result)
 if a.output:
  if a.output.exists():p.error("Preserve prior report")
  a.output.parent.mkdir(parents=True,exist_ok=True);dump(a.output,result)
 print(json.dumps({k:v for k,v in result.items() if k not in ("remaining_api","original_math_deficits_at_r43","evidence_sources","api_dispositions")},ensure_ascii=False))
 return 0
if __name__=="__main__":raise SystemExit(main())
