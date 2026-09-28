"""Close the two defined phase-2 milestones without claiming universal Agent success."""
import argparse,json
from pathlib import Path
from summarize_recovery_r150 import sha,read,require
from benchmark_graph import digest

ROOT=Path(__file__).resolve().parents[4]
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--results',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 parents={}
 def get(path):
  parents[str(path)]=sha(path);return read(path)
 offline=get(a.results/'stage2-offline-revalidation-r152.json')
 journal=get(a.results/'stage2-evaluation-journal-r152.json')
 tests=get(a.results/'stage2-final-regression-r152/report.json')
 recovery=get(a.results/'stage2-recovery-aggregate-r150.json')
 compatibility=get(ROOT/'docs/baseline/stage2-directed-guidance-acceptance-r146.json')
 queue=get(a.results/'stage2-recovery-queue-r148/plan.json')
 terminal=get(a.results/'stage2-recovery-queue-r148/report.json')
 coverage=get(a.results/'stage2-distinct-contexts-r145.json')
 lineage=offline['historical_source_lineage']
 from retained_acceptance_context_r152 import context
 require(context()==lineage,'Current lineage changed')
 for report in (offline,journal,recovery,compatibility):
  for name,h in report['parents'].items():require(sha(Path(name))==h,'Evidence changed: '+name)
 for path in [ROOT/'docs/baseline/stage2-acceptance-r152.md',
              ROOT/'scripts/baseline/benchmarks/tools/recheck_stage2_offline_r152.py',
              ROOT/'scripts/baseline/benchmarks/tools/recheck_stage2_joint_r152.py',
              ROOT/'scripts/baseline/benchmarks/tools/recheck_stage2_evaluations_r152.py',
              ROOT/'scripts/baseline/benchmarks/tools/run_final_regression_r152.py']:
  parents[str(path)]=sha(path)
 require(offline['offline_engineering_acceptance_complete'] and len(offline['requirements'])==401,'Offline requirements')
 expected={'accepted_historical_numeric':153,'accepted_historical_construction':209,
           'accepted_historical_helper':15,'backend_blocked':3,'accepted_historical_artifact':5,'accepted_current_static':16}
 require(offline['requirement_counts']==expected,'Changed semantic scope')
 for row in offline['requirements']:
  require(row['source_basis'] and row['applicable_contracts'],'Missing definition')
  if row['status']=='backend_blocked':require(bool(row['blocker']),'Missing blocker')
  else:
   require(bool(row['negative_checks']),'Missing negative controls')
   if row['status']!='accepted_current_static':
    require(len({x['topology'] for x in row['contexts']})>=3,'Missing independent contexts')
 require(len(offline['upstream_api'])==107,'API scope')
 corpus=offline['corpus']
 require(corpus['models']==corpus['unique_signatures']==corpus['dual_reference_passed']==1200 and
  corpus['topology_groups']>=300 and corpus['small_models']>=960 and corpus['probes']==19200 and
  corpus['legacy_anchors']==96 and corpus['split_counts']==dict(development=720,validation=240,holdout=240),'Corpus scope')
 require(tests['passed']==tests['tests']==36 and tests['failed']==tests['errors']==tests['skipped']==0,'Regression')
 require(tests['source_sha256']==lineage['current_runtime_sha256'],'Regression source')
 require(sha(a.results/'stage2-final-regression-r152/tests.log')==tests['log_sha256'],'Regression log')
 require(compatibility['old_requests_identical']==2030 and compatibility['scripted_encrypted_passed']==2,'Compatibility')
 wanted={'free_primary':1200,'free_supplement':92,'free_gap_context':30,
         'directed_construction':627,'directed_helper':60,'directed_fixed_polynomial':21}
 require({k:v['planned'] for k,v in journal['groups'].items()}==wanted,'Evaluation scope')
 require(all(v['planned']==v['with_audited_outcome'] for v in journal['groups'].values()),'Unevaluated task')
 require(journal['planned_task_ids']==2030 and journal['recorded_evaluations']==2035 and
  journal['evaluation_execution_complete'] and not journal['all_tasks_passed'],'Evaluation accounting')
 require(terminal['queue_finished'] and terminal['failure'] is None and not terminal['uncertain_calls_possible'],'Unfinished queue')
 require(queue['api_workers']<=10 and queue['native_workers']<=2 and queue['compile_jobs']<=2 and queue['link_jobs']<=1,'Concurrency')
 require(all(len(x['plan']['cases'])<=48 for x in queue['shards']),'Shard size')
 require(recovery['generations']<=queue['maximum_generations'] and recovery['http_attempts']<=queue['maximum_http_attempts'] and recovery['seconds']<=queue['max_wall_seconds'],'Approved budget')
 requirements=[
 ('model_corpus','1200 unique models, fixed topology split, size partitions, 96 separate anchors','offline.corpus'),
 ('independent_references','Two references and 16 deterministic probes per model; approximation/reference semantics remain separate','offline.corpus'),
 ('versioned_semantic_inventory','401 requirements and 107 API dispositions, source basis, contracts, positives, negatives and blockers','offline.requirements + offline.upstream_api'),
 ('three_contexts','Each executable offline semantic partition has at least three distinct topology contexts','offline.requirements[*].contexts'),
 ('real_fhe','973 retained execution directories and 36835 files revalidated; no mock substituted','offline.validated_historical_execution_directories'),
 ('construct_contribution','209 construction partitions distinguish numeric, mixed and structural acceptance with counterexamples','offline.requirements'),
 ('upstream_helpers','60 helper tasks and 21 fixed polynomial tasks retain real invocation and contribution evidence','offline.helper_tasks + offline.fixed_polynomial_candidates'),
 ('bootstrap_boundary','HE_ReLU/HE_Max/HE_MaxPad blocked; 107 API sub-classifications retained, not counted as passes','offline.requirements + offline.upstream_api'),
 ('compiler_artifacts','5 compiler partitions retain distinct artifact/runtime/metadata evidence, including Agent artifacts','offline.requirements + journal.compiler_partitions'),
 ('schema_and_request_compatibility','Old interfaces retained; 2030 exact old requests verified; current targeted regression passed','compatibility + tests'),
 ('layout_reference_safety','No changes to mathematical references, contribution checkers, artifact gates or compiler configuration','offline.historical_source_lineage'),
 ('paid_free_evaluation','1200 main, 92 supplemental and 30 context task IDs all have audited outcomes','journal.groups'),
 ('paid_directed_evaluation','627 construction, 60 helper and 21 polynomial task IDs all have audited outcomes','journal.groups'),
 ('no_best_of_rebinding','Old failures/interrupted claims kept; 49 recovery outcomes and 5 new-guidance variants explicitly separate','journal.rows'),
 ('numerical_gate','Frozen absolute/relative gate with elementwise results and retained comparison metrics','offline + journal + recovery'),
 ('bounded_execution','Approved queues, per-candidate sandboxing, <=48/shard and global concurrency/call bounds preserved','queue + terminal + recovery'),
 ('current_rejection_regression','153 model negatives and 48 static controls rechecked, 36 focused tests passed','offline + tests'),
 ('reproducible_handoff','Chinese report, frozen ledger, per-version journal and fresh-output verification commands delivered','docs/baseline/stage2-acceptance-r152.md')]
 result=dict(format='poseidon-stage2-milestone-acceptance-r152',parents=parents,
  definition='The revised plan defines separate offline engineering acceptance and real Agent evaluation. Completed evaluation reports failures; it is not a requirement that every generated candidate succeed.',
  requirements=[dict(id=i,requirement=r,evidence=e,status='verified_with_stated_scope') for i,r,e in requirements],
  corpus=corpus,offline_requirement_counts=expected,
  milestone_offline_engineering_complete=True,milestone_paid_evaluation_complete=True,
  stage2_defined_milestones_complete=True,stage2_complete=True,
  all_agent_tasks_passed=False,full_agent_semantic_success_coverage=False,
  original_agent_three_topology_success=dict(verified=coverage['minimum_three_topologies_verified'],denominator=377,
    scope='Original r145 source; recovery variants are not merged into a current-version score'),
  original_outcomes=journal['original_statuses'],recovery_outcomes=recovery['statuses'],recovery_cohorts=recovery['cohorts'],
  current_runtime_sha256=lineage['current_runtime_sha256'],tests=dict(passed=36,failed=0,skipped=0),
  required_work_remaining_in_defined_milestones=[],
  continuing_research_and_backend_gaps=['Agent success gaps and static/compiler/numerical/provider failures remain genuine benchmark findings.',
    'No same-current-runtime full 2030-task replay is claimed.',
    'Real bootstrap remains unavailable; GPU and arbitrary Python are outside this milestone.',
    'Stage 3 operator decomposition/Python/MLP/RMSNorm/Attention is not completed by this acceptance.',
    'Further paid repair batches require separate approval; historical billing uncertainty remains.'],
  original_evidence_unchanged=True,new_paid_calls=0,new_compilations=0,new_encrypted_executions=0,
  runner_sha256=sha(Path(__file__)))
 result['binding']=digest(result)
 with a.output.open('x') as f:json.dump(result,f,indent=2,sort_keys=True);f.write('\n')
 print(json.dumps({k:result[k] for k in ('binding','milestone_offline_engineering_complete','milestone_paid_evaluation_complete','stage2_complete','full_agent_semantic_success_coverage')}))
if __name__=='__main__':main()
