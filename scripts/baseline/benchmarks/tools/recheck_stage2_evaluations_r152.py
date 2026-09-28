"""Version-qualified evaluation journal; failures stay failures, no best-of selection."""
import argparse,json
from collections import Counter
from pathlib import Path
from summarize_recovery_r150 import read,sha,require,build
from benchmark_graph import digest

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--results',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 oldpath=a.results/'stage2-agent-progress-r145.json';old=read(oldpath)
 recoverypath=a.results/'stage2-recovery-aggregate-r150.json';recovery=read(recoverypath)
 require(build(a.results)==recovery,'Recovery aggregate not reproducible')
 require(old['runtime_source_sha256']=='80738a33351d8c532cd8e4b032de63cc6785d6e5cf8a349f78e52aab5fffd58e','Historical version')
 parents={str(oldpath):sha(oldpath),str(recoverypath):sha(recoverypath)}
 audits={};files=0
 for name,h in old['parents'].items():
  path=Path(name);require(sha(path)==h,'Changed historical parent: '+name)
  if name.endswith('.audit.json'):
   audit=read(path);key=audit['case_id'];require(key not in audits,'Duplicate pass proof')
   for rel,expected in audit['files'].items():
    folder=Path(audit['evidence']);f=(folder/rel).resolve()
    require(f.is_relative_to(folder.resolve()) and sha(f)==expected,'Changed successful artifact')
    files+=1
   require(audit['comparison']['passed'] and audit['comparison']['atol']==1e-5 and audit['comparison']['rtol']==1e-4,'Changed tolerance')
   audits[key]=audit
 require(len(audits)==1692,'Historical pass denominator')
 histories={x['id']:x for x in old['rows']}
 require(len(histories)==len(old['rows'])==2030,'Task uniqueness')
 additions={x['id']:x for x in recovery['rows']}
 require(len(additions)==54,'Recovery uniqueness')
 journal=[]
 for ident,row in sorted(histories.items()):
  status=row['status'];outcomes=[]
  if status in ('passed','failed'):
   ar=row['audited_result'];folder=Path(ar['evidence'])
   require(sha(folder/'report.json')==ar['report_sha256'],'Changed candidate report')
   if status=='passed':
    proof=audits[ident]
    require(proof['request_id']==row['request_id'] and proof['model_sha256']==row['model_sha256'],'Historical pass identity')
   outcomes.append(dict(status=status,request_id=row['request_id'],runtime_source_sha256=old['runtime_source_sha256'],
      evidence=ar['evidence'],report_sha256=ar['report_sha256'],version='original_r145'))
  else:require(status=='launched_unconfirmed','Unexpected original state')
  if ident in additions:
   new=additions[ident]
   require(new['model_sha256']==row['model_sha256'],'Recovery model changed')
   if new['cohort']=='interrupted_recovery':
    require(status=='launched_unconfirmed' and new['request_id']==row['request_id'],'Recovery scope/request')
   else:require(status=='failed' and new['cohort']=='explicit_v4_pilot','Pilot scope')
   outcomes.append(dict(status=new['status'],request_id=new['request_id'],
     runtime_source_sha256=new['runtime_source_sha256'],evidence=new['evidence'],
     report_sha256=new['report_sha256'],version=new['cohort']))
  require(outcomes,'No audited result for '+ident)
  journal.append(dict(id=ident,group=row['group'],model_sha256=row['model_sha256'],
   original_status=status,evaluations=outcomes,all_evaluations_passed=all(x['status']=='passed' for x in outcomes)))
 require(sum(len(x['evaluations']) for x in journal)==2035,'Outcome denominator')
 compilerpath=a.results/'stage2-math-pilot-r117-compiler.json';compiler=read(compilerpath)
 parents[str(compilerpath)]=sha(compilerpath)
 for name,h in compiler['parents'].items():require(sha(Path(name))==h,'Compiler parent changed')
 require(len(compiler['partitions'])==5 and all(x['three_contexts_verified'] and len(set(x['topologies']))>=3 for x in compiler['partitions'].values()),'Compiler contexts')
 groups={}
 for row in journal:
  g=groups.setdefault(row['group'],dict(planned=0,with_audited_outcome=0))
  g['planned']+=1;g['with_audited_outcome']+=bool(row['evaluations'])
 result=dict(format='poseidon-agent-evaluation-journal-r152',parents=parents,
  original_statuses=old['summary']['statuses'],recovery_statuses=recovery['statuses'],recovery_cohorts=recovery['cohorts'],
  planned_task_ids=2030,recorded_evaluations=2035,groups=groups,rows=journal,
  original_parent_hashes_rechecked=len(old['parents']),historical_pass_artifact_sets_rechecked=len(audits),
  historical_artifact_files_rechecked=files,compiler_partitions=compiler['partitions'],
  original_source=old['runtime_source_sha256'],recovery_source=recovery['runtime_source_sha256'],
  all_planned_task_ids_have_audited_outcomes=True,all_tasks_passed=False,
  same_source_full_evaluation=False,full_agent_semantic_success_coverage=False,
  evaluation_execution_complete=True,stage2_complete=False,
  new_paid_calls=0,new_compilations=0,new_encrypted_executions=0,
  automatic_retry_authorized=False,billing_uncertainty_resolved=False,
  limitation='Outcome completeness across disclosed versions is not a same-version success score. Original failures and interrupted-call records remain untouched; no best-of selection.',
  runner_sha256=sha(Path(__file__)))
 result['binding']=digest(result)
 with a.output.open('x') as f:json.dump(result,f,indent=2,sort_keys=True);f.write('\n')
 print(json.dumps({k:result[k] for k in ('binding','planned_task_ids','recorded_evaluations','historical_pass_artifact_sets_rechecked','historical_artifact_files_rechecked','evaluation_execution_complete')}))
if __name__=='__main__':main()
