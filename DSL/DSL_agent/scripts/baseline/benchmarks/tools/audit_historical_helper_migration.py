"""Supplement historical source origins and finite helper return influence."""
import argparse,json,copy,contextlib,shlex,sys,hashlib
from pathlib import Path
ROOT=(Path(__file__).resolve().parents[6] if Path(__file__).resolve().parents[4].name == "DSL_agent" and Path(__file__).resolve().parents[5].name == "DSL" else Path(__file__).resolve().parents[4])
sys.path[:0]=[str(ROOT/'scripts/baseline'),str(Path(__file__).resolve().parent)]
def main():
 from hecate_python_env import WORK,VENV,enter_nix
 p=argparse.ArgumentParser();p.add_argument('--inside',action='store_true');p.add_argument('--output',type=Path,required=True)
 a=p.parse_args()
 if not a.inside:
  cmd='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join([str(VENV/'bin/python'),'-B',str(Path(__file__).resolve()),'--inside','--output',str(a.output)])
  return enter_nix(cmd,seconds=300,environment='poseidon-cpu')
 from audit_historical_delivery import verify_delivery
 from component_contract import request_options
 from unified_graph_contract import prepare
 from compiler_configuration import configuration
 from upstream_helper_coverage import check_exercise
 from benchmark_graph import require,canonical,digest
 from semantic_benchmark_execution import runtime_sources
 from poseidon_backend_identity import identity
 from requalify_historical_agent_sources import verify_frozen_files,sha
 source=WORK/'results/poseidon-historical-helper-replay-r375'
 r=json.loads((source/'report.json').read_text());plan=json.loads((source/'plan.json').read_text())
 require(r['passed']==13 and r['failed']==r['blocked']==r['not_run']==0,'Completed original cohort')
 require(runtime_sources()==plan['source_hashes'] and identity()==plan['backend_identity'],'Current source/backend binding')
 require(not a.output.exists(),'Fresh audit')
 a.output.mkdir();rows=[]
 for row in r['rows']:
  checked=verify_delivery(source/row['id'])
  verify_frozen_files(Path(row['historical_evidence']),row['historical_files'])
  result=dict(id=row['id'],origin=checked,status='passed',encrypted_execution_added=False,compiler_calls=0,paid_calls=0)
  file=Path(row['evidence'])/'attempt-00/output/upstream-calls.json'
  if file.exists():
   records=json.loads(file.read_text());req=json.loads((Path(row['program'])/'request.json').read_text())
   require(records['request_id']==req['request_id'] and records['source_sha256']==row['source_sha256'],'Original actual frontend trace identity')
   names=sorted({x['callee'] for x in records['calls'] if x.get('actual_upstream') is True})
   require(bool(names),'Actual upstream call')
   options=request_options(req);name=options.pop('compiler_configuration');options['helper_exercise']=names
   witness_req=prepare(copy.deepcopy(req['model']),req['compiler_profile_sha256'],
      configuration(name,schema=req['compiler_configuration']['schema']),
      constant_policy=req['constant_origins'].get('policy'),**options)
   require(all(witness_req[k]==v for k,v in req.items() if k not in ['request_id','rules','upstream_exercise']),'Supplemental witness changed semantic request')
   text=(Path(row['program'])/'candidate.py').read_text()
   witness=check_exercise(text,witness_req)
   require(witness['source_sha256']==records['source_sha256'],'Unmodified witness source')
   for name,item in witness['witnesses'].items():
    expected={k:v for k,v in item['trace'].items() if k!='kind'}
    require(any(canonical(c)==canonical(dict(expected,actual_upstream=True)) for c in records['calls']),'Actual influencing callsite')
   result.update(helper_influence=dict(original_trace_request_id=records['request_id'],
      supplemental_requirement_request_id=witness_req['request_id'],
      original_trace_sha256=sha(file),source_sha256=row['source_sha256'],witness=witness,
      actual_original_frontend_callsite_checked=True,original_trace_rewritten=False,
      finite_return_intervention=True,all_input_proof=False,private_reference_used=False))
  rows.append(result);print('HISTORICAL AUDIT',row['id'],'passed',flush=True)
 require(runtime_sources()==plan['source_hashes'] and identity()==plan['backend_identity'],'Frozen identity drift')
 summary=dict(format='poseidon-historical-helper-source-audit-v1',rows=rows,planned=13,passed=13,failed=0,skipped=0,
  helper_tasks=sum('helper_influence' in x for x in rows),actual_compile=False,encrypted_execution_added=False,
  paid_calls=0,new_generation=False,source_hashes=plan['source_hashes'],backend_identity=plan['backend_identity'],
  original_report=str(source/'report.json'),original_report_sha256=sha(source/'report.json'),
  tool_sha256=sha(Path(__file__)))
 summary['binding']=digest(summary);(a.output/'report.json').write_bytes(canonical(summary))
 return 0
if __name__=='__main__':raise SystemExit(main())
