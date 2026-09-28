"""Concat acceptance and retained numerical failure audit. No execution or APIs."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from benchmark_graph import digest,require,samples
from benchmark_math import evaluate
from benchmark_torch import evaluate as torch_reference
from benchmark_runner import dump,DEFAULT,load
from semantic_benchmark_execution import runtime_sources
from audit_unified_candidate import verify_candidate,verify_files,read,sha
from upstream_concat_cases import cases
from unified_graph_contract import prepare,validate_request,validate_candidate
from compiler_configuration import verify_artifact_configuration
from candidate_contract import request_input_names,request_rotations
from cipher_abi import artifact_options
from seal_artifact_gate import inspect_artifacts
from seal_cpu_golden import PROFILE,compare
from hecate_python_env import WORK,ROOT

p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
require(not a.output.exists(),'Preserve audit');w=WORK/'results'
batch=w/'upstream-concat-r18';catalog=w/'upstream-helper-r18-catalog';control=w/'upstream-concat-r18-controls';reg=w/'upstream-concat-r18-regression'
source_sha=digest(runtime_sources());report=read(batch/'report.json');definitions={r['name']:r for r in cases()};records=[];blocked=[];refmax=0.
require(report['source_sha256']==source_sha,'Changed runtime source')
for row in report['rows']:
    if row['status']=='blocked':
        req=read(batch/row['id']/'blocked-request.json');validate_request(req)
        require(req['model']==definitions[row['id']]['model'] and 'HE_Concat0' not in req['upstream_helpers']['helpers'],'Unbound blocker')
        blocked.append(row);continue
    folder=Path(row['evidence']);rec=verify_candidate(folder);rec['case_id']=row['id'];records.append(rec)
    model=definitions[row['id']]['model']
    require(model==read(folder/'model.json') and definitions[row['id']]['source']==(folder/'attempt-00/candidate.py').read_text(),'Fixture binding')
    for inputs in samples(model,16):
        first=evaluate(model,inputs);second=torch_reference(model,inputs)
        for name in first:
            np.testing.assert_allclose(first[name],second[name],atol=1e-12,rtol=1e-12)
            refmax=max(refmax,float(np.max(np.abs(first[name]-second[name]))))
require(len(records)==15 and len(blocked)==1 and blocked[0]['id']=='concat_frozen_7','Complete concat denominator')
cat=read(catalog/'report.json');require(cat['source_sha256']==source_sha,'Catalog source')
cat_records=[verify_candidate(Path(row['evidence'])) for row in cat['rows']]
controls=read(control/'report.json');require(controls['source_sha256']==source_sha,'Control source')
control_records=[dict(id=row['id'],configuration=row['configuration'],**verify_candidate(Path(row['evidence']))) for row in controls['rows']]
def from_log(path):return Path(next(line.split('Candidate evidence: ',1)[1] for line in path.read_text().splitlines() if line.startswith('Candidate evidence: ')))
native=verify_candidate(from_log(reg/'native.log'))
failed=from_log(reg/'public-v1.log');fr=read(failed/'report.json');attempt=fr['attempts'][0];out=failed/'attempt-00/output'
verify_files(ROOT/'scripts/baseline',fr['source_hashes']);verify_files(failed,fr['frozen_hashes']);verify_files(out,attempt['artifact_hashes'])
require(fr['agent_calls']==0 and fr['status']!='passed' and attempt['compiled'] and attempt['executed'] and not attempt['numerically_correct'],'Failure status changed')
require(read(failed/'key-cleanup-outcome.json')['complete'],'Failed-case keys not cleaned')
req=read(failed/'request.json');validate_request(req);validate_candidate(dict(schema=1,request_id=req['request_id'],hecate_source=(failed/'attempt-00/candidate.py').read_text()),req)
require(attempt['trace']['frontend']=='real_Hecate' and read(out/'execution.json')==attempt['execution'] and attempt['execution']['encrypted_execution'],'Failure real execution')
model=read(failed/'model.json');refs=[];packed=[];period=req['layout']['input_slot_period']
for inputs in samples(model,4):
    first=evaluate(model,inputs);second=torch_reference(model,inputs)
    for name in first:np.testing.assert_allclose(first[name],second[name],atol=1e-12,rtol=1e-12)
    refs.append(np.concatenate([first[o['name']].reshape(-1) for o in model['outputs']]))
    packed.append(np.stack([np.pad(inputs[i['name']].reshape(-1),(0,period-inputs[i['name']].size)) for i in model['inputs']]))
with np.load(failed/'arrays.npz',allow_pickle=False) as arrays:
    require(np.array_equal(arrays['reference'],refs) and np.array_equal(arrays['inputs'],packed),'Failed-case independent arrays')
comparison=compare(np.load(out/'decrypted.npy',allow_pickle=False),np.array(refs),1e-5,1e-4)
require(comparison==attempt['comparison'] and not comparison['passed'],'Frozen failure disappeared')
gate=inspect_artifacts((out/'lowered._hecate_golden.hevm').read_bytes(),(out/'_hecate_golden.cst').read_bytes(),rotation_steps=request_rotations(req),expected_inputs=len(request_input_names(req)),**artifact_options(req['layout']))
verify_artifact_configuration(req,gate,sha(PROFILE));require(gate==attempt['artifact_gate'],'Failed-case gate')
old=Path(next(r for r in read(w/'upstream-helper-r17-regression/report.json')['records'] if r['id']=='public-v1')['evidence'])
identical={name:sha(old/name)==sha(failed/name) for name in ['request.json','attempt-00/candidate.py','attempt-00/output/lowered._hecate_golden.hevm','attempt-00/output/_hecate_golden.cst']}
require(all(identical.values()),'Prior artifact identity differs')
old_requests=0
for path in [w/'upstream-candidate-r15-final/report.json',w/'upstream-bn-candidate-r16/report.json',w/'upstream-helper-directed-r17-final/report.json']:
    for row in read(path)['rows']:
        r=read(Path(row['evidence'])/'request.json');new=prepare(r['model'],r['compiler_profile_sha256'],r.get('compiler_configuration'),helper_profile=r['upstream_helpers']['profile'],helper_exercise=r.get('upstream_exercise',{}).get('required_helpers'))
        require(new==r,'Old request hash changed');old_requests+=1
old_cov=read(ROOT/'scripts/baseline/benchmarks/semantic-v1-helper-r17/coverage.json');new_cov=read(DEFAULT/'coverage.json');load(DEFAULT)
require(old_cov['helper_directed_tasks']==new_cov['helper_directed_tasks'][:8],'Old helper tasks changed')
summary=dict(schema=1,source_sha256=source_sha,concat_passed=15,concat_blocked=1,concat_failed=0,concat_records=records,blocked=blocked,catalog_passed=len(cat_records),catalog_records=cat_records,
    initial_regression=dict(passed=1,failed=1,not_run=3,native=native),control_records=control_records,
    retained_numerical_failure=dict(evidence=str(failed),comparison=comparison,prior_artifacts_identical=identical,files={str(p.relative_to(failed)):sha(p) for p in failed.rglob('*') if p.is_file()},not_overridden_by_later_passes=True),
    reference_models=15,reference_probes=240,max_reference_difference=refmax,old_requests_unchanged=old_requests,
    old_directed_tasks_unchanged=8,agent_calls=0,all_helpers_supported=False,default_precision_stability_proven=False,
    checker_sha256=sha(Path(__file__)),batch_sha256=sha(batch/'report.json'),catalog_sha256=sha(catalog/'report.json'))
dump(a.output,summary);print(json.dumps({k:v for k,v in summary.items() if k not in ('concat_records','blocked','catalog_records','initial_regression','control_records','retained_numerical_failure')},indent=2))
