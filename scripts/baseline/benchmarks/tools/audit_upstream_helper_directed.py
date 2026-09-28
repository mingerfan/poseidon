"""Independent final helper witness/artifact audit; no provider or compiler calls."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from benchmark_graph import digest,require,samples
from benchmark_math import evaluate
from benchmark_torch import evaluate as torch_reference
from benchmark_runner import dump,load,DEFAULT
from semantic_benchmark_execution import runtime_sources
from upstream_helper_directed_cases import cases,tasks
from audit_unified_candidate import read,verify_candidate,sha
from unified_graph_contract import prepare

p=argparse.ArgumentParser();p.add_argument('--batch',type=Path,required=True);p.add_argument('--regression',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--old-report',type=Path,action='append',default=[]);a=p.parse_args()
require(not a.output.exists(),'Preserve audit')
plan=read(a.batch/'plan.json');report=read(a.batch/'report.json');reg=read(a.regression/'report.json')
require(runtime_sources()==plan['sources'] and digest(runtime_sources())==report['source_sha256']==reg['source_sha256'],'Source drift')
load(DEFAULT);require(plan['tasks']==tasks(),'Frozen directed task drift')
fixtures={r['name']:r for r in cases()};records=[];reference_max=0.;approximations=[]
for row in report['rows']:
    require(row['status']=='passed','Unpassed task must remain in denominator')
    definition=fixtures[row['id']];folder=Path(row['evidence']);model=read(folder/'model.json')
    require(model==definition['model'] and (folder/'attempt-00/candidate.py').read_text()==definition['source'],'Fixture binding')
    rec=verify_candidate(folder);rec['case_id']=row['id'];records.append(rec)
    require(set(rec['helper_coverage']['witnesses'])==set(definition['required_helpers']),'Missing helper partition')
    for probe in samples(model,16):
        math=evaluate(model,probe);torch=torch_reference(model,probe)
        for name in math:
            np.testing.assert_allclose(math[name],torch[name],atol=1e-12,rtol=1e-12)
            reference_max=max(reference_max,float(np.max(np.abs(math[name]-torch[name]))))
    if row['id'].startswith('silu'):
        probes=[]
        for inputs in samples(model,4):
            x=inputs['input0'];ideal=x/(1+np.exp(-x))
            if row['id']=='silu_nested_native':ideal=ideal*.5+.125
            ideals=[ideal]
            if row['id']=='silu_shared_rotation':ideals.append(np.roll(ideal,-1))
            poly=evaluate(model,inputs);errors=[]
            for o,exact in zip(model['outputs'],ideals):
                predicted=poly[o['name']];error=np.abs(predicted-exact)
                errors.append(dict(output=o['name'],original_function_reference=exact.tolist(),polynomial_reference=predicted.tolist(),approximation_absolute_error=error.tolist()))
            probes.append(dict(input=x.tolist(),outputs=errors))
        approximations.append(dict(id=row['id'],rows=probes,not_encrypted_execution_error=True))
regression=[dict(id=r['id'],**verify_candidate(Path(r['evidence']))) for r in reg['records']]
unchanged=0
for old in a.old_report:
    for row in read(old)['rows']:
        request=read(Path(row['evidence'])/'request.json')
        actual=prepare(request['model'],request['compiler_profile_sha256'],request.get('compiler_configuration'),helper_profile=request['upstream_helpers']['profile'])
        require(actual==request,'Changed old free helper request');unchanged+=1
# Check the actual archived batch runner, since nested tools are separately frozen.
import tarfile
with tarfile.open(a.batch/'frozen-source.tar.gz') as tar:
    runner='scripts/baseline/benchmarks/tools/run_upstream_helper_directed.py'
    archived=tar.extractfile(runner).read()
    archived_sha=hashlib.sha256(archived).hexdigest()
output=dict(schema=1,source_sha256=report['source_sha256'],directed_passed=len(records),failed=0,skipped=0,
    records=records,regression=regression,reference_models=len(records),reference_probes=16*len(records),
    max_reference_difference=reference_max,approximations=approximations,old_free_requests_unchanged=unchanged,
    directed_evidence_scope='actual_frontend_and_finite_return_intervention_plus_independent_encrypted_numerics',
    all_input_proof=False,agent_calls=0,batch_report_sha256=sha(a.batch/'report.json'),regression_report_sha256=sha(a.regression/'report.json'),
    archived_batch_runner_sha256=archived_sha,current_batch_runner_sha256=sha(Path(runner)),checker_sha256=sha(Path(__file__)))
dump(a.output,output);print(json.dumps({k:v for k,v in output.items() if k not in ('records','regression','approximations')},indent=2))
