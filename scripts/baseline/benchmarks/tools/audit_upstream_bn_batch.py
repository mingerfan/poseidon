"""Independent BN binding/CST/numeric evidence and separate SiLU approximation audit."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from audit_unified_candidate import verify_candidate,sha,read
from benchmark_graph import require,samples,digest
from benchmark_math import evaluate
from benchmark_torch import evaluate as torch_reference
from semantic_benchmark_execution import runtime_sources
from benchmark_runner import dump
from upstream_bn_candidate_cases import cases
from unified_graph_contract import prepare

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--batch',type=Path,required=True);p.add_argument('--regression',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--legacy-report',type=Path);a=p.parse_args()
    require(not a.output.exists(),'Preserve audit')
    plan=read(a.batch/'plan.json');report=read(a.batch/'report.json');reg=read(a.regression/'report.json')
    require(runtime_sources()==plan['sources'] and digest(runtime_sources())==report['source_sha256']==reg['source_sha256'],'Source identity')
    definitions={r['name']:r for r in cases()};records=[];reference_max=0.;bn_calls=0;conversions=0
    for row in report['rows']:
        require(row['status']=='passed','Keep failures separate; this auditor requires passed batch')
        folder=Path(row['evidence']);definition=definitions[row['id']];model=read(folder/'model.json')
        require(model==definition['model'] and (folder/'attempt-00/candidate.py').read_text()==definition['source'],'Manual fixture binding')
        record=verify_candidate(folder);record['case_id']=row['id'];records.append(record)
        for probe in samples(model,16):
            first=evaluate(model,probe);second=torch_reference(model,probe)
            for name in first:
                np.testing.assert_allclose(first[name],second[name],atol=1e-12,rtol=1e-12)
                reference_max=max(reference_max,float(np.max(np.abs(first[name]-second[name]))))
        trace=read(folder/'attempt-00/output/upstream-calls.json');bn_calls+=len(trace['bound_calls'])
        for c in read(folder/'attempt-00/output/constant-layout.json').values():conversions+=len(c['changes'])
    regression=[];approx=[]
    for record in reg['records']:
        folder=Path(record['evidence']);replay=verify_candidate(folder);regression.append(dict(id=record['id'],**replay))
        if record['id'].startswith('silu'):
            model=read(folder/'model.json');rows=[]
            for probe in samples(model,4):
                x=next(iter(probe.values()));ideal=x/(1.+np.exp(-x))
                if record['id']=='silu_then_bn':
                    node=model['nodes'][-1];require(node['op']=='batch_norm' and len(model['nodes'])==4,'Approximation fixture identity')
                    mean,var,gamma,beta=[np.asarray(model['constants'][n]) for n in node['inputs'][1:]]
                    shape=(1,len(mean))+ (1,)*(x.ndim-2)
                    ideal=(ideal-mean.reshape(shape))*gamma.reshape(shape)/np.sqrt(var.reshape(shape)+node['attrs']['eps'])+beta.reshape(shape)
                poly=next(iter(evaluate(model,probe).values()));err=np.abs(poly-ideal)
                rows.append(dict(input=x.tolist(),polynomial_reference=poly.tolist(),original_function_reference=ideal.tolist(),approximation_absolute_error=err.tolist()))
            values=[v for row in rows for v in np.asarray(row['approximation_absolute_error']).reshape(-1)]
            approx.append(dict(id=record['id'],rows=rows,maximum=float(max(values)),mae=float(np.mean(values)),not_execution_error=True))
    # Old identities are checked as requests only, never counted as fresh execution.
    unchanged=0
    if a.legacy_report is not None:
        for r in read(a.legacy_report)['rows']:
            request=read(Path(r['evidence'])/'request.json')
            require(prepare(request['model'],request['compiler_profile_sha256'],request.get('compiler_configuration'),helper_profile=request['upstream_helpers']['profile'])==request,'Old SiLU request changed')
            unchanged+=1
    dump(a.output,dict(schema=1,source_sha256=report['source_sha256'],bn_records=records,regression_records=regression,approximations=approx,
        bn_passed=len(records),regression_passed=len(regression),failed=0,skipped=0,bn_actual_calls=bn_calls,lossless_full_vector_conversions=conversions,
        reference_models=len(records),reference_probes=16*len(records),max_reference_difference=reference_max,
        old_v1_requests_unchanged=unchanged,agent_calls=0,directed_contribution_proven=False,
        checker_sha256=sha(Path(__file__)),batch_report_sha256=sha(a.batch/'report.json'),regression_report_sha256=sha(a.regression/'report.json')))
    print(json.dumps(dict(bn_passed=len(records),regression_passed=len(regression),bn_actual_calls=bn_calls,lossless_conversions=conversions,old_v1_requests_unchanged=unchanged,reference_max=reference_max,agent_calls=0),indent=2))
