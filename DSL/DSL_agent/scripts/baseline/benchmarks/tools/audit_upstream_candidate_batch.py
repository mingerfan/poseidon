"""Independent helper candidate audit: failures and approximation stay separate."""
import argparse,json,math
from pathlib import Path
import numpy as np
from audit_unified_candidate import read,sha,verify_files,verify_candidate
from benchmark_graph import require,samples,digest,signature
from benchmark_math import evaluate
from benchmark_torch import evaluate as reference
from benchmark_runner import ROOT,DEFAULT,load,dump
from semantic_benchmark_execution import runtime_sources
from unified_graph_contract import validate_request,validate_candidate
from upstream_candidate_helpers import verify_sources,verify_events
from poly_dependencies import verify
from upstream_candidate_cases import cases


def audit(batch):
    plan=read(batch/'plan.json');report=read(batch/'report.json')
    require(runtime_sources()==plan['sources'] and digest(runtime_sources())==report['source_sha256'],'Batch source binding')
    require(verify_sources()==plan['helper_sources'] and verify()==plan['dependency'],'Helper dependency binding')
    require(len(report['rows'])==report['planned'],'Incomplete batch')
    definitions={r['name']:r for r in cases()};rows=[];approximations=[]
    frozen,_=load(DEFAULT)
    signatures={signature(r['model']):r['model']['id'] for r in frozen}
    for row in report['rows']:
        case=definitions[row['id']];folder=Path(row['evidence']);r=read(folder/'report.json')
        request=read(folder/'request.json');validate_request(request)
        require(read(folder/'model.json')==request['model']==case['model'],'Authoritative model binding')
        require(r['status']==row['status'] and r['agent_calls']==0 and not r['llm_generation_validated'],'Manual status binding')
        verify_files(ROOT/'scripts/baseline',r['source_hashes']);verify_files(folder,r['frozen_hashes'])
        require(read(folder/'key-cleanup-outcome.json')['complete'],'Key cleanup')
        source=(folder/'attempt-00/candidate.py').read_text();require(source==case['source'],'Manual candidate binding')
        checked=validate_candidate(dict(schema=1,request_id=request['request_id'],hecate_source=source),request)
        events=read(folder/'attempt-00/output/upstream-calls.json');trace=verify_events(events,request,source)
        require(trace['actual_upstream_calls']==case['expected_calls'],'Invocation count')
        reference_max=0.;base=[]
        for i,inputs in enumerate(samples(case['model'],16)):
            x=evaluate(case['model'],inputs);y=reference(case['model'],inputs)
            for name in x:
                np.testing.assert_allclose(x[name],y[name],atol=1e-12,rtol=1e-12)
                reference_max=max(reference_max,float(np.max(np.abs(x[name]-y[name]))))
            if i<4 and case.get('ideal')=='silu':
                z=next(iter(inputs.values()));ideal=z/(1.+np.exp(-z));poly=next(iter(x.values()))
                base.append(dict(input=z.tolist(),polynomial_reference=poly.tolist(),original_silu_reference=ideal.tolist(),
                                 approximation_absolute_error=np.abs(poly-ideal).tolist()))
        if base:
            errors=[float(v) for q in base for v in np.asarray(q['approximation_absolute_error']).reshape(-1)]
            approximations.append(dict(id=row['id'],probes=base,mae=sum(errors)/len(errors),maximum=max(errors),
                                       is_encrypted_execution_error=False))
        item=dict(id=row['id'],status=row['status'],reference_probes=16,reference_max=reference_max,
                  upstream_trace=trace,frozen_corpus_model=signatures.get(signature(case['model'])),
                  contribution_negative=case.get('contribution_negative',False),evidence=str(folder))
        if row['status']=='passed':item['replay']=verify_candidate(folder)
        else:
            require(len(r['attempts'])==1 and r['attempts'][0]['failure_layer']=='compiler' and
                    not r['attempts'][0]['compiled'] and not r['attempts'][0]['executed'], 'Unexpected failure layer')
            item.update(failure_layer='compiler',diagnostic=r['attempts'][0]['diagnostic'],
                        compiler_log=(folder/'attempt-00/compile.log').read_text(),
                        files={str(p.relative_to(folder)):sha(p) for p in folder.rglob('*') if p.is_file()})
        rows.append(item)
    return dict(schema=1,source_sha256=plan['source_sha256'],rows=rows,approximations=approximations,
                approximation_is_separate_from_execution_error=True,agent_calls=0,
                passed=sum(r['status']=='passed' for r in rows),failed=sum(r['status']!='passed' for r in rows),
                skipped=0,reference_passed=len(rows),reference_probes=16*len(rows),
                all_helper_coverage=False,directed_contribution_proven=False,
                checker_sha256=sha(Path(__file__)),batch_report_sha256=sha(batch/'report.json'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--batch',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();require(not a.output.exists(),'Preserve audit');result=audit(a.batch);dump(a.output,result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows','approximations')},indent=2))
