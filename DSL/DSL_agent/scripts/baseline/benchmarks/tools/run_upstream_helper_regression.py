"""Bounded final-source BN/SILU/default-profile regressions; no API or installs."""
import argparse,json,subprocess,unittest,time
from pathlib import Path
from hecate_python_env import ROOT,WORK,VENV
from benchmark_runner import dump
from benchmark_graph import digest,require
from semantic_benchmark_execution import runtime_sources
from upstream_candidate_helpers import PROFILE,BN_PROFILE
from unified_graph_contract import prepare
from compiler_configuration import configuration,PROFILE_SHA256
from upstream_candidate_cases import cases,polynomial
from upstream_bn_candidate_cases import bn
from benchmark_suite import Builder
from audit_unified_candidate import verify_candidate

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output
    require(out.resolve().is_relative_to(WORK/'results') and not out.exists(),'Preserve results');out.mkdir()
    sources=runtime_sources();dump(out/'plan.json',dict(sources=sources,source_sha256=digest(sources),agent_calls=0,max_seconds=900,native_concurrency=1))
    names=['test_upstream_helper_coverage','test_semantic_benchmark','test_unified_native_coverage','test_upstream_candidate_bn','test_upstream_candidate_helpers','test_unified_graph','test_unified_constants','test_native_trace_contract','test_candidate_pipeline','test_native_public_loops','test_native_function_calls','test_native_array_mutation','test_native_starred','test_unified_composites','test_unified_public','test_unified_lambda_binding']
    with (out/'unit.log').open('w') as log:
        unit=unittest.TextTestRunner(stream=log,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromNames(names))
    dump(out/'unit.json',dict(total=unit.testsRun,passed=unit.testsRun-len(unit.skipped)-len(unit.failures)-len(unit.errors),failed=len(unit.failures)+len(unit.errors),skipped=len(unit.skipped),skip_reasons=[(str(t),why) for t,why in unit.skipped]))
    require(unit.wasSuccessful(),'Unit regression failed')
    records=[];start=time.monotonic()
    for profile in ['native','public-v1']:
        command=[str(VENV/'bin/python'),'-B','scripts/baseline/run_candidate.py','--inside','--case','scripts/baseline/cases/unified-two-input-two-output.json','--self-test','--unified-profile',profile]
        with (out/(profile+'.log')).open('w') as log:r=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=300)
        require(r.returncode==0,'Default candidate regression failure')
        folder=Path(next(s.split('Candidate evidence: ',1)[1] for s in (out/(profile+'.log')).read_text().splitlines() if s.startswith('Candidate evidence: ')))
        records.append(dict(id=profile,**verify_candidate(folder)));print(profile,'passed',flush=True)
    base=cases()[0];b=Builder([(1,2)]);h=polynomial(b,'input0');h=bn(b,h);model=b.finish(h);model['id']='silu_then_bn'
    selected=[('silu_v1',base['model'],base['source'],PROFILE),('silu_v2',base['model'],base['source'],BN_PROFILE),
              ('silu_then_bn',model,'@hc.func("c,c")\ndef golden(x,zero_ct):\n    return HE_BN0(HE_SiLU(x))\n',BN_PROFILE)]
    failures=[]
    for name,model,source,profile in selected:
        require(time.monotonic()-start<900 and runtime_sources()==sources,'Time/source bound')
        folder=out/name;folder.mkdir();dump(folder/'model.json',model);(folder/'candidate.py').write_text(source)
        request=prepare(model,PROFILE_SHA256,configuration('seal-cpu-eva-w40-v1'),helper_profile=profile)
        dump(folder/'responses.json',[json.dumps(dict(schema=1,request_id=request['request_id'],hecate_source=source))])
        command=[str(VENV/'bin/python'),'-B','scripts/baseline/run_candidate.py','--inside','--case',str(folder/'model.json'),'--replay',str(folder/'responses.json'),'--max-repairs','0','--unified-helpers',profile,'--compiler-configuration','seal-cpu-eva-w40-v1']
        with (folder/'run.log').open('w') as log:r=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=300)
        evidence=Path(next(s.split('Candidate evidence: ',1)[1] for s in (folder/'run.log').read_text().splitlines() if s.startswith('Candidate evidence: ')))
        if r.returncode==0:records.append(dict(id=name,**verify_candidate(evidence)))
        else:failures.append(dict(id=name,evidence=str(evidence),report=json.loads((evidence/'report.json').read_text())))
        print(name,'passed' if r.returncode==0 else 'failed',flush=True)
    require(runtime_sources()==sources,'Source changed')
    dump(out/'report.json',dict(passed=len(records),failed=len(failures),skipped=0,records=records,failures=failures,seconds=time.monotonic()-start,source_sha256=digest(sources),agent_calls=0))
    raise SystemExit(bool(failures))
