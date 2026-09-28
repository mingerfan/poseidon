"""Read-only compiler diagnosis under the original immutable w40 security profile.

Manual mathematical variants. No key generation, HEVM execution or API call.
"""
import argparse,json,time,tarfile
from pathlib import Path
from benchmark_suite import Builder
from upstream_candidate_cases import polynomial
from unified_graph_contract import prepare,validate_candidate
from upstream_candidate_helpers import PROFILE,verify_events,verify_sources
from poly_dependencies import verify
from compiler_configuration import configuration,PROFILE_SHA256
from benchmark_runner import dump
from benchmark_graph import digest,require
from hecate_python_env import ROOT,WORK,VENV
from semantic_benchmark_execution import runtime_sources
from candidate_sandbox import run
from native_execution_slots import native_slot

def cases():
    result=[]
    for mode in ['raw','pre_add','pre_multiply','pre_sum','pre_sum_multiply','post_multiply','native_boundary','original_residual']:
        two=mode in ('pre_sum','pre_sum_multiply','original_residual')
        b=Builder([(2,),(2,)] if two else [(2,)])
        x='input0';expr='x';prefix=''
        if two:x=b.node('add',[x,'input1']);expr='x+y'
        if mode=='pre_add':x=b.node('add',[x,b.const(.125)]);expr='x+0.125'
        if mode in ('pre_multiply','pre_sum_multiply','native_boundary','original_residual'):
            x=b.node('multiply',[x,b.const(.25)]);expr='('+expr+')*0.25'
        out=polynomial(b,x)
        body='HE_SiLU('+expr+')'
        if mode=='post_multiply':out=b.node('multiply',[out,b.const(.25)]);body+='*0.25'
        if mode=='native_boundary':
            prefix='@hc.func("c")\ndef f(v):\n    return HE_SiLU(v)\n';body='f('+expr+')'
        if mode=='original_residual':
            out=b.node('add',[out,'input0']);other=b.node('subtract',['input0','input1']);graph=b.finish(out,other);body='['+body+'+x,x-y]'
        else:graph=b.finish(out)
        graph['id']='silu_depth_'+mode
        source=prefix+('@hc.func("c,c,c")\ndef golden(x,y,zero_ct):\n' if two else '@hc.func("c,c")\ndef golden(x,zero_ct):\n')+'    return '+body+'\n'
        result.append((mode,graph,source))
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    require(a.output.resolve().is_relative_to(WORK/'results') and not a.output.exists(),'Preserve results')
    out=a.output;out.mkdir();sources=runtime_sources();start=time.monotonic()
    dump(out/'plan.json',dict(source_sha256=digest(sources),sources=sources,helper_sources=verify_sources(),dependency=verify(),cases=[x[0] for x in cases()],max_seconds=900,agent_calls=0,native_concurrency=1,encrypted_execution=False))
    with tarfile.open(out/'frozen-source.tar.gz','w:gz') as tar:
        for name in sources:tar.add(ROOT/name,arcname=name)
        tar.add(Path(__file__),arcname=str(Path(__file__).relative_to(ROOT)))
    rows=[]
    for name,model,source in cases():
        require(time.monotonic()-start<900 and runtime_sources()==sources,'Budget/source guard')
        folder=out/name;folder.mkdir();output=folder/'output';output.mkdir()
        request=prepare(model,PROFILE_SHA256,configuration('seal-cpu-eva-w40-v1'),helper_profile=PROFILE)
        candidate=dict(schema=1,request_id=request['request_id'],hecate_source=source)
        validate_candidate(candidate,request);dump(folder/'payload.json',dict(request=request,candidate=candidate));(folder/'candidate.py').write_text(source)
        with native_slot(WORK/'cache/agent-native-slots'):
            traced=run(folder/'payload.json',output,[str(VENV/'bin/python'),'/app/candidate_trace.py'],folder/'trace.log',helpers=True)
        require(traced==0,'Trace failed')
        trace=verify_events(json.loads((output/'upstream-calls.json').read_text()),request,source)
        command=['/hecate-opt','/out/candidate_trace.mlir','--eva','--ckks-config=/profile.json','--waterline=40','--enable-debug-printer','--mlir-disable-threading','--verify-each','--mlir-print-ir-after-failure','-o','/out/lowered.mlir']
        with native_slot(WORK/'cache/agent-native-slots'):
            compiled=run(folder/'payload.json',output,command,folder/'compile.log')
        rows.append(dict(id=name,traced=True,compile_exit=compiled,compiled=compiled==0,actual_helper_trace=trace,encrypted_execution=False,files={str(f.relative_to(folder)):__import__('hashlib').sha256(f.read_bytes()).hexdigest() for f in folder.rglob('*') if f.is_file()}))
        print(name,'compiled' if compiled==0 else 'compiler_failed',flush=True)
    require(runtime_sources()==sources,'Source changed')
    dump(out/'report.json',dict(rows=rows,seconds=time.monotonic()-start,source_sha256=digest(sources),agent_calls=0,encrypted_execution=False))
