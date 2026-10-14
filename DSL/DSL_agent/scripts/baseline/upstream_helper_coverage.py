"""Finite public helper-return interventions, separate from math/FHE references.

No candidate Python execution and no expected test answers. A witness needs a
reachable individual return AND simultaneous same-callee influence, plus a real
frontend callsite. This is finite evidence, not equivalence for all inputs.
"""
import ast
import hashlib
import json
import math
from pathlib import Path
from benchmark_graph import canonical,digest,require
from decorated_functions import _analyze
from packed_native_exercises import Cell
from unified_native_coverage import run_probe

RULES='''Directed upstream helper exercise: every callee in
upstream_exercise.required_helpers must actually be called and contribute to a
named output. Unused definitions, discarded results and cancelling calls do not
satisfy the exercise. Requirements do not change the model, layout or reference.
Acceptance combines bounded public return interventions, real frontend records,
and the ordinary encrypted numerical checks; it is not an all-input proof.
'''

def exercise_spec(names,capability):
    require(type(names) is list and 1<=len(names)<=5 and
            all(type(n) is str and n in capability['helpers'] for n in names) and
            len(set(names))==len(names),'Unknown or repeated directed helper')
    return dict(schema=1,contract='upstream-return-influence-v1',required_helpers=sorted(names),
                probes=3,individual_return=True,simultaneous_callee=True,
                actual_frontend_required=True,encrypted_numerics_required=True)


def helper_evaluator(request):
    """Full-period probe semantics; never generate reference arrays with this."""
    specs=request['upstream_helpers']['helpers'];period=request['layout']['input_slot_period']
    coeff=None
    def evaluate(name,args):
        nonlocal coeff
        require(len(args)==len(specs[name]['parameters']) and all(type(x) is Cell and x.kind=='c' for x in args),'Helper probe type')
        x=args[0].values
        if request["upstream_helpers"]["profile"]=="upstream-poly-fixed-polynomials-v10":
            from upstream_adapters.fixed_polynomial import probe
            y=probe(name,x)
        elif name=='HE_SiLU':
            if coeff is None:
                from upstream_candidate_helpers import LOCK
                relative='third_party/dacapo/python/poly/poly/data/coeffStr.txt'
                lock=json.loads(LOCK.read_text())
                path=Path('/upstream-poly/poly/data/coeffStr.txt')
                if not path.is_file():
                    from workspace_paths import ROOT
                    path=ROOT/relative
                raw=path.read_bytes()
                require(hashlib.sha256(raw).hexdigest()==lock['sources'][relative],'Changed influence polynomial source')
                coeff=[float(v) if i%2 else 0. for i,v in enumerate(raw.decode().splitlines()) if v.strip()]
                require(len(coeff)==96 and all(math.isfinite(c) for c in coeff),'Influence coefficient budget')
            def polynomial(v):
                # Independent recurrence for the fixed odd-leaf Chebyshev polynomial.
                a,b=1.,v;total=coeff[0]+(coeff[1]*v if len(coeff)>1 else 0.)
                for c in coeff[2:]:a,b=b,2*v*b-a;total+=c*b
                return v*(total+.5)
            y=tuple(polynomial(v) for v in x)
        elif specs[name].get("binding",{}).get("adapter")=="chunked-virtual-prefix-v1":
            from upstream_adapters.chunk_virtual_node import probe
            y=probe(specs[name]["binding"],[a.values for a in args[:-1]],args[-1].values)
        elif specs[name].get('binding',{}).get('adapter')=='virtual-prefix-v1':
            from upstream_adapters.virtual_node import probe
            y=probe(specs[name]['binding'],x,args[1].values)
        elif specs[name].get('binding',{}).get('adapter')=='mapped-downsample-v1':
            from upstream_adapters.downsample_node import probe
            y=probe(specs[name]['binding'],x)
        elif specs[name].get('binding',{}).get('adapter')=='fused-conv-bn-v1':
            from upstream_adapters.fused_conv_bn import probe
            y=probe(specs[name]['binding'],x)
        elif specs[name].get('binding',{}).get('adapter')=='windowed-spatial-v1':
            from upstream_adapters.spatial_mapped import probe
            y=probe(specs[name]['binding'],x)
        elif name.startswith(('HE_Conv','HE_Avg','HE_Pool')):
            binding=specs[name]['binding'];geometry=binding['geometry']
            ci,hi,wi=geometry['ci'],geometry['hi'],geometry['wi'];kind=binding['kind']
            out=[];params=binding['parameters']
            if kind=='global_pool':
                for c in range(ci):out.append(sum(x[c*hi*wi:(c+1)*hi*wi])/(hi*wi))
            else:
                co,ho,wo=geometry['co'],geometry['ho'],geometry['wo']
                fh,fw,stride=geometry['fh'],geometry['fw'],geometry['s']
                ph,pw=(fh//2,fw//2) if kind=='conv' or fw!=2 else (0,0)
                for c in range(co):
                    for h in range(ho):
                        for w in range(wo):
                            total=params['bias'][c] if kind=='conv' else 0.
                            for j in (range(ci) if kind=='conv' else [c]):
                                for u in range(fh):
                                    for v in range(fw):
                                        yy,xx=h*stride-ph+u,w*stride-pw+v
                                        if 0<=yy<hi and 0<=xx<wi:
                                            value=x[(j*hi+yy)*wi+xx]
                                            total+=value*params['weight'][c][j][u][v] if kind=='conv' else value/(fh*fw)
                            out.append(total)
            y=tuple(out+[0.]*(period-len(out)))
        elif name.startswith('HE_Concat'):
            binding=specs[name]['binding'];closure=binding['closure_period'];count=binding['logical_input_elements']
            other=args[1].values
            y=tuple((x[i] if i%closure<count else other[(i-count)%period] if i%closure<2*count else 0.) for i in range(period))
        else:
            binding=specs[name]['binding'];view=binding['mathematical_view'];node=view['nodes'][0]
            mean,var,gamma,beta=[view['constants'][ref] for ref in node['inputs'][1:]]
            spatial=math.prod(binding['input_shape'][2:]);closure=binding['closure_period']
            gains=[];biases=[]
            for m,v,g,b in zip(mean,var,gamma,beta):
                gain=g/math.sqrt(v+node['attrs']['eps'])
                gains.extend([gain]*spatial);biases.extend([b-m*gain]*spatial)
            # ParBNConst pads channels/spatial prefix to closure period, then repeats.
            gains.extend([0.]*(closure-len(gains)));biases.extend([0.]*(closure-len(biases)))
            require(period%closure==0 and len(gains)==closure,'Influence BN period')
            y=tuple(v*gains[i%closure]+biases[i%closure] for i,v in enumerate(x))
        require(len(y)==period and all(math.isfinite(v) for v in y),'Nonfinite helper influence')
        return Cell('c',y)
    return evaluate


def check_exercise(source,request):
    from unified_graph_contract import validate_request
    from candidate_contract import request_input_names
    validate_request(request)
    nodes,plan=_analyze(source,request['public_constants'],request['layout']['output_ciphertexts'],
        request_input_names(request),array_mutation=True,slot_period=request['layout']['input_slot_period'],
        upstream_helpers=True,upstream_request=request)
    loops={fn.name:[(n.lineno,n.end_lineno) for n in ast.walk(fn) if type(n) is ast.For]
           for fn in ast.parse(source).body}
    evaluator=helper_evaluator(request);budget=[0]
    def run(k,intervention=None,group=None):
        return run_probe(nodes,plan,request,k,budget,loops,intervention,
                         helper_evaluator=evaluator,helper_group=group)
    baseline=[run(k) for k in range(3)];events=baseline[0][1]
    require(all(es==events for _,es in baseline),'Helper probe schedule changed')
    # Physical ciphertext indices no longer equal named output indices when
    # one logical tensor spans multiple chunks. The layout was validated above.
    from unified_chunk_layout import ABI as CHUNK_ABI
    if request["layout"]["execution_abi"]==CHUNK_ABI:
        output_names={c["ciphertext"]:o["name"] for o in request["layout"]["outputs"] for c in o["chunks"]}
    else:output_names={i:o["name"] for i,o in enumerate(request["layout"]["outputs"])}
    def delta(changed):
        by_output={o['name']:0. for o in request['layout']['outputs']}
        for new,(old,_) in zip(changed,baseline):
            for i,(a,b) in enumerate(zip(new,old)):
                output=output_names[request['layout']['output_selectors'][i][0]]
                by_output[output]=max(by_output[output],abs(a-b))
        return by_output
    witnesses={}
    for name in request['upstream_exercise']['required_helpers']:
        matches=[e for e in events if e['trace']['kind']=='upstream' and e['trace']['callee']==name]
        require(matches,'No reachable upstream helper: '+name)
        grouped=delta([run(k,group=name)[0] for k in range(3)])
        require(max(grouped.values())>1e-9,'No simultaneous helper contribution: '+name)
        for event in matches:
            individual=delta([run(k,(event['id'],0))[0] for k in range(3)])
            outputs=[n for n in individual if individual[n]>1e-9 and grouped[n]>1e-9]
            if outputs:
                witnesses[name]=dict(trace=event['trace'],event_id=event['id'],outputs=outputs,
                    individual_max_deltas=individual,simultaneous_max_deltas=grouped,
                    reachable_invocations=len(matches))
                break
        require(name in witnesses,'No individual helper return contribution: '+name)
    return dict(schema=1,contract='upstream-return-influence-v1',request_id=request['request_id'],
        source_sha256=hashlib.sha256(source.encode()).hexdigest(),exercise_sha256=digest(request['upstream_exercise']),
        witnesses=witnesses,probe_count=3,work=budget[0],plaintext_reference_used=False,
        candidate_python_executed=False,real_frontend_checked=False,all_input_proof=False)


def verify_trace(checked,records):
    require(checked['request_id']==records['request_id'] and checked['source_sha256']==records['source_sha256'],
            'Helper witness trace identity')
    for name,witness in checked['witnesses'].items():
        expected={k:v for k,v in witness['trace'].items() if k!='kind'}
        require(any(canonical(c)==canonical(dict(expected,actual_upstream=True)) for c in records['calls']),
                'Missing contributing actual helper call: '+name)
    return dict(schema=1,contract=checked['contract'],request_id=checked['request_id'],
        source_sha256=checked['source_sha256'],exercise_sha256=checked['exercise_sha256'],
        witnesses=checked['witnesses'],actual_frontend_checked=True,finite_return_influence_checked=True,
        independent_encrypted_numerics_required=True,all_input_proof=False,agent_generation_proven=False)
