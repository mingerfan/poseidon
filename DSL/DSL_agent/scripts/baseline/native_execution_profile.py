"""Explicit bounded Native request/compile profile; no backend or key operations.

Only newly prepared requests selecting this configuration use these probes.
All old request IDs, samplers, compiler defaults and numerical references remain
with their existing implementation. Profile preparation is not FHE acceptance.
"""
import copy
import hashlib
import json
import math
from pathlib import Path

NAME='poseidon-cpu-eva-w44-native-epoch-v1'
PROFILE='273e3a20a734a3beb71c4eafc5c29a21e49db8bd5c64576e30ef13c159c2c621'
SPEC='a1e74e356276cda1141e3b651e72ba8a5456dabeed03120830f6f0b371c55841'
CAPABILITY='e4e8fe348b7f875d7af1283ca4c51da23090c17f591f42f677a5b4d56d914c9e'
SOURCE_LOCK='243946ab90446743f0063b7b71bdbfd6ba46f0bab63e0ca6eecf57af9eef0526'
EPOCH='ad7439a7013ff2cf8f215d97daad155a'
RUNTIME='1719b145c72ff0d26fc3b515e82369abf59fed69baa71423c612c728d26f9bb6'
CLIENT='a4fe8901cce504f82c308d72549f67933b672fa9f9fddd37ad84eeb699c1454d'
POLICY='poseidon-native-certified-unused-input-link-prototype-v3'
DOMAIN=dict(format='poseidon-native-bounded-input-profile-v1',minimum=-.5,maximum=.5,
            max_slot_period=256,max_logical_inputs=1,
            validation_sampler='native-bounded-four-groups-v1')
OPS=frozenset({'square','multiply','power','flatten','reshape'})


def require(condition,reason):
    if not condition:raise ValueError('Native execution profile: '+reason)


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def configuration(profile_sha256=None,*,schema=None):
    require(schema is None or type(schema)is int and schema==6,'configuration version')
    require(profile_sha256 in(None,PROFILE),'physical compiler profile')
    body=dict(schema=6,name=NAME,backend='poseidon_cpu',pipeline='eva-runtime-plan',
        execution_profile='original-epoch-public-native-v1',ckks_config_sha256=PROFILE,waterline=44,
        runtime_commit='f7e9115eaec22e44c8e12ff9f53528e421691688',
        dacapo_commit='39c1fb255adddbfb9d7180f9d1b8535a31241bff',
        api_commit='957ff9c05181f47485d2555ab52b2bedcbdb60d6',
        parameter_identity='poseidon-agent-n15-q16-p2-tc128-original-epoch-v1',
        source_lock_sha256=SOURCE_LOCK,operator_spec_sha256=SPEC,
        runtime_plan_entry_policy='hecate-private-symbol-dce-v1',
        trace_serialization_policy='hecate-dense-f64-shortest-v1',
        runtime_plan_serialization_policy='runtime-plan-json-files-v1',
        native_capability_binding=CAPABILITY,native_adapter_policy=POLICY,
        native_key_epoch=EPOCH,native_runtime_sha256=RUNTIME,native_client_sha256=CLIENT,
        native_input_profile=copy.deepcopy(DOMAIN),native_Boot_invocations_per_process=2,
        compiler_output_val=0,source_input_physical_level=1,source_input_scale_log2=44)
    return dict(body,identity_sha256=hashlib.sha256(
        b'poseidon-compiler-configuration-v6\0'+canonical(body)).hexdigest())


def selected(request):
    value=request.get('compiler_configuration')
    return type(value)is dict and value.get('name')==NAME


def validate_configuration(value):
    require(type(value)is dict and canonical(value)==canonical(configuration()),'immutable configuration')
    return copy.deepcopy(value)


def validate_model(model):
    from benchmark_graph import validate
    check=validate(model)
    require(len(model['inputs'])==1 and not model['constants']and
            len(model['outputs'])==1,'one logical input/output, no uncertified plaintext operators')
    require(math.prod(model['inputs'][0]['shape'])<=256,'certified logical period bound')
    intervals={model['inputs'][0]['name']:(-.5,.5)}
    for node in model['nodes']:
        op=node['op'];require(op in OPS,'operation outside current native fragment profile')
        operands=[intervals[name]for name in node['inputs']]
        if op in ('square','power'):
            exponent=2 if op=='square'else node['attrs']['exponent']
            lo,hi=operands[0]
            result=(0. if lo<=0<=hi else min(lo**exponent,hi**exponent),
                    max(lo**exponent,hi**exponent))
        elif op=='multiply':
            a,b=operands
            products=[x*y for x in a for y in b];result=(min(products),max(products))
        else:
            result=operands[0]
        require(all(math.isfinite(x)for x in result),'finite declared model interval')
        for name in node['outputs']:intervals[name]=result
    output=intervals[model['outputs'][0]['value']]
    require(max(abs(x)for x in output)<=1,'model output exceeds compiler output-val=0 bound')
    return dict(model_validation=check,input_interval=[-.5,.5],output_interval=list(output),
                compiler_output_val=0,numerical_or_encrypted_acceptance=False)


def validate_request(request):
    validate_configuration(request.get('compiler_configuration'))
    require(not any(key in request for key in ('upstream_helpers','construction_profile',
            'construction_exercise','capability_composition')),'initial Native fragment construction scope')
    require(request.get('compiler_profile_sha256')==PROFILE,'request profile identity')
    validate_model(request['model'])
    layout=request['layout']
    require(type(layout['input_slot_period'])is int and
            1<=layout['input_slot_period']<=256 and
            layout['input_slot_period']&(layout['input_slot_period']-1)==0,
            'bounded power-of-two slot period')
    require(len(layout['inputs'])==1 and layout['output_ciphertexts']==1,
            'single physical input/output within current certified invocation profile')


def samples(model):
    """Four explicit Native probes, independently referenced by original math."""
    validate_model(model)
    import numpy as np
    spec=model['inputs'][0];count=math.prod(spec['shape']);groups=[]
    signed=[.125,-.25,.375,-.5]
    boundary=[-.5,.5,-.499,.499,0.,1/1024,-1/1024,.25]
    seed=233;random_values=[]
    for _ in range(count):
        seed=(1664525*seed+1013904223)&0xffffffff
        random_values.append(float(seed>>8)/16777215-.5)
    for values in [[0.]*count,[signed[i%4]for i in range(count)],random_values,
                   [boundary[i%8]for i in range(count)]]:
        array=np.asarray(values,dtype=np.float64).reshape(spec['shape'])
        groups.append({spec['name']:array})
    return groups


def command(value):
    validate_configuration(value)
    return ['/hecate-opt','/out/candidate_trace.entry.mlir',
        '--runtime-plan-format=json','--runtime-plan-bundle-format=files','--symbol-dce',
        '--eva','--ckks-config=/profile.json','--waterline=44','--output-val=0',
        '--enable-debug-printer','--mlir-disable-threading','--verify-each',
        '--runtime-plan-id=1','--runtime-plan-target-id=poseidon-ckks-cpu',
        '--runtime-plan-capability-version=1',
        '--runtime-plan-operator-spec-id=B-original-epoch-native-odd-cpu-v1',
        '--runtime-plan-operator-spec-version=1','--runtime-plan-operator-spec-sha256=sha256:'+SPEC,
        '--runtime-plan-context-id=ctx-original-epoch','--runtime-plan-device-count=0',
        '--runtime-plan-ntt=true','--runtime-plan-boot-implementation=native',
        '--runtime-plan-inline-payload-max-bytes=4096','-o','/out/lowered.mlir']


def prepare_entrypoint(output,value):
    validate_configuration(value)
    from poseidon_runtime_compile import compact_f64_trace,entrypoint_trace
    source=(Path(output)/'candidate_trace.mlir').read_text()
    compact,serialization=compact_f64_trace(source)
    selected,receipt=entrypoint_trace(compact)
    receipt['trace_serialization']=serialization
    receipt['native_compiler_configuration_sha256']=value['identity_sha256']
    (Path(output)/'candidate_trace.compact.mlir').write_text(compact)
    (Path(output)/'candidate_trace.entry.mlir').write_text(selected)
    (Path(output)/'entry-selection.json').write_text(json.dumps(receipt,sort_keys=True,indent=2)+'\n')
    return receipt
