"""Trusted reference and fixed-input preparation for external candidate evaluation."""
import json
import math
from pathlib import Path
import numpy as np
from benchmark_graph import samples, validate, require
from benchmark_math import evaluate as mathematical
from benchmark_torch import evaluate as torch_reference, Model
from unified_graph_contract import layout


def prepare_case(graph,folder,chunk_period=None,*,capacity_abi=None,output_binding_policy=None,
                 input_profile=None,evaluation_inputs=None,compile_only=False):
    require(type(compile_only) is bool,'Compile preparation mode')
    check=validate(graph);plan=layout(graph,chunk_period,capacity_abi=capacity_abi,output_binding_policy=output_binding_policy);period=plan['input_slot_period']
    np.savez(folder/'weights.npz',**{k:np.asarray(v,dtype=np.float64) for k,v in graph['constants'].items()})
    binding_text=json.dumps(plan,indent=2)
    from candidate_contract import MAX_BYTES
    if len(binding_text.encode())>MAX_BYTES:
        binding_text=json.dumps(plan,separators=(',',':'),allow_nan=False)
    if len(binding_text.encode())>MAX_BYTES:
        raise ValueError('Logical layout binding size')
    (folder/'logical-bindings.json').write_text(binding_text)
    dataset=None
    if evaluation_inputs is not None:
        from hecate_python_env import WORK
        from frozen_evaluation_inputs import read_manifest
        dataset=read_manifest(evaluation_inputs,allowed_root=WORK/'results')
        require(dataset.get('model_id')==graph['id'],'Evaluation dataset model identity')
        (folder/'evaluation-inputs.json').write_text(json.dumps(dataset,sort_keys=True,allow_nan=False))
    if compile_only:
        return Model(graph).eval()
    inputs=[];references=[]
    cached=None
    if dataset is not None:
        from frozen_evaluation_inputs import load
        from hecate_python_env import WORK
        require(input_profile is None,'Fixed inputs and the Native sampler are separate selections')
        output_elements=sum(math.prod(check['shapes'][o['value']]) for o in graph['outputs'])
        resolved=load(dataset,graph['inputs'],output_elements,allowed_root=WORK/'results')
        probes=[{k:np.asarray(v,dtype=np.float64) for k,v in row['values'].items()} for row in resolved['cases']]
        cached=[row['reference'] for row in resolved['cases']]
    else:
        probes=samples(graph,4)
        if input_profile is not None:
            from native_execution_profile import DOMAIN,samples as native_samples
            from benchmark_graph import canonical
            require(canonical(input_profile)==canonical(DOMAIN),'Explicit Native bounded sampler identity')
            probes=native_samples(graph)
    for index,probe in enumerate(probes):
        retained=None if cached is None else cached[index]
        if retained is None:
            expected=mathematical(graph,probe);observed=torch_reference(graph,probe)
            for name in expected:
                np.testing.assert_allclose(observed[name],expected[name],atol=1e-12,rtol=1e-12)
            reference=np.concatenate([expected[o['name']].reshape(-1) for o in graph['outputs']])
        else:
            reference=np.asarray(retained,dtype=np.float64)
        vectors=[]
        for binding in plan['inputs']:
            flat=probe[binding['name']].reshape(-1)
            start=binding.get('offset',0);count=binding.get('elements',flat.size)
            vectors.append(np.pad(flat[start:start+count],(0,period-count)))
        inputs.append(np.stack(vectors));references.append(reference)
    packed=np.stack(inputs)
    if len(plan['inputs'])==1:packed=packed[:,0,:]
    np.savez(folder/'arrays.npz',inputs=packed,reference=np.stack(references))
    return Model(graph).eval()


def prepare_request(request, folder, *, evaluation_inputs=None, compile_only=False):
    """Prepare compiler metadata or a complete external numerical input set."""
    from benchmark_graph import require
    from unified_graph_contract import validate_request
    from unified_output_aliases import policy as output_policy
    validate_request(request)
    binding=request['layout']
    period=binding['input_slot_period'] if 'logical_inputs' in binding else None
    abi=binding['execution_abi'] if 'resource_profile' in binding else None
    input_profile=None
    from native_execution_profile import selected,validate_request as validate_native_request
    if selected(request):
        validate_native_request(request)
        input_profile=request['compiler_configuration']['native_input_profile']
    model=prepare_case(request['model'],folder,period,capacity_abi=abi,
                       output_binding_policy=output_policy(binding),input_profile=input_profile,
                       evaluation_inputs=evaluation_inputs,compile_only=compile_only)
    require(json.loads((folder/'logical-bindings.json').read_text())==binding,
            'Prepared model/layout binding mismatch')
    return model
