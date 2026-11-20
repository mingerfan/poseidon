"""Public computation obligations before one-shot DSL generation.

This inspects the supplied graph and registered bindings. It creates no DSL,
chooses no refresh positions and reads no compiler, execution or reference data.
"""
import json
from collections import Counter

VERSION = 'compiler-managed-computational-obligations-v1'


def require(value, message):
    if not value:
        raise ValueError(message)


def obligations(request):
    from unified_contract_scope_guidance import (effective_contract, NATIVE_CONTROL,
        NATIVE_HELPERS, NATIVE_INTEGER, route_semantics)
    model=request['model'];layout=request['layout']
    inputs={s['name'] for s in model['inputs']};constants=set(model['constants'])
    nodes=model['nodes'];producers={};ids=set()
    require(type(nodes) is list, 'Public computation graph')
    available=inputs|constants
    for node in nodes:
        require(node['id'] not in ids and all(name in available for name in node['inputs']),
                'Ordered public computation dependencies')
        ids.add(node['id'])
        for output in node['outputs']:
            require(output not in available, 'Distinct public computation value')
            producers[output]=node;available.add(output)
    outputs=[o['value'] for o in model['outputs']]
    require(all(name in available for name in outputs), 'Declared public model outputs')
    pending=list(outputs);seen=set();live=set();used_constants=set()
    while pending:
        value=pending.pop()
        if value in seen:continue
        seen.add(value)
        if value in constants:used_constants.add(value)
        node=producers.get(value)
        if node is not None:
            live.add(node['id']);pending.extend(node['inputs'])
    ordered=[node for node in nodes if node['id'] in live]
    arguments=[s['dsl_name'] for s in layout['inputs']]
    arguments.extend(s['dsl_name'] for s in layout.get('auxiliary_ciphertexts',[]))
    require(arguments and len(arguments)==len(set(arguments)), 'Public entry argument names')
    header='@hc.func("'+','.join('c' for unused in arguments)+'")\ndef golden('+', '.join(arguments)+'):'
    contract=effective_contract(request)
    syntax=dict(contract=contract,entry_header=header,output_ciphertexts=layout['output_ciphertexts'],
                registered_public_names_only=True,model_constant_ids_are_not_implicit_DSL_names=True)
    if contract=='native':
        syntax.update(control=NATIVE_CONTROL,helpers=NATIVE_HELPERS,static_integers=NATIVE_INTEGER)
    else:
        syntax['control']='Use the existing public construction profile and its own declared syntax; native-only restrictions do not change that profile.'
    counts=dict(sorted(Counter(n['op'] for n in ordered).items()))
    flow=[[n['id'],n['op'],list(n['inputs']),list(n['outputs'])] for n in ordered]
    joins=[n['id'] for n in ordered if sum(name not in constants for name in n['inputs'])>1]
    polynomial=[dict(node=n['id'],basis=n['attrs']['basis'],coefficients=n['inputs'][1])
                for n in ordered if n['op']=='polynomial']
    packing=request.get('constant_origins',{}).get('capacity_tensor_packing',{})
    bank=packing.get('scalar_bank');access=None
    if packing.get('version')=='capacity-public-tensors-v11' and bank is not None:
        pages=bank['pages'];segments=bank['segments'];period=layout['input_slot_period']
        names=set(request['public_constants']);examples=[]
        require(all(page in names for page in pages), 'Registered public coefficient pages')
        ordered_refs=[]
        for node in ordered:
            for name in node['inputs']:
                if name in used_constants and name not in ordered_refs:ordered_refs.append(name)
        for name in ordered_refs[:2]:
            segment=segments[name];offset=segment['offset'];page,index=divmod(offset,period)
            require(0<=page<len(pages), 'Public coefficient page range')
            examples.append(dict(model_constant=name,flat_index=0,expression=pages[page]+'['+str(index)+']',shape=segment['shape']))
        access=dict(version='capacity-public-tensors-v11',page_count=len(pages),page_length=period,
            coordinate_rule='For a static C-order element k, t=scalar_bank.segments[model_constant].offset+k; read scalar_bank.pages[t//P][t%P] as a registered page name with a static integer index. Resolve page names and indices using the disclosed metadata.',
            examples=examples,
            value_rule='A model constant ID labels a logical tensor. A registered vector used as a Plain Expr and a scalar loaded from a public coefficient page have distinct meanings. Preserve the complete tensor and every required coefficient.')
    return dict(version=VERSION,syntax=syntax,reachable_node_count=len(ordered),
        required_node_ids=[n['id'] for n in ordered],dataflow_columns=['node','operator','input_values','output_values'],
        required_dataflow=flow,operator_counts=counts,branch_join_nodes=joins,declared_polynomials=polynomial,
        logical_outputs=model['outputs'],required_constant_ids=sorted(used_constants),
        coefficient_access=access,bound_route_semantics=route_semantics(request),
        implementation_obligation='Construct the complete computation of every requested output from its actual dependencies. Fusion and algebraic simplification must preserve the declared function on the complete valid domain. Placeholder features, skipped nonlinearities and an output head without its producer computation do not implement that function.',
        implementation_choices_owner='Agent: computational implementation, fusion, layout and blocking',
        ciphertext_state_owner='Frozen external compiler: rescale, modswitch, bootstrap placement and ciphertext state',
        candidate_source_created=False,compiler_or_execution_feedback_used=False,reference_values_used=False,
        numeric_or_backend_certification=False)


def render(request):
    return 'Public computation obligations before emitting the final DSL:\n'+json.dumps(
        obligations(request),sort_keys=True,separators=(',',':'),allow_nan=False)
