"""External, data-only program target construction from audited unit compilers.

This preserves program namespaces and all selected cells. It is not a proof of
whole-source decomposition, a live compiler/HE host witness or native admission.
"""
import copy
import importlib.util
import math
from pathlib import Path

from delivery_component import bound, canonical, checked_bound, read_bytes, require, sha
from component_contract import reconstruct_request
from benchmark_graph import validate
from candidate_contract import request_input_names
import compiler_target_coordinates as coordinates

FORMAT = 'poseidon-compiler-program-target-fragment-v1'
COMPOSED = 'poseidon-compiler-program-qualified-final-correspondence-v1'
TRANSPORT = Path('/home/lhohy/poseidon-work/platforms/aarch64-linux/results/agent-public-bundle-readonly-view-controls-r2588/revision-corrected-carry-helper-v2/controls/program_transport.py')
TRANSPORT_SHA = 'a83b4c79b9527a56348416ec1762f615365f0cf59fa8b2fac5958de8bc3c7e2b'
MAX_METADATA = 1024**2
MAX_ALIASES = 16  # Original B decoder v1 and target bridge v2; never truncate.


class ProgramTargetBlocked(ValueError):
    def __init__(self, reason):
        self.reason = reason
        super().__init__('Program target remains blocked: '+reason)


def decoder_period_requirements(period):
    require(type(period)is int and 1<=period<=16384 and 16384%period==0,'Original declared slot period')
    count=16384//period-1
    return bound(dict(format='poseidon-complete-periodic-output-decoder-requirements-v1',input_slot_period=period,
        required_explicit_aliases_per_chunk=count,original_decoder_alias_limit=MAX_ALIASES,
        fits_original_decoder=count<=MAX_ALIASES,aliases_shortened_or_sampled=False,
        numeric_values_decoded=0,native_or_first_delivery_success_certified=False))


def transport():
    """Only the independently pinned existing project metadata validator."""
    code=read_bytes(TRANSPORT,MAX_METADATA)
    require(sha(code)==TRANSPORT_SHA,'Original program metadata validator changed')
    spec=importlib.util.spec_from_file_location('poseidon_original_program_metadata',str(TRANSPORT))
    module=importlib.util.module_from_spec(spec)
    exec(compile(code,str(TRANSPORT),'exec'),module.__dict__)
    return module


def _bounded(value):
    require(len(canonical(value))<=MAX_METADATA,'Original final-client per-program metadata1MiB limit')


def _origin(row):
    return dict(value=row['logical_value'],chunk=row['chunk_index'])


def _physical_type(value, expected):
    require(value['kind']==expected['kind']=='ciphertext'and value['components']==expected['components']
        and value['level']==expected['level']and value['scale_log2']==expected['scale_log2'],
        'Actual compiled boundary types, without header/scale retagging')


def build(request,source,container_raw,program,unit_receipts,plan_bytes,selection,expected):
    """Construct target rows and maps; no bytes from numeric/native outputs.

    expected comes from the frozen controller, independently of the objects.
    The resulting composite return IDs explicitly qualify their unit and do
    not pretend to be a single compiler's original local ValueIds.
    """
    require(type(expected)is dict and set(expected)=={'request_id','source_sha256','container_sha256',
        'container_binding','program_binding','selection_binding','unit_receipt_bindings'},'Closed original program freeze')
    require(type(source)is bytes and source and sha(source)==expected['source_sha256']
        and request['request_id']==expected['request_id'],'Original whole-request and untouched DSL bytes')
    reconstruct_request(copy.deepcopy(request));checked_model=validate(request['model'])
    original=transport();container=original.decode(container_raw,expected['container_sha256'])
    original.binding(container,expected['container_binding']);checked_bound(program,expected['program_binding'])
    checked_bound(selection,expected['selection_binding']);_bounded(program);_bounded(selection)
    require(container['format']=='poseidon-bound-tensor-multiunit-source-container-v1'
        and container.get('execution_authority')is False and program['binding_stage']=='physical_bound',
        'Original physical program and non-authorizing source container')
    require(container['scope']==program['scope'] and container['key_epoch']==program['key']['epoch']
        and container['key_origin']==program['key']['origin'],'Original program source scope and key')
    require(program['source']==dict(container_sha256=sha(container_raw),origins_sha256=sha(container_raw),
        model_sha256=sha(canonical(request['model'])),layout_sha256=sha(canonical(request['layout'])))
        and container['source_model_sha256']==program['source']['model_sha256']
        and container['parent_layout_sha256']==program['source']['layout_sha256'],
        'Whole original model/layout/source-container identities')
    units=program['units'];require(1<=len(units)==len(container['units'])==len(unit_receipts)<=16,
        'Complete original program unit inventory')
    require(len(expected['unit_receipt_bindings'])==len(units),'One independently frozen compiler receipt per unit')
    require(set(plan_bytes)=={u['plan_sha256']for u in units},'Exact complete program plan inventory')
    require(all(type(raw)is bytes and len(raw)<=MAX_METADATA for raw in plan_bytes.values()),
        'Original client plan size, without silently accepting larger transport files')
    checked=original.validate_program(program,expected['program_binding'],plan_bytes)
    require([r['logical']for r in program['inputs']]==[_origin(r)for r in container['inputs']]
        and program['final_outputs']==[_origin(r)for r in container['outputs']],
        'Original external inputs and complete final output order')
    layout=request['layout'];period=layout['input_slot_period']
    require(type(period)is int and 1<=period<=16384 and 16384%period==0,'Original physical slot period')
    physical_inputs=list(layout['inputs'])+list(layout.get('auxiliary_ciphertexts',[]))
    require(len(physical_inputs)==len(program['inputs'])<=16 and len(physical_inputs)==len(request_input_names(request)),
        'All original physical input roles, including encrypted zero')
    for role,row in zip(physical_inputs,program['inputs']):
        if role.get('kind')=='encrypted_zero':
            require(row['role']=='encrypted_zero'and row['logical']==dict(value=role['dsl_name'],chunk=0),
                'Original trusted encrypted-zero role')
        else:
            require(row['role']=='data'and row['logical']==dict(value=role['name'],chunk=role.get('offset',0)//period),
                'Original logical input name/chunk; no substituted input samples')
    maps=[];by_unit={};compiler_identity=None
    for index,(unit,source_unit,receipt,pin)in enumerate(zip(units,container['units'],unit_receipts,expected['unit_receipt_bindings'])):
        _bounded(source_unit);checked_bound(receipt,pin)
        current_identity=(receipt['original_compiler_sha256'],receipt['original_backend_identity_sha256'])
        require(compiler_identity is None or compiler_identity==current_identity,'All units use the original compiler/backend identity')
        compiler_identity=current_identity
        coordinate=coordinates.make(receipt,pin,request_id=receipt['request_id'],source_sha256=receipt['source_sha256'])
        require(coordinate['input_slot_period']==period,'No per-unit packing-period substitution')
        local=coordinates.check_program_unit(coordinate,coordinate['binding'],unit,source_unit,unit_index=index)
        plan=checked['plans'][unit['unit_id']];values={r['id']:r for r in plan['values']}
        require(receipt['DAG']['value_count']==len(plan['values'])
            and receipt['DAG']['instruction_count']==sum(len(plan[k])for k in ('initialization','execution','finalization')),
            'Full compiler instruction/value inventory for the actual unit plan')
        for side,rows in [('physical_inputs',coordinate['physical_inputs']),('physical_outputs',coordinate['physical_outputs'])]:
            for row in rows:_physical_type(values[row['value_id']],row['type'])
        maps.append(dict(unit_id=unit['unit_id'],compiler_receipt_binding=pin,coordinate_binding=coordinate['binding'],
            local_origin_binding=local['binding'],request_id=receipt['request_id'],source_sha256=receipt['source_sha256']))
        by_unit[unit['unit_id']]=dict(coordinates=coordinate,plan=plan,source=source_unit)
    require(selection.get('format')=='poseidon-program-final-selection-v1'
        and selection['program_binding']==program['binding']and selection['key']==program['key'],
        'Original selected program and same-key metadata')
    original.hex_value(selection['case_binding']);original.hex_value(selection['input_bank_binding'])
    finals=program['final_outputs'];selected=selection['outputs']
    require(len(selected)==len(finals)==layout['output_ciphertexts']<=16,'Every final physical ciphertext')
    require(len(selection['terminals'])==len(units)
        and [r['unit_id']for r in selection['terminals']]==[u['unit_id']for u in units],
        'Complete original unit terminal metadata, not proof of live host execution')
    for terminal,unit in zip(selection['terminals'],units):
        checked_bound(terminal)
        require(terminal['program_binding']==program['binding']and terminal['case_binding']==selection['case_binding']
            and terminal['input_bank_binding']==selection['input_bank_binding']and terminal['unit_plan_sha256']==unit['plan_sha256'],
            'Original program/case/bank/unit terminal metadata scope')
    returned=[];file_names=set()
    for index,(logical,row)in enumerate(zip(finals,selected)):
        tag=original.logical(logical);require(tag in checked['namespace'],'Final must be produced by a source unit')
        uid,plan_hash,local_id=checked['namespace'][tag]
        require(row['logical']==logical and row['unit_id']==uid and row['unit_plan_sha256']==plan_hash
            and row['local_value_id']==local_id,'Exact source-produced selected final; no hand-written runtime label')
        filename=row['decoded_file'];require(type(filename)is str and filename and Path(filename).name==filename
            and filename not in file_names,'Distinct original physical decoded files')
        file_names.add(filename)
        coordinate=by_unit[uid]['coordinates'];matches=[r for r in coordinate['physical_outputs']if r['value_id']==local_id]
        require(len(matches)==1,'Selected final is an actual compiler return')
        actual=matches[0];returned.append(dict(ssa='%'+uid+'.'+actual['ssa'].removeprefix('%'),
            value_id=uid+':'+local_id,type=copy.deepcopy(actual['type']),unit_id=uid,local_value_id=local_id))
    require(len({r['value_id']for r in returned})==len(returned)
        and len({r['ssa']for r in returned})==len(returned),'Unit-qualified physical returns stay distinct')
    composed=bound(dict(format=COMPOSED,source_kind='audited_unit_compiler_and_program_origin_data',
        program_binding=program['binding'],source_container_binding=container['binding'],unit_maps=maps,
        returns=returned,value_count=checked['value_count'],instruction_count=checked['instruction_count'],
        one_compiler_global_DAG_or_whole_source_decomposition_claimed=False,
        generation_to_unit_source_derivation_or_live_compiler_host_certified=False))
    owners={};outputs=[];targets=[];decoder_rows=[];selectors=[];physical_cells={}
    require(len(request['model']['outputs'])==len(layout['outputs']),'Every original named model output')
    for model_row,declared in zip(request['model']['outputs'],layout['outputs']):
        name,value=model_row['name'],model_row['value'];shape=list(checked_model['shapes'][value]);size=math.prod(shape)
        require(declared['name']==name and declared['shape']==shape,'Original whole-model output shape/name')
        first=owners.get(value,name)
        require(declared.get('alias_of')==(None if first==name else first),'Original exact logical alias owner')
        chunks=declared.get('chunks',[dict(ciphertext=declared.get('ciphertext'),offset=0,elements=size)])
        compiler_segments=[];native_segments=[];cursor=0;runtime_value=None
        for chunk_index,chunk in enumerate(chunks):
            ct,count,start=(chunk[k]for k in ('ciphertext','elements','offset'))
            require(type(ct)is int and 0<=ct<len(finals)and type(count)is int and 1<=count<=period
                and start==cursor,'Complete original logical output chunks and tails')
            require(finals[ct]==dict(value=value,chunk=chunk_index),'Final namespace must retain original model value and chunk')
            runtime_value=finals[ct]['value']if runtime_value is None else runtime_value
            require(runtime_value==finals[ct]['value'],'One original runtime owner per logical tensor')
            requirements=decoder_period_requirements(period)
            if not requirements['fits_original_decoder']:
                raise ProgramTargetBlocked('all_periodic_aliases_exceed_original_decoder_limit:%d>%d'%(
                    requirements['required_explicit_aliases_per_chunk'],MAX_ALIASES))
            aliases=list(range(period,16384,period))
            compiler_segments.append(dict(ciphertext=ct,value_id=returned[ct]['value_id'],logical_start=start,slot_start=0,elements=count))
            native_segments.append(dict(file=selected[ct]['decoded_file'],logical_start=start,count=count,slot_start=0,alias_starts=aliases))
            for slot in range(count):
                key=(ct,slot);require(key not in physical_cells or physical_cells[key]==value and first!=name,
                    'Only exact model-value aliases may share cells')
                physical_cells[key]=value;selectors.append([ct,slot])
            cursor+=count
        require(cursor==size,'No missing logical output cells')
        owners.setdefault(value,name)
        outputs.append(dict(name=name,model_value=value,shape=shape,alias_of=None if first==name else first,segments=compiler_segments))
        targets.append(dict(request_name=name,model_value=value,runtime_value=runtime_value,decoder_name=name,shape=shape,physical_owner=first))
        decoder_rows.append(dict(name=name,shape=shape,segments=native_segments))
    require(selectors==layout['output_selectors']and set(ct for ct,slot in physical_cells)==set(range(len(finals))),
        'Every original logical and unique physical final cell')
    io=bound(dict(format='poseidon-generic-compiler-logical-I-O-map-v1',origin='audited_program_composition_not_single_compiler_DAG',
        compiler_DAG_binding=composed['binding'],request_id=request['request_id'],source_sha256=sha(source),
        source_model_sha256=sha(canonical(request['model'])),source_container_binding=container['binding'],
        outputs=outputs,logical_values=len(selectors),unique_physical_cells=len(physical_cells),alias_cells=len(selectors)-len(physical_cells),
        output_selectors_sha256=sha(canonical(selectors)),whole_source_model_semantics_or_native_execution_certified=False))
    periodic=sum(s['count']*len(s['alias_starts'])for row in decoder_rows for s in row['segments'])
    fragment=dict(format=FORMAT,request_id=request['request_id'],source_sha256=sha(source),
        source_model_sha256=sha(canonical(request['model'])),source_container_binding=container['binding'],program_binding=program['binding'],
        final_selection_binding=selection['binding'],unit_compiler_receipt_bindings=expected['unit_receipt_bindings'],
        compiler_program_map=composed,logical_IO=io,target_outputs=targets,decoder_outputs=decoder_rows,
        physical_identity=dict(program_binding=program['binding'],case_binding=selection['case_binding'],
            input_bank_binding=selection['input_bank_binding'],key=copy.deepcopy(program['key'])),
        logical_values=len(selectors),unique_physical_cells=len(physical_cells),logical_alias_cells=len(selectors)-len(physical_cells),
        periodic_alias_values=periodic,complex_components_with_periodic_aliases=2*(len(selectors)+periodic),
        original_unit_count=len(units),original_program_values=checked['value_count'],original_program_instructions=checked['instruction_count'],
        numerical_rows_decoded=0,model_or_DSL_forward_executed=False,old_source_or_plan_or_selection_modified=False,
        original_transport_source_sha256=TRANSPORT_SHA,complete_decoder_contract_or_native_eligibility_created=False,
        missing_external_witnesses=['whole Agent source to unit compiler derivation/lowering',
            'actual compiler frontend and host execution','raw/logical/encrypted input and same-key final-only output',
            'original host/resource terminal scope','independent mathematics/effective domain/full application'],
        first_original_or_complex_whole_model_or_CKKS_success_certified=False)
    _bounded(io);_bounded(composed)
    return bound(fragment)
