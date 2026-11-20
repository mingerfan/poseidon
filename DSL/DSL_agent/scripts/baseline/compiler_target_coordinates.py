"""Source-bound compiler I/O coordinates, independent of native eligibility.

The caller supplies the independently frozen audit binding. These coordinates
are not program namespace labels, a decoder selection or an execution witness.
"""
import copy
import math
import re

from delivery_component import bound, canonical, checked_bound, require, sha
from retained_compiler_provenance import FORMAT as COMPILER
from compiler_plan_provenance import FORMAT as DAG

FORMAT = 'poseidon-compiler-audited-target-coordinates-v1'


def _hash(value):
    require(type(value) is str and re.fullmatch('[0-9a-f]{64}', value), 'Frozen target hash')
    return value


def make(receipt, expected_binding, *, request_id, source_sha256):
    """Use the compiler's actual return IDs and the original logical layout."""
    checked_bound(receipt, _hash(expected_binding))
    require(receipt.get('format') == COMPILER and receipt.get('request_id') == _hash(request_id)
        and receipt.get('source_sha256') == _hash(source_sha256), 'Original compiler target/source')
    graph = checked_bound(receipt['DAG']); layout = checked_bound(receipt['logical_IO'])
    require(graph.get('format') == DAG and layout.get('format') == 'poseidon-generic-compiler-logical-I-O-map-v1'
        and layout.get('compiler_DAG_binding') == graph['binding']
        and layout.get('request_id') == request_id and layout.get('source_sha256') == source_sha256,
        'Nested compiler DAG and logical target binding')
    period = checked_bound(receipt['periodic_binding'])
    require(period.get('format') == 'poseidon-original-periodic-payload-correspondence-v1'
        and period.get('all_non_payload_fields_unchanged') is True
        and period.get('plaintext_values_changed') is False, 'Original constant binding')
    width = period['input_slot_period']
    require(type(width) is int and 1 <= width <= 16384 and 16384 % width == 0, 'Original slot period')
    parameters, returns = graph['parameters'], graph['returns']
    require(1 <= len(parameters) <= 17 and 1 <= len(returns) <= 16, 'Original physical I/O limits')
    require(len({r['value_id'] for r in parameters}) == len(parameters)
        and len({r['value_id'] for r in returns}) == len(returns), 'Unique physical input/return ValueIds')
    require(len(layout['inputs']) == len(parameters), 'All physical parameter roles')
    inputs = []
    for index, (row, parameter) in enumerate(zip(layout['inputs'], parameters)):
        require(row['abi_index'] == index and row['ssa'] == parameter['ssa']
            and row['value_id'] == parameter['value_id'] and parameter['type']['kind'] == 'ciphertext',
            'Compiler parameter identity and order')
        inputs.append(dict(row, type=copy.deepcopy(parameter['type'])))
    physical = [dict(ciphertext=i, **copy.deepcopy(row)) for i, row in enumerate(returns)]
    require(all(r['type']['kind'] == 'ciphertext' for r in physical), 'Encrypted physical results')
    outputs, names, owners, cells, selectors = [], set(), {}, {}, []
    require(1 <= len(layout['outputs']) <= 16, 'All bounded logical outputs')
    for row in layout['outputs']:
        name, value, shape = row['name'], row['model_value'], row['shape']
        require(type(name) is str and name and name not in names and type(value) is str and value,
            'Distinct original logical output names')
        names.add(name)
        require(type(shape) is list and 1 <= len(shape) <= 8
            and all(type(n) is int and 1 <= n <= 1048576 for n in shape)
            and math.prod(shape) <= 1048576, 'Complete bounded output tensor')
        owner = owners.get(value)
        require(row.get('alias_of') == (owner['name'] if owner else None), 'Exact original model-value owner')
        segments, cursor = [], 0
        require(type(row['segments']) is list and row['segments'], 'Explicit output chunks')
        for segment in row['segments']:
            ct, count = segment['ciphertext'], segment['elements']
            require(type(ct) is int and 0 <= ct < len(returns)
                and segment['value_id'] == returns[ct]['value_id']
                and type(count) is int and 1 <= count <= width
                and segment['logical_start'] == cursor and segment['slot_start'] == 0,
                'Compiler return/ordered chunk/tail correspondence')
            segments.append(copy.deepcopy(segment))
            for slot in range(count):
                key = (ct, slot)
                require(key not in cells or owner is not None and cells[key] == value,
                    'Different expressions cannot share sampled-equal cells')
                cells[key] = value; selectors.append([ct, slot])
            cursor += count
        require(cursor == math.prod(shape), 'Complete logical tensor, including tail cells')
        if owner is not None:
            require(shape == owner['shape'] and segments == owner['segments'], 'Alias shape and complete physical mapping')
        result = dict(name=name, model_value=value, shape=copy.deepcopy(shape),
            alias_of=row.get('alias_of'), segments=segments)
        owners.setdefault(value, result); outputs.append(result)
    require(set(ct for ct, slot in cells) == set(range(len(returns)))
        and sha(canonical(selectors)) == layout['output_selectors_sha256']
        and len(selectors) == layout['logical_values'] and len(cells) == layout['unique_physical_cells']
        and len(selectors)-len(cells) == layout['alias_cells'], 'All original physical/logical/alias cells')
    return bound(dict(format=FORMAT, compiler_receipt_binding=expected_binding,
        compiler_DAG_binding=graph['binding'], logical_IO_binding=layout['binding'],
        request_id=request_id, source_sha256=source_sha256,
        original_compiler_proof_binding=receipt['original_proof_binding'],
        original_backend_identity_sha256=receipt['original_backend_identity_sha256'],
        original_compiler_sha256=receipt['original_compiler_sha256'],
        raw_plan_file_sha256=receipt['files_read']['attempt-00/output/lowered._hecate_golden.runtime-plan.json']['sha256'],
        bound_plan_file_sha256=receipt['files_read']['attempt-00/output/bound/plan.json']['sha256'],
        physical_inputs=inputs, physical_outputs=physical, logical_outputs=outputs,
        input_slot_period=width, logical_values=len(selectors), unique_physical_cells=len(cells),
        alias_cells=len(selectors)-len(cells),
        program_namespace_assigned=False, decoder_or_final_selection_constructed=False,
        input_encryption_or_compiler_host_or_same_key_final_output_certified=False,
        mathematical_or_CKKS_or_first_delivery_success_certified=False))


def check_program_unit(coordinates, expected_binding, program_unit, source_unit, *, unit_index):
    """Check local coordinates against both original program origin maps.

    The source container must be frozen independently by the caller. Matching
    local rows is not a proof of the multiunit model decomposition or execution.
    """
    checked_bound(coordinates, expected_binding); checked_bound(source_unit)
    require(coordinates.get('format') == FORMAT and type(unit_index) is int
        and 0 <= unit_index < 16 and source_unit['unit_index'] == unit_index,
        'Bounded original program unit')
    require(program_unit['source_origin_map_sha256'] == sha(canonical(source_unit)), 'Original source unit bytes')
    require(program_unit['raw_plan_sha256'] == source_unit['source_plan_sha256'] == coordinates['raw_plan_file_sha256']
        and program_unit['plan_sha256'] == source_unit['derived_plan_sha256'] == coordinates['bound_plan_file_sha256'],
        'Exact compiler raw/bound plan file identities')
    rows = {}
    for side, physical in [('inputs', coordinates['physical_inputs']), ('outputs', coordinates['physical_outputs'])]:
        actual, original = program_unit[side], source_unit[side]
        require(len(actual) == len(original) == len(physical), 'All local program inputs/results')
        selected = []
        for local, origin, position in zip(actual, original, physical):
            require(local['local_value_id'] == origin['local_value_id'] == position['value_id'],
                'No hand-assigned or reordered compiler runtime IDs')
            logical = dict(value=origin['logical_value'], chunk=origin['chunk_index'])
            require(local['logical'] == logical and type(logical['value']) is str and logical['value']
                and type(logical['chunk']) is int and 0 <= logical['chunk'] <= 8191,
                'Exact original program namespace/chunk')
            selected.append(dict(local_value_id=position['value_id'], logical=logical))
        rows[side] = selected
    require(len({(r['logical']['value'], r['logical']['chunk']) for r in rows['outputs']}) == len(rows['outputs']),
        'Unique physical produced logical chunks')
    return bound(dict(format='poseidon-compiler-program-local-origin-correspondence-v1',
        coordinates_binding=expected_binding, source_unit_binding=source_unit['binding'],
        unit_index=unit_index, unit_id=program_unit['unit_id'], **rows,
        proves_original_model_decomposition_or_native_execution=False,
        complete_program_target_or_decoder_eligibility_granted=False))
