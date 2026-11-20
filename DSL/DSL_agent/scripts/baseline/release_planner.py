"""Data-only sparse Release insertion for the existing RuntimePlan v2 API.

The original v1 checker is a pinned dependency. No arithmetic is evaluated,
library changed, input re-encrypted, or execution authority granted here.
"""
import copy
import hashlib
import cpu_ssa_linker as link


def nominal_bytes(value):
    return 8 * 32768 * (value['level'] + 1) * value['components']


def phases(plan):
    return plan['initialization'] + plan['execution'] + plan['finalization']


def verify_transform(original_raw, transformed_raw, spec_raw, rotations):
    old = link.loads(original_raw)
    values, _, _ = link.verify_plan(old, spec_raw, rotations)
    new = link.loads(transformed_raw)
    link.require(type(new) is dict and type(new.get('format_version')) is int and new['format_version'] == 2,
                 'Release profile requires format2')
    link.require(set(new) == set(old), 'unchanged plan fields')
    link.require(all(type(new[phase]) is list for phase in
                     ('initialization', 'execution', 'finalization')), 'phase list')
    operations = phases(new)
    link.require(len(operations) <= link.MAX_OPS, 'original total instruction bound including Release')
    inverse = copy.deepcopy(new)
    inverse['format_version'] = 1
    ordinal = 0
    for phase in ('initialization', 'execution', 'finalization'):
        restored = []
        link.require(type(new[phase]) is list, 'phase list')
        for op in new[phase]:
            link.require(type(op) is dict, 'instruction object')
            if op.get('kind') == 'release':
                link.require(phase == 'execution', 'Release only in execution')
                link.exact(op, {'ordinal', 'kind', 'value'}, 'Release exact fields')
                link.value_id(op['value'])
            else:
                restored.append(dict(op, ordinal=ordinal))
                ordinal += 1
        inverse[phase] = restored
    link.require(link.canonical(inverse) == link.canonical(old),
                 'original operations metadata order and payload bytes unchanged')
    protected = set(old['external_inputs']) | set(old['final_outputs'])
    live = set(old['external_inputs'])
    released = []
    total = sum(nominal_bytes(values[key]) for key in live)
    peak = total
    events = []
    for ordinal, op in enumerate(operations):
        link.require(type(op.get('ordinal')) is int and op['ordinal'] == ordinal,
                     'contiguous transformed ordinal')
        if op['kind'] == 'release':
            key = op['value']
            link.require(key in live and key not in protected and
                         values[key]['kind'] == 'ciphertext', 'live nonfinal internal ciphertext release')
            live.remove(key)
            total -= nominal_bytes(values[key])
            released.append(key)
        else:
            if op['kind'] == 'compute':
                link.require(all(key in live for key in op['inputs']), 'use of released value')
            key = op['output']
            link.require(key not in live and key not in released, 'single original definition')
            live.add(key)
            total += nominal_bytes(values[key])
            peak = max(peak, total)
        events.append([ordinal, total])
    link.require(set(old['final_outputs']) <= live, 'final outputs remain live')
    before = sum(nominal_bytes(row) for row in values.values())
    report = dict(format='poseidon-sparse-v2-Release-structural-proof-v1',
        original_raw_sha256=hashlib.sha256(original_raw).hexdigest(),
        transformed_raw_sha256=hashlib.sha256(transformed_raw).hexdigest(),
        spec_raw_sha256=hashlib.sha256(spec_raw).hexdigest(),
        original_instruction_count=len(phases(old)), transformed_instruction_count=len(operations),
        release_count=len(released), released_value_ids=released,
        exact_inverse_to_original_v1=True, same_original_encode_compute_order=True,
        same_values_target_bundle_externals_final_outputs=True,
        released_only_internal_ciphertexts=True, all_uses_precede_release=True,
        original4096_instruction_bound=True,
        nominal_all_values_coefficient_bytes=before,
        nominal_store_peak_coefficient_bytes_after_sparse_release=peak,
        nominal_store_final_coefficient_bytes=total,
        nominal_liveness_events_sha256=hashlib.sha256(link.canonical(events)).hexdigest(),
        memory_scope='Metadata coefficient-storage accounting only; excludes keys, allocator, decoded bundle, input owner, temporaries and process overhead; not actual memory or execution fit',
        original_CPP_checked=False, current_execution_dependencies=False,
        execution_authority=False, encrypted_execution=False, numerical_sum=False,
        library_modified=False, parameters_changed=False)
    report['binding'] = hashlib.sha256(link.canonical(report)).hexdigest()
    return report


def insert_sparse_releases(original_raw, spec_raw, rotations):
    old = link.loads(original_raw)
    values, _, _ = link.verify_plan(old, spec_raw, rotations)
    operations = phases(old)
    protected = set(old['external_inputs']) | set(old['final_outputs'])
    last = {}
    for op in operations:
        last[op['output']] = op['ordinal']
        if op['kind'] == 'compute':
            for key in op['inputs']:
                last[key] = op['ordinal']
    candidates = [key for key in values if key not in protected and
                  values[key]['kind'] == 'ciphertext']
    headroom = link.MAX_OPS - len(operations)
    link.require(headroom > 0 and candidates, 'no instruction budget or internal ciphertext for Release')
    # Prefer large values which would otherwise remain retained longest. This
    # is a deterministic heuristic, never an optimal-peak or real-memory claim.
    ranked = sorted(candidates, key=lambda key:
                    (-nominal_bytes(values[key]) * (len(operations) - last[key]), int(key)))
    selected = ranked[:headroom]
    after = {}
    for key in selected:
        after.setdefault(last[key], []).append(key)
    new = copy.deepcopy(old)
    new['format_version'] = 2
    execution = []
    ordinal = len(new['initialization'])
    for op in old['execution']:
        execution.append(dict(copy.deepcopy(op), ordinal=ordinal))
        ordinal += 1
        for key in sorted(after.get(op['ordinal'], []), key=int):
            execution.append(dict(ordinal=ordinal, kind='release', value=key))
            ordinal += 1
    new['execution'] = execution
    raw = link.canonical(new)
    proof = verify_transform(original_raw, raw, spec_raw, rotations)
    link.require(proof['release_count'] == len(selected), 'all selected Releases emitted')
    return dict(plan=new, raw=raw, proof=proof)
