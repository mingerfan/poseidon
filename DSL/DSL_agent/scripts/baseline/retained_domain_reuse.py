"""Bind retained measured domain checks to a new, explicit evaluation policy.

No domain/model/DSL forward or inference of extrema from theorem bounds occurs.
The old observation and its absent historical policy remain unchanged.
"""
import copy
import math
from pathlib import Path
import re

from delivery_component import bound, canonical, checked_bound, read_bytes, require, sha, write_once
from first_output_assessment import MATH, CKKS
import retained_math_domains as domains
import benchmark_math

ENTRY = 'poseidon-retained-measured-domain-entry-v1'
CATALOG = 'poseidon-retained-measured-domain-catalog-v1'
SELECTION = 'poseidon-retained-measured-domain-policy-selection-v1'


def implementation_sources():
    return {str(Path(m.__file__).absolute()): sha(read_bytes(m.__file__))
        for m in (domains, benchmark_math)}


def freeze_entry(metadata, expected_bindings, *, test_only=False, theorem_annotation=None):
    """Normalize one frozen B observation without inventing its old request.

    This importer uses the published domain-profile metadata ABI, not a model
    name or a generated implementation route. The theorem annotation is opaque,
    separately typed archival material; only actual observations may be reused.
    """
    require(type(metadata) is dict and set(metadata) == {'contract','observation','input','origin','plan'}
        and set(expected_bindings) == set(metadata) and type(test_only) is bool, 'Explicit domain evidence set')
    for key, value in metadata.items(): checked_bound(value, expected_bindings[key])
    contract, observation, source, origin, plan = (metadata[k] for k in ('contract','observation','input','origin','plan'))
    require(contract['format'] == domains.FORMAT and observation['format'] == domains.RESULT
        and observation['domain_contract_binding'] == contract['binding']
        and observation['declared_reference_model_sha256'] == contract['declared_reference_model_sha256'],
        'Original declared function and complete domain contract')
    require(origin['A_actual_observation_binding'] == observation['binding']
        and origin['A_domain_contract_binding'] == contract['binding']
        and origin['input_record_binding'] == source['binding'] and origin['plan_binding'] == plan['binding'],
        'Original observation/input/plan linkage')
    logical = source['original_logical_input']
    require(source['case_id'] == logical['case_id'] == origin['case_id']
        and logical == plan['original_logical_input'] and logical['dtype'] == source['dtype'] == '<f8'
        and logical['shape'] == source['shape']
        and type(logical['bytes']) is int and logical['bytes'] == math.prod(logical['shape'])*8,
        'Original retained case and declared binary64 bytes')
    require(type(logical['sha256']) is str and re.fullmatch('[0-9a-f]{64}', logical['sha256'])
        and type(logical['shape']) is list and logical['shape']
        and all(type(n) is int and n > 0 for n in logical['shape']), 'Original declared input identity')
    require(plan['A_domain_contract']['binding'] == contract['binding']
        and origin['original_operation_source'] == plan['original_operation_source'], 'Original domain implementation identity')
    sources = implementation_sources()
    for description in (plan['A_declared_domain_source'], plan['original_operation_source']):
        require(sources.get(description['path']) == description['sha256']
            and plan['source_sha256'].get(description['path']) == description['sha256'],
            'Original domain/math implementation changed')
    require(origin['actually_evaluated_case_count'] == 1
        and origin['actual_polynomial_argument_observations'] == len(contract['predicates'])
        and origin['standalone_development_observation_not_a_pre_generation_policy_record'] is True,
        'Original measured scope, without fabricated historical policy')
    require(type(observation['maximum_scalar_operations']) is int and 1 <= observation['maximum_scalar_operations'] <= 2**28
        and type(observation['accounted_scalar_work']) is int
        and 0 <= observation['accounted_scalar_work'] <= observation['maximum_scalar_operations'], 'Original finite domain work')
    require(origin['maximum_scalar_operations'] == observation['maximum_scalar_operations']
        and origin['accounted_scalar_work'] == observation['accounted_scalar_work'], 'Original recorded work budget')
    require(all(observation.get(k) is False for k in ('candidate_python_executed','CKKS_outputs_used',
        'encrypted_execution','runtime_or_refresh_domain_certified')), 'Domain observation is not native evidence')
    cells = observation['observations']; predicates = contract['predicates']; results = []
    require(type(cells) is list and len(cells) == len(predicates), 'All measured domain predicates')
    for cell, predicate in zip(cells, predicates):
        require(cell['predicate_id'] == predicate['id'] and cell['reference'] == predicate['reference']
            and cell['expected_interval'] == predicate['interval'], 'Original predicate order and declared bounds')
        if cell['all_finite'] is True:
            require(type(cell['elements']) is int and cell['elements'] > 0
                and all(type(cell[k]) in (int,float) and math.isfinite(cell[k]) for k in ('minimum','maximum'))
                and cell['minimum'] <= cell['maximum'], 'Measured finite extrema, not theorem intervals')
            ok = predicate['interval'][0] <= cell['minimum'] and cell['maximum'] <= predicate['interval'][1]
        else:
            require(cell['all_finite'] is False and cell['minimum'] is cell['maximum'] is None, 'Nonfinite measured domain')
            ok = False
        require(cell['passed'] is bool(ok), 'Domain cell status contradicts actual extrema'); results.append(ok)
    overall = observation['failed_reference_node'] is None and all(results)
    require(observation['passed'] is bool(overall) and origin['actual_domain_passed'] is bool(overall),
        'Domain result contradicts actual predicates')
    annotation = None
    if theorem_annotation is not None:
        value, expected = theorem_annotation
        checked_bound(value, expected)
        require(value['actual57_observation']['binding'] == observation['binding']
            and value['case_id'] == source['case_id'], 'Separate theorem annotation refers to another observation')
        annotation = dict(binding=expected, role='opaque_theorem_metadata_not_measured_extrema',
            never_substituted_for_actual_observations=True)
    return bound(dict(format=ENTRY, case_id=source['case_id'], domain_contract=copy.deepcopy(contract),
        domain_observation=copy.deepcopy(observation), declared_input=copy.deepcopy(logical),
        original_evidence_bindings=copy.deepcopy(expected_bindings), implementation_sources=sources,
        theorem_annotation=annotation, new_request_or_policy_bound=False,
        historical_policy_or_generation_binding_fabricated=False, measured_predicates=len(cells),
        declared_model_shape_revalidation_still_required=True, test_only=test_only,
        actual_encrypted_input_or_CKKS_or_first_delivery_certified=False))


def catalog(entries, expected_bindings):
    require(type(entries) is list and 1 <= len(entries) <= 64 and len(entries) == len(expected_bindings), 'Finite retained domain cases')
    seen = set()
    for entry, expected in zip(entries, expected_bindings):
        checked_bound(entry, expected)
        require(entry['format'] == ENTRY and entry['case_id'] not in seen, 'Distinct measured case identities')
        seen.add(entry['case_id'])
        require(entry['domain_contract'] == entries[0]['domain_contract'] and entry['test_only'] is entries[0]['test_only'],
            'One original domain/function and explicit test scope')
        require(entry['implementation_sources'] == implementation_sources(), 'Domain implementation changed since import')
    return bound(dict(format=CATALOG, entries=copy.deepcopy(entries), test_only=entries[0]['test_only'],
        domain_contract_binding=entries[0]['domain_contract']['binding'],
        actual_cases=len(entries), domain_prefix_reexecutions=0,
        whole_required_case_set_or_native_or_first_delivery_certified=False))


def attach(policy, value, expected_binding):
    """Create a new policy version before generation; never alter the old one."""
    checked_bound(policy); checked_bound(value, expected_binding)
    spec = checked_bound(policy['retained_mathematical_producer'])
    require(value['format'] == CATALOG and policy['independent_math'] == MATH and policy['CKKS'] == CKKS,
        'Original mathematical and CKKS thresholds')
    require(catalog(value['entries'], [e['binding'] for e in value['entries']]) == value, 'Catalog source and entry integrity')
    require('retained_domain_reuse' not in policy, 'Domain reuse selection is immutable once attached')
    require(value['test_only'] is spec['test_only'] and spec['domain']['effective_declared_domain'] is not None
        and value['domain_contract_binding'] == spec['domain']['effective_declared_domain']['binding'], 'Same frozen effective domain')
    path = str(Path(__file__).absolute())
    require(spec['producer_sources'].get(path) == sha(read_bytes(path)), 'Freeze the reuse producer before generation')
    require(all(spec['producer_sources'].get(p) == h for p,h in implementation_sources().items()),
        'New policy must freeze the unchanged original domain/math implementations')
    by_case = {r['case_id']: r for r in spec['input_bindings']}
    for entry in value['entries']:
        row = by_case.get(entry['case_id'])
        require(row is not None and row['declared_input_sha256'] == entry['declared_input']['sha256']
            and entry['domain_contract'] == spec['domain']['effective_declared_domain']
            and entry['domain_observation']['declared_reference_model_sha256'] == spec['declared_reference_model_sha256']
            and entry['domain_observation']['maximum_scalar_operations'] == spec['maximum_scalar_operations'],
            'Retained observation must match the new policy function/input/domain/work exactly')
    selection = bound(dict(format=SELECTION, catalog=copy.deepcopy(value), catalog_binding=expected_binding,
        missing_required_case_ids=[r['case_id'] for r in spec['input_bindings'] if r['case_id'] not in {e['case_id'] for e in value['entries']}],
        selected_before_generation=True, original_case_count=len(spec['input_bindings']),
        historical_request_or_policy_or_generation_binding_created=False))
    unsigned = {k: copy.deepcopy(v) for k,v in policy.items() if k != 'binding'}
    unsigned['retained_domain_reuse'] = selection
    return bound(unsigned)


def rows(policy, indices):
    """Match all three new input identities while retaining the old observation."""
    checked_bound(policy); spec = checked_bound(policy['retained_mathematical_producer'])
    selection = checked_bound(policy['retained_domain_reuse']); value = checked_bound(selection['catalog'], selection['catalog_binding'])
    require(selection['format'] == SELECTION and value['format'] == CATALOG, 'Explicit retained-domain reuse policy')
    original = bound({k:v for k,v in policy.items() if k not in ('binding','retained_domain_reuse')})
    require(attach(original,value,value['binding']) == policy, 'Frozen reuse policy/function/sources changed')
    require(type(indices) is list and 1 <= len(indices) <= 16 and indices == sorted(set(indices))
        and all(type(i) is int and 0 <= i < len(spec['input_bindings']) for i in indices), 'Original per-record subset limit')
    by_case = {e['case_id']:e for e in value['entries']}; result = []
    for index in indices:
        identity = spec['input_bindings'][index]; entry = by_case.get(identity['case_id'])
        require(entry is not None, 'Missing measured case; no hull substitution or shortening required cases')
        result.append(dict(copy.deepcopy(identity), domain_observation=copy.deepcopy(entry['domain_observation']),
            retained_entry_binding=entry['binding'], observation_origin='retained_actual_domain_prefix',
            actual_domain_prefix_reexecuted=False, historical_pre_generation_binding_existed=False))
    return result


def produce(request, selection, samples, raw_cases, policy, output, *, indices):
    """External evaluation only, after the original full input/source revalidation."""
    import retained_math_producer as producer
    from retained_math_evidence import verify_domain_record
    spec, state = producer.check_policy(policy,request,selection,samples,raw_cases)
    selected = rows(policy,indices)
    path = Path(output).absolute()
    require(path.suffix == '.json' and path.parent.resolve() == path.parent and not path.exists(), 'New domain binding record only')
    write_once(path.with_name(path.name+'.attempt.json'),canonical(bound(dict(format='poseidon-retained-domain-reuse-attempt-v1',
        policy_sha256=sha(canonical(policy)),input_indices=indices))))
    record = bound(dict(format=producer.DOMAIN_RECORD, mode='new_policy_binding_of_retained_actual_domain_observations',
        request_id=request['request_id'],policy_sha256=sha(canonical(policy)),producer_spec_binding=spec['binding'],
        domain_contract_binding=spec['domain']['effective_declared_domain']['binding'],producer_sources=spec['producer_sources'],
        declared_reference_model_sha256=spec['declared_reference_model_sha256'],input_indices=indices,inputs=selected,
        retained_domain_reuse_selection_binding=policy['retained_domain_reuse']['binding'],
        candidate_python_executed=False,CKKS_outputs_used=False,encrypted_execution=False,
        runtime_or_refresh_domain_certified=False,test_only=spec['test_only'],new_domain_prefix_executions=0,
        historical_request_policy_generation_or_observation_modified=False))
    producer.check_policy(policy,request,selection,samples,raw_cases); state[0].verify_sources()
    write_once(path,canonical(record)); descriptor=dict(path=str(path),sha256=sha(read_bytes(path)))
    # The original verifier still checks model shape, all predicate/extrema,
    # original work/tolerances and every raw/logical/physical identity.
    verify_domain_record(descriptor,policy,spec)
    return descriptor
