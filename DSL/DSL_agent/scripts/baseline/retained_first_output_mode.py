"""Explicit future retained-reference mode, selected before single delivery.

This module generates no candidate itself and performs no post-output numeric
work. Its new delivery entry only delegates to the frozen original Provider path.
"""
import copy
from pathlib import Path
import re

from delivery_component import bound, canonical, checked_bound, read_bytes, require, sha
import lossless_host_bound_delivery as host_entry
import prepared_first_delivery as prepared

VERSION = 'poseidon-retained-first-output-mode-v1'
REAL = 'original-real-slot-qualification-v1'
COMPLEX = 'program-complex-final-output-v1'


def source_manifest():
    import retained_math_producer
    import retained_math_evidence
    import retained_math_domains
    import retained_dual_reference_rows
    import native_dual_reference_evidence
    import first_output_numerical_evidence
    import shared_execution
    paths = prepared.source_manifest()
    for module in (retained_math_producer, retained_math_evidence, retained_math_domains,
                   retained_dual_reference_rows, native_dual_reference_evidence,
                   first_output_numerical_evidence, shared_execution):
        path = Path(module.__file__).absolute(); paths[str(path)] = sha(read_bytes(path))
    for path in (Path(__file__).absolute(), Path(__file__).with_name('retained_external_assessment.py').absolute()):
        paths[str(path)] = sha(read_bytes(path))
    return paths


def native_mode(abi, *, external_batch_name, native_evidence_root):
    require(abi in (REAL, COMPLEX), 'Explicit audited native ABI selection required')
    require(type(external_batch_name) is str and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}', external_batch_name),
            'Frozen external batch name')
    root = Path(native_evidence_root).absolute()
    require(root.parent.resolve() == root.parent and root.name not in ('.env', '.', '..'), 'Direct external evidence parent')
    return bound(dict(format='poseidon-retained-native-mode-v1', abi=abi,
        external_batch_name=external_batch_name, native_evidence_root=str(root),
        complex_bridge_is_not_compiler_layout_input_host_or_resource_proof=True))


def _numeric(policy, request, mechanism):
    from first_output_assessment import MATH, CKKS
    from retained_math_producer import SPEC
    from retained_math_evidence import _sources
    checked_bound(policy); spec = checked_bound(policy['retained_mathematical_producer'])
    require(spec['format'] == SPEC and spec['request_id'] == request['request_id']
            and spec['request_sha256'] == sha(canonical(request))
            and spec['request_model_sha256'] == sha(canonical(request['model']))
            and spec['whole_model_required'] is policy['whole_model_required']
            and type(spec['test_only']) is bool and policy['independent_math'] == MATH and policy['CKKS'] == CKKS,
            'Frozen retained policy/request/threshold identity')
    _sources(policy, spec)
    require(all(mechanism['source_sha256'].get(p) == v for p, v in spec['producer_sources'].items()),
            'Complete retained mathematical sources absent from generator freeze')
    require(policy['inputs'] == [r['physical_input_sha256'] for r in spec['input_bindings']]
            and len(set(policy['inputs'])) == len(policy['inputs'])
            and policy['frozen_domain_sha256'] == sha(canonical(spec['domain']))
            and policy['original_model_sha256'] == spec['original_model_sha256'], 'Frozen input/function/domain scope')
    return spec


def _record(native, numeric, spec, mechanism):
    checked_bound(native)
    require(native == native_mode(native['abi'], external_batch_name=native['external_batch_name'],
        native_evidence_root=native['native_evidence_root']), 'Native mode fields changed')
    reason = ('complex_compiler_layout_input_host_adapter_not_yet_audited' if native['abi'] == COMPLEX
              else 'original_real_slot_input_group_limit' if len(numeric['inputs']) > 16 else None)
    status = 'backend_blocked' if native['abi'] == COMPLEX else 'resource_blocked' if reason else 'prepared'
    return bound(dict(native_mode=native, numeric_policy_sha256=sha(canonical(numeric)),
        mathematical_producer_binding=spec['binding'], native_backend_identity_sha256=sha(canonical(mechanism['host_backend_identity'])),
        generation_status=status, blocking_reason=reason,
        fixture_policy=spec['test_only'], retained_input_groups=len(numeric['inputs'])))


def bind(mechanism, policy, modes):
    """No old response can acquire this new selection retroactively."""
    prepared.verify_selection(mechanism); checked_bound(policy)
    require('retained_first_output_contract' not in mechanism, 'Retained mode selected once before generation')
    require(sha(canonical(policy)) == mechanism['evaluation_policy_sha256'], 'Frozen request policy changed')
    keys = set(mechanism['host_request_bindings'])
    require(type(modes) is dict and keys == set(modes) == set(policy['slots']), 'One ABI per complete frozen request set')
    sources = source_manifest()
    require(all(mechanism['source_sha256'].get(p) == v for p, v in sources.items()), 'New mode absent from generator source freeze')
    records = {}
    for key, native in modes.items():
        request = host_entry.resolve_host(mechanism['host_request_bindings'][key], mechanism=mechanism)
        numeric = policy['slots'][key]['policy']; spec = _numeric(numeric, request, mechanism)
        require(policy['slots'][key]['request_sha256'] == mechanism['host_request_bindings'][key]['public_canonical_sha256'],
                'Frozen public slot changed')
        records[key] = _record(native, numeric, spec, mechanism)
    return bound(dict({k: v for k, v in mechanism.items() if k != 'binding'}, retained_first_output_contract=VERSION,
        retained_first_output_sources=sources, retained_first_output_slots=records))


def verify_selection(mechanism, policy):
    prepared.verify_selection(mechanism); checked_bound(policy)
    require(mechanism.get('retained_first_output_contract') == VERSION and
        mechanism.get('retained_first_output_sources') == source_manifest(), 'Retained mode was not frozen before this delivery')
    require(sha(canonical(policy)) == mechanism['evaluation_policy_sha256'], 'Frozen external policy changed')
    require(set(mechanism['retained_first_output_slots']) == set(policy['slots']) == set(mechanism['host_request_bindings']),
            'Frozen retained request set changed')
    # Recheck frozen identities without reconstructing the full host again.
    # prepared.generate_once owns the one actual full-host/backend admission.
    from retained_math_evidence import _sources
    for key, selected in mechanism['retained_first_output_slots'].items():
        numeric = checked_bound(policy['slots'][key]['policy']); spec = checked_bound(numeric['retained_mathematical_producer'])
        _sources(numeric, spec)
        require(spec['request_id'] == key and spec['request_sha256'] == mechanism['host_request_bindings'][key]['host_canonical_sha256']
                and spec['request_model_sha256'] == mechanism['host_pre_generation_plans'][key]['model_sha256']
                and all(mechanism['source_sha256'].get(p) == v for p, v in spec['producer_sources'].items()),
                'Frozen retained host/model/source identity changed')
        require(selected == _record(selected['native_mode'], numeric, spec, mechanism), 'Retained ABI/backend/producer selection changed')


def generate_once(host_request, provider, output, *, mechanism, plan, backend, backend_identity,
                  batch=None, checkpoint=None, test_only=False):
    """Delivery only. Unavailable native modes are recorded before any API call."""
    verify_selection(mechanism, plan)
    selected = mechanism['retained_first_output_slots'][host_request['request_id']]
    require(type(test_only) is bool and (test_only or not selected['fixture_policy']), 'Fixture math cannot authorize live delivery')
    if selected['generation_status'] != 'prepared':
        host, wire, binding, planning = host_entry._slot(host_request, mechanism, plan, output)
        from planned_delivery_component import _blocked
        return _blocked(wire, mechanism, plan, output, planning, selected['generation_status'])
    return prepared.generate_once(host_request, provider, output, mechanism=mechanism, plan=plan, backend=backend,
        backend_identity=backend_identity, batch=batch, checkpoint=checkpoint, test_only=test_only)
