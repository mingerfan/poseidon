"""Future prepared single-output delivery, with byte-only post-output checking.

One full backend admission occurs before generation. Compilation/numerical work
is external; this entry never calls an evaluator, qualifier or feedback repair.
"""
import copy
from dataclasses import asdict
from pathlib import Path
import time

import delivery_component as delivery
from delivery_component import bound, canonical, checked_bound, read_bytes, require, sha, write_once
import lossless_host_bound_delivery as host_entry
import completed_delivery_startup as startup
import bounded_generation_context as context
import bounded_generation_transport as transport
from generation_protocol import Config

VERSION = 'poseidon-prepared-bounded-first-delivery-v1'


def source_manifest():
    paths = transport.source_manifest()
    paths.update(startup.source_manifest())
    paths.update(startup.external.source_manifest())
    paths[str(Path(__file__).absolute())] = sha(read_bytes(__file__))
    return paths


def bind(mechanism, selections):
    """Select a new mode before generation; never re-sign an old delivery."""
    startup.verify_selection(mechanism)
    require('prepared_delivery_contract' not in mechanism, 'Select prepared delivery once')
    require(type(selections) is dict and set(selections) == set(mechanism['host_request_bindings']), 'One context per frozen request')
    sources = source_manifest()
    require(all(mechanism['source_sha256'].get(p) == v for p, v in sources.items()), 'Prepared entry not in frozen source closure')
    config = Config(**mechanism['provider_configuration'])
    bodies = {}
    for key, selected in selections.items():
        host = host_entry.resolve_host(mechanism['host_request_bindings'][key], mechanism=mechanism)
        wire = context.verify(selected, host_entry.public_request(host), host=host)
        bodies[key] = transport.freeze(selected, wire, config)
        provenance = selected['verified_origin_provenance']
        require(provenance is None or mechanism['source_sha256'].get(provenance['path']) == provenance['sha256'],
                'Model origin provenance asset absent from generator freeze')
    return bound(dict({k: v for k, v in mechanism.items() if k != 'binding'},
        prepared_delivery_contract=VERSION, prepared_delivery_sources=sources,
        bounded_generation_contexts=copy.deepcopy(selections), frozen_generation_bodies=bodies))


def verify_selection(mechanism):
    startup.verify_selection(mechanism)
    sources = source_manifest()
    require(mechanism.get('prepared_delivery_contract') == VERSION and mechanism.get('prepared_delivery_sources') == sources,
            'Prepared mode/source was not frozen before generation')
    require(all(mechanism['source_sha256'].get(p) == v for p, v in sources.items()), 'Prepared source changed')
    keys = set(mechanism['host_request_bindings'])
    require(keys == set(mechanism['bounded_generation_contexts']) == set(mechanism['frozen_generation_bodies']), 'Prepared request sets differ')


def verify_preserved(folder, expected_binding):
    """Saved-byte verification only; no host reconstruction or backend calls."""
    receipt, wire, source, mechanism = delivery.verify_delivery(folder, expected_binding)
    key = receipt['request_id']
    require(mechanism.get('prepared_delivery_contract') == VERSION, 'Original prepared mechanism required')
    selected = checked_bound(mechanism['bounded_generation_contexts'][key])
    frozen = checked_bound(mechanism['frozen_generation_bodies'][key])
    folder = Path(folder).absolute()
    admission_path = host_entry._admission_path(folder)
    record = checked_bound(host_entry._json(admission_path/'prepared.json'))
    host_record = checked_bound(host_entry._json(admission_path/'admission.json'))
    attempt = checked_bound(host_entry._json(folder/'attempt.json'))
    require(record['format'] == VERSION and record['mechanism_binding'] == mechanism['binding']
            and record['context_binding'] == selected['binding'] and record['body_binding'] == frozen['binding']
            and record['request_id'] == key and record['recorded_at_ns'] <= attempt['started_at_ns']
            and record['test_only'] == host_record['test_only'], 'Prepared admission must precede this output')
    require(host_record['format'] == host_entry.ADMISSION and host_record['request_id'] == key
            and host_record['mechanism_binding'] == mechanism['binding']
            and host_record['host_binding'] == mechanism['host_request_bindings'][key]['binding']
            and host_record['planning_binding'] == mechanism['host_pre_generation_plans'][key]['binding']
            and host_record['host_sha256'] == mechanism['host_request_bindings'][key]['host_canonical_sha256']
            and host_record['public_sha256'] == sha(canonical(wire))
            and host_record['backend_identity'] == mechanism['host_backend_identity']
            and host_record['recorded_at_ns'] <= attempt['started_at_ns']
            and host_record['Provider_calls'] == host_record['compiler_calls'] == host_record['encrypted_calls'] == 0,
            'Original pre-output full-host admission binding')
    require(selected['public_request_sha256'] == sha(canonical(wire))
            and frozen['context_binding'] == selected['binding'] and frozen['request_id'] == key,
            'Saved public/context/body identities')
    require(sha(read_bytes(admission_path/'body.json')) == frozen['actual_body_sha256'], 'Frozen actual wire bytes changed')
    require(attempt['generations'] == attempt['HTTP_attempts'] == 1 and attempt['retries'] == 0
            and attempt['feedback'] == [] and attempt['compilation'] == attempt['encrypted_execution'] == 0
            and attempt['configuration'] == frozen['configuration'] == mechanism['provider_configuration'], 'Single output without feedback')
    require(not receipt['real_Agent'] or record['test_only'] is False, 'Fixture cannot certify live delivery')
    require(receipt['origin'] == 'provider_first_output' and receipt['post_output_repairs'] == 0, 'Unmodified prospective source only')
    return receipt, wire, source, mechanism


def generate_once(host_request, provider, output, *, mechanism, plan, backend,
                  backend_identity, batch=None, checkpoint=None, test_only=False):
    verify_selection(mechanism)
    host, wire, binding, planning = host_entry._slot(host_request, mechanism, plan, output)
    require(type(test_only) is bool and (test_only or not planning['test_only']), 'Fixture cannot authorize live delivery')
    if planning['status'] != 'prepared':
        from planned_delivery_component import _blocked
        return _blocked(wire, mechanism, plan, output, planning, planning['status'])
    key = host['request_id']; selected = mechanism['bounded_generation_contexts'][key]
    context.verify(selected, wire, host=host)
    transport.checked_provider(provider, selected, wire, mechanism['frozen_generation_bodies'][key], test_only=test_only)
    require(backend is not None and backend_identity == mechanism['host_backend_identity'], 'Frozen original backend required')
    if not test_only:
        from normal_cpu_overlay_backend import trusted_backend
        trusted_backend(backend, approved_overlay_required=planning['constraints'].get('SDK_format') == 'poseidon-normal-cpu-approved-overlay-v1')
    require(backend.identity_manifest() == backend_identity, 'Backend changed before host admission')
    try:
        backend.check_request(copy.deepcopy(host), 'numerical')
    except ValueError:
        from planned_delivery_component import _blocked
        return _blocked(wire, mechanism, plan, output, planning, 'backend_blocked')
    require(backend.identity_manifest() == backend_identity, 'Backend changed during host admission')
    path = host_entry._admission_path(output)
    require(path.parent.resolve() == path.parent, 'Indirect admission path')
    if not test_only:
        require(batch is not None and any(path.is_relative_to(a['path']) for a in batch.record()['artifacts']), 'Admission outside accounted batch')
    path.mkdir(mode=0o700)
    write_once(path/'admission.json', canonical(bound(dict(format=host_entry.ADMISSION,
        request_id=key, mechanism_binding=mechanism['binding'], host_binding=binding['binding'],
        host_sha256=binding['host_canonical_sha256'], public_sha256=binding['public_canonical_sha256'],
        planning_binding=planning['binding'], backend_identity=backend_identity, recorded_at_ns=time.time_ns(),
        test_only=test_only, Provider_calls=0, compiler_calls=0, encrypted_calls=0))))
    write_once(path/'prepared.json', canonical(bound(dict(format=VERSION, request_id=key,
        mechanism_binding=mechanism['binding'], context_binding=selected['binding'],
        body_binding=mechanism['frozen_generation_bodies'][key]['binding'],
        recorded_at_ns=time.time_ns(), test_only=test_only))))
    write_once(path/'body.json', context.bodies(selected, wire, provider.config)[1])
    receipt = delivery.generate_once(host, provider, output, mechanism=mechanism, plan=plan, batch=batch, checkpoint=checkpoint)
    verify_preserved(output, receipt['binding'])
    return receipt
