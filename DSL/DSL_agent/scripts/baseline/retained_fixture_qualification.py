"""External development compilation of one unchanged retained candidate.

Only the input preparation is replaced by verified, literal retained bytes.
The existing session, sandbox probe, validation adapter and compiler are used.
This is neither Agent generation nor numerical/first-delivery qualification.
"""
import copy
import json
from pathlib import Path
import tempfile

from delivery_component import bound, canonical, checked_bound, read_bytes, require, sha, write_once

INPUTS = ('model.json', 'weights.npz', 'arrays.npz', 'request.json', 'logical-bindings.json')


def retained_inputs(descriptor, request, source, *, evidence_root):
    """Validate all donor bytes before returning any bytes for a fresh directory."""
    from first_output_backend import _json
    from deepseek_provider import public_request
    require(type(descriptor) is dict and set(descriptor) == {'path', 'sha256'},
            'Independently frozen retained source-binding descriptor')
    proof_bytes = read_bytes(descriptor['path'], 4 * 1024**2)
    require(sha(proof_bytes) == descriptor['sha256'], 'Retained source-binding bytes changed')
    proof = checked_bound(_json(Path(descriptor['path'])))
    root = Path(proof['evidence']).absolute()
    parent = Path(evidence_root).absolute()
    require(root.resolve() == root and parent.resolve() == parent and root.is_dir()
            and root != parent and root.is_relative_to(parent), 'Direct retained evidence root')
    require(type(source) is bytes and source and proof['source_sha256'] == sha(source)
            and proof['request_id'] == request['request_id'], 'Exact retained source/request identity')
    require(proof['candidate_preserved'] is True and proof['compiler_input_bound'] is True
            and proof['real_compilation_observed'] is True
            and proof['real_encrypted_execution_observed'] is False,
            'A compiled-only retained donor, without numerical promotion')
    values = {}
    for name in (*INPUTS, 'attempt-00/candidate.py'):
        item = proof['files'][name]
        require(type(item) is dict and set(item) == {'sha256', 'bytes'}, 'Complete retained file descriptor')
        raw = read_bytes(root / name, 4 * 1024**2)
        require(len(raw) == item['bytes'] and sha(raw) == item['sha256'],
                'Retained input bytes changed: ' + name)
        values[name] = raw
    require(values.pop('attempt-00/candidate.py') == source, 'Retained candidate must remain byte-identical')
    stored_request = _json(root / 'request.json')
    require(canonical(public_request(stored_request)) == canonical(public_request(request)),
            'Retained fixtures belong to another public request')
    require(_json(root / 'model.json') == request['model']
            and _json(root / 'logical-bindings.json') == request['layout'],
            'Retained model/layout differs from the request')
    receipt = bound(dict(format='poseidon-retained-fixture-byte-intake-v1', donor=copy.deepcopy(descriptor),
        donor_binding=proof['binding'], donor_evidence=str(root), request_id=request['request_id'],
        source_sha256=sha(source), files={name: copy.deepcopy(proof['files'][name]) for name in INPUTS},
        model_or_reference_forwards=0, array_deserializations=0, new_generation_or_repair=False,
        mathematical_or_CKKS_or_first_delivery_certified=False))
    return values, receipt


def install_inputs(result, values):
    result = Path(result)
    require(result.resolve() == result and result.is_dir() and not any(result.iterdir()),
            'Fresh empty direct qualification directory')
    require(type(values) is dict and set(values) == set(INPUTS)
            and all(type(v) is bytes for v in values.values()), 'Exactly the five retained byte inputs')
    for name in INPUTS:
        write_once(result / name, values[name])
    return {name: sha(values[name]) for name in INPUTS}


def qualify_retained(backend, donor, request, source, *, artifact_observer, native_observer=None):
    """One original fixed-candidate session; never call its reference-building factory.

    native_observer, when provided, wraps one original sandbox invocation with
    external passive observation. It cannot choose or modify its command.
    The original metadata auditor is deliberately left to the outer controller.
    """
    from candidate_contract import ReplayProvider, run_feedback_loop
    from component_contract import reconstruct_request
    from unified_graph_contract import validate_candidate
    from qualification_session import QualificationSession, SOURCE, read_json
    from hecate_python_env import WORK
    from numerical_results import dump
    from platform_config import identity, require_python_packages
    from native_execution_slots import native_slot
    import normal_cpu_overlay_backend as overlay_adapter
    import normal_cpu_overlay_sandbox as sandbox

    reconstruct_request(request)
    candidate = dict(schema=1, request_id=request['request_id'], hecate_source=source.decode('utf-8'))
    validate_candidate(candidate, request)
    backend.check_request(request, 'compiled')
    import numpy as np
    import torch
    require_python_packages(torch, np)
    torch.set_num_threads(2)
    values, intake = retained_inputs(donor, request, source, evidence_root=WORK / 'results')
    result = Path(tempfile.mkdtemp(prefix='candidate-replay-', dir=WORK / 'results'))
    artifact_observer(result)
    frozen = install_inputs(result, values)
    write_once(result / 'RETAINED_FIXTURE_INTAKE.json', canonical(intake))
    intake_hash = sha(read_bytes(result / 'RETAINED_FIXTURE_INTAKE.json'))
    report = dict(validation_level='compiled', feedback_policy='legacy-v1', platform_identity=identity(),
        status='running', provider='scripted_replay', agent_calls=0, backend='poseidon_cpu',
        poseidon_gpu_validated=False, attempts=[], artifact_replay_only=True,
        llm_generation_validated=False, rule_baseline_status='not_requested_by_component',
        external_retained_fixture_intake=dict(receipt_binding=intake['binding'],
            original_reference_building_factory_calls=0, original_fixed_candidate_sessions=1,
            model_or_reference_forwards=0, first_delivery_evaluation=False))
    measured = backend._evidence()[1]
    identity_frozen = backend.identity_manifest()
    adapter_sources = overlay_adapter._sources()
    own_source = sha(read_bytes(__file__))

    class RetainedOverlaySession(QualificationSession):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._overlay_record = dict(SDK=copy.deepcopy(backend.overlay), identity=identity_frozen,
                adapter_source_sha256=adapter_sources, original_fixed_candidate_session=True, post_output_repairs=0)
            self.report['normal_CPU_approved_overlay'] = copy.deepcopy(self._overlay_record)

        def verify_frozen(self):
            super().verify_frozen()
            require(self.report['normal_CPU_approved_overlay'] == self._overlay_record
                    and backend.identity_manifest() == identity_frozen
                    and overlay_adapter._sources() == adapter_sources and backend._evidence()[1] == measured,
                    'Original overlay admission/source identity changed')
            require(sha(read_bytes(__file__)) == own_source
                    and sha(read_bytes(result / 'RETAINED_FIXTURE_INTAKE.json')) == intake_hash,
                    'External retained adapter/intake changed')

        def native_run(self, *args, **kwargs):
            context, _ = backend._evidence()
            require('selected_library' not in kwargs, 'Selected library belongs to the original SDK')
            if self.helper_environment is not None:
                kwargs['helpers'] = True
            kwargs['selected_library'] = context['library']
            def invoke():
                with native_slot(WORK / 'cache/agent-native-slots', metrics=self.report['native_resources']):
                    return sandbox.run(*args, **kwargs)
            return invoke() if native_observer is None else native_observer(invoke, args, copy.deepcopy(kwargs))

    phase = 'qualification_setup'
    try:
        helpers = None
        if 'upstream_helpers' in request:
            from upstream_candidate_helpers import verify_sources
            from poly_dependencies import verify as verify_dependency
            helpers = dict(sources=verify_sources(backend='poseidon_cpu'), dependency=verify_dependency())
            report['upstream_helper_environment'] = copy.deepcopy(helpers)
        session = RetainedOverlaySession(result, request, report, frozen, read_json(SOURCE / 'fixtures.json'),
            request['compiler_configuration']['waterline'], helper_environment=helpers, validation_level='compiled')
        from application_capabilities import assess_request
        report['capability_report'] = assess_request(request)
        phase = 'sandbox_preflight'
        session.preflight()
        phase = 'fixed_candidate_validation'
        def record(index, raw, feedback):
            attempt = result / ('attempt-%02d' % index)
            write_once(attempt / 'response.txt', raw.encode('utf-8'))
            dump(attempt / 'feedback.json', feedback)
            dump(result / 'report.json', report)
        report['loop'] = run_feedback_loop(request, ReplayProvider([json.dumps(candidate)]),
            session.evaluate, record, max_repairs=0)
        report['status'] = report['loop']['status']
        session.verify_frozen()
        require((result / 'attempt-00/candidate.py').read_bytes() == source,
                'Compiler candidate differs from retained source')
    except Exception as error:
        report.update(status='infrastructure_failed', failure_layer=phase, diagnostic=str(error))
    finally:
        report['frozen_hashes'] = frozen
        dump(result / 'report.json', report)
    return (0 if report['status'] == 'passed' else 1), report, result
