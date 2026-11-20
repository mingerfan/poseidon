"""External Poseidon qualification bound to the exact frozen delivery bytes.

No generation, feedback, candidate repair, compiler substitution or SDK fallback.
Retained evidence is inspectable without executing the compiler or a reference.
Numerical evidence must be supplied by the independent external assessment;
this module never fills missing mathematical outputs with CKKS/reference data.
"""
import copy
import hashlib
import json
from pathlib import Path
import time

from delivery_component import (bound, canonical, checked_bound, read_bytes,
    require, sha, write_once)
from deepseek_provider import public_request

VERSION = 'poseidon-first-source-qualification-binding-v1'


def _json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'Duplicate artifact JSON key')
            result[key] = value
        return result
    def invalid(_):
        raise ValueError('Nonfinite artifact JSON')
    value = json.loads(read_bytes(path).decode('utf-8'), object_pairs_hook=pairs, parse_constant=invalid)
    from public_basis_storage import decode_data_file
    return decode_data_file(value,Path(path).name)


def _relative(root, name):
    require(type(name) is str and name and not Path(name).is_absolute()
            and '..' not in Path(name).parts and '.' not in Path(name).parts,
            'Relative artifact path required')
    path = root/name
    require(path.resolve() == path and path.is_relative_to(root), 'Indirect evidence path')
    return path


def _file(root, name, inventory, expected=None):
    p = _relative(root, name)
    raw = read_bytes(p)
    actual = sha(raw)
    require(expected is None or actual == expected, 'Retained artifact hash changed: '+name)
    inventory[name] = dict(sha256=actual, bytes=len(raw))
    return raw


def bind_qualification(evidence, request, source, *, expected_report_sha256,
                       expected_backend_identity=None, evidence_root=None):
    """Inspect exactly one qualifier attempt, including numerical failures.

    A successful later attempt is never selected. Source/payload checks precede
    compiler/native success observations. The returned receipt records retained
    provenance, not a new compilation or independent acceptance.
    """
    require(type(source) is bytes and bool(source), 'Frozen source bytes required')
    root = Path(evidence).absolute()
    require(root.resolve() == root and root.is_dir(), 'Indirect qualification directory')
    if evidence_root is not None:
        parent = Path(evidence_root).absolute()
        require(root != parent and root.is_relative_to(parent) and parent.resolve() == parent,
                'Qualification outside registered evidence root')
    files = {}
    _file(root, 'report.json', files, expected_report_sha256)
    report = _json(root/'report.json')
    require(report.get('backend') == 'poseidon_cpu', 'No fallback backend')
    require(report.get('request_id') == request['request_id'], 'Qualification request identity')
    require(report.get('qualification', {}).get('generation_performed') is False,
            'Qualifier cannot generate or repair a candidate')
    require(report.get('qualification', {}).get('format') in
            ('poseidon-qualification-session-v1', 'poseidon-native-qualification-session-v1'),
            'Original qualification session required')
    require(report.get('tolerance') == dict(atol=1e-5, rtol=1e-4), 'Original CKKS tolerance')
    identity = report.get('backend_identity')
    require(type(identity) is dict and bool(identity), 'Measured backend identity required')
    require(expected_backend_identity is None or identity == expected_backend_identity, 'Frozen backend changed')
    for field in ('compiler_sha256', 'runtime_sha256'):
        value = report.get(field)
        require(type(value) is str and len(value) == 64 and all(c in '0123456789abcdef' for c in value), 'Backend artifact identity')
    attempts = report.get('attempts')
    require(type(attempts) is list and len(attempts) == 1 and attempts[0].get('index') == 0,
            'Exactly one original qualification attempt required; no success selection')
    attempt = attempts[0]
    require(report.get('failure_layer') != 'integrity' and attempt.get('failure_layer') != 'integrity',
            'Qualification integrity was rejected')
    frozen = report.get('frozen_hashes')
    required_inputs={'request.json', 'model.json', 'weights.npz', 'logical-bindings.json'}
    if report.get('validation_level','numerical')=='numerical':required_inputs.add('arrays.npz')
    require(type(frozen) is dict and required_inputs <= set(frozen), 'Frozen qualifier inputs required')
    for name, expected in frozen.items():
        _file(root, name, files, expected)
    stored_request = public_request(_json(root/'request.json'))
    require(canonical(stored_request) == canonical(public_request(request)), 'Qualifier used another public request')
    if attempt.get('source_parsed') is not True or attempt.get('checked') is not True:
        require(attempt.get('compiled') is not True and attempt.get('executed') is not True,
                'Invalid source cannot claim compilation or execution')
        return bound(dict(format=VERSION, origin='retained_qualification', evidence=str(root),
            source_sha256=sha(source), request_id=request['request_id'], files=files,
            candidate_preserved=False, compiler_input_bound=False, real_compilation_observed=False,
            real_encrypted_execution_observed=False, failure_layer=attempt.get('failure_layer'),
            backend_identity_sha256=sha(canonical(identity)), numerical_evidence_complete=False))
    candidate = _file(root, 'attempt-00/candidate.py', files)
    require(candidate == source, 'Actual qualifier candidate differs from frozen first output')
    payload_name = 'attempt-00/trace-payload.json'
    require(payload_name in frozen, 'Compiler trace input was not frozen')
    payload = _json(root/payload_name)
    require(type(payload) is dict and set(payload) == {'request', 'candidate'}, 'Original trace payload')
    parsed = payload['candidate']
    require(type(parsed) is dict and set(parsed) == {'schema', 'request_id', 'hecate_source'}
            and type(parsed['schema']) is int and parsed['schema'] == 1
            and parsed['request_id'] == request['request_id']
            and parsed['hecate_source'].encode('utf-8') == source, 'Tracer used a replacement DSL')
    require(canonical(public_request(payload['request'])) == canonical(public_request(request)),
            'Tracer used another request')
    artifacts = attempt.get('artifact_hashes', {})
    require(type(artifacts) is dict, 'Compiled artifact inventory')
    for name, expected in artifacts.items():
        _file(root, 'attempt-00/output/'+name, files, expected)
    compiled = attempt.get('compiled') is True
    native = attempt.get('executed') is True
    compiler_command = attempt.get('compile_command')
    if compiled:
        _file(root, 'attempt-00/output/trace-evidence.json', files)
        trace = _json(root/'attempt-00/output/trace-evidence.json')
        require(trace.get('request_id') == request['request_id'] and trace.get('frontend') == 'real_Hecate'
                and trace.get('candidate_python_executed') is False, 'Actual Hecate trace required')
        require(type(compiler_command) is list and compiler_command and compiler_command[0] == '/hecate-opt',
                'Original compiler command required')
        required = {'candidate_trace.mlir', 'lowered.earth.mlir', 'lowered.ckks.mlir',
                    'lowered._hecate_golden.runtime-plan.json'}
        require(required <= set(artifacts), 'Compiled output hashes missing')
        entry = compiler_command[1] if len(compiler_command) > 1 else ''
        require(type(entry) is str and entry.startswith('/out/') and entry[5:] in artifacts,
                'Actual compiler input missing from artifact binding')
    require(not native or compiled, 'Execution without a compiler input binding')
    execution = None
    if native:
        _file(root, 'attempt-00/output/execution.json', files)
        _file(root, 'attempt-00/output/decrypted.npy', files)
        _file(root, 'attempt-00/output/inputs.json', files)
        execution = _json(root/'attempt-00/output/execution.json')
        require(type(execution) is dict and execution.get('backend') == 'poseidon_cpu'
                and execution.get('encrypted_execution') is True
                and execution.get('executor_secret_key') is False
                and execution.get('security') == 'tc128', 'Actual CKKS execution metadata required')
        require(type(attempt.get('comparison')) is dict, 'Original numerical comparison missing')
    # Recheck everything after reading. A receipt never upgrades old SDK scope.
    require(all(sha(read_bytes(_relative(root, name))) == info['sha256'] for name, info in files.items()),
            'Evidence changed during source binding')
    return bound(dict(format=VERSION, origin='retained_qualification', evidence=str(root),
        source_sha256=sha(source), request_id=request['request_id'], files=files,
        candidate_preserved=True, compiler_input_bound=compiled, compiler_input=compiler_command[1] if compiled else None,
        real_compilation_observed=compiled, real_encrypted_execution_observed=native,
        failure_layer=attempt.get('failure_layer'), backend_identity_sha256=sha(canonical(identity)),
        compiler_sha256=report['compiler_sha256'], runtime_sha256=report['runtime_sha256'],
        numerical_evidence_complete=False, numerical_scope='Native observations only; independent math/domain/application evidence still required'))


def verify_qualification_binding(binding, request, source):
    checked_bound(binding)
    require(binding.get('format') == VERSION, 'Qualification source binding format')
    report = binding['files']['report.json']['sha256']
    actual = bind_qualification(binding['evidence'], request, source, expected_report_sha256=report)
    require(actual == binding, 'Qualification source binding changed')
    return binding


class FrozenSourceBackendEvaluator:
    """One external qualifier call under the original shared lock and host guard.

    A separate audited numerical reader must supply all three numerical layers.
    Its absence returns backend_blocked, never a compile-only delivery success.
    """
    def __init__(self, backend, *, batch, checkpoint, frozen_identity, output,
                 numerical_reader=None, test_only=False):
        self.backend = backend
        self.batch = batch
        self.checkpoint = checkpoint
        self.frozen_identity = copy.deepcopy(frozen_identity)
        self.output = Path(output).absolute()
        self.numerical_reader = numerical_reader
        self.test_only = test_only
        self.used = False

    def __call__(self, request, source):
        require(not self.used, 'External qualification cannot retry or repair')
        self.used = True
        require(type(source) is bytes and callable(self.checkpoint), 'Frozen source and host guard')
        require(self.output.parent.resolve() == self.output.parent, 'Indirect evaluator output')
        self.output.mkdir(mode=0o700)
        write_once(self.output/'candidate.py', source)
        write_once(self.output/'request.json', canonical(request))
        start = bound(dict(format='poseidon-first-source-external-call-v1', request_id=request['request_id'],
            source_sha256=sha(source), backend_identity=self.frozen_identity, started_at_ns=time.time_ns(),
            generations=0, feedback_repairs=0, qualification_calls_max=1))
        write_once(self.output/'before-call.json', canonical(start))
        witness = dict(source_sha256=sha(source), compiler_source_sha256=None, compiled_validated=False,
            real_compiler=False, encrypted_execution=False, real_CKKS=False, test_only=self.test_only,
            intermediate_decryptions=None, reference_or_tolerance_changed=False, inputs=[])
        self.checkpoint()
        require(self.backend.identity_manifest() == self.frozen_identity, 'Frozen qualifier identity changed')
        if not self.test_only:
            from normal_cpu_overlay_backend import trusted_backend
            from shared_execution import Batch
            trusted_backend(self.backend,approved_overlay_required=
                'normal_CPU_approved_overlay' in self.frozen_identity.get('backend',{}))
            require(isinstance(self.batch, Batch), 'Original shared batch required')
            row = self.batch.record()
            require(row['plan']['stages']['qualify'] == 1
                    and all(v == 0 for k, v in row['plan']['stages'].items() if k != 'qualify'),
                    'External batch may qualify exactly once; no Provider')
            require(any(self.output != Path(a['path']) and self.output.is_relative_to(a['path']) for a in row['artifacts']),
                    'Unaccounted evaluator output')
        try:
            self.backend.check_request(request, 'numerical')
        except ValueError:
            witness.update(failure_category='backend_blocked', failure_code='SDK_or_request_not_certified')
            write_once(self.output/'result.json', canonical(bound(witness)))
            return witness
        candidate = dict(schema=1, request_id=request['request_id'], hecate_source=source.decode('utf-8'))
        if self.test_only:
            result = self.backend.qualify(request, candidate, 'numerical')
        else:
            from shared_execution import CoordinatedBackend
            result = CoordinatedBackend(self.batch, self.backend).qualify(request, candidate, 'numerical')
        write_once(self.output/'qualification-result.json', canonical(result))
        require(self.backend.identity_manifest() == self.frozen_identity, 'Backend changed during qualification')
        require(read_bytes(self.output/'candidate.py') == source, 'Frozen external input changed')
        self.checkpoint()
        evidence = Path(result['evidence']).absolute()
        proof = bind_qualification(evidence, request, source,
            expected_report_sha256=sha(read_bytes(evidence/'report.json')),
            evidence_root=None if self.test_only else self.batch.coordinator.roots[0])
        write_once(self.output/'source-binding.json', canonical(proof))
        witness.update(compiler_source_sha256=sha(source) if proof['compiler_input_bound'] else None,
            compiled_validated=proof['real_compilation_observed'], real_compiler=proof['real_compilation_observed'],
            encrypted_execution=proof['real_encrypted_execution_observed'], real_CKKS=proof['real_encrypted_execution_observed'],
            backend_binding=proof)
        if not proof['real_compilation_observed']:
            layer = proof.get('failure_layer')
            witness.update(failure_category='contract_rejected' if layer in ('response_parse', 'source_parse', 'static_check') else 'evaluation_failed',
                           failure_code='qualification_'+str(layer or 'incomplete'))
        elif not proof['real_encrypted_execution_observed']:
            witness.update(failure_category='resource_blocked' if proof.get('failure_layer') == 'resource_gate' else 'backend_blocked',
                           failure_code='encrypted_execution_not_completed')
        elif self.numerical_reader is None:
            witness.update(failure_category='backend_blocked', failure_code='independent_numerical_witness_required')
        else:
            # This reader is trusted, separately frozen assessment code, not a plugin.
            extra = self.numerical_reader(request, source, proof)
            required = {'inputs', 'intermediate_decryptions', 'reference_or_tolerance_changed'}
            optional = {'mathematical_evidence','native_output_profile','complete_output_read'}
            require(type(extra) is dict and required <= set(extra) <= required | optional,
                    'Independent numerical reader fields')
            require(('native_output_profile'in extra)==('complete_output_read'in extra),
                    'Complete-output selection and decoded observation required together')
            witness.update(extra)
        verify_qualification_binding(proof, request, source)
        write_once(self.output/'result.json', canonical(bound(witness)))
        return witness
