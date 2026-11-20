"""Delivery boundary: one public request -> one frozen, unedited DSL answer.

There is deliberately no compiler, executor, evaluator or repair callback here.
The existing Application synthesis/qualification loop is a development service.
Live callers retain the existing host admission guard and shared API lease.
"""
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import time

from candidate_contract import MAX_BYTES, canonical, strict_json
from deepseek_provider import DeepSeekProvider, HTTPSTransport, SYSTEM, public_request
from generation_protocol import ProviderError

VERSION = 'poseidon-first-output-delivery-v2'
SOURCE_FILE = 'candidate.py'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def bound(value):
    return dict(value, binding=sha(canonical(value)))


def checked_bound(value, expected=None):
    require(type(value) is dict, 'Bound object required')
    binding = value.get('binding')
    require(binding == sha(canonical({k: v for k, v in value.items() if k != 'binding'})),
            'Object binding changed')
    require(expected is None or binding == expected, 'Unexpected frozen binding')
    return value


def read_bytes(path, limit=32*1024**2):
    path = Path(path).absolute()
    require(path.resolve() == path and not path.is_symlink() and path.is_file(), 'Indirect or absent artifact')
    with path.open('rb') as stream:
        raw = stream.read(limit + 1)
    require(len(raw) <= limit, 'Artifact byte limit')
    return raw


def write_once(path, raw):
    """Exclusive, synced byte preservation; no newline or encoding normalization."""
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, 'O_NOFOLLOW', 0)
    fd = os.open(path, flags, 0o400)
    with os.fdopen(fd, 'wb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def generator_manifest(version, source_paths, *, baseline_binding, capabilities_sha256,
                       evaluation_policy_sha256, provider_config=None, compiler_evaluation_profile=None):
    """Freeze a caller-reviewed source closure; this is not SDK certification.

    The caller must supply the full relevant closure, including planning/rules.
    The mandatory files below prevent changing the delivery/transport contract
    without changing its version identity. Independent cohort sealing is external.
    """
    import candidate_contract
    import deepseek_provider
    import generation_protocol
    mandatory = [Path(m.__file__).absolute() for m in
                 (candidate_contract, deepseek_provider, generation_protocol)] + [Path(__file__).absolute()]
    prompt=SYSTEM
    if compiler_evaluation_profile is not None:
        import compiler_managed_evaluation
        compiler_managed_evaluation.verify_profile(compiler_evaluation_profile)
        mandatory += [Path(compiler_managed_evaluation.__file__).absolute()]
        prompt=deepseek_provider.COMPILER_MANAGED_SYSTEM
    paths = sorted(set([Path(p).absolute() for p in source_paths] + mandatory))
    require(type(version) is str and 0 < len(version) <= 128, 'Generator version')
    require(all(p.suffix in ('.py', '.json', '.md') for p in paths), 'Public source files only')
    for value in (baseline_binding, capabilities_sha256, evaluation_policy_sha256):
        require(type(value) is str and len(value) == 64 and all(c in '0123456789abcdef' for c in value),
                'Frozen dependency hash')
    require(provider_config is None or provider_config.max_calls == 1 and provider_config.provider_retries == 0,
            'Frozen single-generation Provider configuration')
    result=dict(format='poseidon-first-output-generator-v2', version=version,
        frozen_at_ns=time.time_ns(), baseline_binding=baseline_binding,
        source_sha256={str(p): sha(read_bytes(p)) for p in paths},
        system_prompt_sha256=sha(prompt.encode('utf-8')),
        capabilities_sha256=capabilities_sha256, evaluation_policy_sha256=evaluation_policy_sha256,
        provider_configuration=asdict(provider_config) if provider_config is not None else None,
        full_source_closure_review_required=True, independent_cohort_seal_required=True)
    if compiler_evaluation_profile is not None:
        result['compiler_evaluation_profile']=compiler_evaluation_profile
    return bound(result)


def verify_generator(mechanism):
    import candidate_contract
    import deepseek_provider
    import generation_protocol
    checked_bound(mechanism)
    require(mechanism.get('format') == 'poseidon-first-output-generator-v2', 'Generator contract')
    profile=mechanism.get('compiler_evaluation_profile')
    prompt=SYSTEM
    if profile is not None:
        import compiler_managed_evaluation
        compiler_managed_evaluation.verify_profile(profile)
        require(str(Path(compiler_managed_evaluation.__file__).absolute()) in mechanism['source_sha256'],
                'Compiler evaluation configuration absent from source freeze')
        prompt=deepseek_provider.COMPILER_MANAGED_SYSTEM
    require(mechanism['system_prompt_sha256'] == sha(prompt.encode('utf-8')), 'Prompt changed')
    require(type(mechanism['source_sha256']) is dict and bool(mechanism['source_sha256']), 'Source closure')
    for name, expected in mechanism['source_sha256'].items():
        require(sha(read_bytes(name)) == expected, 'Generator source changed')
    mandatory = [Path(m.__file__).absolute() for m in
                 (candidate_contract, deepseek_provider, generation_protocol)] + [Path(__file__).absolute()]
    require(all(str(p) in mechanism['source_sha256'] for p in mandatory), 'Delivery/Provider source absent from closure')


def delivery_plan(slots, *, independent_seal_verified=False):
    """Freeze one output location and one numeric policy for each request.

    All cases share the same generator version while retaining their own input
    sets. The fixed output slot prevents selecting a second response under the
    same mechanism/request identity. Seal provenance is reviewed by the host.
    """
    require(type(slots) is list and bool(slots), 'Frozen request slots required')
    records = {}
    for slot in slots:
        request = public_request(slot['request'])
        key = request['request_id']
        require(key not in records, 'Duplicate delivery request slot')
        checked_bound(slot['policy'])
        output = Path(slot['delivery_path']).absolute()
        require(output.parent.resolve() == output.parent, 'Indirect delivery slot')
        require(slot['split'] in ('development', 'regression', 'acceptance'), 'Delivery split')
        require(type(slot.get('unseen', False)) is bool, 'Unseen flag')
        require(not slot.get('unseen') or slot['split'] == 'acceptance', 'Unseen split')
        records[key] = dict(request_id=key, request_sha256=sha(canonical(request)),
            delivery_path=str(output), policy=slot['policy'], split=slot['split'], unseen=slot.get('unseen', False))
    require(type(independent_seal_verified) is bool, 'Independent seal flag')
    return bound(dict(format='poseidon-first-output-delivery-plan-v1', frozen_at_ns=time.time_ns(),
        slots=records, independent_seal_verified=independent_seal_verified))


def _folder(output):
    p = Path(output).absolute()
    require(p.parent.resolve() == p.parent, 'Indirect delivery parent')
    p.mkdir(mode=0o700)  # Existing attempts, including failed attempts, cannot be overwritten.
    return p


def _preserve_answer(folder, request, mechanism, raw, *, origin, live, metrics):
    receipt = dict(format=VERSION, mode='delivery_generation' if origin == 'provider_first_output' else 'retrospective_capture',
        origin=origin, request_id=request['request_id'], mechanism_binding=mechanism['binding'],
        first_generation_index=0 if origin == 'provider_first_output' else None,
        post_output_repairs=0, real_Agent=live, external_evaluation_performed=False,
        request_sha256=sha(read_bytes(folder/'request.json')), generator_sha256=sha(read_bytes(folder/'generator.json')),
        response_sha256=None, source_sha256=None, source_bytes=0, status='contract_rejected',
        first_output_eligible=origin == 'provider_first_output' and live, provider_metrics=metrics)
    receipt['delivery_plan_sha256'] = sha(read_bytes(folder/'delivery-plan.json')) if (folder/'delivery-plan.json').exists() else None
    if raw is None and origin == 'provider_first_output':
        receipt['status'] = 'provider_failed'
        calls = metrics.get('calls', []) if type(metrics) is dict else []
        if calls and calls[-1].get('error') in ('incomplete_or_refused_response', 'unexpected_response_action'):
            receipt['status'] = 'refused'
    if type(raw) is str:
        response = raw.encode('utf-8')
        require(len(response) <= MAX_BYTES, 'Response byte limit')
        write_once(folder/'response.txt', response)
        receipt['response_sha256'] = sha(response)
        try:
            parsed = strict_json(raw)
            require(type(parsed) is dict and set(parsed) == {'schema', 'request_id', 'hecate_source'}, 'Response fields')
            require(type(parsed['schema']) is int and parsed['schema'] == 1 and parsed['request_id'] == request['request_id'], 'Response identity')
            source = parsed['hecate_source']
            require(type(source) is str and bool(source.strip()), 'Empty DSL')
            source_bytes = source.encode('utf-8')
            # Freeze directly after response-contract parsing, before any DSL checks.
            write_once(folder/SOURCE_FILE, source_bytes)
            receipt.update(source_sha256=sha(source_bytes), source_bytes=len(source_bytes), status='dsl_delivered')
        except (ValueError, TypeError, KeyError, UnicodeError):
            pass
    saved = bound(receipt)
    write_once(folder/'delivery.json', canonical(saved))
    return saved


def generate_once(request, provider, output, *, mechanism, plan=None, batch=None, checkpoint=None):
    """The delivery API. A fresh approved Provider, one generation, zero feedback.

    This API makes no success claim about the delivered DSL. External assessment
    receives the returned binding and must verify it before and after evaluation.
    """
    require(type(provider) is DeepSeekProvider, 'Original approved Provider required')
    require(provider.config.max_calls == 1 and provider.config.provider_retries == 0, 'One delivery attempt, no retries')
    require(provider.generation_attempts == provider.request_attempts == 0, 'Fresh Provider session required')
    require('generate' not in vars(provider), 'Provider generation method must be original')
    verify_generator(mechanism)
    request = public_request(request)
    from compiler_configuration import COMPILER_MANAGED_CONFIGURATION
    if request.get('compiler_configuration',{}).get('name')==COMPILER_MANAGED_CONFIGURATION:
        require(mechanism.get('compiler_evaluation_profile',{}).get('configuration')==request['compiler_configuration'],
                'Freeze the independent compiler evaluation profile before generation')
    elif mechanism.get('compiler_evaluation_profile')is not None:
        require(False,'Compiler-managed mechanism used another request configuration')
    live = isinstance(provider._transport, HTTPSTransport)
    require(not (provider.kind == 'deepseek_api') or live, 'A fixture cannot claim a real Agent')
    frozen_config = mechanism['provider_configuration']
    require(frozen_config is None or frozen_config == asdict(provider.config), 'Provider settings differ from frozen mechanism')
    require(not live or frozen_config is not None and plan is not None, 'Live delivery needs frozen Provider settings and request slots')
    if plan is not None:
        checked_bound(plan)
        require(plan['format'] == 'poseidon-first-output-delivery-plan-v1'
                and sha(canonical(plan)) == mechanism['evaluation_policy_sha256'], 'Delivery plan differs from mechanism freeze')
        slot = plan['slots'].get(request['request_id'])
        require(slot is not None and slot['request_sha256'] == sha(canonical(request)), 'Request absent from frozen plan')
        require(slot['delivery_path'] == str(Path(output).absolute()), 'A frozen request has exactly one delivery slot')
        require(slot['split'] != 'acceptance' or plan['independent_seal_verified'], 'Independent acceptance seal missing')
    if live:
        from shared_execution import Batch
        require(isinstance(batch, Batch) and callable(checkpoint), 'Existing shared batch and host guard required')
        row = batch.record()
        require(row['plan']['source_binding'] == mechanism['binding'], 'Shared plan generator identity')
        require(row['plan']['stages']['api'] == 1 and all(v == 0 for k, v in row['plan']['stages'].items() if k != 'api'),
                'Delivery batch cannot compile, execute or repair')
        target = Path(output).absolute()
        require(any(target != Path(a['path']) and target.is_relative_to(a['path']) for a in row['artifacts']), 'Delivery outside accounted batch')
        checkpoint()
    folder = _folder(output)
    write_once(folder/'request.json', canonical(request))
    write_once(folder/'generator.json', canonical(mechanism))
    if plan is not None:
        write_once(folder/'delivery-plan.json', canonical(plan))
    write_once(folder/'attempt.json', canonical(bound(dict(format=VERSION, started_at_ns=time.time_ns(),
        request_id=request['request_id'], mechanism_binding=mechanism['binding'], configuration=asdict(provider.config),
        generations=1, HTTP_attempts=1, retries=0, feedback=[], compilation=0, encrypted_execution=0))))
    try:
        if live:
            with batch.stage('api'):
                raw = provider.generate(request, [])
        else:
            raw = provider.generate(request, [])
    except ProviderError:
        # The Provider owns redacted failure codes and the consumed HTTP ledger.
        result = _preserve_answer(folder, request, mechanism, None,
            origin='provider_first_output', live=live, metrics=provider.metrics())
        return result
    result = _preserve_answer(folder, request, mechanism, raw,
        origin='provider_first_output', live=live, metrics=provider.metrics())
    if live:
        checkpoint()
    return result


def capture_response(request, raw, output, *, mechanism):
    """Audit retained responses without relabeling them as prospective delivery."""
    verify_generator(mechanism)
    request = public_request(request)
    folder = _folder(output)
    write_once(folder/'request.json', canonical(request))
    write_once(folder/'generator.json', canonical(mechanism))
    return _preserve_answer(folder, request, mechanism, raw,
        origin='retained_response', live=False, metrics=None)


def verify_delivery(folder, expected_binding):
    folder = Path(folder).absolute()
    receipt = checked_bound(strict_json(read_bytes(folder/'delivery.json').decode('utf-8')), expected_binding)
    require(receipt['format'] in ('poseidon-first-output-delivery-v1', VERSION), 'Delivery format')
    request_raw = read_bytes(folder/'request.json')
    require(sha(request_raw) == receipt['request_sha256'], 'Frozen request changed')
    request = json.loads(request_raw)
    generator_raw = read_bytes(folder/'generator.json')
    require(sha(generator_raw) == receipt['generator_sha256'], 'Frozen generator changed')
    generator = checked_bound(json.loads(generator_raw), receipt['mechanism_binding'])
    if receipt.get('delivery_plan_sha256') is not None:
        plan_raw = read_bytes(folder/'delivery-plan.json')
        require(sha(plan_raw) == receipt['delivery_plan_sha256'] == generator['evaluation_policy_sha256'], 'Frozen request slots changed')
        plan = checked_bound(json.loads(plan_raw))
        require(plan['slots'][receipt['request_id']]['request_sha256'] == sha(request_raw), 'Delivery slot request changed')
    require(request['request_id'] == receipt['request_id'], 'Frozen request identity')
    if receipt['response_sha256'] is not None:
        raw = read_bytes(folder/'response.txt', MAX_BYTES)
        require(sha(raw) == receipt['response_sha256'], 'Frozen response changed')
    if receipt['source_sha256'] is not None:
        source = read_bytes(folder/SOURCE_FILE, MAX_BYTES)
        require(sha(source) == receipt['source_sha256'] and len(source) == receipt['source_bytes'], 'Frozen DSL changed')
        parsed = strict_json(raw.decode('utf-8'))
        require(parsed['hecate_source'].encode('utf-8') == source and parsed['request_id'] == receipt['request_id'], 'DSL differs from original response')
    else:
        source = None
        require(not (folder/SOURCE_FILE).exists(), 'Unexpected replacement DSL')
    return receipt, request, source, generator
