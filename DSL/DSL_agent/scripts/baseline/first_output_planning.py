"""Generic planning before delivery, using the existing layout/feature services.

No model names select routes. No candidate, compiler, native result or numerical
reference is consulted. Numeric capability evidence remains distinct from SDK
metadata and from an admitted logical layout.
"""
import copy
from pathlib import Path
import time

from delivery_component import bound, canonical, checked_bound, read_bytes, require, sha
from first_output_backend import _json

FORMAT = 'poseidon-pre-generation-plan-v1'


def backend_constraints(sdk_path, *, sdk_sha256, negative_boundary=None):
    """Read frozen public SDK facts; this cannot grant a numerical certificate."""
    path = Path(sdk_path).absolute()
    require(path.suffix == '.json' and sha(read_bytes(path)) == sdk_sha256, 'Frozen SDK descriptor changed')
    sdk = checked_bound(_json(path))
    from native_sdk_manifest import ready_for_agent_qualification
    require(sdk.get('format') == 'poseidon-runtime-toolchain-native-v5', 'Explicitly audited SDK constraint adapter required')
    readiness = sdk['native']['readiness']
    evidence = {str(path): sdk_sha256}
    artifacts = {}
    refresh = []
    if negative_boundary is not None:
        require(type(negative_boundary) is dict and set(negative_boundary) == {'path', 'sha256'}, 'Negative boundary descriptor')
        negative_path = Path(negative_boundary['path']).absolute()
        require(negative_path.suffix == '.json' and sha(read_bytes(negative_path)) == negative_boundary['sha256'], 'Negative boundary changed')
        boundary = checked_bound(_json(negative_path))
        profile = boundary['profile']
        require(profile['numeric_gate'] == 'closed' and profile['numerically_certified_refresh_profiles'] == 0,
                'Expected observed negative refresh boundary')
        candidate = sdk['library_overlay']['candidate']
        library = Path(candidate['path']).absolute()
        require(library.name.startswith('libposeidon') and library.suffix == '.so', 'Original SDK library path')
        library_hash = sha(read_bytes(library, 128*1024**2))
        require(profile['library_sha256'] == library_hash == candidate['sha256']
                == sdk['artifacts']['poseidon_library'], 'Failure boundary belongs to another library')
        artifacts[str(library)] = library_hash
        evidence[str(negative_path)] = negative_boundary['sha256']
        refresh.append(dict(profile=profile, boundary_binding=boundary['binding'],
            scope='exact tested profile only; other profiles remain uncertified, not inferred failed'))
    return bound(dict(format='poseidon-frozen-generation-backend-constraints-v1', evidence_sha256=evidence,
        artifact_sha256=artifacts,
        SDK_binding=sdk['binding'], SDK_format=sdk['format'], numeric_policy=sdk['native']['numeric_policy'],
        normal_Application_ready=ready_for_agent_qualification(sdk) and readiness['normal_Application_profile'],
        metadata_attestation=readiness['compiler_and_origin_attestation'],
        native_completion_observed=readiness['standalone_native_runtime_measured'],
        certified_refresh_profiles=[], observed_failed_refresh_profiles=refresh,
        source_of_authority='original SDK readiness function plus frozen public boundary; no promotion', test_only=False))


def verify_constraints(value):
    checked_bound(value)
    require(value.get('format') == 'poseidon-frozen-generation-backend-constraints-v1', 'Backend constraints format')
    require(type(value.get('test_only')) is bool and type(value.get('normal_Application_ready')) is bool,
            'Explicit backend constraint scope')
    require(type(value.get('evidence_sha256')) is dict and all(Path(p).suffix == '.json'
            and sha(read_bytes(p)) == h for p, h in value['evidence_sha256'].items()), 'Backend evidence changed')
    require(type(value.get('artifact_sha256', {})) is dict and all(Path(p).name.startswith('libposeidon')
            and Path(p).suffix == '.so' and sha(read_bytes(p, 128*1024**2)) == h
            for p, h in value.get('artifact_sha256', {}).items()), 'Frozen backend artifact changed')
    if not value['test_only']:
        if value.get('source_of_authority') == 'poseidon-normal-CPU-scoped-canary-admission-v1':
            from normal_cpu_overlay_planning import verify
            verify(value)
        else:
            require(value['normal_Application_ready'] is False and value['certified_refresh_profiles'] == [],
                    'SDK metadata cannot grant numeric execution authority')
    return value


def source_authority():
    from application_planning.layout import source_authority as retained_authority
    from application_planning.resources import source_authority as resource_authority
    return dict(layout=retained_authority(), resources=resource_authority(),
        new_sources={str(Path(__file__).absolute()): sha(read_bytes(__file__))})


def _features(model):
    from benchmark_graph import validate
    from application_planning.resources import logical_graph
    checked = validate(model)
    facts = logical_graph(model, checked)
    uses = {}
    for node in model['nodes']:
        for name in node['inputs']: uses[name] = uses.get(name, 0)+1
    by_value = {v: node for node in model['nodes'] for v in node['outputs']}
    affine = []
    for node in model['nodes']:
        if node['op'] != 'batch_norm' or not node['inputs']: continue
        parent = by_value.get(node['inputs'][0])
        if parent is not None and parent['op'] in ('linear', 'conv1d', 'conv2d') and uses[node['inputs'][0]] == 1:
            affine.append(dict(producer=parent['id'], consumer=node['id'],
                decision='public candidate for affine fusion; source implementation still belongs to Agent'))
    facts['affine_fusion_candidates'] = affine
    facts['fanout_values'] = sum(count > 1 for count in uses.values())
    facts['scope'] = 'public logical graph only; syntactic depth is not a CKKS-level or refresh estimate'
    return facts


def prepare_plan(model, constraints, *, options=None, candidate_limit=8, wall_seconds=120,
                 memory_budget_bytes=4*1024**3, model_origin=None, test_only=False):
    """Select among registered layouts by generic costs before any Provider call."""
    require(type(test_only) is bool and (test_only or not constraints['test_only']), 'Fixture capability cannot authorize live planning')
    verify_constraints(constraints)
    require(type(candidate_limit) is int and 1 <= candidate_limit <= 8
            and type(wall_seconds) is int and 1 <= wall_seconds <= 120, 'Finite pre-generation planning budget')
    from application_planning.backend_parameters import native_address_limit
    require(type(memory_budget_bytes) is int and 1 <= memory_budget_bytes <= native_address_limit(), 'Original native memory ceiling')
    from application_planning.layout import candidate_options, request_resources, _score
    from application_policy import prepare
    from component_contract import OPTION_KEYS
    require(options is None or type(options) is dict and set(options) <= OPTION_KEYS, 'Existing explicit preparation options only')
    options = copy.deepcopy(options or {})
    source = source_authority(); started = time.monotonic()
    specs = candidate_options(model, options, candidate_limit)
    rows, requests, seen = [], {}, {}
    for index, selection in enumerate(specs):
        require(time.monotonic()-started < wall_seconds, 'Pre-generation planning wall budget exhausted')
        row = dict(index=index, options=selection, status='rejected', request_id=None,
                   resources=None, score=None, failure='layout_binding_rejected')
        try:
            request, _ = prepare(model, 'poseidon_cpu', 'numerical', selection, model_origin=model_origin)
            resources, stages = request_resources(request)
            key = request['request_id']
            row.update(status='admitted' if key not in seen else 'duplicate', request_id=key,
                       resources=resources, score=_score(resources), failure=None)
            if key not in seen:
                seen[key] = index; requests[key] = request
        except (ValueError, TypeError, KeyError, IndexError, OverflowError, RecursionError):
            pass
        rows.append(row)
    admitted = [row for row in rows if row['status'] == 'admitted']
    require(admitted, 'No registered logical layout fits this request')
    selected = min(admitted, key=lambda row: (row['score'], row['index']))
    request = requests[selected['request_id']]
    layout = request['layout']; cfg = request['compiler_configuration']
    minimum_payload = (len(layout['inputs'])+len(layout['auxiliary_ciphertexts']))*layout['input_slot_period']*8
    refresh_calls = cfg.get('native_Boot_invocations_per_process', 0)
    require(type(refresh_calls) is int and refresh_calls >= 0, 'Declared refresh requirement')
    failures = []
    if not constraints['normal_Application_ready']: failures.append('normal_SDK_not_certified')
    if refresh_calls and not constraints['certified_refresh_profiles']:
        failures.append('required_refresh_has_no_numeric_certificate')
    if minimum_payload > memory_budget_bytes: failures.append('input_float64_payload_exceeds_memory_budget')
    status = ('backend_blocked' if any(x != 'input_float64_payload_exceeds_memory_budget' for x in failures)
              else 'resource_blocked' if failures else 'prepared')
    require(source_authority() == source, 'Planning source changed')
    result = bound(dict(format=FORMAT, model_sha256=sha(canonical(model)), request_id=request['request_id'],
        request_sha256=sha(canonical(request)), constraints=constraints, source_authority=source,
        candidate_options=specs, candidates=rows, selection=selected,
        public_model_origin=request.get('generation_model_origin'),
        features=_features(model), status=status, failure_reasons=failures, test_only=test_only,
        memory_budget_bytes=memory_budget_bytes, minimum_input_float64_payload_bytes=minimum_payload,
        required_refresh_calls_per_process=refresh_calls,
        CKKS_level_consumption=None, refresh_schedule=None, actual_runtime_peak_bytes=None,
        limitations='layout and payload facts do not establish CKKS feasibility, optimized depth or peak memory',
        Agent_receives='existing validated model, constants, selected layout and mandatory request rules; no new wire fields',
        compiler_calls=0, encrypted_calls=0, Provider_calls=0, post_output_feedback=0))
    return request, result


def verify_plan(value, request):
    checked_bound(value); verify_constraints(value['constraints'])
    require(value['format'] == FORMAT and value['request_id'] == request['request_id']
            and value['request_sha256'] == sha(canonical(request))
            and value['model_sha256'] == sha(canonical(request['model'])), 'Frozen planning request/model changed')
    require(value['source_authority'] == source_authority(), 'Frozen planning code changed')
    require(value['selection']['request_id'] == request['request_id']
            and value.get('public_model_origin') == request.get('generation_model_origin')
            and value['features'] == _features(request['model']), 'Frozen planning selection/features changed')
    require(value['status'] in ('prepared', 'backend_blocked', 'resource_blocked')
            and (value['status'] == 'prepared') == (not value['failure_reasons']), 'Planning status boundary')
    require(value['CKKS_level_consumption'] is value['refresh_schedule'] is value['actual_runtime_peak_bytes'] is None,
            'Unknown physical feasibility cannot become an execution certificate')
    return value
