"""Explicit, source-bound public addendum for a future single-output mechanism.

The original request, instruction string, checker and byte limits are unchanged.
Only this version's exact addendum may be used; it contains no candidate/reference.
"""
import copy
from pathlib import Path

from delivery_component import bound, canonical, checked_bound, read_bytes, require, sha
from deepseek_provider import public_request, SYSTEM, MAX_BODY_BYTES
from candidate_contract import MAX_BYTES
from public_weight_context import CONTEXT_BYTES
import public_loop_constraints as loops

VERSION = 'poseidon-bounded-generation-context-v1'
INTRO = ('\nSingle-output delivery: plan the implementation before answering. Return one original DSL '
         'in the required response contract. Compilation, execution, numerical checking and any '
         'feedback repair belong to a separate external development evaluator. No output feedback '
         'or second answer is available in this delivery. Existing public checker constraints: ')


def source_manifest():
    import generation_model_origin
    paths = [Path(__file__).absolute(), Path(loops.__file__).absolute(),
             Path(loops.native_public_loops.__file__).absolute(), Path(generation_model_origin.__file__).absolute()]
    return {str(p): sha(read_bytes(p)) for p in paths}


def _public(origin):
    facts = loops.describe()
    return dict(format=VERSION, native_loop_limits=facts['limits'], meanings=facts['meanings'],
        existing_bounds_only=True, compiler_or_CKKS_feasibility_certified=False,
        mathematical_model_origin=copy.deepcopy(origin))


def addendum(public):
    return INTRO + canonical(public).decode('utf-8')


def _selection(wire, origin, provenance, host_sha256):
    public = _public(origin)
    extra = addendum(public).encode('utf-8')
    # Count the entire new system addendum against the original public-context
    # budget. Moving data to another message cannot evade that budget.
    count = len(canonical(wire)) + len(extra)
    require(count <= CONTEXT_BYTES, 'Combined original public context and complete addendum budget')
    return bound(dict(format=VERSION, request_id=wire['request_id'], public_request_sha256=sha(canonical(wire)),
        public=public, addendum_sha256=sha(extra), combined_public_context_bytes=count,
        public_context_limit=CONTEXT_BYTES, source_sha256=source_manifest(),
        verified_origin_provenance=copy.deepcopy(provenance), verified_host_sha256=host_sha256,
        old_request_or_system_changed=False, post_output_feedback_available=False,
        native_or_first_output_success_certified=False))


def _provenance(descriptor):
    require(type(descriptor) is dict and set(descriptor) == {'path', 'sha256'}, 'Frozen public model provenance asset')
    path = Path(descriptor['path']).absolute()
    require(path.suffix == '.json' and path.name != '.env', 'Model provenance JSON only')
    raw = read_bytes(path)
    require(sha(raw) == descriptor['sha256'], 'Public model provenance changed')
    from candidate_contract import strict_json
    return strict_json(raw.decode('utf-8'))


def freeze(request, *, host=None, provenance=None):
    wire = public_request(copy.deepcopy(request))
    origin = None
    host_hash = None
    if provenance is not None or 'generation_model_origin' in wire:
        require(host is not None and provenance is not None, 'Model origin requires independently retained host provenance')
        require(canonical(public_request(host)) == canonical(wire), 'Model origin host/public relationship')
        from generation_model_origin import build, verify
        value = _provenance(provenance)
        origin = build(value, host)
        if 'generation_model_origin' in wire:
            verify(wire['generation_model_origin'], value, host)
            require(wire['generation_model_origin'] == origin, 'Different selected model origin')
        host_hash = sha(canonical(host))
    else:
        require(host is None, 'Host is only an explicit model-origin verification input')
    return _selection(wire, origin, provenance, host_hash)


def verify(selection, request, *, host=None):
    checked_bound(selection)
    wire = public_request(copy.deepcopy(request))
    origin = selection['public']['mathematical_model_origin']
    provenance = selection['verified_origin_provenance']
    host_hash = selection['verified_host_sha256']
    if origin is None:
        require(provenance is None and host_hash is None and 'generation_model_origin' not in wire,
                'Unverified or omitted model origin')
    else:
        from generation_model_origin import checked_public
        checked_public(origin)
        require(provenance is not None and type(host_hash) is str and len(host_hash) == 64,
                'Origin verification provenance missing')
        _provenance(provenance)
        if 'generation_model_origin' in wire:
            require(wire['generation_model_origin'] == origin, 'Public origin changed')
        if host is not None:
            require(freeze(wire, host=host, provenance=provenance) == selection, 'Host-verified model origin changed')
    expected = _selection(wire, origin, provenance, host_hash)
    require(selection == expected, 'Frozen public constraints or source changed')
    return wire


def bodies(selection, request, config):
    wire = verify(selection, request)
    require(config.max_calls == 1 and config.provider_retries == 0, 'Single output, zero retry configuration')
    base = dict(config.settings(), messages=[dict(role='system', content=SYSTEM),
        dict(role='user', content=canonical(wire).decode('utf-8'))])
    outgoing = copy.deepcopy(base)
    outgoing['messages'][0]['content'] += addendum(selection['public'])
    require(all(len(m['content'].encode('utf-8')) <= MAX_BYTES for m in outgoing['messages']), 'Original message byte limit')
    baseline, actual = canonical(base), canonical(outgoing)
    require(max(len(baseline), len(actual)) <= MAX_BODY_BYTES, 'Original Provider body limit')
    return baseline, actual
