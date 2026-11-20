"""Choose an admitted existing request representation before Agent generation.

Mandatory rules, host/wire limits and backend constraints are unchanged. Optional
guidance may be omitted when it does not fit. No response or feedback is read.
"""
import copy
from pathlib import Path
import time

from delivery_component import bound, canonical, require, sha, read_bytes
from first_output_planning import prepare_plan, verify_constraints
from generation_profile_selection import select, verify

VERSION = 'poseidon-bounded-context-preparation-v1'


class PreparationRejected(ValueError):
    def __init__(self, report):
        self.report = report
        super().__init__('No existing request representation fits the frozen context/resource limits')


def prepare(model, constraints, *, options=None, candidate_limit=8, wall_seconds=120):
    require(type(candidate_limit) is int and 1 <= candidate_limit <= 8, 'Finite representation candidate budget')
    require(type(wall_seconds) is int and 1 <= wall_seconds <= 120, 'Finite representation wall budget')
    verify_constraints(constraints)
    from application_planning.layout import candidate_options
    from component_contract import OPTION_KEYS
    from deepseek_provider import public_request
    from public_weight_context import TRUSTED_BYTES
    require(options is None or type(options) is dict and set(options) <= OPTION_KEYS,
            'Existing preparation options only')
    base = copy.deepcopy(options or {})
    profile = select(model)
    verify(profile, model)
    if 'generation_guidance' in base:
        guidance = [base['generation_guidance']]
    else:
        guidance = [profile['preparation_options']['generation_guidance'], None]
    slots = max(1, candidate_limit // len(guidance))
    specs = candidate_options(model, base, slots)
    candidates = [dict(spec, generation_guidance=g) for g in guidance for spec in specs][:candidate_limit]
    rows, accepted = [], []
    started = time.monotonic()
    for index, spec in enumerate(candidates):
        if time.monotonic() - started >= wall_seconds:
            rows.append(dict(index=index, options=spec, status='resource_blocked', failure='planning_wall_limit'))
            break
        row = dict(index=index, options=spec, status='rejected')
        try:
            host, plan = prepare_plan(model, constraints, options=spec, candidate_limit=1,
                                      wall_seconds=max(1, int(wall_seconds - (time.monotonic()-started))))
            host_bytes = len(canonical(host)) + 1
            require(host_bytes <= TRUSTED_BYTES, 'Original complete host storage limit')
            wire = public_request(copy.deepcopy(host))
            require(wire['request_id'] == host['request_id'], 'Original public projection request identity')
            row.update(status=plan['status'], request_id=host['request_id'], host_bytes=host_bytes,
                       public_bytes=len(canonical(wire)), score=plan['selection']['score'],
                       selected_guidance=host.get('generation_guidance', {}).get('version'))
            if plan['status'] == 'prepared':
                accepted.append((row, host, plan))
        except (ValueError, TypeError, KeyError, IndexError, OverflowError, RecursionError) as exc:
            row.update(failure=str(exc), failure_type=type(exc).__name__)
        rows.append(row)
    selected = min(accepted, key=lambda item:(item[0]['score'], item[0]['public_bytes'], item[0]['index'])) if accepted else None
    report = bound(dict(format=VERSION, model_sha256=sha(canonical(model)), constraints_binding=constraints['binding'],
        preferred_generation_profile=profile, candidates=rows, candidate_limit=candidate_limit,
        selected=None if selected is None else selected[0], status='prepared' if selected else 'blocked',
        selector_sha256=sha(read_bytes(Path(__file__).absolute())),
        optional_guidance_fallback='existing optional guidance only; mandatory rules and all storage limits retained',
        actual_provider_configuration=profile['provider_configuration'], model_name_dispatch=False,
        compiler_or_numeric_feedback_used=False, API_calls=0, compiler_calls=0, encrypted_calls=0,
        numeric_certificate=False, completion_guaranteed=False))
    if selected is None:
        raise PreparationRejected(report)
    return selected[1], selected[2], report
