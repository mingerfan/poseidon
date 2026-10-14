"""Typed object unary witnesses; never infer cell types from source spelling."""
import hashlib
from benchmark_graph import digest
from seal_artifact_gate import require
from object_unary_exercises import perturb
from unified_public_contract import normalize

CELL_KINDS={"cipher_cells":"cipher","plain_cells":"plain",
            "public_cells":"number","boolean_cells":"bool"}
FEATURES=frozenset((*CELL_KINDS,"operator.UAdd","operator.USub","call.UAdd","call.USub",
                    "fresh.UAdd","fresh.USub","input_view"))


def check(source,request,reference,events,feature,*,max_spans,max_attempts):
    from unified_public_coverage import fingerprint
    candidates=[];seen=set()
    for event in events:
        position=tuple(event['span'])
        if ('event.object_unary' in event['features'] and feature in event['features']
                and position not in seen):
            candidates.append(event);seen.add(position)
    attempts=0
    for event in candidates[:max_spans]:
        attempts+=1;require(attempts<=max_attempts,'Public influence work bound')
        probe=None
        try:
            if feature in CELL_KINDS:
                # The existing trusted probe changes only results from cells of
                # this actual interpreter type, leaving other result cells intact.
                changed=source;probe=dict(span=event['span'],kind=CELL_KINDS[feature])
                intervention=dict(kind='typed_unary_result',probe=probe)
            else:
                fresh=feature.startswith('fresh.')
                changed=perturb(source,event,fresh=fresh)
                intervention=dict(kind='fresh_storage_vs_alias' if fresh else 'unary_result')
            expanded=normalize(changed,request,_unary_probe=probe)
            actual=fingerprint(expanded,request)
        except (ValueError,TypeError,KeyError,IndexError,ZeroDivisionError,OverflowError):
            continue
        require(len(actual)==len(reference),'Changed influence ABI')
        delta=max((abs(a-b) for a,b in zip(actual,reference)),default=0.)
        if delta>1e-9:
            change_hash=hashlib.sha256(changed.encode()).hexdigest()
            return dict(span=event['span'],intervention=intervention,
                        changed_source_sha256=change_hash,
                        intervention_sha256=digest(dict(source_sha256=change_hash,intervention=intervention)),
                        changed_normalized_sha256=expanded['construction']['normalized_sha256'],
                        max_probe_delta=delta,
                        evidence='finite_typed_result_or_storage_influence_not_per_cell_or_all_input_proof'),attempts
    return None,attempts
