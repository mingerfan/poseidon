"""Dtype context + numeric consumer, or initialized allocation shape witnesses.

Dtype aliases do not have distinct mathematical values. This records the actual
float64 constructor context and separately proves its result has finite influence.
"""
import hashlib
import math
from benchmark_graph import digest
from seal_artifact_gate import require
from unified_public_contract import normalize

FEATURES=frozenset(('attr.double','attr.float64','call.empty'))
SCOPES={'attr.double':'typed_float64_constructor_output',
        'attr.float64':'typed_float64_constructor_output',
        'call.empty':'initialized_object_allocation_shape'}


def check(source,request,reference,events,feature,*,max_spans,max_attempts):
    from unified_public_coverage import fingerprint
    kind='object_allocation' if feature=='call.empty' else 'numeric_constructor'
    candidates=[];seen=set()
    for event in events:
        position=tuple(event['span'])
        if 'event.'+kind in event['features'] and feature in event['features'] and position not in seen:
            candidates.append(event);seen.add(position)
    attempts=0
    for event in candidates[:max_spans]:
        attempts+=1;require(attempts<=max_attempts,'Public influence work bound')
        if feature=='call.empty' and (not event['facts']['shape'] or math.prod(event['facts']['shape'])==0):continue
        probe=dict(span=event['span'],kind='allocation_shape' if feature=='call.empty' else 'numeric_constructor_result')
        try:
            expanded=normalize(source,request,_storage_probe=probe)
            actual=fingerprint(expanded,request)
        except (ValueError,TypeError,KeyError,IndexError,ZeroDivisionError,OverflowError):
            continue
        require(len(actual)==len(reference),'Changed influence ABI')
        delta=max((abs(a-b) for a,b in zip(actual,reference)),default=0.)
        if delta>1e-9:
            source_hash=hashlib.sha256(source.encode()).hexdigest()
            return dict(span=event['span'],scope=SCOPES[feature],typed_context=event['facts'],
                        intervention=dict(kind=probe['kind'],probe=probe),
                        intervention_sha256=digest(dict(source_sha256=source_hash,probe=probe)),
                        changed_source_sha256=source_hash,
                        changed_normalized_sha256=expanded['construction']['normalized_sha256'],
                        max_probe_delta=delta,
                        evidence='typed_context_and_constructor_result_influence_not_dtype_alias_numeric_effect'),attempts
    return None,attempts
