"""Observe actual object storage; alias/copy interventions must change output."""
import hashlib
from benchmark_graph import digest
from seal_artifact_gate import require
from unified_public_contract import normalize

FEATURES=frozenset(('storage.copy_independence','storage.reshape_view','storage.transpose_view'))

def check(source,request,reference,events,feature,*,max_spans,max_attempts):
    from unified_public_coverage import fingerprint
    candidates=[];seen=set()
    for event in events:
        position=tuple(event['span'])
        if 'event.object_storage' in event['features'] and feature in event['features'] and position not in seen:
            candidates.append(event);seen.add(position)
    attempts=0
    for event in candidates[:max_spans]:
        attempts+=1;require(attempts<=max_attempts,'Public influence work bound')
        probe=dict(span=event['span'],kind='copy_alias' if feature=='storage.copy_independence' else 'view_detach')
        try:expanded=normalize(source,request,_storage_probe=probe);actual=fingerprint(expanded,request)
        except (ValueError,TypeError,KeyError,IndexError,ZeroDivisionError,OverflowError):continue
        require(len(actual)==len(reference),'Changed influence ABI')
        delta=max((abs(a-b) for a,b in zip(actual,reference)),default=0.)
        if delta>1e-9:
            source_hash=hashlib.sha256(source.encode()).hexdigest()
            return dict(span=event['span'],scope='object_storage_alias_distinction',typed_context=event['facts'],
                        intervention=dict(kind=probe['kind'],probe=probe),
                        intervention_sha256=digest(dict(source_sha256=source_hash,probe=probe)),
                        changed_source_sha256=source_hash,
                        changed_normalized_sha256=expanded['construction']['normalized_sha256'],
                        max_probe_delta=delta,evidence='observable_storage_alias_or_copy_difference_not_all_input_proof'),attempts
    return None,attempts
