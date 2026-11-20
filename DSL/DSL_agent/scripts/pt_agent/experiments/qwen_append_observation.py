"""Bounded latest scalar observations at original DAG append boundaries."""
from inspect import isfunction
import json
import os
from pathlib import Path
import time
import weakref


INTERVAL_NODES = 250000
MAX_SNAPSHOT_BYTES = 2048
SCHEME = 'append-250000-latest-scalars-v1'


def install_append_observation(active, recorder, graph_sequence):
    """Wrap only this instance, preserving its original bound call signature.

    The owner is weak. In particular, neither a bound original method nor a
    recorder/lifetime is captured by the installed function.
    """
    if recorder is None:
        return False
    if type(graph_sequence) is not int or graph_sequence < 1:
        raise ValueError('Observation requires a positive primitive graph sequence')
    original = type(active)._append
    if '_append' in active.__dict__ or not isfunction(original):
        raise ValueError('Observation requires an original unbound append method')
    target = Path(recorder.path).parent/'dag-append-progress.json'
    temporary = target.with_name(target.name+'.tmp')
    owner_ref = weakref.ref(active)
    enabled = True

    def append(kind, args, *, is_input=False):
        nonlocal enabled
        owner = owner_ref()
        if owner is None:
            raise RuntimeError('Observed DAG owner has been released')
        result = original(owner, kind, args, is_input=is_input)
        if enabled:
            nodes = len(owner.kind)
            if nodes % INTERVAL_NODES == 0:
                context = owner.op_context
                valid_context = context is None or type(context) is int
                record = dict(schema_version=1, kind='dag_append_progress', pid=os.getpid(),
                    graph_sequence=graph_sequence, sampled_ns=time.perf_counter_ns(),
                    interval_nodes=INTERVAL_NODES,
                    sampling_point='after_append_before_cse_record_updates',
                    nodes_after_append=nodes, numbering_entries_at_append=len(owner.numbering),
                    constant_names_at_append=len(owner._constants),
                    constant_identity_entries_at_append=len(owner._constant_objects),
                    op_context=context if valid_context else None,
                    op_context_type_valid=valid_context, complete_lowering_claim=False,
                    layer_progress=None)
                serialized = json.dumps(record, separators=(',', ':'), allow_nan=False)+'\n'
                if len(serialized.encode('utf-8')) > MAX_SNAPSHOT_BYTES:
                    enabled = False
                else:
                    try:
                        temporary.write_text(serialized)
                        temporary.replace(target)
                    except InterruptedError:
                        raise
                    except OSError:
                        # Preserve original arithmetic/async exceptions. A
                        # failed optional metadata write disables observation.
                        enabled = False
        return result

    # Do not use wraps(bound_method): __wrapped__ would retain this owner.
    active._append = append
    return True
