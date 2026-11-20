"""Persist a completion endpoint after required artifacts and registry close."""
import os
from pathlib import Path
import time

from .r10_paper_metrics import write_json


def artifact_bytes(directory):
    total = 0
    for parent, _, names in os.walk(directory):
        for name in names:
            try:
                total += (Path(parent) / name).stat().st_size
            except FileNotFoundError:
                pass
    return total


def persist_case_completion(out, *, round_label, started_ns, status, timing,
                            required_artifact_bytes, clock=time.perf_counter_ns):
    """The endpoint includes required persistence and excludes this marker write."""
    ended_ns = clock()
    marker = dict(round=round_label, status=status, started_ns=started_ns, ended_ns=ended_ns,
                  complete_case_elapsed_ns=ended_ns-started_ns,
                  complete_case_elapsed_seconds=(ended_ns-started_ns)/1e9,
                  elapsed_scope='case entry through successful required-artifact and registry persistence; completion marker write excluded',
                  required_artifacts_persisted=True, registry_persisted=True,
                  required_artifact_bytes=required_artifact_bytes, case_artifact_bytes=None,
                  timing_status=timing['status'], exclusive_attribution_ready=timing['attribution_ready'],
                  overlap_and_timing_diagnostics=timing['diagnostics'],
                  phase_spans=[{key: row.get(key) for key in ('event_id', 'stage_id', 'parent_id',
                               'inclusive_ns', 'exclusive_ns', 'status')} for row in timing['spans']])
    target = Path(out) / 'completion.json'
    for _ in range(12):
        write_json(target, marker)
        total = artifact_bytes(out)
        if marker['case_artifact_bytes'] == total:
            return marker
        marker['case_artifact_bytes'] = total
    raise RuntimeError('Completion marker size did not stabilize')
