"""Append scalar paper observations without replacing existing run records."""
import fcntl
import json
import math
import os
from pathlib import Path

STATUSES = {'ok', 'compiler_crash', 'timeout', 'resource_limit', 'unsupported', 'driver_error',
            'no_placement'}
REQUIRED = ('run_id', 'task', 'model', 'system', 'param_group', 'status',
            'status_detail', 'budget_s', 'compile_s', 'peak_rss_gib',
            'boots_compiled', 'boot_sites_static', 'exec_ops', 'pred_latency_s',
            'pred_table', 'ckks_inference_s', 'ckks_scope', 'quality',
            'generation_e2e_s', 'stages_s', 'api_calls', 'source', 'finished_utc')


def empty_record(**values):
    row = dict.fromkeys(REQUIRED)
    row.update(quality=dict(metric=None, split=None, value=None, original=None), stages_s={})
    row.update(values)
    return row


def append_record(path, row):
    if set(REQUIRED) - row.keys() or row['status'] not in STATUSES:
        raise ValueError('Complete paper record and explicit terminal category required')
    if not all(isinstance(row[k], str) and row[k] for k in ('run_id', 'task', 'model', 'system', 'source')):
        raise ValueError('Paper identity and source paths must be explicit')
    text = json.dumps(row, ensure_ascii=False, allow_nan=False)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        stream.seek(0)
        for line in stream:
            if not line.strip():
                continue
            old = json.loads(line)
            if old['run_id'] == row['run_id']:
                if old != row:
                    raise ValueError('Preserve a previously published run identity')
                return False
        stream.write(text + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    return True
