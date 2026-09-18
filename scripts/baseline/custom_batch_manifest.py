"""Bounded data-only custom graph cohort. No credentials, Torch, or API access."""
import hashlib
import json
from pathlib import Path

from model_graph import validate_graph


def load_manifest(path, expected_sha256=None):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 1024**2:
        raise ValueError('Custom manifest must be a regular JSON file at most 1 MiB')
    raw = path.read_bytes()
    if len(raw) > 1024**2:
        raise ValueError('Custom manifest size limit')
    digest = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError('Custom manifest changed before execution')

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('Duplicate manifest JSON key')
            result[key] = value
        return result

    def nonfinite(value):
        raise ValueError('Non-finite manifest JSON number')

    try:
        data = json.loads(raw, object_pairs_hook=pairs, parse_constant=nonfinite)
    except (RecursionError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError('Invalid custom manifest JSON') from error
    if type(data) is not dict or set(data) != {'schema', 'cases'} or type(data['schema']) is not int or data['schema'] != 1:
        raise ValueError('Custom manifest requires schema=1 and cases only')
    cases = data['cases']
    if type(cases) is not list or not 1 <= len(cases) <= 96:
        raise ValueError('Custom manifest requires 1..96 explicit model graphs')
    ids = set()
    for case in cases:
        validate_graph(case)
        if case['id'] in ids:
            raise ValueError('Duplicate custom model id')
        ids.add(case['id'])
    rows = [dict(descriptor=case, benchmark_family='custom_graph', status='pending') for case in cases]
    return rows, data, digest
