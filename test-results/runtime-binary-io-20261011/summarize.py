#!/usr/bin/env python3
import json
import statistics
from pathlib import Path

root = Path(__file__).resolve().parent
records = [json.loads(line) for line in (root / 'loads.jsonl').read_text().splitlines()]
groups = {}
for record in records:
    groups.setdefault((record['mode'], record.get('copies', 0)), []).append(record)
summary = []
for (mode, copies), runs in sorted(groups.items()):
    entry = {'mode': mode, 'copies': copies, 'runs': len(runs), 'bytes': runs[0]['bytes'],
             'records': runs[0].get('instructions', runs[0].get('entries'))}
    for field in ['load_seconds', 'read_seconds', 'hash_seconds', 'parse_build_seconds',
                  'peak_rss_bytes', 'verify_seconds', 'verify_peak_rss_bytes']:
        if field in runs[0]:
            values = [r[field] for r in runs]
            entry[field] = {'median': statistics.median(values), 'min': min(values), 'max': max(values)}
    if 'verify_seconds' in runs[0]:
        entry['load_plus_verify_seconds'] = statistics.median(r['load_seconds'] + r['verify_seconds'] for r in runs)
    summary.append(entry)
(root / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
for row in summary:
    print(row['mode'], row['copies'], 'MB', row['bytes'] / 1e6,
          'load', row['load_seconds']['median'], 'parse', row['parse_build_seconds']['median'],
          'MiB', row['peak_rss_bytes']['median'] / 2 ** 20,
          'load+verify', row.get('load_plus_verify_seconds'))
