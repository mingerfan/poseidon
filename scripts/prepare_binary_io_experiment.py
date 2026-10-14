#!/usr/bin/env python3
"""Build metadata-only benchmark inputs from independent copies of one plan."""
import argparse
import copy
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('plan', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--copies', type=int, nargs='+', default=[81, 324])
    args = parser.parse_args()
    if any(n < 1 for n in args.copies):
        parser.error('counts must be positive')
    args.output.mkdir(parents=True, exist_ok=True)
    source = json.loads(args.plan.read_bytes())
    value_stride = max(int(v['id']) for v in source['values']) + 1
    phases = ['initialization', 'execution', 'finalization']
    transfer_ids = [int(i['transfer_id']) for phase in phases for i in source[phase] if 'transfer_id' in i]
    transfer_stride = max(transfer_ids, default=0) + 1

    def remap(record, index, field):
        record = copy.deepcopy(record)
        delta = value_stride * index
        if field == 'values':
            record['id'] = str(int(record['id']) + delta)
        else:
            for name in ['output', 'value']:
                if name in record:
                    record[name] = str(int(record[name]) + delta)
            for name in ['inputs', 'outputs']:
                if name in record:
                    record[name] = [str(int(v) + delta) for v in record[name]]
            if 'transfer_id' in record:
                record['transfer_id'] = str(int(record['transfer_id']) + transfer_stride * index)
        return record

    def dump(value):
        return json.dumps(value, separators=(',', ':'), allow_nan=False).encode()

    for copies in args.copies:
        path = args.output / f'plan-c{copies}.json'
        metadata = {k: v for k, v in source.items() if k not in ['values', 'external_inputs', *phases, 'final_outputs']}
        with path.open('xb', buffering=1024 * 1024) as output:
            output.write(dump(metadata)[:-1])
            ordinal = 0
            for field in ['values', 'external_inputs', *phases, 'final_outputs']:
                output.write(b',' + dump(field) + b':[')
                first = True
                for index in range(copies):
                    for record in source[field]:
                        if field in ['external_inputs', 'final_outputs']:
                            transformed = str(int(record) + value_stride * index)
                        else:
                            transformed = remap(record, index, field)
                            if field in phases:
                                transformed['ordinal'] = ordinal
                                ordinal += 1
                        if not first:
                            output.write(b',')
                        output.write(dump(transformed))
                        first = False
                output.write(b']')
            output.write(b'}\n')
        print(json.dumps({'file': str(path), 'copies': copies, 'values': len(source['values']) * copies,
                          'instructions': sum(len(source[p]) for p in phases) * copies,
                          'source_plan': str(args.plan), 'bytes': path.stat().st_size}), flush=True)

if __name__ == '__main__':
    main()
