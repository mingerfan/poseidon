"""Count full-slot +/-1 encodings without loading the multi-GB plan again."""
import argparse
import json
from pathlib import Path
from decoded_graph import FULL_UNIT, FULL_NEGATIVE_UNIT

parser = argparse.ArgumentParser()
parser.add_argument('plan', type=Path)
args = parser.parse_args()
counts = {'full_one_encode_count': 0, 'full_minus_one_encode_count': 0}
active = False
record = []
finished = False
with args.plan.open() as stream:
    for line in stream:
        if not active:
            if line == '  "initialization": [\n':
                active = True
            continue
        if line == '  ],\n':
            assert not record
            finished = True
            break
        record.append(line)
        if line in ('    },\n', '    }\n'):
            step = json.loads(''.join(record).rstrip().removesuffix(','))
            if step['kind'] == 'encode' and step['payload']['kind'] == 'bundle':
                content = step['payload']['content']
                if content == FULL_UNIT:
                    counts['full_one_encode_count'] += 1
                elif content == FULL_NEGATIVE_UNIT:
                    counts['full_minus_one_encode_count'] += 1
            record.clear()
assert finished, 'expected a pretty-printed initialization array'
print(json.dumps(counts))
