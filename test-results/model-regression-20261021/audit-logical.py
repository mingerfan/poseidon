#!/usr/bin/env python3
"""Audit compiler-produced logical CKKS types without materializing constants."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re


def audit(path, factor, anchor, lower):
    text = path.read_text()
    poly = re.compile(r'!ckks\.poly<(\d+) \* (\d+) \* (\d+)>')
    header = next(line for line in text.splitlines() if 'func.func ' in line)
    base = int(re.search(r'init_level = (\d+)', header)[1])
    counts = Counter()
    rq = mq = 0
    boot_inputs, boot_outputs = Counter(), Counter()
    min_level = anchor
    outputs = []
    for line in text.splitlines():
        op = re.search(r'"ckks\.([a-zA-Z0-9_]+)"', line)
        if not op:
            if line.strip().startswith('return '):
                outputs = [dict(components=int(c), scale_log2=int(s),
                                physical_level=anchor-(base-int(l))*factor)
                           for c, s, l in poly.findall(line)]
            continue
        name = op[1]
        counts[name] += 1
        types = [(int(c), int(s), int(l)) for c, s, l in poly.findall(line)]
        if not types:
            raise ValueError('CKKS operation without poly types: '+line)
        for c, _, l in types:
            physical = anchor-(base-l)*factor
            if c == 2:
                min_level = min(min_level, physical)
                if physical < lower or physical > anchor:
                    raise ValueError(f'ciphertext outside [{lower},{anchor}]: {line}')
        if name in ('rescalec', 'modswitchc'):
            cost = (types[0][2]-types[-1][2])*factor
            if cost <= 0:
                raise ValueError('non-positive Q consumption: '+line)
            if name == 'rescalec': rq += cost
            else: mq += cost
        if name == 'bootstrapc':
            boot_inputs[(anchor-(base-types[0][2])*factor, types[0][1])] += 1
            boot_outputs[(anchor-(base-types[-1][2])*factor, types[-1][1])] += 1
            if types[-1][1] != 45 or types[-1][2] != base:
                raise ValueError('Boot output differs from level22/scale45')
    states = lambda counter: [dict(level=l, scale_log2=s, count=n)
                              for (l, s), n in sorted(counter.items())]
    return dict(source=str(path), stage='unplaced logical CKKS; nominal scales',
                physical_anchor=anchor, physical_lower_bound=lower,
                levels_per_logical_level=factor, logical_base=base,
                boot_count=counts['bootstrapc'], rescale_q=rq, modswitch_q=mq,
                minimum_physical_level=min_level, output_states=outputs,
                operations=dict(counts), boot_input_states=states(boot_inputs),
                boot_output_states=states(boot_outputs), state_audit_passed=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('source', type=Path)
    p.add_argument('--factor', type=int, default=1)
    p.add_argument('--anchor', type=int, default=22)
    p.add_argument('--lower', type=int, default=4)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    result = audit(a.source, a.factor, a.anchor, a.lower)
    a.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['boot_count','rescale_q','modswitch_q',
                                           'minimum_physical_level','output_states']}))


if __name__ == '__main__':
    main()
