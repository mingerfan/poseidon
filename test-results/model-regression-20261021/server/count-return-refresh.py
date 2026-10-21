#!/usr/bin/env python3
"""Count Boot results having no later arithmetic consumer in emitted CKKS IR."""
import argparse
import json
from pathlib import Path
import re

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('source', type=Path)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
lines = a.source.read_text().splitlines()
instruction = re.compile(r'^\s*(%[\w]+) = "ckks\.([\w]+)"\(([^)]*)\)')
operand = re.compile(r'%[\w]+')
need_arithmetic = set()
boots = 0
without_arithmetic = 0
management = {'bootstrapc', 'modswitchc', 'rescalec', 'relinearize', 'encode'}
for line in reversed(lines):
    m = instruction.match(line)
    if not m: continue
    value, op, inputs = m.groups()
    needed = value in need_arithmetic
    if op == 'bootstrapc':
        boots += 1
        if not needed: without_arithmetic += 1
    if op not in management or needed:
        need_arithmetic.update(operand.findall(inputs))
result = dict(source=str(a.source), boots=boots,
              boots_without_later_arithmetic=without_arithmetic,
              definition='Boot result reaches Return only through Boot/ModSwitch/Rescale/Relinearize; Add/Mul/Rotate/Negate count as arithmetic',
              scope='Use-def structural diagnosis, not a proposed transformed executable')
a.output.write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result))
