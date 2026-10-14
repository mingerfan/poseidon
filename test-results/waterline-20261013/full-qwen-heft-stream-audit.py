#!/usr/bin/env python3
"""Compare one invocation's immediate pre/post Waterline IR, one line at a time.

The only SSA table retained is the current snapshot's integer SSA -> stable ID
map. Canonical foreground records spool to disk; no full IR text is loaded.
"""
import argparse
from array import array
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import time

OP = re.compile(r'^\s*(%[\w.$-]+) = "([^"]+)"\(([^)]*)\)(.*)$')
POLY = re.compile(r'!ckks\.poly<\s*(\d+)\s*\*\s*(\d+)\s*\*\s*(\d+)\s*>')
ARG = re.compile(r'(%[\w.$-]+):\s*(tensor<[^{}]+>)\s*\{([^}]+)\}')
RID = re.compile(r'(?:, )?runtime\.value_id = \d+ : i64')
WATER = re.compile(r',? runtime\.waterline = "(?:\\.|[^"\\])*"')

def integer(text, name, required=True):
    m = re.search(re.escape(name) + r' = (-?\d+) : i64', text)
    if m is None:
        if required:
            raise ValueError('missing ' + name)
        return None
    return int(m.group(1))

def ssa(token):
    token = token.strip()
    return int(token[1:]) if token[1:].isdigit() else token

def stable(packed):
    return ('t:' if packed & 1 else 'v:') + str(packed >> 1)

def encode(record):
    return (json.dumps(record, separators=(',', ':'), ensure_ascii=True) + '\n').encode()

class Snapshot:
    def __init__(self, side, spool, original=None):
        self.side, self.spool, self.original = side, spool, original
        self.values, self.transfers = {}, array('Q')
        self.maximum_logical_id = -1
        self.counts, self.devices = Counter(), Counter()
        self.hashes = {k: hashlib.sha256() for k in ('foreground', 'initialization', 'execution', 'ct_descriptors')}
        self.raw_hash, self.raw_bytes = hashlib.sha256(), 0
        self.records, self.first_mismatch = 0, None
        self.name = None
        self.done = False

    def record(self, kind, item, phase=None):
        raw = encode([kind, item])
        self.hashes['foreground'].update(raw)
        if phase:
            self.hashes[phase].update(raw)
        self.records += 1
        if self.side == 'before':
            self.spool.write(raw)
        else:
            expected = self.spool.readline()
            if raw != expected and self.first_mismatch is None:
                self.first_mismatch = {'record': self.records, 'expected': expected.decode().strip()[:3000],
                                       'actual': raw.decode().strip()[:3000]}

    def descriptor(self, key, poly, attrs):
        self.hashes['ct_descriptors'].update(encode([stable(key), poly,
            integer(attrs, 'dist.rank'), integer(attrs, 'dist.device')]))
        self.counts['ct_values'] += 1

    def transfer_key(self, runtime_id):
        index = runtime_id - self.original.maximum_logical_id - 1
        if index < 0 or index >= len(self.original.transfers):
            raise ValueError('runtime transfer ValueId outside original layout')
        return (self.original.transfers[index] << 1) | 1

    def line(self, line):
        raw = line.encode()
        self.raw_hash.update(raw)
        self.raw_bytes += len(raw)
        text = line.strip()
        if text.startswith('func.func '):
            if self.name is not None:
                raise ValueError('multiple functions in a dump')
            self.name = re.search(r'@([^ (]+)', text).group(1)
            args = list(ARG.finditer(text))
            if not args:
                raise ValueError('function has no typed placed arguments')
            header = WATER.sub('', text)
            for index, match in enumerate(args):
                token, typename, attrs = match.groups()
                key = index << 1
                self.values[ssa(token)] = key
                self.maximum_logical_id = max(self.maximum_logical_id, index)
                header = header.replace(token + ':', stable(key) + ':', 1)
                poly = POLY.search(typename)
                if not poly or int(poly.group(1)) <= 1:
                    raise ValueError('expected ciphertext argument')
                self.descriptor(key, list(map(int, poly.groups())), attrs)
            self.record('signature', header)
            return
        if text == '}':
            self.done = True
            return
        if text.startswith('return '):
            operands, types = text[7:].split(' : ', 1)
            keys = [stable(self.values[ssa(v)]) for v in operands.split(',')]
            self.record('return', [keys, types])
            self.counts['returns'] += 1
            return
        if not text:
            return
        match = OP.match(line)
        if match is None:
            raise ValueError('unsupported IR line: ' + text[:300])
        output, op, operands, tail = match.groups()
        polys = [list(map(int, m)) for m in POLY.findall(tail)]
        if not polys:
            raise ValueError('operation has no CKKS polynomial type: ' + op)
        if op.startswith('ckks.'):
            logical = integer(tail, 'dist.logical_id')
            runtime = integer(tail, 'runtime.value_id', False)
            if runtime is not None and runtime != logical:
                raise ValueError('CKKS runtime/logical ID mismatch')
            self.maximum_logical_id = max(self.maximum_logical_id, logical)
            key = logical << 1
        elif op == 'dist.transfer':
            transfer = integer(tail, 'transfer_id')
            key = (transfer << 1) | 1
            self.transfers.append(transfer)
            if self.side == 'after' and self.transfer_key(integer(tail, 'runtime.value_id')) != key:
                raise ValueError('ciphertext transfer runtime ID changed')
        elif op == 'dist.prepare_plain':
            if self.side != 'after':
                raise ValueError('pre-Waterline dump already contains Prepare')
            key = self.transfer_key(integer(tail, 'runtime.value_id'))
            self.counts['prepare'] += 1
        elif op in ('dist.release', 'dist.fence'):
            self.counts['lifetime'] += 1
            return
        else:
            raise ValueError('unsupported operation: ' + op)
        self.values[ssa(output)] = key
        self.counts['all_ops'] += 1
        if polys[-1][0] <= 1:
            return
        if not (op.startswith('ckks.') or op == 'dist.transfer'):
            raise ValueError('unexpected ciphertext producer: ' + op)
        inputs = [stable(self.values[ssa(v)]) for v in operands.split(',') if v.strip()]
        params = RID.sub('', tail).strip()
        phase = 'initialization' if op == 'dist.transfer' and 'initialization = true' in tail else 'execution'
        self.record('ct_operation', [phase, stable(key), op, inputs, params], phase)
        self.descriptor(key, polys[-1], tail)
        self.counts['ct_transfer' if op == 'dist.transfer' else 'ct_compute'] += 1
        self.counts[op] += 1
        self.devices[str((integer(tail, 'dist.rank'), integer(tail, 'dist.device')))] += 1

    def summary(self):
        return {'function': self.name, 'counts': dict(self.counts), 'device_operation_counts': dict(self.devices),
                'hashes': {k: v.hexdigest() for k, v in self.hashes.items()}, 'records': self.records,
                'max_logical_id': self.maximum_logical_id, 'ir_bytes': self.raw_bytes,
                'ir_sha256': self.raw_hash.hexdigest(), 'first_mismatch': self.first_mismatch}

def audit(path, spool_path):
    snapshots, current = {}, None
    with Path(path).open() as lines, Path(spool_path).open('w+b') as spool:
        for lineno, line in enumerate(lines, 1):
            if 'IR Dump ' in line and '(plan-plaintext-waterline)' in line:
                side = 'before' if 'IR Dump Before ' in line else 'after' if 'IR Dump After ' in line else None
                if side is None or side in snapshots or (current and not current.done):
                    raise ValueError('duplicate, unknown, or overlapping Waterline dump')
                if side == 'after':
                    if 'before' not in snapshots or not snapshots['before'].done:
                        raise ValueError('after dump has no completed independent before dump')
                    # The second snapshot only needs the original transfer
                    # sequence and maximum logical ID, not its SSA table.
                    snapshots['before'].values.clear()
                    spool.flush(); spool.seek(0)
                current = Snapshot(side, spool, snapshots.get('before'))
                snapshots[side] = current
            elif current and not current.done:
                try:
                    current.line(line)
                except Exception as e:
                    raise ValueError(f'{current.side} dump line {lineno}: {e}') from e
        if set(snapshots) != {'before', 'after'} or not all(s.done for s in snapshots.values()):
            raise ValueError('expected exactly one complete before/after pair from one pass invocation')
        before, after = snapshots['before'], snapshots['after']
        extra = bool(spool.readline())
        summaries = {side: snap.summary() for side, snap in snapshots.items()}
        passed = (before.records == after.records and before.hashes['foreground'].digest() == after.hashes['foreground'].digest()
                  and before.hashes['ct_descriptors'].digest() == after.hashes['ct_descriptors'].digest()
                  and after.first_mismatch is None and not extra and before.counts['ct_compute'] > 0
                  and before.counts['ct_transfer'] > 0 and after.counts['prepare'] > 0)
        return {'passed': passed, 'method': 'one invocation immediate pre/post plan-plaintext-waterline IR; stable logical IDs, transfer IDs and original RuntimePlan transfer layout; exact ordered ciphertext operation/descriptor records',
                'scope': 'foreground CT computation/transfers, parameters, operands/results, rank/device, types/scales/levels, HEFT schedule attrs, argument/return signatures and phase-relative order; excludes plaintext preparation/lifetime and newly attached runtime.value_id metadata',
                'snapshots': summaries, 'spool_has_unmatched_before_records': extra}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dump', required=True)
    parser.add_argument('--summary', required=True)
    parser.add_argument('--spool', required=True)
    args = parser.parse_args()
    start = time.monotonic()
    try:
        result = audit(args.dump, args.spool)
    except Exception as e:
        result = {'passed': False, 'error': str(e)}
    result.update(dump_path=str(Path(args.dump).resolve()), scan_seconds=time.monotonic()-start)
    Path(args.summary).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    return 0 if result['passed'] else 1

if __name__ == '__main__':
    raise SystemExit(main())
