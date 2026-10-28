#!/usr/bin/env python3
"""Audit one-prefix pruning; oracle-feasible rejections are recorded as failures.

Replay checks intended plaintext semantics, not CKKS encoding accuracy. In
particular it does not hide or compensate low-precision encoded-one alignment.
"""
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
import re
import subprocess
import tempfile


class Graph:
    def __init__(self):
        self.ops = []
        self.outputs = []

    def op(self, kind, *inputs):
        name = f'v{len(self.ops)}'
        self.ops.append((name, kind, inputs))
        return name

    def source(self, exhausted=0):
        ct = f'tensor<1x!earth.ci<0 * {exhausted}>>'
        pt = f'tensor<1x!earth.pl<0 * {exhausted}>>'
        lines = [f'module {{ func.func @graph(%x: {ct}, %y: {ct}) -> (' +
                 ', '.join([ct] * len(self.outputs)) + ') {',
                 f'%p = "earth.constant"() {{value = dense<1.01> : tensor<1xf64>, rms_var = 1.01 : f64}} : () -> {pt}']
        for name, kind, inputs in self.ops:
            attrs = ' <{offset = array<i64: 1>}>' if kind == 'rotate' else ''
            operands = ', '.join('%' + v for v in inputs)
            types = ', '.join(pt if v == 'p' else ct for v in inputs)
            lines.append(f'%{name} = "earth.{kind}"({operands}){attrs} : ({types}) -> {ct}')
        lines += ['return ' + ', '.join('%' + v for v in self.outputs) + ' : ' +
                  ', '.join([ct] * len(self.outputs)), '}}']
        return '\n'.join(lines) + '\n'

    def expected(self, x=.25, y=.75):
        values = dict(x=x, y=y, p=1.01)
        for name, kind, inputs in self.ops:
            a = [values[v] for v in inputs]
            values[name] = (a[0] * a[1] if kind == 'mul' else
                            a[0] + a[1] if kind == 'add' else
                            -a[0] if kind == 'negate' else a[0])
        return [values[v] for v in self.outputs]

    def minimum_boots(self, budget, boot_scale, exhausted, moduli):
        # Enumerate ALL depth-boundary subsets (including boundary zero).
        # Small DAGs keep this independent of the compiler's DP recurrence.
        needed = set(self.outputs)
        active = []
        for row in reversed(self.ops):
            if row[0] in needed:
                active.append(row)
                needed.update(row[2])
        active.reverse()
        depth = dict(x=0, y=0, p=0)
        uses = {v: [] for v in depth}
        for name, kind, inputs in active:
            depth[name] = max(depth[v] for v in inputs) + (kind == 'mul')
            uses[name] = []
            for v in inputs:
                uses[v].append(name)
        maximum = max(depth.values())
        depth['return'] = maximum + 1
        for v in self.outputs:
            uses[v].append('return')
        # Unary scheduling is the compiler's documented depth-cut policy.
        for name, kind, _ in reversed(active):
            if kind in ('rotate', 'negate') and uses[name]:
                depth[name] = min(depth[user] for user in uses[name])
        ordered = sorted(active, key=lambda row: depth[row[0]])
        best = math.inf
        anchor = 4 + budget
        def normalize(level, scale, nominal):
            while level <= anchor and scale - math.log2(moduli[anchor-level]) >= 40:
                scale -= math.log2(moduli[anchor-level])
                level += 1
                nominal = math.ceil(scale)
            return level, scale, nominal
        refreshed = normalize(0, boot_scale, boot_scale)
        for mask in range(1 << maximum):
            states = dict(x=(exhausted, 40., 40), y=(exhausted, 40., 40))
            cost, valid = 0, True
            for current_depth in range(maximum + 2):
                if current_depth and mask & (1 << (current_depth - 1)):
                    cut = current_depth - 1
                    crossing = [v for v in states if any(depth[u] > cut for u in uses[v])]
                    cost += len(crossing)
                    for v in crossing:
                        states[v] = refreshed
                for name, kind, inputs in ordered:
                    if depth[name] != current_depth:
                        continue
                    operands = [states[v] for v in inputs if v != 'p']
                    level = max(e for e, _, _ in operands)
                    nominal = max(n for _, _, n in operands)
                    scale = max(s + (nominal-n if kind == 'add' and len(operands) == 2 else 0)
                                for _, s, n in operands)
                    if kind == 'mul':
                        scale = sum(s for _, s, _ in operands) + (40 if 'p' in inputs else 0)
                        nominal = sum(n for _, _, n in operands) + (40 if 'p' in inputs else 0)
                        level, scale, nominal = normalize(level, scale, nominal)
                    if level > budget:
                        valid = False
                        break
                    states[name] = (level, scale, nominal)
                if not valid or cost >= best:
                    break
            if valid:
                best = min(best, cost)
        return best


def cases():
    result = []
    g = Graph(); a = g.op('mul', 'x', 'p'); b = g.op('mul', a, 'p'); c = g.op('mul', b, 'p')
    g.outputs = [a, b, c]; result.append(('cp-double-rescale', g, False))
    g = Graph(); a = g.op('mul', 'x', 'p'); b = g.op('mul', 'y', 'p'); c = g.op('mul', a, b)
    g.outputs = [c]; result.append(('cc-double-rescale', g, False))
    g = Graph(); a = g.op('mul', g.op('mul', 'x', 'p'), 'p')
    b = g.op('mul', g.op('mul', 'y', 'p'), 'p'); g.outputs = [g.op('mul', a, b)]
    result.append(('cc-scale60', g, False))
    g = Graph(); a = g.op('mul', 'x', 'p'); b = g.op('mul', a, a); c = g.op('mul', b, b)
    g.outputs = [c]; result.append(('shared-square', g, False))
    g = Graph(); a = g.op('mul', 'x', 'p'); b = g.op('mul', a, 'p'); c = g.op('mul', 'y', 'p')
    d = g.op('add', b, c); e = g.op('mul', d, 'x'); f = g.op('mul', e, 'p')
    g.outputs = [f, c]; result.append(('unequal-depth-residual', g, False))
    g = Graph(); a = g.op('mul', 'x', 'p'); b = g.op('mul', a, 'p'); c = g.op('mul', b, 'p')
    d = g.op('add', c, 'x'); g.outputs = [g.op('mul', d, 'p')]
    result.append(('equal-nominal-residual', g, False))
    g = Graph(); a = g.op('negate', 'x'); b = g.op('add', a, 'y'); c = g.op('mul', b, 'p')
    for _ in range(7):
        c = g.op('mul', c, 'p')
    d = g.op('rotate', a); e = g.op('mul', c, d)
    g.outputs = [e, b]; result.append(('depth-zero-unary-fanout', g, False))
    g = Graph(); g.outputs = [g.op('rotate', g.op('negate', 'x')), 'y']
    result.append(('unary-only', g, False))
    g = Graph(); g.outputs = [g.op('mul', 'x', 'p')]
    result.append(('exhausted-input', g, True))
    g = Graph(); a = g.op('mul', 'x', 'x'); b = g.op('mul', a, a)
    g.outputs = [b]; result.append(('exhausted-shared-input', g, True))
    g = Graph(); a = g.op('mul', 'x', 'p'); g.outputs = [g.op('mul', a, a)]
    result.append(('one-q-before-double-rescale', g, 'two-left'))
    g = Graph(); a = 'x'
    for _ in range(10):
        a = g.op('mul', a, 'p')
    g.outputs = [g.op('negate', 'x')]; result.append(('dead-branch', g, False))
    for seed in range(6):
        rng = random.Random(seed); g = Graph(); previous = 'x'; available = ['x', 'y']
        for i in range(12):
            kind = 'add' if i % 3 == 0 else 'mul'
            rhs = rng.choice(available) if kind == 'add' or i % 4 == 0 else 'p'
            previous = g.op(kind, previous, rhs); available.append(previous)
        g.outputs = [previous]; result.append((f'mixed-{seed}', g, False))
    return result


def check_plan(plan, ir, graph, spec, exhausted):
    assert plan['format_version'] == 5
    assert len(plan['external_inputs']) == 2
    assert len(plan['final_outputs']) == len(graph.outputs)
    desc = {v['id']: v for v in plan['values']}
    nominal = {}
    for line in ir.splitlines():
        match = re.search(r'runtime.value_id = (\d+) : i64.* -> tensor<\d+x!ckks.poly<\d+ \* (\d+) \* (\d+)>>', line)
        if match:
            nominal[match[1]] = int(match[2])
    value = dict(zip(plan['external_inputs'], (.25, .75)))
    for v in plan['external_inputs']:
        nominal[v] = 40
        assert desc[v]['level'] == spec['boot_profiles'][0]['output_level'] - exhausted
    boot = spec['boot_profiles'][0]
    counts = dict(boots=0, rescale_q=0, modswitch_q=0, double_rescales=0)
    producer = {}
    for step in itertools.chain(plan['initialization'], plan['execution'], plan['finalization']):
        kind = step['kind']
        if kind == 'release':
            continue
        out = step['outputs'][0] if kind == 'transfer' else step.get('output')
        if kind in ('encode', 'prepare_plaintext'):
            payload = step['payload']['values']
            assert all(x == payload[0] for x in payload)
            if payload[0] == 1.0:
                assert len(payload) == spec['context']['poly_degree'] // 2
            value[out] = payload[0]
            continue
        ids = step['inputs']; before = [desc[v] for v in ids]; after = desc[out]
        op = step.get('op', kind)
        if op in ('transfer', 'copy'):
            assert before[0]['level'] == after['level']
            assert before[0]['scale_log2'] == after['scale_log2']
            nominal[out] = nominal[ids[0]]; value[out] = value[ids[0]]
            continue
        assert 4 <= after['level'] <= 49, step
        scales = [v['scale_log2'] for v in before]
        scale = scales[0]; level = before[0]['level']; result = value[ids[0]]
        ns = nominal[ids[0]]
        if op in ('mul_cc', 'mul_cp', 'add_cc', 'add_cp'):
            assert before[0]['level'] == before[1]['level'], step
            if op.startswith('mul'):
                scale = sum(scales); ns += nominal[ids[1]]
                result *= value[ids[1]]
                assert after['components'] == before[0]['components'] + before[1]['components'] - 1
            else:
                assert math.isclose(scales[0], scales[1], abs_tol=1e-8), step
                assert nominal[ids[0]] == nominal[ids[1]], step
                result += value[ids[1]]
        elif op == 'rescale':
            assert level - after['level'] == 1, step
            scale -= math.log2(spec['context']['rns_moduli'][level])
            assert scale >= 40, step
            level -= 1; ns = math.ceil(scale); counts['rescale_q'] += 1
            if producer.get(ids[0]) == 'rescale':
                counts['double_rescales'] += 1
            assert step['attrs']['target_level'] == level
            assert math.isclose(step['attrs']['target_scale_log2'], scale, abs_tol=1e-8)
        elif op == 'mod_switch':
            assert after['level'] < level, step
            counts['modswitch_q'] += level - after['level']; level = after['level']
        elif op == 'boot':
            assert boot['input_level_min'] <= level <= boot['input_level_max'], step
            assert before[0]['components'] == 2
            level, scale, ns = boot['output_level'], boot['output_scale_log2'], boot['output_scale_log2']
            counts['boots'] += 1
        else:
            assert op in ('rotate', 'negate', 'relinearize'), step
            if op == 'negate':
                result = -result
            if op == 'relinearize':
                assert before[0]['components'] == 3 and after['components'] == 2
        assert after['level'] == level, step
        assert math.isclose(after['scale_log2'], scale, abs_tol=1e-8), step
        assert nominal[out] == ns, (step, ns, nominal[out])
        if after['kind'] == 'ciphertext':
            capacity = sum(math.log2(q) for q in spec['context']['rns_moduli'][:level+1])
            assert 40 <= scale < capacity, (step, scale, capacity)
        value[out] = result; producer[out] = op
    for v, expected in zip(plan['final_outputs'], graph.expected()):
        assert desc[v]['scale_log2'] >= 40
        assert math.isclose(value[v], expected, abs_tol=1e-12, rel_tol=1e-12)
    return counts


def main():
    p = argparse.ArgumentParser(); p.add_argument('--hecate-opt', type=Path, required=True)
    p.add_argument('--source-dir', type=Path, required=True); p.add_argument('--output-dir', type=Path)
    p.add_argument('--strategies', nargs='+', choices=('greedy', 'depth-dp', 'optimized'), default=['greedy', 'depth-dp', 'optimized'])
    args = p.parse_args(); source_dir = args.source_dir.resolve()
    with tempfile.TemporaryDirectory(prefix='threshold-graphs-') as directory:
        root = args.output_dir.resolve() if args.output_dir else Path(directory)
        root.mkdir(parents=True, exist_ok=True); rows = []
        for boot_scale, budget in ((40, 1), (40, 2), (40, 4), (45, 1), (45, 2), (45, 4), (45, 18), (120, 3), (120, 4), (120, 6)):
            config = json.loads((source_dir / 'config.json').read_text())
            config.update(polynomialDegree=16, rescalingFactor=30, bootstrapLevelLowerBound=1,
                          bootstrapLevelUpperBound=budget+1, levelLowerBound=1, levelUpperBound=budget+1)
            spec = json.loads((source_dir / 'test/runtime-plan/lazy-physical-level-operator-spec.json').read_text())
            for entry in spec['operators'].values():
                entry.update(supported=True, latency_us_by_level=[10] * 46)
            spec['context']['default_scale_log2'] = 40
            spec['context']['rns_moduli'] = [(1 << 30) - (i + 1) * 32768 + 1 for i in range(50)]
            spec['rescale_mode'] = 'native'
            implementation = 'native' if boot_scale in (40, 45) else 'decrypt_reencrypt'
            spec['boot_profiles'] = [dict(profile_id='threshold', implementation=implementation,
                input_level_min=4, input_level_max=49, input_components=2,
                output_level=4+budget, output_scale_log2=boot_scale, output_components=2,
                latency_us_by_input_level=[85000]*46, noise_by_input_level=None,
                host_requirements=dict(needs_secret_key=boot_scale==120, needs_host_compute=boot_scale==120))]
            for name, graph, exhausted in cases():
                exhausted = (budget if exhausted is True else
                             max(0, budget-2) if exhausted == 'two-left' else 0)
                minimum = graph.minimum_boots(budget, boot_scale, exhausted,
                                              spec['context']['rns_moduli'])
                for strategy in args.strategies:
                    out = root / f'{boot_scale}-q{budget}-{strategy}-{name}'; out.mkdir(exist_ok=True)
                    (out/'input.mlir').write_text(graph.source(exhausted))
                    (out/'config.json').write_text(json.dumps(config))
                    (out/'spec.json').write_text(json.dumps(spec))
                    cmd = [str(args.hecate_opt.resolve()), '--dacapo', f'--boot-placement={strategy}',
                        f'--ckks-config={out}/config.json', '--waterline=40', f'--bootstrap-scale={boot_scale}',
                        '--runtime-plan-format=json', '--runtime-plan-memory=true',
                        '--runtime-plan-memory-reuse=true', '--runtime-plan-boot-profile=threshold',
                        f'--runtime-plan-boot-implementation={implementation}',
                        f'--runtime-plan-physical-level-operator-spec-path={out}/spec.json',
                        '--runtime-plan-levels-per-logical-level=1', '--runtime-plan-device-counts=1',
                        '--runtime-plan-device-count=1', f'--runtime-plan-operator-spec-path={out}/spec.json',
                        '--runtime-plan-target-id=lazy-physical-level-test',
                        '--runtime-plan-context-id=lazy-physical-level-test',
                        '--runtime-plan-operator-spec-id=lazy-physical-level-test-v1',
                        '--runtime-plan-operator-spec-version=1', '--runtime-plan-operator-spec-sha256=sha256:'+
                        hashlib.sha256((out/'spec.json').read_bytes()).hexdigest(),
                        str(out/'input.mlir'), '-o', str(out/'result.mlir')]
                    run = subprocess.run(cmd, text=True, capture_output=True, timeout=30)
                    (out/'compile.log').write_text(run.stdout+run.stderr)
                    if math.isinf(minimum):
                        assert run.returncode != 0, out.name
                        assert ('no feasible depth-DP placement' in run.stderr or
                                'multiplication cannot fit after Boot' in run.stderr or
                                'no feasible DaCapo candidate set' in run.stderr), run.stderr
                        rows.append(dict(case=out.name, expected_infeasible=True))
                        continue
                    if run.returncode:
                        assert 'no feasible DaCapo plan' in run.stderr, (out.name, run.stderr)
                        rows.append(dict(case=out.name, oracle_feasible_rejected=True,
                                         minimum_depth_cut_boots=minimum,
                                         compiler_error=run.stderr.splitlines()[0]))
                        print(out.name, 'ORACLE-FEASIBLE PLAN REJECTED', flush=True)
                        continue
                    plan = json.loads((out/'result.graph.runtime-plan.json').read_text())
                    counts = check_plan(plan, (out/'result.mlir').read_text(), graph, spec, exhausted)
                    planned = re.search(r'Number of Bootstrapping: (\d+)', run.stdout)
                    assert planned and int(planned[1]) == counts['boots'], (out.name, run.stdout, counts)
                    if strategy == 'depth-dp':
                        assert counts['boots'] == minimum, (out.name, counts, minimum)
                    if budget >= 4 and name in ('cp-double-rescale', 'cc-double-rescale'):
                        assert counts['double_rescales'] >= 1, (out.name, counts)
                        assert counts['boots'] == 0 and counts['rescale_q'] == 4, counts
                    if name == 'dead-branch':
                        assert counts['boots'] == 0 and counts['rescale_q'] == 0, counts
                    rows.append(dict(case=out.name, minimum_depth_cut_boots=minimum, **counts))
                    print(out.name, counts, flush=True)
        (root/'summary.json').write_text(json.dumps(rows, indent=2)+'\n')
        rejected = sum(row.get('expected_infeasible', False) for row in rows)
        missed = sum(row.get('oracle_feasible_rejected', False) for row in rows)
        print(f'AUDIT: {len(rows)} cases, {len(rows)-rejected-missed} validated plans, '
              f'{rejected} expected infeasible, {missed} oracle-feasible rejected')



if __name__ == '__main__':
    main()
