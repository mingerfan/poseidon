"""Replay regional plans with real primes and fixed-scale Boot inputs.

Plaintext arithmetic treats Boot as an identity. This checks scheduling and
coefficient semantics, not encrypted Boot approximation error.
"""

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import runpy
import subprocess
import tempfile


def replay(plan, graph, spec):
    desc = {v['id']: v for v in plan['values']}
    test_inputs = getattr(graph, 'test_inputs', (.25, .75))
    values = dict(zip(plan['external_inputs'], test_inputs))
    boots = 0
    for s in itertools.chain(plan['initialization'], plan['execution'], plan['finalization']):
        kind = s['kind']
        if kind == 'release':
            continue
        out = s['outputs'][0] if kind == 'transfer' else s.get('output')
        if kind in ('encode', 'prepare_plaintext'):
            payload = s['payload']['values']
            assert all(x == payload[0] for x in payload)
            values[out] = payload[0]
            assert desc[out]['scale_log2'] >= (30 if payload[0] == 1 else 40) - 1e-8, (out, payload[0], desc[out])
            continue
        inputs = s['inputs']
        before = [desc[x] for x in inputs]
        after = desc[out]
        op = s.get('op', kind)
        scale, level, value = before[0]['scale_log2'], before[0]['level'], values[inputs[0]]
        if op in ('mul_cc', 'mul_cp', 'add_cc', 'add_cp'):
            assert before[0]['level'] == before[1]['level']
            if op.startswith('mul'):
                scale += before[1]['scale_log2']
                value *= values[inputs[1]]
            else:
                assert abs(scale - before[1]['scale_log2']) < 1e-8
                value += values[inputs[1]]
        elif op == 'rescale':
            scale -= math.log2(spec['context']['rns_moduli'][level])
            level -= 1
        elif op == 'mod_switch':
            assert after['level'] < level
            level = after['level']
        elif op == 'boot':
            profile = spec['boot_profiles'][0]
            assert profile['input_level_min'] <= level <= profile['input_level_max']
            if 'input_scale_log2' in profile:
                assert abs(scale - profile['input_scale_log2']) < 1e-8
            level, scale = profile['output_level'], profile['output_scale_log2']
            boots += 1
        else:
            assert op in ('rotate', 'negate', 'relinearize', 'transfer', 'copy'), op
            if op == 'negate':
                value = -value
        assert level == after['level'] and abs(scale - after['scale_log2']) < 1e-8
        if after['kind'] == 'ciphertext':
            assert level >= spec['levels']['lower_bound']
            capacity = sum(math.log2(q) for q in spec['context']['rns_moduli'][:level+1])
            assert 40 - 1e-8 <= scale < capacity
        values[out] = value
    for output, expected in zip(plan['final_outputs'], graph.expected(*test_inputs)):
        assert math.isclose(values[output], expected, rel_tol=1e-10, abs_tol=1e-10)
    return boots


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hecate-opt', type=Path, required=True)
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--boot-placement', choices=('greedy', 'optimized'), default='greedy')
    parser.add_argument('--all-candidates', action='store_true')
    parser.add_argument('--beam-width', type=int, default=32)
    args = parser.parse_args()
    source = args.source_dir.resolve()
    definitions = runpy.run_path(str(source/'test/runtime-plan/check-threshold-graphs.py'))
    fixture = json.loads((source.parents[1]/'tests/fixtures/resnet20-manual-q50.json').read_text())
    spec = json.loads((source/'test/runtime-plan/lazy-physical-level-operator-spec.json').read_text())
    spec['rescale_mode'] = 'native'
    spec['context'].update(poly_degree=16, default_scale_log2=40,
                           rns_moduli=fixture['rns_moduli'],
                           rns_moduli_log2=[q.bit_length() for q in fixture['rns_moduli']])
    spec['levels'].update(lower_bound=4, upper_bound=49)
    for entry in spec['operators'].values():
        entry.update(supported=True, latency_us_by_level=[10]*50)
    spec['boot_profiles'] = [dict(profile_id='regional40', implementation='native',
        input_level_min=5, input_level_max=5, input_scale_log2=40, input_components=2,
        output_level=30, output_scale_log2=40, output_components=2,
        latency_us_by_input_level=[85000]*50, noise_by_input_level=None,
        host_requirements=dict(needs_secret_key=False, needs_host_compute=False))]
    config = json.loads((source/'config.json').read_text())
    config.update(polynomialDegree=16, rescalingFactor=30, bootstrapLevelLowerBound=1,
                  bootstrapLevelUpperBound=24, levelLowerBound=1, levelUpperBound=24)
    for name, table in config['latencyTable'].items():
        config['latencyTable'][name] = [max(table)] * 50
    rows = definitions['cases']()
    # Structurally identical regions may share scale plans, but each emitted
    # operation must keep its own coefficient payload and operand binding.
    class RepeatedShape(definitions['Graph']):
        def source(self, exhausted=0):
            text = super().source(exhausted)
            constant = next(line for line in text.splitlines() if line.startswith('%p ='))
            text = text.replace(constant, constant + '\n' +
                                constant.replace('%p =', '%p2 =').replace('1.01', '1.02'))
            return text.replace(f'%{self.outputs[1]} = "earth.mul"(%v2, %p)',
                                f'%{self.outputs[1]} = "earth.mul"(%v2, %p2)')

        def expected(self, x=.25, y=.75):
            return [x*x*1.01, y*y*1.02]

    g = RepeatedShape()
    left = g.op('mul', g.op('mul', 'x', 'x'), 'p')
    right = g.op('mul', g.op('mul', 'y', 'y'), 'p')
    g.outputs = [left, right]
    rows.append(('repeated-shape-distinct-weights', g, False))
    class WideLinearReduction(definitions['Graph']):
        def __init__(self, terms):
            super().__init__()
            self.weights = {}
            shared = self.op('rotate', 'x')
            for branch in range(2):
                total = None
                for index in range(terms):
                    term = self.op('mul', shared, 'p')
                    self.weights[term] = (1 + branch * .2 + index / (terms * 10)) / terms
                    total = term if total is None else self.op('add', total, term)
                self.outputs.append(total)

        def source(self, exhausted=0):
            text = super().source(exhausted)
            constant = next(line for line in text.splitlines() if line.startswith('%p ='))
            constants = [constant]
            for name, weight in self.weights.items():
                constants.append(constant.replace('%p =', f'%p_{name} =')
                                 .replace('1.01', repr(weight)))
                text = text.replace(f'%{name} = "earth.mul"(%v0, %p)',
                                    f'%{name} = "earth.mul"(%v0, %p_{name})')
            return text.replace(constant, '\n'.join(constants), 1)

        def expected(self, x=.25, y=.75):
            values = dict(x=x, y=y)
            for name, kind, inputs in self.ops:
                if kind == 'mul':
                    values[name] = values[inputs[0]] * self.weights[name]
                elif kind == 'add':
                    values[name] = values[inputs[0]] + values[inputs[1]]
                else:
                    values[name] = values[inputs[0]]
            return [values[out] for out in self.outputs]

    for terms in (64, 256, 1024):
        rows.append((f'shared-linear-reduction-{terms}', WideLinearReduction(terms), False))
    g = definitions['Graph']()
    value = g.op('rotate', 'x')
    g.outputs = [g.op('add', value, g.op('mul', value, 'p'))]
    rows.append(('linear-reconvergence', g, False))
    g = definitions['Graph']()
    value = 'x'
    for _ in range(40):
        value = g.op('mul', value, 'p')
    g.outputs = [value]
    rows.append(('long-cp-chain', g, False))
    g = definitions['Graph']()
    value = g.op('rotate', 'x')
    for _ in range(3):
        value = g.op('mul', value, 'p')
    g.outputs = [value]
    rows.append(('terminal-linear', g, True))
    g = definitions['Graph']()
    g.outputs = [g.op('mul', g.op('rotate', 'x'), 'p')]
    rows.append(('terminal-q5', g, 'q7'))
    # A Q9 shortcut can survive until the residual add. Refresh only the
    # main branch after its three-Q linear region, rather than both branches.
    g = definitions['Graph']()
    # Keep the squared branch near one so a wrong result cannot hide under
    # the replay's absolute tolerance after fourteen squares.
    g.test_inputs = (1 / 1.01**2, .75)
    value = g.op('rotate', 'x')
    for _ in range(2):
        value = g.op('mul', value, 'p')
    for _ in range(14):
        value = g.op('mul', value, value)
    value = g.op('mul', value, 'p')
    g.outputs = [g.op('add', value, 'x')]
    rows.append(('residual-selective-boot', g, 'q9'))
    # The raw instruction order interleaves two private linear regions. A
    # candidate grid must include the first region's exit despite that order.
    g = definitions['Graph']()
    g.test_inputs = (1 / 1.01**2, .75)
    left = g.op('rotate', 'x')
    right = g.op('rotate', 'y')
    left = g.op('mul', left, 'p')
    right = g.op('mul', right, 'p')
    left = g.op('mul', left, 'p')
    for _ in range(14):
        left = g.op('mul', left, left)
    right = g.op('mul', right, 'p')
    g.outputs = [g.op('add', g.op('mul', left, 'p'), right)]
    rows.append(('interleaved-linear-bypass', g, 'q9'))
    with tempfile.TemporaryDirectory(prefix='regional-scale-') as directory:
        root = Path(directory)
        (root/'spec.json').write_text(json.dumps(spec))
        (root/'config.json').write_text(json.dumps(config))
        for name, graph, exhausted in rows:
            exhausted = (25 if exhausted is True else 23 if exhausted == 'two-left'
                         else 24 if exhausted == 'q7'
                         else 22 if exhausted == 'q9' else 0)
            (root/'input.mlir').write_text(graph.source(exhausted).replace('tensor<1xf64>', 'tensor<8xf64>'))
            command = [str(args.hecate_opt.resolve()), str(root/'input.mlir'), '--dacapo',
                f'--regional-beam-width={args.beam_width}',
                '--native-scale-policy=regional', f'--boot-placement={args.boot_placement}',
                f'--ckks-config={root}/config.json', '--waterline=40', '--bootstrap-scale=40',
                '--runtime-plan-format=json', '--runtime-plan-boot-profile=regional40',
                f'--runtime-plan-physical-level-operator-spec-path={root}/spec.json',
                f'--runtime-plan-operator-spec-path={root}/spec.json',
                '--runtime-plan-levels-per-logical-level=1',
                '--runtime-plan-target-id=lazy-physical-level-test',
                '--runtime-plan-context-id=lazy-physical-level-test',
                '--runtime-plan-operator-spec-id=lazy-physical-level-test-v1',
                '--runtime-plan-operator-spec-version=1',
                '--runtime-plan-operator-spec-sha256=sha256:'+hashlib.sha256((root/'spec.json').read_bytes()).hexdigest(),
                '-o', str(root/'result.mlir')]
            if name == 'terminal-q5' or name.startswith('shared-linear-reduction-'):
                command.append('--regional-output-scale=40')
            if args.all_candidates:
                command.append('--regional-all-boot-candidates')
            result = subprocess.run(command, capture_output=True, text=True)
            assert result.returncode == 0, (name, result.stderr)
            plan = json.loads((root/'result.graph.runtime-plan.json').read_text())
            boots = replay(plan, graph, spec)
            if name.startswith('shared-linear-reduction-'):
                assert boots == 0, (name, boots)
                steps = [step for step in plan['execution'] if step['kind'] == 'compute']
                assert sum(step['op'] == 'rescale' for step in steps) == 4, name
                assert sum(step['op'] == 'rotate' for step in steps) == 1, name
                desc = {v['id']: v for v in plan['values']}
                input_level = desc[plan['external_inputs'][0]]['level']
                for out in plan['final_outputs']:
                    assert desc[out]['level'] == input_level - 2, (name, desc[out])
                    assert abs(desc[out]['scale_log2'] - 40) < 1e-8, (name, desc[out])
            if name == 'interleaved-linear-bypass' and args.boot_placement == 'optimized':
                assert boots == 1
            if name == 'residual-selective-boot':
                assert boots == (1 if args.boot_placement == 'optimized' else 2)
                if args.boot_placement == 'optimized':
                    steps = [s for s in plan['execution'] if s['kind'] == 'compute']
                    first_boot = next(i for i, s in enumerate(steps) if s['op'] == 'boot')
                    assert sum(s['op'] == 'mul_cp' for s in steps[:first_boot]) == 2
                    add = next(s for s in steps if s['op'] == 'add_cc')
                    producers = {s['output']: s for s in steps}
                    shortcut = producers[add['inputs'][1]]
                    assert shortcut['op'] == 'mod_switch'
                    assert shortcut['inputs'] == [plan['external_inputs'][0]]
            if name == 'terminal-q5':
                assert boots == 0
                desc = {v['id']: v for v in plan['values']}
                output = desc[plan['final_outputs'][0]]
                assert output['level'] == 4 and abs(output['scale_log2'] - 40) < 1e-8
            if name == 'terminal-linear':
                assert boots == 0
                result = subprocess.run(command + ['--regional-output-scale=40'], capture_output=True, text=True)
                assert result.returncode == 0, result.stderr
                plan = json.loads((root/'result.graph.runtime-plan.json').read_text())
                assert replay(plan, graph, spec) == 1
                desc = {v['id']: v for v in plan['values']}
                assert abs(desc[plan['final_outputs'][0]]['scale_log2'] - 40) < 1e-8
        # Qwen's existing profile accepts a range of input levels and does not
        # prescribe an input scale. It must still admit Boot checkpoints.
        profile = spec['boot_profiles'][0]
        del profile['input_scale_log2']
        profile.update(input_level_min=4, input_level_max=49,
                       output_level=22, output_scale_log2=45)
        (root/'spec.json').write_text(json.dumps(spec))
        graph = next(graph for name, graph, _ in rows if name == 'long-cp-chain')
        (root/'input.mlir').write_text(graph.source(0).replace('tensor<1xf64>', 'tensor<8xf64>'))
        command = [('--bootstrap-scale=45' if arg == '--bootstrap-scale=40' else
                    '--runtime-plan-operator-spec-sha256=sha256:' +
                    hashlib.sha256((root/'spec.json').read_bytes()).hexdigest()
                    if arg.startswith('--runtime-plan-operator-spec-sha256=') else arg)
                   for arg in command]
        result = subprocess.run(command + ['--compilation-progress'], capture_output=True, text=True)
        assert result.returncode == 0, ('unconstrained-boot-input', result.stderr)
        if args.boot_placement == 'optimized':
            assert 'initial_boot=1' in result.stderr, result.stderr
        plan = json.loads((root/'result.graph.runtime-plan.json').read_text())
        assert replay(plan, graph, spec) > 0
        # An early output (like saved KV) must not be refreshed just because
        # a later independent chain needs Boot; its required s40 must survive.
        graph = definitions['Graph']()
        saved = graph.op('mul', 'x', 'p')
        value = 'y'
        for _ in range(40):
            value = graph.op('mul', value, 'p')
        graph.outputs = [saved, value]
        (root/'input.mlir').write_text(graph.source(0).replace('tensor<1xf64>', 'tensor<8xf64>'))
        result = subprocess.run(command + ['--regional-output-scale=40', '--compilation-progress'],
                                capture_output=True, text=True)
        assert result.returncode == 0, ('saved-output', result.stderr)
        assert '[compile]' in result.stderr
        plan = json.loads((root/'result.graph.runtime-plan.json').read_text())
        assert replay(plan, graph, spec) > 0
        desc = {v['id']: v for v in plan['values']}
        assert all(abs(desc[v]['scale_log2'] - 40) < 1e-8 for v in plan['final_outputs'])
        producers = {s.get('output'): s for s in plan['execution']}
        current = plan['final_outputs'][0]
        while current in producers:
            step = producers[current]
            assert step.get('op') != 'boot', 'saved output was unnecessarily refreshed'
            current = step['inputs'][0]
        # Saving a value must also preserve its original version when later
        # arithmetic still uses (and may refresh) that value.
        graph.outputs = [saved, graph.op('add', value, saved)]
        (root/'input.mlir').write_text(graph.source(0).replace('tensor<1xf64>', 'tensor<8xf64>'))
        result = subprocess.run(command + ['--regional-output-scale=40'], capture_output=True, text=True)
        assert result.returncode == 0, ('saved-live-output', result.stderr)
        plan = json.loads((root/'result.graph.runtime-plan.json').read_text())
        assert replay(plan, graph, spec) > 0
        desc = {v['id']: v for v in plan['values']}
        assert all(abs(desc[v]['scale_log2'] - 40) < 1e-8 for v in plan['final_outputs'])
        # Multi-exit linear components cannot be fused as one region. Their
        # 64-node private reduction fragments must still share one scale/RS.
        graph = definitions['Graph']()
        base = graph.op('rotate', 'x')
        value = 'y'
        for _ in range(31):
            value = graph.op('add', value, graph.op('mul', base, 'p'))
        value = graph.op('add', base, graph.op('rotate', value))
        other = graph.op('add', base, graph.op('mul', 'y', 'p'))
        graph.outputs = [value, other]
        source_text = graph.source(0).replace('tensor<1xf64>', 'tensor<8xf64>')
        source_text = source_text.replace(') {', ') attributes {'
            'segment_inputType = [tensor<1x!earth.ci<45 * 0>>, tensor<1x!earth.ci<45 * 0>>], '
            'segment_input_scales = array<f64: 45.0, 45.0>} {', 1)
        (root/'input.mlir').write_text(source_text)
        result = subprocess.run(command + ['--regional-output-scale=40'], capture_output=True, text=True)
        assert result.returncode == 0, ('multi-exit-linear', result.stderr)
        plan = json.loads((root/'result.graph.runtime-plan.json').read_text())
        assert replay(plan, graph, spec) == 0
    print(f'{len(rows)+5} regional plan replays passed with fixed and unconstrained Boot inputs')


if __name__ == '__main__':
    main()
