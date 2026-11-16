#!/usr/bin/env python3
"""Standalone CPU-only mixed-attention schema gate; no ignored model goldens.

Run with --runner PATH [--guard PATH] --output-dir PATH. Output paths must be
inside test-results/bert-a800-20261115/multi-plan-build/attention-modes/.
Only --check-only, --help, and the no-argument usage path are ever invoked.
These synthetic assets test validation/contracts, NOT GPU numerical validity,
partial-AV consumer correctness, encrypted equivalence, or performance.

Mode/physical contracts follow the frozen prepared_qk_adapter.h and
prepared_av_adapter.h; the runner must use their authoritative CPU validators.
The formulas below construct their declared direct rotation inventories, not a
replacement manifest validator. The old 37-test script is imported read-only
solely for its synthetic zero matrix and 16 parameter-hash declarations.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

# In particular, importing the existing test helper must not create __pycache__.
sys.dont_write_bytecode = True

REPO_ROOT = Path(__file__).resolve().parents[4]
OUTPUT_ROOT = REPO_ROOT / 'test-results/bert-a800-20261115/multi-plan-build/attention-modes'
ARGS = None
RESULTS = []
Q = [204865537, 205651969, 206307329, 207880193, 209059841,
     210370561, 211025921, 211812353, 214171649, 215482369,
     215744513, 216137729, 216924161]
SLOTS = 32768
INPUT_SCALE = 55.360150528219684
LINEAR_SCALE = 55.352557838953814
COMMON_SCALE = 55.372711913462794
OUTPUT_SCALE = 55.4442212582195
FINAL_SCALE = 56
LABELS = ('a', 'b', 'c')
QK_CALLEE = 'bert.compact_qk'
AV_CALLEE = 'bert.compact_av'
PARTIAL_CALLEE = 'bert.compact_av_packed_partial'
PARTIAL_SEMANTICS = 'live_full_av_gap_B_current_plus_A_next_head_row'
GUARD_LOADED = 'BERT_NO_DEVICE_GUARD loaded device_calls=0'
GUARD_FINISHED = 'BERT_NO_DEVICE_GUARD finished device_calls=0'


def sha(data):
    return 'sha256:' + hashlib.sha256(data).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, separators=(',', ':'), allow_nan=False) + '\n')


def load(path):
    return json.loads(path.read_text())


def canonical_steps(steps):
    # Full-slot left rotations, with the half turn represented positively.
    return sorted({s % SLOTS if s % SLOTS <= SLOTS // 2 else s % SLOTS - SLOTS
                   for s in steps if s % SLOTS})


def linear_steps():
    return canonical_steps(list(range(1, 16)) + [16*g for g in range(-16, 16) if g])


def qk_steps(packed=False):
    raw = [s for b in range(1, 8) for s in (256*b, -b)]
    raw += [s for g in range(1, 16) for s in (-2048*g, 2040*g)]
    raw += [1, 2, 4, 8, 16, 32]
    return canonical_steps(raw + ([-64, 63, 192] if packed else []))


def av_steps(packed=False):
    raw = [s for b in range(1, 8) for s in (256*b, b)]
    raw += [b-128 for b in range(8)]
    raw += [s for g in range(1, 16) for s in (2048*g, -2040*g)]
    return canonical_steps(raw + ([-64, 64, 192] if packed else []))


def input_sha(fixture):
    # Canonical ORIGINAL client order is token -> hidden feature, not the
    # concatenation of three complete packed ciphertext payloads.
    rows = [row['values'] for row in fixture['inputs']]
    return sha(b''.join(struct.pack('<d', rows[g][t*256+x])
                        for t in range(128) for g in range(3) for x in range(256)))


def qk_manifest(packed=False):
    row = dict(call_id='0', id='layer0.qk_pair0', tokens=128, tile=256,
               head_dim=64, baby=8, input_level=8, input_scale_log2=LINEAR_SCALE,
               intermediate_scale_log2=COMMON_SCALE, output_level=4,
               output_scale_log2=OUTPUT_SCALE, pair_products=False, pack_inputs=packed)
    return dict(format_version=2 if packed else 1, calls=[row],
                rotation_steps=qk_steps(packed))


def av_manifest(packed=False):
    row = dict(call_id='0', id='layer0.av_pair0', tokens=128, tile=256,
               head_dim=64, baby=8, score_input_level=10,
               score_input_scale_log2=INPUT_SCALE, value_input_level=8,
               value_input_scale_log2=LINEAR_SCALE, common_level=6,
               common_scale_log2=COMMON_SCALE, output_level=4,
               output_scale_log2=OUTPUT_SCALE,
               algorithm='PackedInput2Partial' if packed else 'Baseline',
               callee=PARTIAL_CALLEE if packed else AV_CALLEE,
               output_semantics=PARTIAL_SEMANTICS if packed else 'full_cyclic_av_all_features',
               pack_inputs=packed, partial_output=packed,
               client_value_gap_zero_required=packed, pair_products=False)
    return dict(format_version=2 if packed else 1, calls=[row],
                rotation_steps=av_steps(packed))


def bind_manifest(root, label, engine, manifest):
    path = root / (label + '.' + engine + '.json')
    dump(path, manifest)
    fixture_path = root / (label + '.fixture.json')
    fixture = load(fixture_path)
    field = 'prepared_' + engine + '_manifest'
    fixture[field] = path.name
    fixture[field + '_sha256'] = sha(path.read_bytes())
    dump(fixture_path, fixture)


def set_modes(root, label, qk_packed, av_packed):
    bind_manifest(root, label, 'qk', qk_manifest(qk_packed))
    bind_manifest(root, label, 'av', av_manifest(av_packed))
    path = root / (label + '.fixture.json')
    fixture = load(path)
    fixture['prepared_qk_required_capabilities'] = ['pack_inputs'] if qk_packed else []
    fixture['prepared_av_required_capabilities'] = ['packed_input2_partial'] if av_packed else []
    fixture['model'].update(qk_pack_inputs=qk_packed, av_pack_inputs=av_packed)
    fixture['prepared_rotation_steps'] = canonical_steps(
        linear_steps() + qk_steps(qk_packed) + av_steps(av_packed))
    dump(path, fixture)
    path = root / (label + '.plan.json')
    plan = load(path)
    plan['execution'][4]['attrs']['callee'] = PARTIAL_CALLEE if av_packed else AV_CALLEE
    dump(path, plan)


def assets(root):
    helper_spec = importlib.util.spec_from_file_location(
        '_attention_mode_seed_assets', Path(__file__).with_name('test_check_only.py'))
    helper = importlib.util.module_from_spec(helper_spec)
    helper_spec.loader.exec_module(helper)
    helper.assets(root)

    spec = load(root / 'spec.json')
    spec['context'].update(poly_degree=65536, rns_moduli_log2=[28]*13,
                           rns_moduli=Q, max_modulus_log2=28, default_scale_log2=28)
    spec['levels'] = dict(lower_bound=0, upper_bound=12)
    spec['boot_profiles'][0].update(input_level_min=2, input_level_max=12,
                                    output_level=10, output_scale_log2=FINAL_SCALE,
                                    latency_us_by_input_level=[1]*13)
    for operator in spec['operators'].values():
        if 'latency_us_by_level' in operator:
            operator['latency_us_by_level'] = [1]*13
    dump(root / 'spec.json', spec)

    linear = load(root / 'linear.json')
    for row in linear['calls']:
        row.update(input_level=10, input_scale_log2=INPUT_SCALE,
                   output_level=8, output_scale_log2=LINEAR_SCALE)
    linear['rotation_steps'] = linear_steps()
    dump(root / 'linear.json', linear)

    host = dict(kind='host', rank=0)
    dev = dict(kind='device', rank=0, index=0)

    def descriptor(ident, place, level, scale):
        return dict(id=str(ident), kind='ciphertext', place=place, context='mock',
                    level=level, scale_log2=scale, ntt=True, components=2)

    values = [descriptor(i, host if i % 2 == 0 else dev, 10, INPUT_SCALE)
              for i in range(6)]
    values += [descriptor(i, dev, 8, LINEAR_SCALE) for i in range(6, 9)]
    values += [descriptor(i, dev, 4, OUTPUT_SCALE) for i in (9, 10)]
    values += [descriptor(i, dev, 10, FINAL_SCALE) for i in (11, 12, 13)]
    initialization = [dict(kind='transfer', ordinal=i, transfer_id=str(i), hint='auto',
                           inputs=[str(2*i)], outputs=[str(2*i+1)], sources=[host],
                           destinations=[dev], output_kinds=['ciphertext']) for i in range(3)]
    execution = [dict(kind='compute', op='ApiCall', ordinal=3+i,
                      inputs=['1', '3', '5'], output=str(6+i), place=dev,
                      attrs=dict(callee='bert.compact_linear', call_id=str(i)))
                 for i in range(3)]
    execution += [dict(kind='compute', op='ApiCall', ordinal=6, inputs=['6', '7'],
                       output='9', place=dev, attrs=dict(callee=QK_CALLEE, call_id='0')),
                  dict(kind='compute', op='ApiCall', ordinal=7, inputs=['1', '8'],
                       output='10', place=dev, attrs=dict(callee=AV_CALLEE, call_id='0'))]
    execution += [dict(kind='compute', op='boot', ordinal=8+i, inputs=[str(source)],
                       output=str(11+i), place=dev,
                       attrs=dict(target_level=10, target_scale_log2=FINAL_SCALE,
                                  target_components=2, operator_profile='mock-native',
                                  implementation='native'))
                  for i, source in enumerate((9, 10, 6))]
    plan = load(root / 'a.plan.json')
    plan.update(values=values, external_inputs=['0', '2', '4'],
                initialization=initialization, execution=execution,
                finalization=[], final_outputs=['11', '12', '13'])
    plan['target']['operator_spec']['source_sha256'] = sha((root / 'spec.json').read_bytes())

    fixture = load(root / 'a.fixture.json')
    fixture['context'].update(poly_degree=65536, q_moduli_bits=[28]*13, q_moduli=Q,
                              p_moduli_bits=[28], p_moduli=[268042241],
                              default_scale_log2=28)
    fixture['model']['config'] = dict(tokens=128, hidden=768, heads=12, tile=256,
                                     slots=SLOTS, intermediate=768, layers=1,
                                     bsgs_baby_step=16, reciprocal_iterations=3,
                                     ln_epsilon=1e-5, ln_variance_center=1.0,
                                     weight_std=0.02, attention_layout='cyclic')
    fixture['model']['seed'] = 20261114
    fixture['inputs'] = [dict(value_id=2*g, values=[(g+1)/8.0]*SLOTS) for g in range(3)]
    fixture['expected_outputs'] = [dict(value_id=i, values=[0.0]*SLOTS) for i in (11, 12, 13)]
    fixture['model']['input_sha256'] = input_sha(fixture)
    for label in LABELS:
        dump(root / (label + '.plan.json'), plan)
        dump(root / (label + '.fixture.json'), fixture)
    set_modes(root, 'a', False, False)
    set_modes(root, 'b', True, False)
    set_modes(root, 'c', True, True)
    dump(root / 'schedule.json', dict(
        plans=[dict(name=x.upper(), plan=x+'.plan.json', spec='spec.json',
                    fixture=x+'.fixture.json') for x in LABELS],
        schedule=[dict(name=x, phase='diagnostic') for x in ('A', 'A', 'B', 'C', 'A', 'C')]))


class AttentionModes(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='attention-mock-', dir=ARGS.output_dir)
        self.root = Path(self.temp.name)
        assets(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def mutate(self, name, fn):
        path = self.root / name
        value = load(path)
        fn(value)
        dump(path, value)

    def manifest(self, label, engine, fn):
        value = load(self.root / (label + '.' + engine + '.json'))
        fn(value)
        bind_manifest(self.root, label, engine, value)

    def cases(self, label, engine, cases):
        """Restore each negative independently, including the bound raw SHA."""
        original = load(self.root / (label + '.' + engine + '.json'))
        for name, change in cases:
            with self.subTest(case=name):
                value = copy.deepcopy(original)
                change(value)
                bind_manifest(self.root, label, engine, value)
                try:
                    self.run_check(reject=True, case=name)
                finally:
                    bind_manifest(self.root, label, engine, original)

    def fixture_cases(self, label, cases):
        path = self.root / (label + '.fixture.json')
        original = load(path)
        for name, change in cases:
            with self.subTest(case=name):
                value = copy.deepcopy(original)
                change(value)
                dump(path, value)
                try:
                    self.run_check(reject=True, case=name)
                finally:
                    dump(path, original)

    def keep_plans(self, labels):
        names = {label.upper() for label in labels}
        self.mutate('schedule.json', lambda s: s.update(
            plans=[p for p in s['plans'] if p['name'] in names],
            schedule=[p for p in s['schedule'] if p['name'] in names]))

    def layers(self, count):
        for label in LABELS:
            def change(f):
                f['model']['config']['layers'] = count
                parameters = f['model']['weight_sha256']['0']
                f['model']['weight_sha256'] = {str(i): copy.deepcopy(parameters) for i in range(count)}
            self.mutate(label + '.fixture.json', change)

    def invoke(self, arguments, case=None, report_path=None):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='')
        if ARGS.guard:
            env['LD_PRELOAD'] = str(ARGS.guard)
        command = [str(ARGS.runner)] + [str(arg) for arg in arguments]
        timed_out = False
        try:
            proc = subprocess.run(command, env=env, capture_output=True, text=True, timeout=120)
            code, stdout, stderr = proc.returncode, proc.stdout, proc.stderr
        except subprocess.TimeoutExpired as error:
            timed_out = True
            code = None
            stdout = error.stdout or ''
            stderr = error.stderr or ''
            if isinstance(stdout, bytes):
                stdout = stdout.decode(errors='replace')
            if isinstance(stderr, bytes):
                stderr = stderr.decode(errors='replace')
        report = None
        report_error = None
        if report_path is not None:
            try:
                report = load(report_path)
            except (OSError, ValueError) as error:
                report_error = str(error)
        loaded = GUARD_LOADED in stderr
        finished = GUARD_FINISHED in stderr
        blocked = 'blocked api=' in stderr
        evidence = dict(invocation=len(RESULTS)+1, test=self.id(), case=case,
                        command=command, exit=code, timeout=timed_out,
                        cuda_visible_devices='', guard=bool(ARGS.guard),
                        guard_path=str(ARGS.guard) if ARGS.guard else None,
                        guard_loaded=loaded, guard_finished=finished,
                        blocked_device_api=blocked,
                        device_calls_zero=loaded and finished and not blocked and code != 99,
                        stdout=stdout, stderr=stderr, report=report, report_read_error=report_error,
                        report_completed=report.get('completed') if isinstance(report, dict) else None,
                        error=report.get('error') if isinstance(report, dict) else None)
        # Save evidence BEFORE assertions, including failed/missing-report runs.
        RESULTS.append(evidence)
        dump(ARGS.output_dir / 'invocations' / ('%04d.json' % evidence['invocation']), evidence)
        self.assertFalse(timed_out, 'check-only process timed out; evidence was retained')
        self.assertNotIn('blocked api=', stderr)
        self.assertNotEqual(code, 99, stderr)
        if ARGS.guard:
            self.assertTrue(loaded, stderr)
            self.assertTrue(finished, stderr)
        if report_path is not None:
            self.assertIsNone(report_error, stderr + '\n' + str(report_error))
            self.assertIsInstance(report, dict)
        return code, stdout, stderr, report

    def run_check(self, reject=False, case=None, error_pattern=None):
        report_path = self.root / 'report.json'
        # A failed process must not accidentally reuse a previous subtest report.
        report_path.unlink(missing_ok=True)
        code, _, stderr, report = self.invoke(
            ['--check-only', self.root / 'schedule.json', report_path], case, report_path)
        self.assertFalse(report.get('gpu_executed', False), report)
        for section in ('keys', 'native_boot_profile', 'original_inputs'):
            if isinstance(report.get(section), dict) and 'generation_calls' in report[section]:
                self.assertEqual(report[section]['generation_calls'], 0, report)
        if reject:
            self.assertEqual(code, 1, stderr + '\n' + json.dumps(report))
            self.assertIs(report.get('completed'), False, report)
            self.assertIsInstance(report.get('error'), str, report)
            self.assertTrue(report['error'].strip(), report)
            if error_pattern:
                self.assertRegex(report['error'], error_pattern)
        else:
            self.assertEqual(code, 0, stderr + '\n' + json.dumps(report))
            self.assertIs(report.get('completed'), True, report)
            self.assertIs(report['compatibility']['compatible'], True)
            self.assertIs(report['gpu_executed'], False)
            self.assertEqual(report['keys']['generation_calls'], 0)
            self.assertEqual(report['native_boot_profile']['generation_calls'], 0)
        return report

    def test_mixed_three_plans_reuse_identical_plan_and_attention_ids(self):
        plans = [load(self.root / (label + '.plan.json')) for label in LABELS]
        self.assertEqual({p['plan_id'] for p in plans}, {'1'})
        for engine in ('qk', 'av'):
            rows = [load(self.root / (label + '.' + engine + '.json'))['calls'][0] for label in LABELS]
            self.assertEqual(len({row['id'] for row in rows}), 1)
            self.assertEqual({row['call_id'] for row in rows}, {'0'})
        self.run_check()

    def test_baseline_and_qk_packed_two_plans(self):
        self.keep_plans(('a', 'b'))
        self.run_check()

    def test_baseline_and_qk_av_packed_two_plans(self):
        self.keep_plans(('a', 'c'))
        self.run_check()

    def test_per_plan_namespaces_and_exact_rotation_unions(self):
        report = self.run_check()
        compatibility = report['compatibility']
        candidates = compatibility['plans']
        self.assertEqual({p['attention_adapter_namespace'] for p in candidates}, {'A', 'B', 'C'})
        self.assertEqual(len(candidates), 3)
        union = set()
        for candidate, (qk_packed, av_packed) in zip(candidates, ((False, False), (True, False), (True, True))):
            # This report is grouped by adapter engine, not Runtime callee;
            # baseline and partial AV both belong to the compact_av engine.
            expected = {'bert.compact_linear': linear_steps(), QK_CALLEE: qk_steps(qk_packed),
                        AV_CALLEE: av_steps(av_packed)}
            actual = candidate['prepared_rotation_steps_by_engine']
            self.assertEqual(set(actual), set(expected))
            for callee, steps in expected.items():
                self.assertEqual(actual[callee], steps)
                union.update(steps)
            self.assertEqual(candidate['model_seed'], 20261114)
        self.assertEqual(compatibility['union_requirements']['prepared_direct_rotation_steps'], sorted(union))
        self.assertEqual(set(compatibility['prepared_capabilities_union']),
                         {'qk.pack_inputs', 'av.packed_input2_partial'})
        self.assertIs(compatibility['mixed_attention_contracts'], True)

    def test_packed_capabilities_can_be_required_by_model_flags_alone(self):
        for label in LABELS:
            def change(f):
                f.pop('prepared_qk_required_capabilities')
                f.pop('prepared_av_required_capabilities')
            self.mutate(label + '.fixture.json', change)
        self.run_check()

    def test_packed_capabilities_can_be_required_by_lists_alone(self):
        for label in LABELS:
            self.mutate(label + '.fixture.json', lambda f: f['model'].update(
                qk_pack_inputs=False, av_pack_inputs=False))
        self.run_check()

    def test_v2_baseline_without_required_capabilities_is_valid(self):
        self.manifest('a', 'qk', lambda m: m.update(format_version=2))
        self.manifest('a', 'av', lambda m: m.update(format_version=2))
        self.run_check()

    def test_same_baseline_contract_multilayer_retained(self):
        for label in LABELS:
            set_modes(self.root, label, False, False)
        self.layers(2)
        self.run_check()

    def test_same_packed_contract_multilayer_retained(self):
        for label in LABELS:
            set_modes(self.root, label, True, True)
        self.layers(2)
        self.run_check()

    def test_mixed_attention_contracts_reject_multiple_layers(self):
        self.layers(2)
        self.run_check(reject=True)

    def test_qk_v1_true_modes_rejected(self):
        self.cases('a', 'qk', [(field, lambda m, field=field: m['calls'][0].update({field: True}))
                              for field in ('pair_products', 'pack_inputs')])

    def test_av_v1_partial_algorithm_rejected(self):
        self.manifest('c', 'av', lambda m: m.update(format_version=1))
        self.run_check(reject=True)

    def test_qk_invalid_manifest_versions(self):
        self.cases('b', 'qk', [(repr(value), lambda m, value=value: m.update(format_version=value))
                              for value in (0, 3, True, 2.0, '2')])

    def test_av_invalid_manifest_versions(self):
        self.cases('c', 'av', [(repr(value), lambda m, value=value: m.update(format_version=value))
                              for value in (0, 3, True, 2.0, '2')])

    def test_qk_mode_fields_require_json_booleans(self):
        self.cases('b', 'qk', [(field + '=' + repr(value),
                               lambda m, field=field, value=value: m['calls'][0].update({field: value}))
                              for field in ('pair_products', 'pack_inputs') for value in (1, 'true', None)])

    def test_qk_mutually_exclusive_modes_rejected(self):
        self.manifest('b', 'qk', lambda m: m['calls'][0].update(pair_products=True))
        self.run_check(reject=True)

    def test_av_unknown_or_wrong_type_algorithm_rejected(self):
        self.cases('c', 'av', [(repr(value), lambda m, value=value: m['calls'][0].update(algorithm=value))
                              for value in ('Unknown', 'packed_input2_partial', True, 2, None)])

    def test_av_mode_fields_require_json_booleans(self):
        self.cases('c', 'av', [(field + '=' + repr(value),
                               lambda m, field=field, value=value: m['calls'][0].update({field: value}))
                              for field in ('pack_inputs', 'partial_output', 'client_value_gap_zero_required', 'pair_products')
                              for value in (1, 'true')])

    def test_av_algorithm_mode_mismatches_rejected(self):
        self.cases('c', 'av', [(field, lambda m, field=field: m['calls'][0].update({field: False}))
                              for field in ('pack_inputs', 'partial_output', 'client_value_gap_zero_required')])

    def test_av_baseline_cannot_claim_packed_or_pair_products(self):
        self.cases('a', 'av', [(field, lambda m, field=field: m['calls'][0].update({field: True}))
                              for field in ('pack_inputs', 'partial_output', 'client_value_gap_zero_required', 'pair_products')])

    def test_required_capability_lists_are_strict(self):
        self.fixture_cases('c', [(field + '=' + repr(value),
                                 lambda f, field=field, value=value: f.update({field: value}))
                                for field in ('prepared_qk_required_capabilities', 'prepared_av_required_capabilities')
                                for value in ('pack_inputs', {}, [True], ['unknown'])])

    def test_model_mode_flags_require_json_booleans(self):
        self.fixture_cases('c', [(field + '=' + repr(value),
                                 lambda f, field=field, value=value: f['model'].update({field: value}))
                                for field in ('qk_pack_inputs', 'av_pack_inputs') for value in (1, 'true', None)])

    def test_false_claimed_capabilities_rejected_even_in_v2(self):
        self.manifest('a', 'qk', lambda m: m.update(format_version=2))
        self.manifest('a', 'av', lambda m: m.update(format_version=2))
        cases = []
        for engine, capability, flag in (('qk', 'pack_inputs', 'qk_pack_inputs'),
                                         ('av', 'packed_input2_partial', 'av_pack_inputs')):
            cases.append((engine + '-list', lambda f, engine=engine, capability=capability:
                          f.update({'prepared_' + engine + '_required_capabilities': [capability]})))
            cases.append((engine + '-model', lambda f, flag=flag: f['model'].update({flag: True})))
        self.fixture_cases('a', cases)

    def test_required_capabilities_need_manifests(self):
        self.fixture_cases('c', [(engine, lambda f, engine=engine: f.pop('prepared_' + engine + '_manifest'))
                                for engine in ('qk', 'av')])

    def test_av_manifest_algorithm_callee_mismatch(self):
        self.cases('c', 'av', [('baseline-callee', lambda m: m['calls'][0].update(callee=AV_CALLEE)),
                              ('wrong-semantics', lambda m: m['calls'][0].update(output_semantics='full_cyclic_av_all_features')),
                              ('missing-callee', lambda m: m['calls'][0].pop('callee')),
                              ('missing-semantics', lambda m: m['calls'][0].pop('output_semantics'))])

    def test_partial_av_runtime_callee_must_match_manifest(self):
        self.mutate('c.plan.json', lambda p: p['execution'][4]['attrs'].update(callee=AV_CALLEE))
        self.run_check(reject=True)

    def test_baseline_av_runtime_callee_must_match_manifest(self):
        self.mutate('a.plan.json', lambda p: p['execution'][4]['attrs'].update(callee=PARTIAL_CALLEE))
        self.run_check(reject=True)

    def add_unused_partial_row(self, required):
        def change(m):
            row = av_manifest(True)['calls'][0]
            row.update(id='unused-partial-av', call_id='1')
            m.update(format_version=2, rotation_steps=av_steps(True))
            m['calls'].append(row)
        self.manifest('a', 'av', change)
        def fixture(f):
            f['prepared_rotation_steps'] = canonical_steps(linear_steps() + qk_steps() + av_steps(True))
            if required:
                f['prepared_av_required_capabilities'] = ['packed_input2_partial']
                f['model']['av_pack_inputs'] = True
        self.mutate('a.fixture.json', fixture)

    def test_unused_partial_manifest_row_rejected(self):
        self.add_unused_partial_row(required=False)
        self.run_check(reject=True)

    def test_unused_partial_row_cannot_satisfy_required_capability(self):
        self.add_unused_partial_row(required=True)
        self.run_check(reject=True)

    def test_duplicate_attention_ids_and_call_ids_within_plan_rejected(self):
        for engine in ('qk', 'av'):
            def duplicate_id(m):
                row = copy.deepcopy(m['calls'][0]); row['call_id'] = '1'
                m['calls'].append(row)
            def duplicate_call(m):
                row = copy.deepcopy(m['calls'][0]); row['id'] = 'another-physical-id'
                m['calls'].append(row)
            self.cases('c', engine, [(engine + '-id', duplicate_id), (engine + '-call-id', duplicate_call)])

    def test_missing_per_plan_actual_rotation_keys_rejected(self):
        # Other candidates still declare these keys: an all-plan union must not
        # conceal an incomplete individual fixture's actual preparation list.
        for label, step in (('b', 63), ('c', -121)):
            self.fixture_cases(label, [(label, lambda f, step=step: f['prepared_rotation_steps'].remove(step))])

    def test_manifest_inventory_cannot_hide_required_rotation_keys(self):
        for engine, step in (('qk', 63), ('av', -121)):
            manifest_path = self.root / ('c.' + engine + '.json')
            fixture_path = self.root / 'c.fixture.json'
            original_manifest, original_fixture = load(manifest_path), load(fixture_path)
            with self.subTest(engine=engine):
                self.manifest('c', engine, lambda m, step=step: m['rotation_steps'].remove(step))
                self.mutate('c.fixture.json', lambda f, step=step: f['prepared_rotation_steps'].remove(step))
                try:
                    self.run_check(reject=True, case=engine)
                finally:
                    dump(manifest_path, original_manifest)
                    dump(fixture_path, original_fixture)

    def test_qk_per_plan_physical_layout_rejected(self):
        self.manifest('b', 'qk', lambda m: m['calls'][0].update(head_dim=32))
        self.run_check(reject=True)

    def test_qk_real_two_prime_intermediate_contract_required(self):
        self.cases('b', 'qk', [('missing', lambda m: m['calls'][0].pop('intermediate_scale_log2')),
                              ('inconsistent', lambda m: m['calls'][0].update(intermediate_scale_log2=COMMON_SCALE+1)),
                              ('unrepresentable', lambda m: m['calls'][0].update(intermediate_scale_log2=-2000))])

    def test_av_per_plan_physical_layout_and_types_rejected(self):
        self.cases('c', 'av', [('common-level', lambda m: m['calls'][0].update(common_level=5)),
                              ('integer-required', lambda m: m['calls'][0].update(common_level=True)),
                              ('scale-number-required', lambda m: m['calls'][0].update(common_scale_log2='55'))])

    def test_av_actual_prime_scale_equation_rejected(self):
        self.manifest('c', 'av', lambda m: m['calls'][0].update(common_scale_log2=COMMON_SCALE+1))
        self.run_check(reject=True)

    def test_av_equations_do_not_replace_capacity_headroom_validation(self):
        # Keep actual score/value input metadata unchanged, avoiding a spurious
        # cross-plan input/Linear rejection. The adapter intentionally permits
        # noncanonical mask scales if its physical equations/capacity hold.
        # Here output=2*common-drop(Q6,Q5), but the common CC has no headroom.
        common = 82.0
        output = 2*common - math.log2(Q[6]) - math.log2(Q[5])
        self.manifest('c', 'av', lambda m: m['calls'][0].update(
            common_scale_log2=common, output_scale_log2=output))
        self.mutate('c.plan.json', lambda p: next(v for v in p['values'] if v['id'] == '10').update(scale_log2=output))
        self.run_check(reject=True, error_pattern='capacity|headroom')

    def test_attention_output_descriptor_must_match_its_own_plan(self):
        self.mutate('b.plan.json', lambda p: next(v for v in p['values'] if v['id'] == '9').update(scale_log2=OUTPUT_SCALE+1))
        self.run_check(reject=True)

    def test_shared_linear_contract_is_not_relaxed_with_attention_namespaces(self):
        manifest = load(self.root / 'linear.json')
        for row in manifest['calls']:
            row['output_scale_log2'] += 1
        dump(self.root / 'other-linear.json', manifest)
        self.mutate('c.fixture.json', lambda f: f.update(compact_linear_manifest='other-linear.json'))
        self.run_check(reject=True)

    def test_mixed_attention_original_input_stale_hash_rejected(self):
        self.mutate('c.fixture.json', lambda f: f['inputs'][0]['values'].__setitem__(256, 0.333))
        self.run_check(reject=True)

    def test_mixed_attention_different_valid_canonical_input_rejected(self):
        def change(f):
            f['inputs'][0]['values'][256] = 0.333
            f['model']['input_sha256'] = input_sha(f)
        self.mutate('c.fixture.json', change)
        self.run_check(reject=True)

    def test_mixed_attention_full_config_still_shared(self):
        self.fixture_cases('c', [(field, lambda f, field=field, value=value: f['model']['config'].update({field: value}))
                                for field, value in (('ln_epsilon', 2e-5), ('ln_variance_center', 1.5),
                                                     ('weight_std', 0.03), ('intermediate', 1536),
                                                     ('head_dim', 64))])

    def test_mixed_attention_seed_still_shared(self):
        self.mutate('c.fixture.json', lambda f: f['model'].update(seed=20261115))
        self.run_check(reject=True)

    def test_seed_is_required_and_strictly_integer(self):
        cases = [('missing', lambda f: f['model'].pop('seed'))]
        cases += [(repr(value), lambda f, value=value: f['model'].update(seed=value))
                  for value in (True, '20261114', -1, 1.5)]
        self.fixture_cases('c', cases)

    def test_mixed_attention_all_parameter_hashes_still_shared(self):
        self.mutate('c.fixture.json', lambda f: f['model']['weight_sha256']['0'].update(ffn_ln_gamma=sha(b'different-ln')))
        self.run_check(reject=True)

    def test_mixed_attention_actual_linear_weights_still_checked(self):
        path = self.root / 'weights.bin'
        data = bytearray(path.read_bytes())
        data[0] = 1
        path.write_bytes(data)
        self.run_check(reject=True)

    def test_mixed_attention_context_still_shared(self):
        self.mutate('c.fixture.json', lambda f: f['context'].update(default_scale_log2=27))
        self.run_check(reject=True)

    def test_mixed_attention_initial_physical_descriptors_still_shared(self):
        def change(p):
            for value in p['values']:
                if value['id'] in ('0', '1'):
                    value['scale_log2'] += 1
        self.mutate('c.plan.json', change)
        self.run_check(reject=True)

    def test_noargs_and_help_are_guarded(self):
        for arguments in ([], ['--help']):
            with self.subTest(arguments=arguments):
                code, stdout, stderr, _ = self.invoke(arguments, case='help' if arguments else 'noargs')
                self.assertEqual(code, 0, stderr)
                self.assertIn('usage:', stdout)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runner', required=True, type=Path)
    parser.add_argument('--guard', type=Path)
    parser.add_argument('--output-dir', type=Path, default=OUTPUT_ROOT / 'tests')
    ARGS, remaining = parser.parse_known_args()
    ARGS.runner = ARGS.runner.resolve()
    if ARGS.guard:
        ARGS.guard = ARGS.guard.resolve()
    ARGS.output_dir = ARGS.output_dir.resolve()
    if not ARGS.output_dir.is_relative_to(OUTPUT_ROOT.resolve()):
        parser.error('--output-dir must be inside ' + str(OUTPUT_ROOT))
    if not ARGS.runner.is_file():
        parser.error('--runner must name an existing executable')
    if ARGS.guard and not ARGS.guard.is_file():
        parser.error('--guard must name an existing preload library')
    ARGS.output_dir.mkdir(parents=True, exist_ok=True)
    (ARGS.output_dir / 'invocations').mkdir(exist_ok=True)
    program = unittest.main(argv=[__file__] + remaining, exit=False)
    all_guarded = bool(ARGS.guard) and bool(RESULTS) and all(row['device_calls_zero'] for row in RESULTS)
    dump(ARGS.output_dir / 'no-device-gate.json', dict(
        passed=program.result.wasSuccessful(), tests_run=program.result.testsRun,
        invocation_count=len(RESULTS), guard_requested=bool(ARGS.guard),
        all_invocations_guarded_with_zero_device_calls=all_guarded,
        gpu_execution_or_numerical_equivalence_claimed=False, invocations=RESULTS))
    raise SystemExit(0 if program.result.wasSuccessful() else 1)
