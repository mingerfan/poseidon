import copy
import csv
from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from pt_agent.data import cifar
from pt_agent.dsl.ir import canonical
from pt_agent.dsl.parser import parse
from pt_agent.dsl.printer import print_program
from pt_agent.dsl.typecheck import check
from pt_agent.experiments import r9a_motivation as motivation
from pt_agent.experiments.r9a_sampler import (
    DEGS, MARGIN_RANGE, act_ops, anchor_configs, apply_config, random_config,
)
from pt_agent.intake import Graph, Node, TensorInfo
from pt_agent.paths import pt_result
from pt_agent.planners.rule import plan
from pt_agent.tests._artifacts import DATA, RESULTS_DIR, require


@require(DATA, RESULTS_DIR)
class MotivationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        names = ('x', 'pre_relu', 'relu_out', 'pre_silu', 'silu_out')
        cls.graph = Graph(
            name='tiny_mixed',
            nodes=[
                Node('linear_a', 'linear', ['x'], ['pre_relu'],
                     dict(weight='a.weight', bias=None), 'synthetic'),
                Node('relu', 'act', ['pre_relu'], ['relu_out'], dict(fn='relu'), 'synthetic'),
                Node('linear_b', 'linear', ['relu_out'], ['pre_silu'],
                     dict(weight='b.weight', bias=None), 'synthetic'),
                Node('silu', 'act', ['pre_silu'], ['silu_out'], dict(fn='silu'), 'synthetic'),
            ],
            tensors={name: TensorInfo(name, (1, 4), 'tensor') for name in names},
            weights={'a.weight': np.eye(4), 'b.weight': np.eye(4)},
            inputs=['x'], outputs=['silu_out'],
        )
        cls.calibration = {'pre_relu': dict(min=-2., max=3.),
                           'pre_silu': dict(min=-5., max=2.)}
        cls.ranges = [cls.calibration['pre_relu'], cls.calibration['pre_silu']]
        cls.base = parse(plan(cls.graph, cls.calibration, deg=63, margin=1.5))
        cls.weight_shapes = {name: weight.shape for name, weight in cls.graph.weights.items()}

    def test_01_seeded_sampling_and_non_relu_methods(self):
        acts = act_ops(self.base)
        first_rng = np.random.default_rng(20261023)
        second_rng = np.random.default_rng(20261023)
        first = [random_config(acts, first_rng) for _ in range(100)]
        second = [random_config(acts, second_rng) for _ in range(100)]
        self.assertEqual(first, second)
        self.assertEqual(DEGS, (15, 31, 63, 127, 255))
        self.assertEqual(MARGIN_RANGE, (1.0, 1.6))
        for config in first:
            self.assertEqual(len(config), len(acts))
            for op, settings in zip(acts, config):
                self.assertIn(settings['method'], ('cheb', 'minimax_comp'))
                self.assertGreaterEqual(settings['margin'], MARGIN_RANGE[0])
                self.assertLessEqual(settings['margin'], MARGIN_RANGE[1])
                self.assertEqual(settings['margin'], round(settings['margin'], 2))
                if op.attrs['fn'] != 'relu':
                    self.assertEqual(settings['method'], 'cheb')
                if settings['method'] == 'cheb':
                    self.assertIn(settings['deg'], DEGS)
                else:
                    self.assertIsNone(settings['deg'])

    def assert_structure_and_typecheck(self, changed):
        self.assertTrue(check(changed, self.weight_shapes).ok)
        restored = canonical(changed)
        original = canonical(self.base)
        self.assertEqual(len(restored['ops']), len(original['ops']))
        permitted = {'method', 'deg', 'degrees', 'domain'}
        for old, new in zip(original['ops'], restored['ops']):
            if old[0] == 'act':
                self.assertEqual(set(old[3]), set(new[3]))
                for key in set(old[3])-permitted:
                    self.assertEqual(old[3][key], new[3][key])
                new[3] = copy.deepcopy(old[3])
        self.assertEqual(restored, original)
        reparsed = parse(print_program(changed))
        self.assertEqual(canonical(reparsed), canonical(changed))
        self.assertTrue(check(reparsed, self.weight_shapes).ok)

    def test_02_apply_config_domains_and_structure(self):
        original = canonical(self.base)
        config = [dict(method='cheb', deg=31, margin=1.2),
                  dict(method='cheb', deg=127, margin=1.1)]
        changed = apply_config(self.base, self.ranges, config)
        self.assertIsNot(changed, self.base)
        self.assertEqual(canonical(self.base), original)
        for op, bounds, settings in zip(act_ops(changed), self.ranges, config):
            self.assertEqual(op.attrs['method'], 'cheb')
            self.assertEqual(op.attrs['deg'], settings['deg'])
            self.assertIsNone(op.attrs['degrees'])
            self.assertEqual(op.attrs['domain'],
                             (bounds['min']*settings['margin'], bounds['max']*settings['margin']))
        self.assert_structure_and_typecheck(changed)

        mixed = [dict(method='minimax_comp', deg=None, margin=1.35),
                 dict(method='cheb', deg=15, margin=1.05)]
        changed = apply_config(self.base, self.ranges, mixed)
        first, second = act_ops(changed)
        bound = 1.35*max(abs(self.ranges[0]['min']), abs(self.ranges[0]['max']))
        self.assertEqual(first.attrs['method'], 'minimax_comp')
        self.assertIsNone(first.attrs['deg'])
        self.assertEqual(tuple(first.attrs['degrees']), (15, 15, 27))
        self.assertEqual(first.attrs['domain'], (-bound, bound))
        self.assertEqual(second.attrs['domain'], (-5.*1.05, 2.*1.05))
        self.assert_structure_and_typecheck(changed)
        self.assertEqual(canonical(self.base), original)

    def test_03_anchor_counts_fallback_and_rule_equivalence(self):
        for name in ('resnet20', 'mlp'):
            with self.subTest(model=name):
                program = parse(pt_result(
                    f'r8e7-reconfirm-20261022T122558/models/{name}/program.ptdsl').read_text())
                self.assertEqual(len(anchor_configs(act_ops(program))), 12)
        anchors = anchor_configs(act_ops(self.base))
        expected = {f'uniform-cheb-{deg}-{margin}' for deg in DEGS for margin in (1.2, 1.5)}
        expected |= {f'uniform-minimax-{margin}' for margin in (1.2, 1.5)}
        self.assertEqual(set(anchors), expected)
        chosen = apply_config(self.base, self.ranges, anchors['uniform-cheb-63-1.5'])
        reference = parse(plan(self.graph, self.calibration, deg=63, margin=1.5))
        self.assertEqual(canonical(chosen), canonical(reference))
        self.assertTrue(check(chosen, self.weight_shapes).ok)
        for margin in (1.2, 1.5):
            relu, silu = anchors[f'uniform-minimax-{margin}']
            self.assertEqual(relu, dict(method='minimax_comp', deg=None, margin=margin))
            self.assertEqual(silu, dict(method='cheb', deg=63, margin=margin))
        non_relu = [op for op in act_ops(self.base) if op.attrs['fn'] != 'relu']
        self.assertEqual(len(anchor_configs(non_relu)), 10)

    def test_04_count_nonblank_noncomment_lines(self):
        python = ('\n# header\n  # indented comment\ndef example():  # inline comment\n'
                  '    label = "# data"\n\n    # another comment\n    return label\n')
        cpp = ('\n// header\n   // indented comment\nint main() {\n'
               '    return 0; // inline comment\n}\n\n')
        self.assertEqual(motivation.count_lines(python), 3)
        self.assertEqual(motivation.count_lines(cpp, language='cpp'), 3)
        self.assertEqual(motivation.count_lines('\n # comment\n'), 0)

    def test_05_range_ratios_only_use_applicable_sides(self):
        calibration_min = np.array([-2., -4., 0., -4., 2.])
        calibration_max = np.array([4., 0., 4., -1., 5.])
        unseen_min = np.array([-3., -8., 0., -2., 1.])
        unseen_max = np.array([6., -1., 2., -.5, 10.])
        expected = [(1.5, 1.5, 1.5), (None, 2., 2.), (1., None, 1.),
                    (None, 1., 1.), (2., None, 2.)]
        for lo, hi, observed_lo, observed_hi, values in zip(
                calibration_min, calibration_max, unseen_min, unseen_max, expected):
            with self.subTest(calib_min=lo, calib_max=hi):
                result = motivation.range_ratio(lo, hi, observed_lo, observed_hi)
                self.assertEqual(result, dict(zip(('ratio_hi', 'ratio_lo', 'ratio'), values)))

    def test_06_resume_skips_existing_keys_and_appends_new_rows(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary)/'rows.jsonl'
            old = dict(key='old', measured=7)
            path.write_text(json.dumps(old)+'\n')
            items = [dict(key='old', config='keep'), dict(key='new', config='measure')]
            evaluate = Mock(side_effect=lambda item: dict(key=item['key'], measured=9))
            rows = motivation.evaluate_pending(path, items, evaluate)
            self.assertIsInstance(rows, list)
            evaluate.assert_called_once_with(items[1])
            stored = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(stored, [old, dict(key='new', measured=9)])
            evaluate.reset_mock()
            motivation.evaluate_pending(path, items, evaluate)
            evaluate.assert_not_called()
            self.assertEqual([json.loads(line) for line in path.read_text().splitlines()], stored)

    def assert_data_indices(self, loader, expected_indices):
        reads = []
        normalization = ((.1, .2, .3), (1., 2., 3.))

        def read_rows(split, indices):
            indices = list(indices)
            self.assertEqual(split, 'test')
            self.assertTrue(all(8000 <= index < 10000 for index in indices))
            reads.extend(indices)
            return (np.zeros((len(indices), 3, 32, 32), dtype=np.uint8),
                    np.array(indices, dtype=np.int64) % 10)

        with patch.object(cifar, 'read_rows', side_effect=read_rows), \
                patch.object(cifar, 'report_test_set', side_effect=AssertionError('report set must stay unread')):
            images, labels = loader(normalization=normalization)
        self.assertEqual(reads, list(expected_indices))
        self.assertEqual(len(images), len(expected_indices))
        self.assertEqual(images.shape[1:], (3, 32, 32))
        np.testing.assert_array_equal(labels, np.array(list(expected_indices), dtype=np.int64) % 10)
        expected_image = cifar.normalize(np.zeros((1, 3, 32, 32), dtype=np.uint8), normalization)
        np.testing.assert_array_equal(images[:1], expected_image)

    def test_07_e1_reads_only_heldout_200(self):
        self.assert_data_indices(motivation.load_e1_data, range(9000, 9200))

    def test_08_e4_reads_confirm_and_heldout_union(self):
        self.assert_data_indices(motivation.load_e4_data, range(8000, 10000))

    def test_09_e6_summary_retains_failed_key_and_compares_only_successes(self):
        with TemporaryDirectory() as temporary:
            out = Path(temporary)
            directory = out/'e6'
            directory.mkdir()
            failed = dict(key='mlp/random/004', model='mlp', kind='random',
                          error='ValueError: synthetic lowering failure')
            successful = dict(key='mlp/anchor/good', program='mlp/anchor/good', model='mlp',
                              kind='pair', boots_pred=4, boots_lazy=6, boots_plan=4,
                              latency_pred=110., latency_lazy=140., latency_plan=100.)
            rows_path = directory/'rows.jsonl'
            rows_path.write_text(''.join(json.dumps(row)+'\n' for row in (failed, successful)))
            motivation._summarize(out, 'e6')
            summary = json.loads((directory/'summary.json').read_text())
            self.assertEqual(summary['rows'], 2)
            self.assertEqual(summary['errors'], 1)
            self.assertEqual(summary['models']['mlp'], dict(rows=2, errors=1))
            depth = summary['depth-dp']
            self.assertEqual(depth['pairs'], 1)
            self.assertEqual(depth['boots_exact_fraction'], 1.)
            self.assertEqual(depth['boots_mae'], 0.)
            self.assertAlmostEqual(depth['latency_relative_error_median'], .1)
            self.assertAlmostEqual(depth['latency_relative_error_max'], .1)
            lazy = summary['lazy']
            self.assertEqual(lazy['pairs'], 1)
            self.assertEqual(lazy['boots_exact_fraction'], 0.)
            self.assertEqual(lazy['boots_mae'], 2.)
            self.assertAlmostEqual(lazy['latency_relative_error_median'], .4)
            self.assertAlmostEqual(lazy['latency_relative_error_max'], .4)
            with (directory/'pairs.csv').open(newline='') as stream:
                pairs = list(csv.DictReader(stream))
            self.assertEqual([row['program'] for row in pairs], [failed['key'], successful['key']])
            self.assertEqual(pairs[0]['error'], failed['error'])
            self.assertEqual(pairs[1]['boots_plan'], '4')
            self.assertEqual([json.loads(line) for line in rows_path.read_text().splitlines()],
                             [failed, successful])

    def test_10_compile_resume_recovers_plan_without_repeating_native_call(self):
        with TemporaryDirectory() as temporary:
            out = Path(temporary)
            args = SimpleNamespace(out=out, model='mlp', models_dir=out/'models')
            item = dict(key='mlp/anchor/uniform-cheb-63-1.5', model='mlp', kind='anchor',
                        config=anchor_configs(act_ops(self.base))['uniform-cheb-63-1.5'])
            case = out/'e6/programs/mlp/anchor-uniform-cheb-63-1.5'
            case.mkdir(parents=True)
            native = out/'retained-native'
            native.mkdir()
            report_path = case/'REPORT.json'
            original = dict(compile=dict(exit=0, seconds=1.), emit=dict(run_dir=str(native)),
                            complete=False, stage='compile')
            report_path.write_text(json.dumps(original))
            source = print_program(self.base)
            (case/'program.ptdsl').write_text(source)
            predicted = dict(boots_pred=4, latency_pred_us=110.)
            lazy = dict(boots_pred=6, latency_pred_us=140.)
            stats = dict(boots=4, steps=10)
            latency = dict(latency_us=100.)
            pruned = dict(removed=['temporary.mlir'])
            with patch('pt_agent.run._inside', side_effect=AssertionError('native call repeated')) as run, \
                    patch('pt_agent.cost.estimate', side_effect=[predicted, lazy]) as estimate, \
                    patch('pt_agent.toolchain.plan_stats', return_value=stats) as plan_stats, \
                    patch('pt_agent.cost.plan_latency', return_value=latency) as plan_latency, \
                    patch('pt_agent.toolchain.prune', return_value=pruned) as prune:
                row = motivation._compile_pair(args, item, self.graph, self.calibration,
                                               self.base, self.ranges)
            run.assert_not_called()
            self.assertEqual([call.kwargs['placement_model'] for call in estimate.call_args_list],
                             ['depth-dp', 'lazy'])
            plan_stats.assert_called_once_with(native)
            plan_latency.assert_called_once_with(native/'lowered._hecate_golden.runtime-plan.json')
            prune.assert_called_once_with(native)
            self.assertEqual((row['key'], row['program']), (item['key'], item['key']))
            self.assertEqual((row['boots_pred'], row['boots_lazy'], row['boots_plan']), (4, 6, 4))
            self.assertEqual((row['latency_pred'], row['latency_lazy'], row['latency_plan']),
                             (110., 140., 100.))
            self.assertEqual(row['report_path'], str(report_path))
            saved = json.loads(report_path.read_text())
            self.assertEqual(saved['compile'], original['compile'])
            self.assertEqual(saved['emit'], original['emit'])
            self.assertEqual(saved['plan'], stats)
            self.assertEqual(saved['plan_latency'], latency)
            self.assertEqual(saved['pruned'], pruned)
            self.assertTrue(saved['complete'])
            self.assertEqual(saved['stage'], 'complete')
            self.assertEqual((case/'program.ptdsl').read_text(), source)


    def test_11_e1confirm_selects_prescribed_union_once(self):
        def row(suffix, latency, feasible=True, kind='random'):
            return dict(key='mlp/'+suffix, model='mlp', kind=kind,
                        latency_pred_us=latency, feasible_like=feasible)

        frozen = row('reference/r8e7', 100, kind='reference')
        fast = row('random/fast', 90)
        frontier = row('random/frontier', 110)
        anchor = row('anchor/slow', 120, kind='anchor')
        reference = row('reference/other', 300, False, 'reference')
        rejected = [row('random/equal', 100), row('random/slow', 130),
                    row('random/failed-frontier', 10, False),
                    row('anchor/failed', 20, False, 'anchor')]
        rows = [frozen, fast, frontier, anchor, reference, fast, *rejected]
        original = copy.deepcopy(rows)
        selected = motivation.select_e1confirm(rows, {
            'mlp': [frontier['key'], rejected[2]['key']]})
        self.assertEqual({item['key'] for item in selected},
                         {item['key'] for item in (frozen, fast, frontier, anchor, reference)})
        self.assertEqual(len(selected), 5)
        self.assertEqual(rows, original)

    def test_12_e1confirm_reads_only_fixed_2000(self):
        self.assert_data_indices(motivation.load_e1confirm_data, range(8000, 10000))

    def test_13_e6_matched_uses_default_header_dispatch(self):
        from pt_agent.cost import PROFILES, estimate

        with TemporaryDirectory() as temporary:
            source = Path(temporary)/'program.ptdsl'
            for refresh, placement in (('agent', 'lazy'), ('compiler', 'depth-dp')):
                with self.subTest(refresh=refresh):
                    program = replace(self.base, header=replace(
                        self.base.header, refresh=refresh,
                        l_max=PROFILES['qwen48']['input_level'],
                        boot_out_level=PROFILES['qwen48']['boot_out_level']))
                    source.write_text(print_program(program))
                    expected = estimate(program, self.graph.weights, placement_model=placement)
                    item = dict(key='mlp/pair/'+refresh, boots_pred=12, boots_lazy=14,
                                latency_pred=100., latency_lazy=120., boots_plan=11)
                    with patch('pt_agent.cost.estimate', wraps=estimate) as spy:
                        result = motivation._matched_pair(item, self.graph.weights, source)
                    self.assertEqual(spy.call_count, 1)
                    self.assertEqual(spy.call_args.kwargs, {})
                    self.assertEqual(result['placement_matched'], placement)
                    self.assertEqual(result['boots_matched'], expected['boots_pred'])
                    self.assertEqual(result['latency_matched'], expected['latency_pred_us'])
                    for key in ('boots_pred', 'boots_lazy', 'latency_pred', 'latency_lazy', 'boots_plan'):
                        self.assertEqual(result[key], item[key])

    def test_14_expert_mlp_manual_decisions_and_csv_source(self):
        manual = motivation.expert_decisions('MLP', 'h2 = h1 * h1')
        self.assertEqual(motivation.MANUAL_DECISIONS['MLP'], dict(
            activations=1, linear_layers=2,
            note='square activation h1*h1; two hand-written diagonal-style linear loops'))
        self.assertEqual(manual['decisions'], 2)
        self.assertEqual(manual['decisions_source'], 'manual')
        self.assertEqual(manual['source'], 'manual')
        self.assertEqual(manual['approximation_decisions'], 0)
        regex = motivation.expert_decisions('ResNet', 'HE_ReLU(x)\nHE_Conv(x)')
        self.assertEqual(regex, dict(decisions=4, decisions_source='regex'))
        with TemporaryDirectory() as temporary:
            path = Path(temporary)/'loc.csv'
            motivation._csv(path, [dict(model='MLP', **manual)],
                            ('model', 'decisions', 'decisions_source'))
            with path.open(newline='') as stream:
                self.assertEqual(list(csv.DictReader(stream)), [
                    dict(model='MLP', decisions='2', decisions_source='manual')])

    def test_15_e1confirm_summary_confusion_and_failure_denominator(self):
        def row(key, before, after, latency, drop=0., diverged=0):
            return dict(key='mlp/'+key, model='mlp', kind='reference' if key.startswith('reference/') else 'random',
                        feasible_like_200=before, robust_feasible=after,
                        latency_pred_us=latency, acc_drop_pp=drop, n_diverged=diverged)

        rows = [row('reference/r8e7', True, True, 100),
                row('random/only-200', True, False, 50, 1.),
                row('reference/other', False, True, 80, .2),
                row('random/neither', False, False, 70, 2., 1),
                dict(key='mlp/random/error', model='mlp', error='synthetic error')]
        with TemporaryDirectory() as temporary:
            out = Path(temporary)
            directory = out/'e1confirm'
            directory.mkdir()
            (directory/'rows.jsonl').write_text(''.join(json.dumps(item)+'\n' for item in rows))
            motivation._summarize(out, 'e1confirm')
            summary = json.loads((directory/'summary.json').read_text())
            result = summary['models']['mlp']
            self.assertEqual((summary['rows'], summary['errors']), (5, 1))
            self.assertEqual(result['confirmed_count'], 5)
            self.assertEqual(result['robust_feasible_count'], 2)
            self.assertEqual(result['robust_feasible_fraction'], .4)
            self.assertEqual(result['fastest_latency_us'], 80)
            self.assertEqual(result['fastest_to_frozen_ratio'], .8)
            self.assertEqual(result['frozen']['acc_drop_pp'], 0.)
            self.assertEqual(result['frozen']['n_diverged'], 0)
            self.assertEqual(result['confusion'], dict(
                both_feasible=1, only_200_feasible=1, only_2000_feasible=1, neither_feasible=1))
            self.assertEqual(result['confusion_evaluated'], 4)


if __name__ == '__main__':
    unittest.main()
