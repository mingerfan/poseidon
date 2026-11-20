"""Pure-array metadata/math preparation tests; no native, keys or model inputs."""
from array import array
from argparse import Namespace
import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from pt_agent.experiments import r31_lattigo_profile_subset as subset


def coverage_fixture():
    rows = []
    for name in subset.MODELS:
        cells = []
        for cell in subset.EXPECTED_CELLS:
            if name == 'ResNet20' and cell['op'] != 'mod_switch':
                continue
            row = copy.deepcopy(cell)
            row.update(source_count=2, representative_attrs=(dict(steps=1023)
                       if cell['op'] == 'rotate' else copy.deepcopy(cell['attrs'])))
            cells.append(row)
        rows.append(dict(model=name, source_plan=name+'-original.runtime-plan.json',
                         all_metadata_covered=False, uncovered_cells=cells))
    return dict(status='metadata_only_cost_transfer_coverage', rows=rows)


def prepared_fixture(root):
    path = root/'prepared'; path.mkdir()
    binary = path/'untouched-native-client'; binary.write_text('metadata fixture only\n')
    record = dict(status='ok', phase='prepare', client_cache_protocol=9,
                  supported_precision_bits=[43,50], supported_mod1_inv_degrees=[7,15],
                  supported_log_message_ratios=[2], supported_bootstrap_normalizations_log2=[0,1,2,3],
                  supports_input_ssa_check=True, ready_binary=str(binary))
    (path/'result.json').write_text(json.dumps(record)+'\n')
    return path


class ProfileSubsetChecks(unittest.TestCase):
    def test_exact_three_cells_and_original_metadata_preserved(self):
        record = coverage_fixture(); before = copy.deepcopy(record)
        selected = subset.select_missing_cells(record)
        self.assertEqual(record, before)
        self.assertEqual({row['cell']['op'] for row in selected}, {'mod_switch','mul_cp','rotate'})
        drop = next(row for row in selected if row['cell']['op'] == 'mod_switch')
        self.assertEqual(set(drop['source_counts']), set(subset.MODELS))
        selected[0]['cell']['output']['level'] = 99
        self.assertEqual(record, before)

    def test_full_metadata_and_representative_step_are_required(self):
        changes = (
            lambda r:r['rows'][1]['uncovered_cells'][1]['output'].update(scale_log2=50),
            lambda r:r['rows'][1]['uncovered_cells'][2]['representative_attrs'].update(steps=-1023),
            lambda r:r['rows'][1]['uncovered_cells'][0]['attrs'].update(target_level=True),
            lambda r:r['rows'][1]['uncovered_cells'].append(copy.deepcopy(r['rows'][1]['uncovered_cells'][0])),
            lambda r:r['rows'].append(copy.deepcopy(r['rows'][0])),
        )
        for change in changes:
            record = coverage_fixture(); change(record)
            with self.subTest(record=record), self.assertRaises(ValueError):
                subset.select_missing_cells(record)

    def test_all_models_and_positive_counts_required(self):
        record = coverage_fixture(); record['rows'].pop()
        with self.assertRaises(ValueError): subset.select_missing_cells(record)
        record = coverage_fixture(); record['rows'][0]['uncovered_cells'][0]['source_count'] = True
        with self.assertRaises(ValueError): subset.select_missing_cells(record)

    def test_eight_operation_plan_has_three_target_cells_and_strict_full_reference(self):
        selected = subset.select_missing_cells(coverage_fixture())
        source = subset.make_subset_source(selected); before = copy.deepcopy(source)
        binding = subset.NativeBinding(50, mod1_inv_degree=15, log_message_ratio=2,
                                       bootstrap_normalization_log2=3)
        plan, manifest, metadata = subset.make_profile_plan(source, binding=binding)
        inventory = subset.plan_inventory(plan, manifest, binding=binding)
        self.assertEqual(source, before)
        self.assertEqual((metadata['cell_count'], metadata['setup_and_aggregation_count'],
                          inventory['execution_count'], inventory['bootstrap_count']), (3,5,8,0))
        self.assertEqual(inventory['rotation_steps'], [1023])
        self.assertEqual(inventory['encode_count'], 1)
        target_ordinals = {row['target_ordinal'] for row in metadata['target_operations']}
        self.assertEqual(len(target_ordinals), 3)
        self.assertEqual(sum(op['ordinal'] not in target_ordinals for op in plan['execution']), 5)
        packed = subset.deterministic_fixture()
        constants = {row['content']:array('d', [row['value']])*subset.SLOTS
                     for row in metadata['constant_vectors']}
        observed, summary = subset.replay_profile(plan, constants, packed)
        direct = subset.direct_profile_sum(metadata, packed, precision_bits=50)
        comparison = subset.compare_references(direct, observed)
        self.assertTrue(comparison['passed'])
        self.assertEqual(comparison['shape'], [16384])
        self.assertEqual(comparison['max_absolute_error'], 0.)
        self.assertEqual(summary['execution_completed'], 8)
        # Independent explicit equation: drop(x) + x*0.125 + rotate_left(x,1023).
        expected = [(packed[i]+packed[i]*.125)+packed[(i+1023)%subset.SLOTS]
                    for i in range(subset.SLOTS)]
        self.assertEqual(list(observed), expected)
        self.assertEqual(subset.compare_references([0.], [1e-11])['passed'], False)

    def test_complete_serialization_and_reused_native_command_without_execution(self):
        with TemporaryDirectory() as temp:
            root = Path(temp); coverage = root/'coverage.json'
            original = coverage_fixture(); coverage.write_text(json.dumps(original)+'\n')
            prepared = prepared_fixture(root); original_prepared = json.loads((prepared/'result.json').read_text())
            args = Namespace(coverage=coverage, prepared=prepared, out=root/'new-preparation')
            result = subset.prepare(args)
            self.assertEqual(result['status'], 'prepared', result)
            self.assertEqual(json.loads(coverage.read_text()), original)
            self.assertEqual(json.loads((prepared/'result.json').read_text()), original_prepared)
            self.assertEqual(result['reference_comparison']['shape'], [16384])
            self.assertEqual((result['cell_count'],result['execution_count']), (3,8))
            self.assertEqual(Path(result['input']).stat().st_size, 16384*8)
            plan = json.loads(Path(result['output_plan']).read_text())
            manifest = json.loads((Path(result['output_bundle'])/'manifest.json').read_text())
            inventory = subset.plan_inventory(plan, manifest, binding=subset.NativeBinding(50))
            self.assertEqual(inventory, result['inventory'])
            expected = json.loads(Path(result['expected']).read_text())
            self.assertEqual(expected['indices'], list(range(16384)))
            self.assertEqual(len(expected['values']), 16384)
            self.assertEqual(result['prepared_client']['binary_executions'], 0)
            self.assertTrue(all(result[key] == 0 for key in ('native_executions','keygen_runs',
                'encryption_runs','bootstrap_runs','decryption_runs','model_forwards','calibration_reads',
                'heldout_reads','llm_calls','downloads','installations','extra_integrity_checks')))
            argv = result['future_execute_argv']
            self.assertEqual(argv[4], 'execute')
            self.assertEqual(argv[argv.index('--prepared')+1], str(prepared))
            self.assertNotIn('prepare', argv)
            self.assertNotIn('--allow-bootstrap', argv)
            self.assertEqual(argv[argv.index('--seconds')+1], '300')
            self.assertEqual(argv[argv.index('--native-rss-gib')+1], '2')
            self.assertLess(sum(path.stat().st_size for path in args.out.rglob('*') if path.is_file()), subset.CASE_BYTES)

    def test_invalid_prepared_support_and_source_save_partial_without_native(self):
        with TemporaryDirectory() as temp:
            root = Path(temp); prepared = prepared_fixture(root)
            metadata = json.loads((prepared/'result.json').read_text())
            metadata['supported_mod1_inv_degrees'] = [7]
            (prepared/'result.json').write_text(json.dumps(metadata))
            with self.assertRaises(ValueError): subset.validate_prepared(prepared)
            source = root/'coverage.json'; source.write_text(json.dumps(coverage_fixture()))
            out = root/'failed-preparation'
            result = subset.prepare(Namespace(coverage=source,prepared=prepared,out=out))
            self.assertEqual(result['status'], 'error')
            self.assertEqual(result['native_executions'], 0)
            self.assertTrue((out/'preparation.json').is_file())
            self.assertFalse((out/'input.bin').exists())
            with self.assertRaises(FileExistsError):
                subset.prepare(Namespace(coverage=source,prepared=prepared,out=out))

    def test_future_allowance_is_finite_and_extra_subset_cells_rejected(self):
        selected = subset.select_missing_cells(coverage_fixture())
        with self.assertRaises(ValueError): subset.make_subset_source(selected+selected[:1])
        for seconds in (0,901,True):
            with self.assertRaises(ValueError):
                subset.execution_command({},out=Path('unused'),prepared=Path('unused'),seconds=seconds)


if __name__ == '__main__':
    unittest.main()
