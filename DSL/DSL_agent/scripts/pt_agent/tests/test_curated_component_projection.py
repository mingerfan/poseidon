"""Synthetic metadata checks for admitting an exact-source component point."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pt_agent.experiments import experiment_delivery_curated_refresh as subject


COMPONENTS = (
    'covered_operator_core_seconds',
    'bootstrap_galois_generation_seconds',
    'residual_galois_generation_seconds',
    'fixed_key_generation_seconds',
    'constructor_dft_seconds',
    'constructor_other_seconds',
    'plaintext_read_encode_transfer_seconds',
    'cold_parameters_input_encryption_final_seconds',
)


class CuratedComponentProjectionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.results = self.root / 'results'
        self.results.mkdir()
        self.case = self.results / 'cold-vgg'
        self.plan = self.case / 'post-freeze/emit/model.runtime-plan.json'
        self.record = dict(compiled_bootstraps=156, phases=dict(compile=dict(
            runtime_plan=str(self.plan), compiled_bootstraps=156,
            execution_ops=dict(boot=156, rotate=3, mul_cp=2))))
        parts = {name: float(index) for index, name in enumerate(COMPONENTS, 1)}
        self.artifact = dict(
            rows=[dict(model='VGG16', source_case=str(self.case),
                source_result=str(self.case / 'result.json'), source_plan=str(self.plan),
                source_check=dict(compiled_path_matches=True, clean_native_compile=True,
                    same_model_quality_program_source=True),
                source_operations=161, source_op_counts=dict(boot=156, rotate=3, mul_cp=2),
                bootstrap_count=156, unique_cells=38, covered_cells=38,
                projected_native_variant='synthetic-n16-scale50-c2s52-p61',
                parameter_group='Original compiler profile; independent synthetic N16 projection',
                symbolic_metadata_mapping='43->50 and86->100; unchanged instruction inventory',
                profile_prediction_source=str(self.root / 'independent-prediction.json'),
                profile_audit_source=str(self.root / 'independent-profile-audit.json'),
                component_point_seconds=parts, engineering_component_point_seconds=sum(parts.values()),
                formal_complete_prediction_seconds=None,
                recording_and_checkpoint_overhead_seconds=None,
                native_mathematical_acceptance=None, security_acceptance=None,
                per_cell_repeated_variance=None)],
            measurements_per_cell=1, per_cell_repeated_variance=None,
            full_native_model_clock_inputs=0, formal_complete_prediction_seconds=None,
            task_accuracy_acceptance=None, security_acceptance=None,
            new_actions=dict(model_runs=0, crypto_runs=0, API_calls=0, explicit_integrity_checks=0))
        self.projection = self.root / 'component.json'
        self.write(self.projection, self.artifact)
        self.cell = subject.pending_cell(self.case, {})
        self.cell.update(primary_matrix_cell=True,
            metadata_admission=dict(status='accepted'),
            bootstrap_count=156, native_compile_success=True)
        self.delivery = dict(cells=[self.cell], coverage={}, comparison_rules=['STALE_RULE'],
            attribution_exports=dict(rules=['STALE_ATTRIBUTION']))
        self.manifest = dict(schema='predeclared-vgg-curated-refresh-v1',
            cold_case=str(self.case), retained_compile_case=str(self.results / 'retained'),
            retained_session_case=str(self.results / 'session'), reports=[
                dict(case=str(self.results / 'cold-report'), frozen_case=str(self.case), compiled_case=str(self.case)),
                dict(case=str(self.results / 'retained-report'), frozen_case=str(self.results / 'session'),
                    compiled_case=str(self.results / 'retained'))])
        for target in ('subprocess.Popen', 'socket.getaddrinfo', 'socket.socket.connect',
                       'urllib.request.urlopen'):
            blocker = patch(target, side_effect=AssertionError('No execution or network in metadata tests'))
            blocker.start()
            self.addCleanup(blocker.stop)

    def write(self, path, value):
        Path(path).write_text(json.dumps(value, allow_nan=True))

    def gate(self, artifact=None, record=None):
        return subject.validate_cold_component_projection(
            self.artifact if artifact is None else artifact, case=self.case,
            record=self.record if record is None else record)

    def test_valid_exact_source_returns_point_with_no_formal_acceptance(self):
        before = copy.deepcopy(self.artifact)
        observed = self.gate()
        self.assertEqual(observed['seconds'], 36.0)
        self.assertEqual(observed['source_result'], str(self.case / 'result.json'))
        self.assertEqual(observed['source_plan'], str(self.plan))
        self.assertIsInstance(observed['scope'], str)
        self.assertTrue(observed['scope'])
        self.assertEqual(self.artifact, before)

    def test_only_one_vgg_row_is_accepted(self):
        for rows in ([], self.artifact['rows'] * 2,
                     [dict(self.artifact['rows'][0], model='AlexNet')]):
            with self.subTest(rows=len(rows)), self.assertRaises(ValueError):
                self.gate(dict(self.artifact, rows=rows))

    def test_foreign_case_result_or_plan_is_refused(self):
        for key in ('source_case', 'source_result', 'source_plan'):
            value = copy.deepcopy(self.artifact)
            value['rows'][0][key] = str(self.results / ('foreign-' + key))
            with self.subTest(field=key), self.assertRaises(ValueError):
                self.gate(value)

    def test_compile_plan_change_is_refused_even_when_case_matches(self):
        record = copy.deepcopy(self.record)
        record['phases']['compile']['runtime_plan'] = str(self.case / 'other.runtime-plan.json')
        with self.assertRaises(ValueError):
            self.gate(record=record)

    def test_source_proof_flags_require_native_true(self):
        for key in self.artifact['rows'][0]['source_check']:
            for wrong in (False, 1, None):
                value = copy.deepcopy(self.artifact)
                value['rows'][0]['source_check'][key] = wrong
                with self.subTest(flag=key, value=wrong), self.assertRaises(ValueError):
                    self.gate(value)

    def test_nonzero_actions_and_boolean_or_float_counters_are_refused(self):
        for key, wrong in (('full_native_model_clock_inputs', 1),
                           ('full_native_model_clock_inputs', False),
                           ('full_native_model_clock_inputs', 0.0),
                           ('measurements_per_cell', True), ('measurements_per_cell', 1.0),
                           ('measurements_per_cell', 2)):
            value = copy.deepcopy(self.artifact)
            value[key] = wrong
            with self.subTest(field=key, value=wrong), self.assertRaises(ValueError):
                self.gate(value)
        for actions in ({}, dict(model_runs=1), dict(API_calls=False), dict(crypto_runs=0.0)):
            with self.subTest(actions=actions), self.assertRaises(ValueError):
                self.gate(dict(self.artifact, new_actions=actions))

    def test_formal_accuracy_security_variance_and_recording_claims_are_refused(self):
        top = ('per_cell_repeated_variance', 'formal_complete_prediction_seconds',
               'task_accuracy_acceptance', 'security_acceptance')
        row = ('formal_complete_prediction_seconds', 'recording_and_checkpoint_overhead_seconds',
               'native_mathematical_acceptance', 'security_acceptance', 'per_cell_repeated_variance')
        for location, keys in (('top', top), ('row', row)):
            for key in keys:
                value = copy.deepcopy(self.artifact)
                target = value if location == 'top' else value['rows'][0]
                target[key] = 0
                with self.subTest(location=location, field=key), self.assertRaises(ValueError):
                    self.gate(value)
                missing = copy.deepcopy(self.artifact)
                target = missing if location == 'top' else missing['rows'][0]
                del target[key]
                with self.subTest(location=location, missing=key), self.assertRaises(ValueError):
                    self.gate(missing)

    def test_operation_inventory_must_match_native_compile_counts_and_total(self):
        variants = []
        value = copy.deepcopy(self.artifact)
        value['rows'][0]['source_operations'] += 1
        variants.append(value)
        value = copy.deepcopy(self.artifact)
        value['rows'][0]['source_op_counts']['rotate'] += 1
        variants.append(value)
        value = copy.deepcopy(self.artifact)
        value['rows'][0]['source_op_counts']['extra'] = 0
        variants.append(value)
        for value in variants:
            with self.subTest(row=value['rows'][0]), self.assertRaises(ValueError):
                self.gate(value)
        for bad in (True, 3.0, -1):
            record = copy.deepcopy(self.record)
            record['phases']['compile']['execution_ops']['rotate'] = bad
            with self.subTest(counter=bad), self.assertRaises(ValueError):
                self.gate(record=record)

    def test_bootstrap_count_requires_agreement_at_all_three_sources(self):
        value = copy.deepcopy(self.artifact)
        value['rows'][0]['bootstrap_count'] = 155
        with self.assertRaises(ValueError):
            self.gate(value)
        for key in ('top', 'compile'):
            record = copy.deepcopy(self.record)
            target = record if key == 'top' else record['phases']['compile']
            target['compiled_bootstraps'] = 155
            with self.subTest(record=key), self.assertRaises(ValueError):
                self.gate(record=record)

    def test_complete_38cell_coverage_is_required(self):
        for key in ('unique_cells', 'covered_cells'):
            for wrong in (37, 39, 38.0, True):
                value = copy.deepcopy(self.artifact)
                value['rows'][0][key] = wrong
                with self.subTest(field=key, value=wrong), self.assertRaises(ValueError):
                    self.gate(value)

    def test_component_keys_finiteness_nonnegativity_and_sum_are_checked(self):
        for bad in (float('nan'), float('inf'), -1.0, True):
            value = copy.deepcopy(self.artifact)
            value['rows'][0]['component_point_seconds'][COMPONENTS[0]] = bad
            with self.subTest(component=bad), self.assertRaises(ValueError):
                self.gate(value)
        for change in ('missing', 'extra', 'sum'):
            value = copy.deepcopy(self.artifact)
            row = value['rows'][0]
            if change == 'missing':
                del row['component_point_seconds'][COMPONENTS[0]]
            elif change == 'extra':
                row['component_point_seconds']['unknown_seconds'] = 0.0
            else:
                row['engineering_component_point_seconds'] += .25
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.gate(value)

    def test_legacy_manifest_and_optional_absolute_projection_path(self):
        observed = subject.manifest_value(self.manifest, self.results)
        self.assertNotIn('cold_component_projection', observed)
        observed = subject.manifest_value(dict(self.manifest,
            cold_component_projection=str(self.projection)), self.results)
        self.assertEqual(observed['cold_component_projection'], str(self.projection))
        with self.assertRaises(ValueError):
            subject.manifest_value(dict(self.manifest, cold_component_projection='relative.json'), self.results)

    def test_unadmitted_cold_cell_does_not_read_projection(self):
        for status in ('not_admitted', 'prepared', None):
            self.cell['metadata_admission'] = dict(status=status)
            self.cell['predicted_latency_seconds'] = 77.0
            manifest = dict(self.manifest, cold_component_projection=str(self.projection))
            with self.subTest(status=status), patch.object(subject, 'selected',
                    side_effect=AssertionError('Unadmitted cell must not read projection')):
                subject.curate_cold_projection(self.delivery, manifest, self.record)
            self.assertEqual(self.cell['predicted_latency_seconds'], 77.0)

    def test_missing_or_invalid_optional_projection_keeps_historical_point(self):
        self.cell['predicted_latency_seconds'] = 77.0
        self.cell['predicted_latency_scope'] = 'existing source-specific point'
        subject.curate_cold_projection(self.delivery, self.manifest, self.record)
        self.assertEqual(self.cell['predicted_latency_seconds'], 77.0)
        self.assertEqual(self.cell['component_projection_admission']['status'], 'not_configured')
        self.assertTrue(self.cell['component_projection_admission'].get('reason'))
        manifest = dict(self.manifest, cold_component_projection=str(self.projection))
        wrong = copy.deepcopy(self.artifact)
        wrong['rows'][0]['source_plan'] = str(self.results / 'foreign-plan')
        self.write(self.projection, wrong)
        subject.curate_cold_projection(self.delivery, manifest, self.record)
        self.assertEqual(self.cell['predicted_latency_seconds'], 77.0)
        self.assertEqual(self.cell['predicted_latency_scope'], 'existing source-specific point')
        admission = self.cell['component_projection_admission']
        self.assertNotEqual(admission['status'], 'accepted')
        self.assertTrue(admission.get('reason'))

    def test_accepted_projection_updates_only_its_cold_cell(self):
        historical = subject.pending_cell(self.results / 'historical-215', {})
        historical.update(predicted_latency_seconds=215.0,
            predicted_latency_scope='retained source-specific estimate')
        self.delivery['cells'].append(historical)
        before = copy.deepcopy(historical)
        subject.curate_cold_projection(self.delivery, dict(self.manifest,
            cold_component_projection=str(self.projection)), self.record)
        self.assertEqual(self.cell['predicted_latency_seconds'], 36.0)
        self.assertTrue(self.cell['predicted_latency_scope'])
        self.assertEqual(self.cell['component_projection_admission']['status'], 'accepted')
        self.assertEqual(historical, before)
        self.assertFalse(self.cell['formal_matrix_acceptance'])

    def test_report_inventory_recomputes_four_then_five_including_ablation(self):
        self.delivery['cells'] = []
        for index in range(5):
            row = subject.pending_cell(self.results / ('report-' + str(index)), {})
            row.update(primary_matrix_cell=index < 3,
                quality_observations=[dict(kind='independent frozen report')] if index < 4 else [])
            self.delivery['cells'].append(row)
        self.delivery['coverage']['experimental_inventory_report8000_cells_including_ablation'] = 3
        subject.coverage(self.delivery)
        self.assertEqual(self.delivery['coverage']['experimental_inventory_report8000_cells_including_ablation'], 4)
        self.assertEqual(self.delivery['coverage']['main_method_independent_report8000_cells'], 3)
        self.delivery['cells'][-1]['quality_observations'] = [dict(kind='independent frozen report')]
        subject.coverage(self.delivery)
        self.assertEqual(self.delivery['coverage']['experimental_inventory_report8000_cells_including_ablation'], 5)
        self.assertEqual(self.delivery['coverage']['main_method_independent_report8000_cells'], 3)

    def test_metadata_rules_replace_stale_rules_without_changing_history(self):
        history = copy.deepcopy(self.delivery['cells'])
        tables = dict(main=[], stages=[dict(stage_id='one')],
            rounds=[dict(candidate_kind='raw'), dict(candidate_kind='tuned')], counts=[])
        subject.metadata_rules(self.delivery, tables)
        self.assertNotIn('STALE_RULE', self.delivery['comparison_rules'])
        self.assertNotIn('STALE_ATTRIBUTION', self.delivery['attribution_exports']['rules'])
        self.assertTrue(all(isinstance(x, str) for x in self.delivery['comparison_rules']))
        self.assertTrue(all(isinstance(x, str) for x in self.delivery['attribution_exports']['rules']))
        self.assertEqual(self.delivery['cells'], history)

    def test_closed_cold_refresh_integration_preserves_historical_rows(self):
        # Compose the existing fixture; do not inherit or collect its old tests.
        from pt_agent.tests import test_experiment_delivery_curated_refresh as fixture_module
        helper = fixture_module.CuratedRefreshTests('runTest')
        helper.setUp()
        self.addCleanup(helper.doCleanups)
        record = helper.cold_success(count=156)
        record['phases']['compile']['execution_ops'] = dict(boot=156, rotate=3, mul_cp=2)
        helper.write(helper.cold / 'result.json', record)
        artifact = copy.deepcopy(self.artifact)
        row = artifact['rows'][0]
        row.update(source_case=str(helper.cold), source_result=str(helper.cold / 'result.json'),
            source_plan=record['phases']['compile']['runtime_plan'])
        path = helper.root / 'projection.json'
        helper.write(path, artifact)
        helper.manifest['cold_component_projection'] = str(path)
        helper.write(helper.manifest_path, helper.manifest)
        helper.run_once()
        observed = helper.cold_cell()
        self.assertEqual(observed['predicted_latency_seconds'], 36.0)
        self.assertTrue(helper.original_ids.issubset({x['cell_id'] for x in helper.load()['cells']}))
        csv_rows, _ = subject.read_csv(helper.delivery / (subject.BASE + '.csv'))
        output = next(x for x in csv_rows if x['cell_id'] == observed['cell_id'])
        self.assertEqual(output['predicted_latency_seconds'], '36.0')
        self.assertEqual(output['bootstrap_count'], '156')


if __name__ == '__main__':
    unittest.main()
