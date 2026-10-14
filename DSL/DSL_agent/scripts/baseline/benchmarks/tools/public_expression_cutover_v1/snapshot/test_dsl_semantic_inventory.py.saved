"""Inventory completeness means mapped source names, not proven FHE semantics."""
import ast
import importlib.util
from pathlib import Path
import re
import unittest

import dsl_semantic_inventory as inventory

ROOT = Path(__file__).resolve().parents[2]
DACAPO = ROOT / "third_party/dacapo"


class InventoryTests(unittest.TestCase):
    def test_native_packed_manual_and_paid_coverage_are_distinct(self):
        entry=inventory.inventory()['native_packed_evidence']
        self.assertTrue(entry['native_v7_composed_with_packed'] and entry['old_contracts_unchanged'])
        self.assertEqual(entry['tested_periods'],[8,16,32,64,128,256])
        self.assertEqual((entry['correct_passed'],entry['wrong_alias_or_order_rejected']),(6,2))
        self.assertEqual((entry['live_first_passes'],entry['live_final_passes'],entry['live_api_calls']),(3,6,10))
        self.assertEqual((entry['live_helper_programs'],entry['live_recorded_helper_calls']),(2,22))
        self.assertEqual(entry['live_array_or_loop_or_augmented_or_starred_programs'],0)
        self.assertFalse(entry['all_native_constructs_live_covered'] or entry['full_semantics_proven'] or
                         entry['poseidon_gpu_validated'] or entry['security_or_tolerance_changed'])

    def test_layout_guidance_is_new_evidence_not_rewritten_history(self):
        entry=inventory.inventory()['node_layout_guidance_evidence']
        self.assertEqual((entry['guidance_schema'],entry['revision']),(2,'node-layout-v1'))
        self.assertEqual((entry['old_failure_retests'],entry['new_compositions']),(2,4))
        self.assertEqual((entry['correct_passed'],entry['live_first_passes'],entry['live_final_passes'],entry['live_api_calls']),(6,6,6,6))
        self.assertEqual((entry['live_repairs'],entry['live_transport_retries']),(0,0))
        self.assertTrue(entry['old_guidance_preserved'] and entry['ast_and_abi_unchanged'])
        self.assertFalse(entry['historical_rates_rewritten'] or entry['statistically_proven_improvement'] or
                         entry['poseidon_gpu_validated'] or entry['full_semantics_proven'])

    def test_gpu_primitive_is_separate_from_dsl_execution(self):
        for name in ("add", "modswitch"):
            self.assertEqual(inventory.SEMANTICS[name]["poseidon_gpu_primitive"], "tc128_add_drop_prefix_fixture")
            self.assertEqual(inventory.SEMANTICS[name]["poseidon_gpu_schedule"], "explicit_physical_q_add_drop_fixture_not_hevm")
            self.assertEqual(inventory.SEMANTICS[name]["poseidon_gpu"], "not_validated")
        self.assertFalse(inventory.inventory()["all_semantics_verified"])

    def test_each_row_has_real_test_location_and_explicit_gpu_gap(self):
        for name, row in inventory.SEMANTICS.items():
            with self.subTest(semantic=name):
                self.assertTrue(row["note"])
                self.assertEqual(row["poseidon_gpu"], "not_validated")
                self.assertTrue(row["tests"])
                for test in row["tests"]:
                    parts = test.split(".")
                    source = Path(importlib.util.find_spec(parts[0]).origin)
                    tree = ast.parse(source.read_text())
                    if len(parts) == 3:
                        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == parts[1])
                        self.assertIn(parts[2], {n.name for n in cls.body if isinstance(n, ast.FunctionDef)})

    @unittest.skipUnless(DACAPO.is_dir(), "requires pinned Dacapo checkout")
    def test_python_opcode_table_fully_mapped(self):
        tree = ast.parse((DACAPO / "python/hecate/hecate/expr.py").read_text())
        names = set()
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in
                    ("toUnary", "toBinary", "toInnerUnary") for t in node.targets):
                names.update(ast.literal_eval(node.value))
        self.assertEqual(names, set(inventory.PYTHON_OPS))

    @unittest.skipUnless(DACAPO.is_dir(), "requires pinned Dacapo checkout")
    def test_all_earth_and_ckks_operations_mapped(self):
        for dialect, mapping in (("Earth", inventory.EARTH_OPS), ("CKKS", inventory.CKKS_OPS)):
            source = DACAPO / f"include/hecate/Dialect/{dialect}/IR/{dialect}Ops.td"
            names = set(re.findall(r"^def\s+(\w+Op)\s*:", source.read_text(), re.MULTILINE))
            self.assertEqual(names, set(mapping))

    @unittest.skipUnless(DACAPO.is_dir(), "requires pinned Dacapo checkout")
    def test_all_poly_public_wrappers_mapped_but_not_enabled(self):
        tree = ast.parse((DACAPO / "python/poly/poly/Func.py").read_text())
        names = {n.name for n in tree.body if isinstance(n, ast.FunctionDef) and n.name.startswith("HE_")}
        self.assertEqual(names, set(inventory.POLY_HELPERS))
        self.assertEqual(inventory.POLY_HELPER_EXECUTION, "none_verified_or_allowed_in_generated_programs")

    def test_targets_use_frozen_baselines_not_relabeling(self):
        for key, count in inventory.TARGET_COUNTS.items():
            self.assertGreaterEqual(count, inventory.BASELINE_COUNTS[key] * 2)
        for mapping in (inventory.PYTHON_OPS, inventory.EARTH_OPS, inventory.CKKS_OPS, inventory.POLY_HELPERS):
            self.assertTrue(set(mapping.values()) <= set(inventory.SEMANTICS))
        self.assertFalse(inventory.inventory()["all_semantics_verified"])
        self.assertFalse(inventory.inventory()["all_goal_requirements_complete"])
        current = inventory.inventory()
        permutation=current['tensor_permutation_evidence']
        self.assertEqual((permutation['logical_rank'],permutation['elements_limit']),([1,4],256))
        self.assertEqual((permutation['correct_passed'],permutation['wrong_rotation_or_order_rejected']),(12,2))
        self.assertEqual((permutation['live_first_passes'],permutation['live_final_passes'],permutation['live_api_calls']),(10,12,15))
        self.assertEqual((permutation['live_repairs'],permutation['failed_attempts']),(3,3))
        self.assertTrue(permutation['transpose_is_not_reshape'] and permutation['original_reference_independent_of_lowering'])
        self.assertFalse(permutation['full_semantics_proven'] or permutation['arbitrary_shape_supported'] or
                         permutation['poseidon_gpu_validated'] or permutation['security_or_tolerance_changed'] or
                         permutation['counted_as_new_upstream_dsl_primitive'])
        composition=current['packed_composition_evidence']
        self.assertEqual((composition['bn_elements_limit'],composition['concat_scalar_output_limit']),(256,16))
        self.assertEqual((composition['correct_passed'],composition['wrong_coefficients_or_order_rejected']),(12,2))
        self.assertEqual((composition['live_first_passes'],composition['live_final_passes'],composition['live_api_calls']),(12,12,12))
        self.assertTrue(composition['public_fixed_statistics_only'] and composition['original_reference_independent_of_lowering'])
        self.assertFalse(composition['full_semantics_proven'] or composition['arbitrary_shape_supported'] or
                         composition['poseidon_gpu_validated'] or composition['security_or_tolerance_changed'] or
                         composition['upstream_poly_helpers_allowed'])
        spatial=current['packed_spatial_evidence']
        self.assertEqual((spatial['input_elements_limit'],spatial['scalar_neuron_total_limit']),(256,16))
        self.assertEqual((spatial['correct_passed'],spatial['wrong_group_or_window_rejected']),(16,2))
        self.assertEqual((spatial['live_first_passes'],spatial['live_final_passes'],spatial['live_api_calls']),(16,16,16))
        self.assertTrue(spatial['optional_leading_batch'] and spatial['original_window_reference_independent_of_lowering'])
        self.assertFalse(spatial['full_semantics_proven'] or spatial['arbitrary_shape_supported'] or
                         spatial['poseidon_gpu_validated'] or spatial['security_or_tolerance_changed'] or
                         spatial['upstream_poly_helpers_allowed'])
        tensor=current['tensor_input_evidence']
        self.assertTrue(tensor['linear_applies_to_last_axis'] and tensor['row_major_shape_preserved'])
        self.assertEqual((tensor['correct_passed'],tensor['wrong_reduction_or_group_rejected']),(12,2))
        self.assertEqual((tensor['live_first_passes'],tensor['live_final_passes'],tensor['live_api_calls']),(12,12,12))
        self.assertFalse(tensor['full_semantics_proven'] or tensor['arbitrary_shape_supported'] or
                         tensor['poseidon_gpu_validated'] or tensor['security_or_tolerance_changed'])
        packed=current['periodic_packed_input_evidence']
        self.assertEqual((packed['model_schema'],packed['logical_elements'],packed['logical_rank']),(5,[1,256],[1,4]))
        self.assertEqual((packed['correct_passed'],packed['wrong_reduction_counterexamples_rejected']),(12,2))
        self.assertEqual((packed['live_first_passes'],packed['live_final_passes'],packed['live_api_calls']),(12,12,12))
        self.assertFalse(packed['full_semantics_proven'] or packed['all_shape_sizes_supported'] or
                         packed['poseidon_gpu_validated'] or packed['security_profile_changed'] or
                         packed['native_construction_combined_with_packed_abi_validated'])
        chunked=current['chunked_input_evidence']
        self.assertEqual(chunked['model_schema'],4)
        self.assertEqual(chunked['logical_elements'],[5,16])
        self.assertEqual((chunked['correct_passed'],chunked['wrong_chunk_counterexamples_rejected']),(10,2))
        self.assertTrue(chunked['original_logical_reference_independent'])
        self.assertTrue(chunked['live_agent_validated'])
        self.assertEqual((chunked['live_first_passes'],chunked['live_final_passes'],chunked['live_api_calls']),(7,10,14))
        self.assertFalse(chunked['default_configuration_changed'] or chunked['all_shape_sizes_supported'] or
                         chunked['full_semantics_proven'])
        configs=current['compiler_configuration_evidence']
        self.assertEqual(configs['configurations'],['seal-cpu-eva-w40-v1','seal-cpu-eva-w45-v1'])
        self.assertEqual((configs['explicit45_correct_programs'],configs['explicit45_correct_passed'],
                          configs['wrong_bound_rejected'],configs['real_worker_configuration_rejections']),(8,8,1,3))
        self.assertTrue(configs['legacy_request_identity_preserved'] and configs['original_failure_preserved'])
        self.assertFalse(configs['default_profile_changed'] or configs['named_profile_live_agent_validated'] or
                         configs['full_semantics_proven'])
        graphs=current['hevm_graph_precision_evidence']
        self.assertEqual((graphs['waterlines'],graphs['matched_key_sets'],graphs['model_programs']),([40,45],3,8))
        self.assertEqual((graphs['program_executions'],graphs['passed_program_executions'],
                          graphs['compared_values'],graphs['agent_calls']),(48,48,576,0))
        self.assertTrue(graphs['finite_broader_graph_precision_verified'] and graphs['actual_runtime_scales_verified'])
        self.assertFalse(graphs['semantic_hevm_instructions_identical'] or graphs['production_profile_changed'] or
                         graphs['threshold_changed'] or graphs['stable_all_input_precision_proven'])
        waterline=current['hevm_waterline_precision_evidence']
        self.assertEqual((waterline['waterlines'],waterline['matched_key_sets'],waterline['model_programs']),
                         ([40,45,50],3,1))
        self.assertEqual((waterline['compared_values'],waterline['agent_calls'],waterline['production_waterline']),
                         (144,0,40))
        self.assertTrue(waterline['semantic_hevm_instructions_identical'] and waterline['actual_runtime_scales_verified']
                        and waterline['original_failure_preserved'])
        self.assertFalse(waterline['production_profile_changed'] or waterline['threshold_changed'] or
                         waterline['broader_graph_precision_verified'] or waterline['stable_all_input_precision_proven'])
        loops=current['native_public_loop_evidence']
        self.assertEqual((loops['correct_programs_tested'],loops['correct_programs_passed'],
                          loops['correct_program_numerical_failures'],loops['agent_calls']), (8,7,1,0))
        self.assertEqual(loops['batch_status'],'failed')
        self.assertFalse(loops['all_goldens_passed'] or loops['live_agent_validated'] or loops['full_semantics_proven'])
        precision=current['rotation_precision_diagnostic']
        self.assertEqual((precision['native_stage_observations'],precision['key_sets']), (72,2))
        self.assertFalse(precision['original_failed_key_reproduced'] or precision['production_profile_changed']
                         or precision['threshold_changed'] or precision['stable_precision_budget_verified'])
        arithmetic = current['native_array_arithmetic_evidence']
        self.assertEqual((arithmetic['manual_cases'],arithmetic['detected_broadcast_counterexamples'],arithmetic['agent_calls']),
                         (8,1,0))
        self.assertTrue(arithmetic['outer_expr_array_broadcast'] and arithmetic['zero_dim_result_is_expr'])
        self.assertTrue(arithmetic['scalar_left_array_rejected'])
        self.assertFalse(arithmetic['packed_slot_broadcast'] or arithmetic['live_agent_validated'] or arithmetic['full_semantics_proven'])
        mutation = current['native_array_mutation_evidence']
        self.assertEqual((mutation['manual_cases'],mutation['detected_counterexamples'],mutation['agent_calls']),(13,2,0))
        self.assertEqual((mutation['input_executions'],mutation['compared_values']),(52,208))
        self.assertTrue(mutation['typed_alias_heap'] and mutation['actual_alias_cell_observations'] and mutation['keep_order_layout'])
        self.assertFalse(mutation['python_class_is_ir_type'] or mutation['subscript_writes_enabled'] or
                         mutation['live_agent_validated'] or mutation['full_semantics_proven'])
        augmented = current['native_scalar_augmented_evidence']
        self.assertEqual((augmented['manual_cases'],augmented['detected_counterexamples'],augmented['agent_calls']),
                         (7,2,0))
        self.assertEqual((augmented['input_executions'],augmented['compared_values']),(28,112))
        self.assertTrue(augmented['actual_augmented_dispatch'] and augmented['unchanged_expr_aliases'])
        self.assertFalse(augmented['array_inplace_mutation_enabled'] or augmented['live_agent_validated'] or
                         augmented['full_semantics_proven'])
        targeted = current['native_star_construction_evidence']
        self.assertEqual((targeted['manual_cases'],targeted['detected_counterexamples'],targeted['live_agent_cases']),
                         (8,1,0))
        self.assertEqual((targeted['unique_features'],targeted['finite_argument_influence_features'],
                          targeted['structural_features']), (9,8,1))
        self.assertTrue(targeted['actual_frontend_expansion_observation'])
        self.assertFalse(targeted['full_semantics_proven'])
        self.assertEqual(targeted['compiler_configuration'],'seal-cpu-eva-w45-v1')
        stars = current['native_starred_argument_evidence']
        self.assertEqual((stars['manual_cases'],stars['detected_order_counterexamples'],stars['agent_calls']), (8,1,0))
        self.assertTrue(stars['upstream_scalar_abi_unchanged'] and stars['first_axis_iteration'])
        self.assertFalse(stars['implicit_flattening'] or stars['expr_array_as_one_native_argument'])
        self.assertFalse(stars['live_agent_validated'] or stars['full_semantics_proven'])
        arrays = current['native_array_construction_evidence']
        self.assertEqual((arrays['manual_cases'],arrays['detected_counterexamples'],arrays['live_agent_cases']), (10,1,0))
        self.assertEqual((arrays['unique_features'],arrays['finite_cell_influence_features'],arrays['structural_features']),
                         (14,13,1))
        self.assertTrue(arrays['actual_frontend_storage_observation'])
        self.assertFalse(arrays['full_semantics_proven'])
        native = current['native_function_agent_evidence']
        self.assertEqual((native['cases'],native['first_successes'],native['api_calls']), (11,10,12))
        self.assertEqual((native['feature_count'],native['output_influence_features'],native['trace_structural_features']),
                         (11,10,1))
        self.assertFalse(native['native_array_live_agent_validated'])
        self.assertFalse(native['full_semantics_proven'])
        self.assertFalse(current["current_goal_scope"]["poseidon_gpu_required"])
        self.assertEqual(current["current_goal_scope"]["execution_backend"], "upstream_SEAL_HEVM_CPU")
        self.assertEqual(current["grammar_coverage"]["tool"], "scripts/baseline/audit_dsl_coverage.py")
        self.assertEqual(current['grammar_coverage']['agent_observed_feature_partitions'], 29)
        self.assertEqual(current['grammar_coverage']['feature_partition_count'], 32)
        zero = current['encrypted_zero_evidence']
        self.assertEqual((zero['agent_cases'], zero['agent_first_successes']), (7, 7))
        self.assertTrue(zero['logical_inputs_unchanged'])
        self.assertFalse(zero['full_algebraic_cancellation_supported'])
        model = current['model_capability_evidence']
        self.assertEqual((model['total_agent_cases'],model['user_graph_cases'],model['legacy_catalog_cases']), (126,78,48))
        self.assertEqual(model['missing_live_graph_operators'], [])
        self.assertTrue(current['explicit_power_evidence']['live_agent_validated'])
        advanced=current['advanced_agent_evidence']
        self.assertEqual((advanced['cases'],advanced['first_successes'],advanced['api_calls']), (12,12,12))
        self.assertEqual((advanced['api_concurrency'],advanced['native_execution_concurrency']), (10,2))
        self.assertEqual(advanced['model'], 'deepseek-flash')
        audited=current['manual_semantic_audit']
        self.assertEqual((audited['audited_programs'],audited['correct_goldens'],audited['detected_counterexamples']),
                         (61,42,19))
        self.assertEqual((audited['new_fhe_executions'],audited['new_agent_calls']),(0,0))
        self.assertEqual(audited['legacy_v1_key_probe_missing'],3)
        self.assertTrue(audited['actual_decrypted_arrays_recomputed'])
        self.assertFalse(audited['full_semantics_proof'])
        manual = current['manual_broadcast_evidence']
        self.assertEqual((manual['golden_passes'], manual['correctly_rejected_after_real_execution'], manual['agent_calls']),
                         (5, 5, 0))
        self.assertEqual(len(manual['features']), 6)
        self.assertEqual(len(manual['sha256']), 64)
        self.assertEqual(current["current_catalog_model_families"], 8)
        suite=current['self_contained_model_suite']
        self.assertEqual((suite['cases'],suite['model_families'],suite['migrated_rule_cases']),(96,16,48))
        self.assertFalse(suite['catalog_lookup_required'])
        self.assertFalse(suite['historical_agent_reports_reclassified'])
        self.assertFalse(suite['new_96_case_live_agent_cohort_validated'])
        self.assertEqual(suite['migrated_rule_agent_calls'],0)
        import hashlib
        self.assertEqual(hashlib.sha256((ROOT/arrays['catalog']).read_bytes()).hexdigest(),arrays['catalog_sha256'])
        self.assertEqual(hashlib.sha256((ROOT/targeted['catalog']).read_bytes()).hexdigest(),targeted['catalog_sha256'])
        for name in ('manifest','provenance'):
            self.assertEqual(hashlib.sha256((ROOT/suite[name]).read_bytes()).hexdigest(),suite[name+'_sha256'])
        self.assertEqual(current["expanded_benchmark_model_families"], 16)
        self.assertGreaterEqual(current["current_graph_operator_count"], 12)
        self.assertTrue(current["expanded_benchmark_live_agent_validated"])
        self.assertEqual(current["expanded_benchmark_live_agent_validation_scope"],
                         "cumulative_heterogeneous_cpu_96_cases_16_families")
        self.assertFalse(current["single_configuration_cohort_passed"])
        self.assertEqual(len(current["live_agent_evidence"]["sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
