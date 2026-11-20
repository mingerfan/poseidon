"""Project the measured two-hour BERT timeout without replacing warm science."""
import copy
from datetime import datetime, timezone
from pathlib import Path

from .bert_three_hour_terminal_metadata import require

FRESH2 = 'bert-sst2-fresh-cold-agent-2h-20261117'
NATIVE2 = 'bert-rule63-current-shared-optimized-qwen48-20261117'


def project(selector, comparison, native_row, native_result, native_completion, *, selected):
    require(comparison.get('schema') == 'bert-two-hour-fresh-vs-independent-native-comparison-v1',
            'Original two-hour comparison required')
    observed = comparison['dacapo_current_rule63_search']
    proof = observed.get('native_pipeline_proof') or {}
    case = Path(native_row['source']).parent
    require(case.name == NATIVE2 and native_row.get('run_id') == NATIVE2
            and native_row.get('model') == 'bert_tiny' and native_row.get('system') == 'dacapo_search_ours'
            and native_row.get('param_group') == 'ours-N16-boot5' and native_row.get('budget_s') == 7200
            and observed.get('state') == 'terminal' and observed.get('classification') == native_row.get('status')
            and proof.get('paper_record_original') == native_row
            and proof.get('registry_fields') == 8 and proof.get('registry_run_id') == NATIVE2
            and proof.get('source_result') == str(case/'result.json')
            and proof.get('source_completion') == str(case/'completion.json')
            and (proof.get('producer_closure') or {}).get('closed') is True,
            'Exact measured terminal native row and original closure proof required')
    require(native_completion.get('registry_persisted') is True
            and native_completion.get('required_artifacts_persisted') is True
            and native_result.get('status') == native_completion.get('status') == 'error'
            and native_row.get('status') == 'timeout', 'This projection records the observed two-hour timeout')
    native = (native_result.get('phases', {}).get('compile') or {}).get('native_process') or {}
    duration = (native.get('ended_ns', 0)-native.get('started_ns', 0))/1e9
    require(native.get('timed_out') is True and native.get('remaining_pids') == []
            and (native.get('limits') or {}).get('wall_seconds') == 7200
            and duration == native_row.get('compile_s') and duration >= 7200,
            'Use the actual full7200second native search clock')
    value = copy.deepcopy(selector)
    found = [c for c in value['cells'] if c.get('model') == 'bert_tiny' and c.get('system') == 'dacapo_search_ours']
    require(len(found) == 1 and found[0].get('selected_run_id') in (None, NATIVE2),
            'Preserve any independently selected later BERT native result')
    cell = found[0]
    selected(cell, native_row, native_result, native_completion)
    cell.update(clock_kind='actual_compile_only', generation_e2e_s=None,
                actual_compile_only_case_s=native_completion['complete_case_elapsed_seconds'],
                reason='Actual current Rule63 native search exhausted its7200second allocation; original input preparation and terminal persistence are recorded separately',
                cold=False, source_generation_reused=True, independent_of_Agent_success=True,
                original_math_quality_unmeasured=True, output_kind=None,
                boots_compiled=None, exec_ops=None, pred_latency_s=None,
                time_outcome_scope='native_search_timeout_after_complete7200second_allocation',
                native_exit_code=native['exit_code'], native_signal_name=None,
                timeout_signals_sent=copy.deepcopy(native.get('signals_sent', [])),
                AS_gib=native['limits'].get('memory_gib'), RSS_guard_gib=14,
                numerical_bootstrap_and_prediction_eligible=False)
    fresh = comparison.get('fresh_arranger') or {}
    require(fresh.get('state') == 'terminal' and fresh.get('usable_within_two_hours') is False
            and Path(fresh.get('case', '')).name == FRESH2,
            'Retain the actual first fresh two-hour policy outcome')
    require(all(c.get('scientific_run_id') != NATIVE2 or c is cell for c in value['cells']),
            'One native observation must fill one exact compiler cell')
    value['BERT_budget_observations'] = dict(
        original_two_hour_comparison=copy.deepcopy(comparison),
        fresh_two_hour_outcome_scope='One fixed7200second pipeline policy with5400second search and1800second tail reserve; actual early search stop retains its original clock',
        retained_warm_program_scope='Existing frozen9180 program and independent872-row report remain their original observations; cold generation time remains unmeasured',
        three_hour_progress_file='bert_3h_comparison.json',
        three_hour_terminal_file='bert_3h_comparison_terminal.json',
        three_hour_outcomes_assumed=False,
        old_frozen_Agent_cell_replaced=False, original_scientific_rows_modified=False)
    value['generated_utc'] = datetime.now(timezone.utc).isoformat()
    value['BERT_two_hour_timeout_projection'] = dict(metadata_only=True, new_scientific_actions=0,
        source_case=str(case), original_budget=7200, actual_native_seconds=duration,
        preserves_all_other_cells=True, original_input_ledger_modified=False)
    return value
