"""Audit and publish closed S2 observations without running scientific work."""
import argparse
import csv
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import time

from .. import paths
from .r10_paper_metrics import write_json
from .s2_heir_compile import inventory, MODEL_NAMES
from .s2_heir_publication import actual_processes

MODELS = ('resnet20', 'vgg16', 'alexnet', 'bert_tiny')


def read(path):
    return json.loads(Path(path).read_text())


def ledger(path):
    with Path(path).open() as stream:
        fcntl.flock(stream, fcntl.LOCK_SH)
        return [json.loads(line) for line in stream if line.strip()]


def check(checks, condition, name, **details):
    checks.append(dict(check=name, passed=bool(condition), **details))
    if not condition:
        raise ValueError(name)


def audit(args):
    checks, rows, by_model = [], ledger(args.paper / 'runs.jsonl'), {}
    registry = ledger(paths.RESULTS_DIR / 'registry.jsonl')
    selector = read(args.paper / 'paper_cell_selector.json')
    before = read(args.publisher / 'before-C-paper_cell_selector.json')
    proof = read(args.source_check)
    queue = read(args.queue / 'result.json')
    check(checks, queue['status'] == 'complete', 'Serial queue reached its complete endpoint')
    check(checks, read(args.publisher / 'result.json')['status'] == 'ok', 'Metadata publisher closed normally')
    check(checks, not actual_processes(), 'No S2 compiler or controller remains active')
    with (paths.RESULTS_DIR / 'autonomous-heavy-experiments.lock').open() as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        check(checks, True, 'Serial heavy-work lease is available')
    for model in MODELS:
        selected = [x for x in rows if x.get('task') == 'S2-HEIR' and x.get('system') == 'heir' and x['model'] == model]
        check(checks, len(selected) == 1, model + ': exactly one S2 observation')
        row = selected[0]
        case = Path(row['source']).parent
        result, completion = read(case / 'result.json'), read(case / 'completion.json')
        check(checks, row == read(case / 'paper-record.json'), model + ': published record matches closed case')
        check(checks, row['status'] == result['status'] == completion['status'], model + ': terminal outcome agrees')
        check(checks, completion['registry_persisted'] and completion['required_artifacts_persisted'], model + ': durable completion')
        check(checks, len([x for x in registry if x['run_id'] == row['run_id'] and x['artifacts_dir'] == str(case)]) == 1,
              model + ': exactly one results-registry entry')
        check(checks, row['param_group'] == 'ours-N16-boot5', model + ': intended parameter-group label retained')
        check(checks, datetime.fromisoformat(row['finished_utc']) < datetime(2026, 11, 18, 12, tzinfo=timezone.utc),
              model + ': completed before submission data cutoff')
        obs = result['observation']
        check(checks, obs['remaining_pids'] == [], model + ': native descendants closed')
        check(checks, result.get('API') == 0 and result.get('CKKS') == 0 and result.get('model_forwards') == 0 and
              result.get('sample_reads') == 0, model + ': no new model, API, or ciphertext execution')
        cell = next(x for x in selector['cells'] if x['model'] == model and x['system'] == 'heir')
        check(checks, cell['cell_state'] == 'terminal' and cell['selected_run_id'] == row['run_id'] and
              cell['source_result'] == row['source'], model + ': current formal cell selects this observation')
        for field in ('compile_s', 'boots_compiled', 'boot_sites_static', 'peak_rss_gib', 'pred_latency_s'):
            check(checks, cell.get(field) == row[field], model + ': formal ' + field + ' matches record')
        if model != 'bert_tiny':
            native_seconds = (obs['ended_ns'] - obs['started_ns']) / 1e9
            check(checks, row['compile_s'] == result['compile_s'] == native_seconds, model + ': actual compiler wall clock')
            check(checks, result['compiler_attempts'] == 1 and result['limits']['native_seconds'] == 7200 and
                  result['limits']['RSS_gib'] == 14, model + ': one attempt with original time and RSS guards')
            check(checks, result['monitor_stopped'] and not result['program_representation_changed'] and
                  not result['scalar_approximation_changed'], model + ': unchanged arithmetic and closed monitor')
            contract = result['parameter_contract']
            check(checks, contract['native_parameters']['logN'] == 16 and contract['scale'] == 43 and
                  contract['greedy_level_budget'] == 5 and contract['bootstrap_waterline'] == 1,
                  model + ': N16, scale43 and five-level placement policy')
            check(checks, contract['exact_qwen48_QP_equality_verified'] is None and
                  contract['full_bootstrap_algorithm_moduli_verified'] is False,
                  model + ': unresolved full-parameter equivalence remains explicit')
            original = next(x for x in proof['rows'] if x['model'] == model)
            check(checks, original['exact_current_approximation_fields_match'] and not original['changed_contracts'] and
                  result['source'] == original, model + ': source approximation proof bound to original input')
            check(checks, row['pred_latency_s'] is None and bool(row['prediction_unavailable_reason']),
                  model + ': missing complete cost mapping remains unmeasured')
            if row['status'] == 'ok':
                emitted = result['native_artifact']['parameters'][0]
                expected = contract['native_parameters']
                check(checks, all(emitted[k] == expected[k] for k in ('logN', 'Q', 'P')) and
                      emitted['log_default_scale'] == expected['logDefaultScale'],
                      model + ': emitted residual parameters match the actual native parameter source')
                inv = inventory(case / 'model-lattigo.mlir', MODEL_NAMES[model])
                check(checks, inv == result['compiled_inventory'] == read(case / 'compiled-inventory.json'),
                      model + ': bootstrap inventory recomputed from actual output IR')
                check(checks, row['boots_compiled'] == inv['boots_compiled'] and row['boot_kind'] == inv['boot_kind'],
                      model + ': execution-count scope matches control-flow proof')
            else:
                check(checks, row['boots_compiled'] is None and bool(row['status_detail']),
                      model + ': unsuccessful compilation has no invented output count')
        else:
            check(checks, row['status'] == 'unsupported' and row['compile_s'] is None and
                  result['compiler_attempts'] == 0 and row['own_frontend_path_unavailable'],
                  'BERT: existing-project entry unavailable; no native compile attempted')
            check(checks, not result['official_HEIR_model_unsupported_proven'] and
                  result['official_same_release_GELU_softmax_paths_exist'],
                  'BERT: project adapter result does not claim global HEIR operator absence')
        by_model[model] = row
    for old in before['cells']:
        if old['system'] == 'heir' and old['model'] in MODELS:
            continue
        now = next(x for x in selector['cells'] if (x['model'], x['system']) == (old['model'], old['system']))
        check(checks, old == now, 'Preserved other formal cell: ' + old['model'] + '/' + old['system'])
    for name in ('S2_HEIR_PROGRESS.json', 'paper_cell_selector.json', 'current_runs.json',
                 'current_experiment_evidence.json', 'EXPERIMENT_DELIVERY.json'):
        check(checks, read(args.paper / name) == read(args.mirror / name), 'Publication mirrors agree: ' + name)
    return checks, rows, by_model


def derived_stages(rows, by_model, paper_runs):
    extension, prefixes = [], []
    for model in MODELS[:3]:
        current = by_model[model]
        candidates = []
        for source in rows:
            if source.get('model') != model or source.get('system') != 'dacapo_search_ours':
                continue
            for field in ('derived_stages_s', 'stages_s'):
                stages = source.get(field) or {}
                if stages.get('shared_generation_prefix') is not None:
                    candidates.append((source, field, stages['shared_generation_prefix']))
        if not candidates or len({x[2] for x in candidates}) != 1:
            raise ValueError('One preserved shared A2b prefix required for ' + model)
        source, field, prefix = candidates[-1]
        native = current['compile_s']
        total = prefix + native
        prefixes.append(dict(model=model, shared_generation_prefix_s=prefix, native_compile_s=native,
                             derived_shared_generation_plus_compile_s=total,
                             actual_generation_e2e_s=None, clock_kind='derived_shared_generation',
                             prefix_ledger=str(paper_runs), prefix_scientific_source=source['source'],
                             prefix_record_run_id=source['run_id'], prefix_record_field=field + '/shared_generation_prefix',
                             compiler_source=current['source'], success=current['status'] == 'ok',
                             scope='Reuse the archived A2b precompile prefix and add this native compiler wall time. '
                                   'This is a derived comparison, with HEIR adapter construction outside the clock. '
                                   'An unsuccessful route measures elapsed time to failure only.'))
        for stage, value in (('shared_generation_prefix', prefix), ('placement_compile', native)):
            from_prefix = stage == 'shared_generation_prefix'
            extension.append(dict(model=model, system='heir', run_id=current['run_id'], scientific_run_id=current['run_id'],
                clock_kind='derived_shared_generation', original_stage=stage, common_category=None,
                exclusive_ns=None, exclusive_s=None, observed_stage_s=value, declared_block_s=value,
                additive_group='derived_shared_prefix_plus_native', total_clock_s=total,
                source=str(paper_runs) if from_prefix else current['source'],
                source_field=field + '/shared_generation_prefix' if from_prefix else 'compile_s',
                native_compile_child_s=native, native_child_is_nested_do_not_add=True,
                is_original_stage_value=True, no_exclusive_partition_claim=True,
                source_kind='jsonl-record' if from_prefix else 'json-field',
                source_record_run_id=source['run_id'] if from_prefix else current['run_id'],
                original_scientific_result=current['source'],
                source_value_scope='Archived generation prefix plus current native compiler; no new cold-clock claim'))
    return extension, prefixes


def publish(args, checks, rows, by_model):
    now = datetime.now(timezone.utc).isoformat()
    extension, prefixes = derived_stages(rows, by_model, args.paper / 'runs.jsonl')
    overview = dict(schema='s2-heir-closed-results-v1', published_utc=now, status='complete',
        task_source=str(args.task), rows=list(by_model.values()), derived_fig6_comparisons=prefixes,
        scope='Native HEIR placement, lowering and MLIR emission; BERT existing-project admission only',
        parameter_comparison_scope='N16 and residual-modulus bit policy match the requested group. '
                                   'The native search budget is five; generated runtime bootstrap parameters '
                                   'specify logN only. Runtime restoration to exactly level five is unverified.',
        runtime_bootstrap_output_level_verified=False,
        limitations=['Concrete full qwen48 Q/P and bootstrap-chain equivalence remain unverified.',
                     'Five-level placement budget was configured; exact runtime 1-to-5 restoration was not verified.',
                     'HEIR native cost mapping is incomplete; prediction values remain null.',
                     'No new CKKS execution or HEIR task-accuracy evaluation was performed.',
                     'BERT unsupported refers to the existing project export path.'],
        new_API=0, new_CKKS=0, new_scientific_actions_in_delivery=0)
    audit_report = dict(schema='s2-heir-closed-source-audit-v1', status='ok', published_utc=now,
                        checks=checks, passed=len(checks), failed=0, scientific_retries=0,
                        scope='S2 extension only; previous 282-check delivery audit is preserved')
    for root in (args.paper, args.mirror):
        for name in ('EXPERIMENT_DELIVERY.json', 'current_experiment_evidence.json', 'fig6_stage_index.json'):
            (args.out / ('before-' + ('C' if root == args.paper else 'D') + '-' + name)).write_text((root / name).read_text())
        write_json(root / 'S2_HEIR_RESULTS.json', overview)
        write_json(root / 'S2_HEIR_SOURCE_AUDIT.json', audit_report)
        with (root / 'S2_HEIR_RESULTS.csv').open('w') as stream:
            fields = ['model', 'status', 'compile_s', 'peak_rss_gib', 'boots_compiled', 'boot_kind',
                      'pred_latency_s', 'generation_e2e_s', 'status_detail', 'source']
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(by_model.values())
        stages = read(root / 'fig6_stage_index.json')
        existing = {(x['run_id'], x['original_stage']) for x in stages['rows']}
        stages['rows'].extend(x for x in extension if (x['run_id'], x['original_stage']) not in existing)
        stages['S2_HEIR_extension'] = dict(rows=6, source_audit=str(args.paper / 'S2_HEIR_SOURCE_AUDIT.json'),
                                          comparison_clock='derived_shared_generation', actual_cold_clock=None)
        write_json(root / 'fig6_stage_index.json', stages)
        csv_path = root / 'fig6_stage_times.csv'
        with csv_path.open() as stream:
            header = next(csv.reader(stream))
        with csv_path.open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=header, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(stages['rows'])
        delivery = read(root / 'EXPERIMENT_DELIVERY.json')
        delivery.update(status='current_required_experiment_scope_complete', last_metadata_revision_utc=now,
                        current_scientific_background_processes=0, global_heavy_lease_available_at_observation=True,
                        S2_HEIR_source_audit=dict(passed=len(checks), failed=0))
        delivery['links'].update(S2_HEIR_results=str(args.paper / 'S2_HEIR_RESULTS.json'),
                                 S2_HEIR_results_csv=str(args.paper / 'S2_HEIR_RESULTS.csv'),
                                 S2_HEIR_source_audit=str(args.paper / 'S2_HEIR_SOURCE_AUDIT.json'))
        delivery['S2_HEIR_parameter_qualification'] = overview['parameter_comparison_scope']
        write_json(root / 'EXPERIMENT_DELIVERY.json', delivery)
        index = read(root / 'current_experiment_evidence.json')
        index.update(status='current_required_experiment_scope_complete', last_metadata_revision_utc=now)
        index['linked_evidence'].update(S2_HEIR_results=str(args.paper / 'S2_HEIR_RESULTS.json'),
                                         S2_HEIR_source_audit=str(args.paper / 'S2_HEIR_SOURCE_AUDIT.json'))
        write_json(root / 'current_experiment_evidence.json', index)
    result = dict(status='ok', source_checks=len(checks), scientific_actions=0,
                  model_statuses={m: r['status'] for m, r in by_model.items()},
                  result=str(args.paper / 'S2_HEIR_RESULTS.json'))
    write_json(args.out / 'result.json', result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('paper', 'mirror', 'queue', 'publisher', 'source-check', 'task', 'out'):
        p.add_argument('--' + key, type=Path, required=True)
    p.add_argument('--wait', action='store_true')
    args = p.parse_args()
    args.out.mkdir(exist_ok=False)
    started = time.monotonic()
    while not (args.queue / 'completion.json').is_file() or not (args.publisher / 'result.json').is_file():
        if not args.wait or time.monotonic() - started > 7 * 3600:
            raise RuntimeError('Scientific queue and metadata publisher have not closed')
        write_json(args.out / 'progress.json', dict(status='waiting_for_original_queue_and_publisher',
                   source_queue=str(args.queue), scientific_actions=0))
        time.sleep(5)
    checks, rows, by_model = audit(args)
    print(json.dumps(publish(args, checks, rows, by_model), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
