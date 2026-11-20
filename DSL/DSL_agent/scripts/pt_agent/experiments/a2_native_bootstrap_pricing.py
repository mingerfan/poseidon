"""Explicit input/output bootstrap table conventions for the A2 continuation.

The original A2 modules keep their legacy input-indexed interpretation. This
entry selects output indexing for DaCapo's native table, reuses the same four
placement algorithms, and re-observes actual RuntimePlans with that policy.
Existing compiled artifacts can also receive metadata-only projection rows.
"""
import ast
from collections import Counter
import copy
from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
import inspect
import json
import math
import os
from pathlib import Path
import sys
import time
from types import FunctionType

from .. import paths
from . import a2_native_profile_driver as original
from . import a2_native_profile_planner as planner
from .a_group_runs import append_record, empty_record
from .completion_metrics import artifact_bytes, persist_case_completion
from .r10_paper_metrics import _process, summarize_events, write_json

MODULE = 'pt_agent.experiments.a2_native_bootstrap_pricing'
ROUND = 'A2-NATIVE-OUTPUT-BOOT-PRICE'
MODE = 'frozen-native-four-placements-output-priced-bootstrap'
CONVENTIONS = ('input', 'output')
FIELDS = {'run_id', 'round', 'model', 'mode', 'git_rev', 'metrics', 'artifacts_dir', 'ts'}
TABLE_SCOPE = ('Unchanged DaCapo HEAAN-HEVM latencyTable; bootstrap indexed by '
               'actual result level; other operations retain the original input-level observer')


def bootstrap_us(table, input_level, output_level, convention):
    """Select one explicitly declared operand side, independent of model name."""
    original.require(convention in CONVENTIONS, 'Explicit bootstrap table convention required')
    level = input_level if convention == 'input' else output_level
    row = table.get('bootstrap_single')
    original.require(type(level) is int and isinstance(row, list) and 0 <= level < len(row),
                     'Bootstrap table level unavailable')
    value = row[level]
    original.require(type(value) in (int, float) and math.isfinite(value) and value >= 0,
                     'Finite nonnegative bootstrap table entry required')
    return value


@dataclass(frozen=True)
class PricedModel(planner.Model):
    bootstrap_pricing: str = 'input'

    @property
    def boot_us(self):
        return (bootstrap_us(self.table, self.boot_in_level, self.raw_boot_out_level,
                             self.bootstrap_pricing)
                + self.at('rescale_single', self.raw_boot_out_level))


def priced_model(config, bootstrap_pricing='input'):
    original.require(bootstrap_pricing in CONVENTIONS, 'Unknown bootstrap convention')
    base = planner.model_for(config)
    return PricedModel(**vars(base), bootstrap_pricing=bootstrap_pricing)


def native_model(config):
    return priced_model(config, 'output')


def plan_observation(plan, profile, bootstrap_pricing='output'):
    """Reprice only serialized Boot operations; keep missing costs explicit."""
    from .r29_local_compile_matrix import plan_observation as observe
    result = observe(plan, profile)
    table = {key.removeprefix('earth.'): value for key, value in profile['latencyTable'].items()}
    delta = 0.; rows = []; old_sum = 0.; new_sum = 0.
    for boot in result['bootstrap_positions']:
        inputs, outputs = boot['input_levels'], boot['output_levels']
        original.require(len(inputs) == len(outputs) == 1,
                         'One actual input/output level required per serialized bootstrap')
        old = bootstrap_us(table, inputs[0], outputs[0], 'input')
        new = bootstrap_us(table, inputs[0], outputs[0], bootstrap_pricing)
        old_sum += old; new_sum += new; delta += new - old
        rows.append(dict(execution_index=boot['execution_index'], input_level=inputs[0],
                         output_level=outputs[0], priced_level=inputs[0] if bootstrap_pricing == 'input' else outputs[0],
                         old_input_price_us=old, price_us=new))
    prediction = result['prediction']
    by_op = prediction['by_op_us']
    if rows:
        by_op['boot'] = by_op.get('boot', 0.) + delta
    table_scope = TABLE_SCOPE if bootstrap_pricing == 'output' else TABLE_SCOPE.replace(
        'actual result level', 'actual input level')
    prediction.update(source=table_scope, bootstrap_pricing_convention=bootstrap_pricing,
        bootstrap_price_observations=rows, legacy_input_priced_boot_subtotal_us=old_sum,
        boot_subtotal_us=new_sum, bootstrap_repricing_delta_us=delta,
        table_known_sum_us=prediction['table_known_sum_us'] + delta)
    if result['predicted_latency_us'] is not None:
        result['predicted_latency_us'] += delta
    return result


def observe_runtime(plan, profile, expected_boots):
    _, levels = original.observe_runtime(plan, profile, expected_boots)
    return plan_observation(plan, profile), levels


def reprice_candidates(candidates, profile):
    """Rank the saved placements without pretending they were planned anew."""
    old, new = priced_model(profile, 'input'), native_model(profile)
    rows = {}
    for name, row in candidates.items():
        original.require(type(row.get('boots')) is int and row['boots'] >= 0,
                         'Saved candidate bootstrap count required')
        old_score = row['latency_us']
        original.require(type(old_score) in (int, float) and math.isfinite(old_score),
                         'Finite saved placement score required')
        rows[name] = dict(row, legacy_input_price_latency_us=old_score,
                         latency_us=old_score + row['boots'] * (new.boot_us - old.boot_us))
    original.require(set(rows) == {'depth-dp', 'depth-dp+prune', 'stage-cut', 'stage-cut+prune'},
                     'All four original placements required')
    return rows, min(rows, key=lambda name: rows[name]['latency_us'])


def admission(args):
    checked = original.admission(args)
    checked.update(bootstrap_pricing_convention='output', bootstrap_price_level=16,
        planning_rescale_reserve_unchanged=True, table_scope=TABLE_SCOPE,
        pricing_adapter=str(Path(__file__)), original_adapter=str(Path(original.__file__)))
    for path in (Path(__file__), Path(original.__file__)):
        target = args.out / ('pricing-source-' + path.name)
        target.write_text(path.read_text())
    return checked


def _clone(function, namespace, *, worker=False, execute=False):
    """Only ordinary A2 functions are cloned; the imported contexts stay intact."""
    tree = ast.parse(inspect.getsource(function))
    model_calls = observation_imports = module_names = table_labels = 0
    for node in ast.walk(tree):
        if worker and isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name) and (node.func.value.id, node.func.attr) == ('planner', 'model_for'):
                node.func = ast.Name(id='native_model', ctx=ast.Load()); model_calls += 1
        if worker and isinstance(node, ast.ImportFrom) and node.module == 'r29_local_compile_matrix':
            original.require(node.level == 1 and [x.name for x in node.names] == ['plan_observation'],
                             'Original observation import changed')
            node.module = 'a2_native_bootstrap_pricing'; observation_imports += 1
        if execute and isinstance(node, ast.Constant):
            if node.value == 'pt_agent.experiments.a2_native_profile_driver':
                node.value = MODULE; module_names += 1
            elif node.value == 'DaCapo original HEAAN-HEVM latencyTable':
                node.value = TABLE_SCOPE; table_labels += 1
    if worker:
        original.require((model_calls, observation_imports) == (1, 1), 'Original A2 worker anchors changed')
    if execute:
        original.require((module_names, table_labels) == (1, 1), 'Original A2 supervisor anchors changed')
    ast.fix_missing_locations(tree)
    exec(compile(tree, str(Path(__file__)) + ':' + function.__name__, 'exec'), namespace)
    return namespace[function.__name__]


def namespace():
    result = dict(vars(original))
    for name, function in tuple(result.items()):
        if isinstance(function, FunctionType) and function.__module__ == original.__name__:
            clone = FunctionType(function.__code__, result, name, function.__defaults__, function.__closure__)
            clone.__kwdefaults__ = function.__kwdefaults__
            result[name] = clone
    result.update(ROUND=ROUND, MODE=MODE, __file__=__file__, native_model=native_model,
                  observe_runtime=observe_runtime, admission=admission)
    result['worker'] = _clone(original.worker, result, worker=True)
    result['execute'] = _clone(original.execute, result, execute=True)
    return result


def closed(case, model, *, registry=None):
    """Metadata-only closed source check; no program, vector or blob is opened."""
    case = Path(case).resolve(); result = original.read(case / 'result.json')
    completion = original.read(case / 'completion.json')
    original.require(result.get('model') == model and result.get('status') == 'ok'
        and completion.get('status') == 'ok' and completion.get('registry_persisted') is True
        and completion.get('required_artifacts_persisted') is True, 'Closed successful source required')
    entries = []
    with Path(registry or paths.RESULTS_DIR / 'registry.jsonl').open() as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                if row.get('run_id') == case.name: entries.append(row)
    original.require(len(entries) == 1 and set(entries[0]) == FIELDS
        and Path(entries[0]['artifacts_dir']).resolve() == case
        and entries[0]['model'] == model and entries[0]['metrics']['status'] == 'ok',
        'One original eight-field source registry identity required')
    if result.get('round') is not None:
        original.require(entries[0]['round'] == result['round'], 'Source registry round differs')
    observations = []
    phases = result.get('phases', [])
    if isinstance(phases, dict): phases = phases.values()
    for phase in phases:
        observations.append(phase.get('process') or {})
        if phase.get('worker', {}).get('native_process'):
            observations.append(phase['worker']['native_process'])
    if result.get('compile_worker', {}).get('process'):
        observations.append(result['compile_worker']['process'])
    original.require(observations and all(row.get('exit_code') == 0 and row.get('signal') is None
        and row.get('timed_out') is False and row.get('remaining_pids') == []
        and row.get('signals_sent') == [] for row in observations), 'Source processes must have clean observations')
    births = set()
    for file in case.rglob('*resources.jsonl'):
        with file.open() as stream:
            for line in stream:
                if not line.strip(): continue
                row = json.loads(line); items = list(row.get('processes') or [])
                if row.get('outer_process'): items.append(row['outer_process'])
                for item in items:
                    if type(item.get('pid')) is int and type(item.get('starttime')) is int:
                        births.add((item['pid'], item['starttime']))
    original.require(births, 'Saved process birth evidence required')
    for pid, birth in births:
        current = _process(pid)
        original.require(not current or current['starttime'] != birth or current['state'] in ('Z', 'X'),
                         'Saved source process birth is still active')
    return result, completion, dict(saved_births=len(births), all_saved_births_closed=True)


def _append_supersession(path, old, new, reason):
    row = dict(old_run_id=old, new_run_id=new, reason=reason,
               old_compile_fact_preserved=True, ts=datetime.now(timezone.utc).isoformat())
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX); stream.seek(0)
        for line in stream:
            if line.strip():
                previous = json.loads(line)
                if (previous.get('old_run_id'), previous.get('new_run_id')) == (old, new): return False
        stream.seek(0, os.SEEK_END); stream.write(json.dumps(row, allow_nan=False) + '\n')
        stream.flush(); os.fsync(stream.fileno())
    return True


def metadata_execute(args):
    """Publish same-table projections only after durable source/row persistence."""
    original.require(args.run and args.reprice_from and args.paper_runs and args.supersessions,
                     'Explicit source/publication destinations and --run required')
    original.require(not args.out.exists() and args.out.parent.resolve() == paths.RESULTS_DIR.resolve(),
                     'Fresh direct results child required')
    args.out.mkdir(); started = time.perf_counter_ns(); rows = []; result = dict(status='running')
    try:
        source, completion, closure = closed(args.reprice_from, args.model)
        original.require(source.get('round') in (original.ROUND, ROUND) and source.get('monitor_stopped') is True,
                         'Original A2 or explicit output-price completed source required')
        original.require(source['attempted'] == source['completed'] == dict.fromkeys(original.PHASES, 1),
                         'Every original A2 phase must be complete')
        checked = source['source']; profile = original.read(checked['profile']); native_model(profile)
        native_case = Path(checked['native_case'])
        native_identity = dict(resnet20_silu='dacapo-original-resnet20-silu',
            vgg16_silu_avgpool='dacapo-original-vgg16-silu-avgpool',
            alexnet_silu_avgpool='dacapo-original-alexnet-silu-avgpool')[args.model]
        native, native_completion, native_closure = closed(native_case, native_identity)
        placement = original.read(args.reprice_from / 'placement.json')
        if source['round'] == ROUND:
            ranked = copy.deepcopy(placement['candidates'])
            selected = min(ranked, key=lambda name: ranked[name]['latency_us'])
        else:
            ranked, selected = reprice_candidates(placement['candidates'], profile)
        original.require(selected == placement['selected'],
                         'Saved selected placement changes under new prices; re-place and compile a fresh case')
        phases = {row['phase']: row for row in source['phases']}
        original.require(set(phases) == set(original.PHASES)
            and all(row.get('status') == 'ok' and row.get('worker', {}).get('status') == 'ok'
                    for row in phases.values())
            and phases['planning']['worker'].get('all_four_verified') is True,
            'All five original phases and four candidate callback proofs required')
        original.require(placement['candidates'][placement['selected']]['boots'] == source['compiled_bootstraps'],
                         'Original selected callback and compiled bootstrap counts differ')
        observed = phases['observe']['worker']; ours_plan = Path(observed['runtime_plan'])
        own_observed, level_audit = observe_runtime(original.read(ours_plan), profile, source['compiled_bootstraps'])
        ref_plan = Path(checked['native_plan']); ref_observed = plan_observation(original.read(ref_plan), profile)
        original.require(ref_observed['compiled_bootstraps'] == native['bootstrap_count'], 'Native reference boots differ')
        native_clock = (native.get('compile_worker') or {}).get('process') or {}
        native_compile = ((native_clock['ended_ns']-native_clock['started_ns'])/1e9
            if type(native_clock.get('started_ns')) is int and type(native_clock.get('ended_ns')) is int
            and native_clock['ended_ns'] >= native_clock['started_ns'] else None)
        observations = [('arranger', ours_plan, own_observed, source['compile_s'], args.reprice_from),
                        ('dacapo_native', ref_plan, ref_observed, native_compile, native_case)]
        short = dict(resnet20_silu='resnet20', vgg16_silu_avgpool='vgg16', alexnet_silu_avgpool='alexnet')[args.model]
        old_case = args.supersedes_case or args.reprice_from
        if old_case.resolve() != args.reprice_from.resolve():
            previous, _, _ = closed(old_case, args.model)
            original.require(previous.get('round') == original.ROUND and all(
                Path(previous['source'][key]).resolve() == Path(checked[key]).resolve()
                for key in ('source_case', 'native_case', 'profile')),
                'Supersession must reference the same original frozen/native source pair')
        for system, plan, observation, historical_compile, artifact_case in observations:
            suffix = '-arranger-output-price-projection' if system == 'arranger' else '-dacapo-native-output-price-projection'
            pred = observation['prediction']; missing = Counter(row['op'] for row in pred['missing'])
            row = empty_record(run_id=args.out.name + suffix, task='A2-corrected-price', model=short,
                system=system, param_group='dacapo-native', status='ok',
                status_detail='Metadata-only re-estimate of an already completed RuntimePlan using the explicit output-level bootstrap convention.',
                budget_s=None, compile_s=None, peak_rss_gib=None, boots_compiled=observation['compiled_bootstraps'],
                exec_ops=observation['execution_steps'], pred_latency_s=None if observation['predicted_latency_us'] is None else observation['predicted_latency_us']/1e6,
                pred_table=checked['profile'] + '; ' + TABLE_SCOPE, generation_e2e_s=None, api_calls=0,
                source=str(args.out/'result.json'), finished_utc=datetime.now(timezone.utc).isoformat())
            row.update(projection_only=True, runtime_plan=str(plan), compiled_artifact_source=str(artifact_case/'result.json'),
                historical_compile_s=historical_compile, historical_compile_clock_reused=historical_compile is not None,
                partial_pred_latency_s=pred['table_known_sum_us']/1e6, prediction_missing_ops=dict(missing),
                bootstrap_pricing_convention='output', bootstrap_price_observations=pred['bootstrap_price_observations'],
                new_compile_attempts=0, new_model_forwards=0, selected_placement=placement['selected'] if system=='arranger' else None,
                quality_inherited=False, quality_evaluations=0)
            old_id = old_case.name if system == 'arranger' else old_case.name+'-dacapo-native-same-table-reference'
            replacement = args.reprice_from.name if system == 'arranger' and source['round'] == ROUND else row['run_id']
            rows.append(dict(record=row, supersedes_run_id=old_id, supersedes_new_run_id=replacement))
        result = dict(status='ok', round=ROUND, mode='metadata-only-output-price-projections', model=args.model,
            source_case=str(args.reprice_from), source_completion=completion, source_closure=closure,
            native_case=str(native_case), native_closure=native_closure, native_completion=native_completion,
            bootstrap_pricing_convention='output', candidates=ranked, selected=selected, native_level_audit=level_audit,
            projections=rows, model_forwards=0, Count=0, emit=0, trace=0, compile=0, api_calls=0, real_ckks_attempts=0)
    except BaseException as exc:
        result = dict(status='error', round=ROUND, mode='metadata-only-output-price-projections', model=args.model,
                      error=type(exc).__name__+': '+str(exc), projections=[], compile=0, api_calls=0)
    write_json(args.out/'result.json', result)
    entry = dict(run_id=args.out.name, round=ROUND, model=args.model, mode=result['mode'], git_rev=None,
                 metrics=dict(status=result['status'], api_calls=0, compile=0), artifacts_dir=str(args.out),
                 ts=datetime.now(timezone.utc).isoformat())
    original._append(paths.RESULTS_DIR/'registry.jsonl', entry)
    timing = summarize_events([])
    persist_case_completion(args.out, round_label=ROUND, started_ns=started, status=result['status'],
                            timing=timing, required_artifact_bytes=artifact_bytes(args.out))
    if result['status'] == 'ok':
        for item in rows:
            # The accepted source and new metadata result are already durable;
            # the exact paper row is persisted before any supersession link.
            append_record(args.paper_runs, item['record'])
            _append_supersession(args.supersessions, item['supersedes_run_id'], item['supersedes_new_run_id'],
                'Bootstrap table uses actual output level16; original compile/count/source fact preserved.')
    return result


def append_actual_record(args, result):
    """Record this fresh compiler attempt, including an unsuccessful attempt."""
    completion = original.read(args.out/'completion.json')
    observed = next((row.get('worker', {}).get('compiled') for row in result.get('phases', [])
                     if row.get('phase') == 'observe'), None) or {}
    status = 'ok' if result['status'] == 'ok' else result.get('failure_class', 'driver_error')
    if status not in ('ok', 'compiler_crash', 'timeout', 'resource_limit', 'unsupported', 'driver_error'):
        status = 'driver_error'
    prediction = observed.get('prediction') or {}
    row = empty_record(run_id=args.out.name, task='A2-corrected-price',
        model=dict(resnet20_silu='resnet20',vgg16_silu_avgpool='vgg16',alexnet_silu_avgpool='alexnet')[args.model],
        system='arranger', param_group='dacapo-native', status=status,
        status_detail=result.get('error') or 'Fresh four-placement native-profile compilation with explicit output-level bootstrap pricing.',
        budget_s=7200, compile_s=result.get('compile_s'), peak_rss_gib=None,
        boots_compiled=result.get('compiled_bootstraps'), exec_ops=observed.get('execution_steps'),
        pred_latency_s=result.get('pred_latency_s'), pred_table=TABLE_SCOPE,
        generation_e2e_s=None, api_calls=0, source=str(args.out/'result.json'),
        finished_utc=datetime.now(timezone.utc).isoformat())
    row.update(bootstrap_pricing_convention='output', projection_only=False,
        warm_compilation_pipeline_s=completion.get('complete_case_elapsed_seconds'),
        warm_timing_scope='Original frozen model/program reused; this fresh attempt includes placement, Count/Emit, trace, compile, observation and persistence. Original Agent generation and intake excluded.',
        cold_end_to_end_s=None, attempted=result.get('attempted'), completed=result.get('completed'),
        failure_class=result.get('failure_class'), partial_pred_latency_s=(
            prediction['table_known_sum_us']/1e6 if prediction.get('table_known_sum_us') is not None else None),
        prediction_missing_ops=dict(Counter(item['op'] for item in prediction.get('missing', []))),
        quality_inherited=False, quality_evaluations=0)
    append_record(args.paper_runs, row)
    return row


def main(argv=None):
    parser = original.parser()
    parser.add_argument('--metadata-only', action='store_true')
    for key in ('reprice-from', 'paper-runs', 'supersessions', 'supersedes-case'):
        parser.add_argument('--'+key, type=Path)
    args = parser.parse_args(argv)
    for key, value in vars(args).items():
        if isinstance(value, Path): setattr(args, key, value.expanduser().resolve())
    if args.metadata_only:
        original.require(not args.worker, 'Metadata-only invocation has no model worker')
        result = metadata_execute(args)
    else:
        original.require(args.worker or args.paper_runs, 'Fresh compiler attempt requires --paper-runs')
        if args.frontend is None: args.frontend = paths.DACAPO_ROOT/'python/hecate/hecate/expr.py'
        dispatch = namespace(); result = dispatch['worker'](args) if args.worker else dispatch['execute'](args)
        if not args.worker:
            append_actual_record(args, result)
    print(json.dumps({key:result.get(key)for key in ('status','error','compiled_bootstraps')}))
    return int(result['status'] != 'ok')


if __name__ == '__main__': raise SystemExit(main())
