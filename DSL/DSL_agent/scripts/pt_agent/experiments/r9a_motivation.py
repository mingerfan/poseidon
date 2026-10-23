"""Append-only motivation measurements for the frozen PT-Agent (no LLM)."""

import argparse
import ast
import csv
from datetime import datetime, timezone
import fcntl
import gc
import json
import os
from pathlib import Path
import re
import resource
import subprocess
import sys
import time
from types import SimpleNamespace
from ..paths import DACAPO_ROOT, DACAPO_UPSTREAM


NAMES = ('resnet20', 'resnet20_silu', 'cnn_plain', 'mlp')
RANDOM_COUNTS = dict(resnet20=150, resnet20_silu=150, cnn_plain=200, mlp=200)
MARGINS = (1.0, 1.05, 1.1, 1.2, 1.3, 1.5)
RUNS = dict(e7='r8e7-reconfirm-20261022T122558', e5='r8e5-dev-20261022T041646',
            graphs='r8a-dev-20261021T074003-v2', r6b='r6b-refresh-20261021T025529',
            r6c='run-resnet20-rule-20261020T165817')
EXPERIMENTS = ('e3', 'e4', 'e1', 'e6')
INNER_SECONDS = 21600
MANUAL_DECISIONS = {
    'MLP': dict(activations=1, linear_layers=2,
                note='square activation h1*h1; two hand-written diagonal-style linear loops'),
}


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def _rows(path):
    path = Path(path)
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def evaluate_pending(rows_path, items, evaluate):
    """Append each completed key once; include failed evaluations as result rows."""
    path = Path(rows_path)
    rows = _rows(path)
    seen = {row['key'] for row in rows}
    path.parent.mkdir(parents=True, exist_ok=True)
    for item in items:
        if item['key'] in seen:
            continue
        try:
            row = evaluate(item)
        except Exception as exc:
            row = dict(item, error=f'{type(exc).__name__}: {exc}')
        row.setdefault('key', item['key'])
        with path.open('a') as stream:
            stream.write(json.dumps(row, allow_nan=False)+'\n')
            stream.flush()
            os.fsync(stream.fileno())
        rows.append(row)
        seen.add(row['key'])
        print(f"ROW {row['key']} error={row.get('error')}", flush=True)
    return rows


def _csv(path, rows, fields):
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def count_lines(source, language='python'):
    prefix = '//' if language in ('cpp', 'c++', 'cu') else '#'
    return sum(bool(line.strip()) and not line.lstrip().startswith(prefix)
               for line in source.splitlines())


def range_ratio(calib_min, calib_max, unseen_min, unseen_max):
    hi = max(float(unseen_max/calib_max), 1.0) if calib_max > 0 else None
    lo = max(float(unseen_min/calib_min), 1.0) if calib_min < 0 else None
    return dict(ratio_hi=hi, ratio_lo=lo,
                ratio=max([1.0]+[value for value in (hi, lo) if value is not None]))


def load_e1_data(normalization=None):
    from ..data.cifar import heldout_set

    return heldout_set(200, **({} if normalization is None else dict(normalization=normalization)))


def load_e4_data(normalization=None):
    import numpy as np
    from ..data.cifar import confirm_set, heldout_set

    kwargs = {} if normalization is None else dict(normalization=normalization)
    first, last = confirm_set(**kwargs), heldout_set(**kwargs)
    return np.concatenate((first[0], last[0])), np.concatenate((first[1], last[1]))


def load_e1confirm_data(normalization=None):
    return load_e4_data(normalization)


def _classes(path, names):
    source = Path(path).read_text()
    lines = source.splitlines(keepends=True)
    return ''.join(''.join(lines[node.lineno-1:node.end_lineno]) for node in ast.parse(source).body
                   if isinstance(node, ast.ClassDef) and node.name in names)


def _decision_count(source):
    acts = len(re.findall(r'\b(?:HE_ReLU|HE_SiLU|HE_Max|relu|silu)\s*\(', source))
    linear = len(re.findall(r'\b(?:HE_Conv|HE_ConvBN|HE_Linear|HE_DwConv|HE_ReshapeLinear)\s*\(', source))
    return 3*acts+linear


def expert_decisions(name, source):
    if name in MANUAL_DECISIONS:
        manual = MANUAL_DECISIONS[name]
        return dict(manual, source='manual', decisions=manual['linear_layers'],
                    decisions_source='manual', approximation_decisions=0)
    return dict(decisions=_decision_count(source), decisions_source='regex')


def _e7(root, model):
    return root/RUNS['e7']/'models'/model


def _raw_source(root, name, selected):
    for line in (root/RUNS['e5']/name/'agent_trace.jsonl').read_text().splitlines():
        row = json.loads(line)
        if row['iteration'] == selected['iteration']:
            content = next(message['content'] for message in reversed(row['messages'])
                           if message['role'] == 'assistant')
            return json.loads(content)['program']
    raise ValueError('Original LLM response is missing')


def _e3(out, root, repo):
    directory = out/'e3'
    directory.mkdir(parents=True, exist_ok=True)
    items = []
    experts = DACAPO_UPSTREAM
    for name in ('MLP', 'ResNet', 'AlexNet', 'VGG16', 'MobileNet', 'SqueezeNet'):
        items.append(dict(key=f'expert/{name}', model=name, artifact='expert_dsl',
                          paths=[str(experts/f'examples/benchmarks/{name}.py')], language='python'))
    items.append(dict(key='expert/shared', model='共享库', artifact='expert_shared', language='python',
                      paths=[str(experts/f'python/poly/poly/{name}.py') for name in ('Func', 'MPCB', 'Poly')]))
    for name in ('Lenet', 'ResNet', 'AlexNet', 'VGG16', 'MobileNet', 'SqueezeNet'):
        items.append(dict(key=f'torch/{name}', model=name, artifact='expert_torch', language='python',
                          paths=[str(experts/f'python/poly/poly/models/{name}.py')]))
    paths = sorted(str(path) for path in (repo/'benchmark/resnet20_gpu').rglob('*')
                   if path.suffix in ('.cpp', '.cu', '.h', '.hpp', '.cuh'))
    items.append(dict(key='cpp/resnet20', model='resnet20', artifact='handwritten_cpp',
                      paths=paths, language='cpp'))
    zoo = repo/'DSL/DSL_agent/scripts/pt_agent/models/zoo.py'
    sources = dict(resnet20=repo/'DSL/DSL_agent/workloads/resnet20/model_reference.py',
                   resnet20_silu=DACAPO_ROOT/'python/poly/poly/models/ResNet.py',
                   cnn_plain=zoo, mlp=zoo)
    for name in NAMES:
        graph = json.loads((root/RUNS['graphs']/name/'graph/graph.json').read_text())
        decisions = sum(3 if node[1] == 'act' else 1 if node[1] in ('linear', 'conv2d') else 0
                        for node in graph['nodes'])
        selected = json.loads((_e7(root, name)/'REPORT.json').read_text())['selected']
        for artifact in ('pytorch', 'frozen_dsl', 'llm_macro'):
            items.append(dict(key=f'dev/{name}/{artifact}', model=name, artifact=artifact,
                              decisions=decisions, selected=selected, paths=[str(sources[name])],
                              language='python'))

    def measure(item):
        row = dict(item)
        if item['artifact'] == 'frozen_dsl':
            row['paths'] = [str(_e7(root, item['model'])/'program.ptdsl')]
            source = Path(row['paths'][0]).read_text()
        elif item['artifact'] == 'llm_macro':
            row['paths'] = [str(root/RUNS['e5']/item['model']/'agent_trace.jsonl')]
            source = _raw_source(root, item['model'], item['selected'])
        elif item['artifact'] == 'pytorch' and item['model'] in ('cnn_plain', 'mlp'):
            source = _classes(item['paths'][0], {'CnnPlain' if item['model'] == 'cnn_plain' else 'Mlp'})
        else:
            source = '\n'.join(Path(path).read_text() for path in item['paths'])
        row['lines'] = count_lines(source, item['language'])
        row['decisions'] = (item['decisions'] if 'decisions' in item else
                            _decision_count(source) if item['artifact'] == 'expert_dsl' else None)
        if item['artifact'] == 'expert_dsl':
            row.update(expert_decisions(item['model'], source))
        row.pop('selected', None)
        return row

    rows = evaluate_pending(directory/'rows.jsonl', items, measure)
    _csv(directory/'loc.csv', rows, ('model', 'artifact', 'lines', 'decisions', 'decisions_source'))
    _write(directory/'summary.json', dict(rows=rows, errors=sum('error' in row for row in rows),
                                        manual_decisions=MANUAL_DECISIONS,
                                        decision_method='expert source-text regex occurrences with documented manual overrides; development counts use retained ATen graph nodes; decisions_source applies to expert DSL rows'))


def _model_context(args, experiment):
    import numpy as np
    from ..analyzer import reference_forward
    from ..models.zoo import get
    from ..planners.agent_tools import AgentContext, VAL_BATCH

    model = get(args.model, args.models_dir)
    graph = model.import_graph()
    calibration = json.loads((_e7(args.assets_root, args.model)/'calibration.json').read_text())
    loader = {'e1': load_e1_data, 'e4': load_e4_data,
              'e1confirm': load_e1confirm_data}[experiment]
    images, labels = loader(normalization=model.normalization)
    images = model.adapt(images)
    acts = [node.inputs[0] for node in graph.nodes if node.op == 'act']
    extrema = {name: dict(min=None, max=None) for name in acts}
    logits = []
    for start in range(0, len(images), VAL_BATCH):
        values = reference_forward(graph, {graph.inputs[0]: images[start:start+VAL_BATCH]}, keep=True)
        logits.append(values[graph.outputs[0]].copy())
        if experiment == 'e4':
            for name in acts:
                lo, hi = float(values[name].min()), float(values[name].max())
                old = extrema[name]
                old['min'] = lo if old['min'] is None else min(old['min'], lo)
                old['max'] = hi if old['max'] is None else max(old['max'], hi)
        del values
    exact = np.concatenate(logits)
    ctx = AgentContext(graph, graph.weights, calibration,
                       {key: value.shape for key, value in graph.weights.items()},
                       images, exact, labels, float(100*np.mean(exact.argmax(axis=1) == labels)),
                       images[:4], [], {}, acts, args.model,
                       val_split='test[9000:9200]' if experiment == 'e1' else 'test[8000:10000]',
                       normalization=model.normalization)
    return graph, calibration, ctx, extrema


def _base(graph, calibration):
    from ..dsl.parser import parse
    from ..planners.rule import plan

    return parse(plan(graph, calibration, deg=63, margin=1.5))


def _configuration_program(item, base, ranges):
    from ..dsl.parser import parse
    from .r9a_sampler import apply_config

    return parse(Path(item['source_path']).read_text()) if item['kind'] == 'reference' else apply_config(base, ranges, item['config'])


def _measure_config(item, base, ranges, ctx):
    from ..cost import estimate
    from ..dsl.typecheck import check
    from ..planners.agent_tools import validate

    program = _configuration_program(item, base, ranges)
    checked = check(program, ctx.weight_shapes)
    if not checked.ok:
        raise ValueError('typecheck: '+str(checked.diags))
    cost = estimate(program, ctx.weights, placement_model='depth-dp')
    result = validate(program, ctx, boot_nodes=cost['boot_nodes'], stop_on_divergence=False)
    return dict(item, **{key: cost[key] for key in ('boots_pred', 'latency_pred_us', 'max_depth')},
                **{key: result[key] for key in ('top1', 'exact_top1', 'acc_drop_pp', 'n_diverged', 'max_u')},
                feasible_like=result['n_diverged'] == 0 and result['acc_drop_pp'] <= 0.5)


def _e1_items(root, name, base):
    import numpy as np
    from ..dsl.parser import parse
    from .r9a_sampler import act_ops, anchor_configs, random_config

    rng = np.random.default_rng(20261023)
    configs = None
    for model in NAMES:
        acts = act_ops(parse((_e7(root, model)/'program.ptdsl').read_text()))
        generated = [random_config(acts, rng) for _ in range(RANDOM_COUNTS[model])]
        if model == name:
            configs = generated
            break
    items = [dict(key=f'{name}/random/{index:03d}', model=name, kind='random', config=config)
             for index, config in enumerate(configs)]
    items += [dict(key=f'{name}/anchor/{key}', model=name, kind='anchor', config=config)
              for key, config in anchor_configs(act_ops(base)).items()]
    items.append(dict(key=f'{name}/reference/r8e7', model=name, kind='reference', config=None,
                      source_path=str(_e7(root, name)/'program.ptdsl')))
    if name == 'resnet20':
        items.append(dict(key=f'{name}/reference/r6c', model=name, kind='reference', config=None,
                          source_path=str(root/RUNS['r6c']/'program.ptdsl')))
    return items


def _e4(args):
    graph, calibration, ctx, extrema = _model_context(args, 'e4')
    base = _base(graph, calibration)
    ranges = [calibration[name] for name in ctx.act_tensors]
    directory = args.out/'e4'
    items = [dict(key=f'{args.model}/range/{index}', model=args.model, kind='range',
                  tensor=name, calibration_min=calibration[name]['min'], calibration_max=calibration[name]['max'],
                  unseen_min=extrema[name]['min'], unseen_max=extrema[name]['max'])
             for index, name in enumerate(ctx.act_tensors)]
    from .r9a_sampler import act_ops
    methods = ['cheb']+(['minimax_comp'] if any(op.attrs['fn'] == 'relu' for op in act_ops(base)) else [])
    for method in methods:
        for margin in MARGINS:
            config = [dict(method=method if op.attrs['fn'] == 'relu' else 'cheb',
                           deg=None if method == 'minimax_comp' and op.attrs['fn'] == 'relu' else 63,
                           margin=margin) for op in act_ops(base)]
            items.append(dict(key=f'{args.model}/margin/{method}/{margin}', model=args.model,
                              kind='margin', method=method, margin=margin, config=config))

    def measure(item):
        if item['kind'] == 'range':
            return dict(item, **range_ratio(item['calibration_min'], item['calibration_max'],
                                           item['unseen_min'], item['unseen_max']))
        return _measure_config(item, base, ranges, ctx)

    evaluate_pending(directory/'rows.jsonl', items, measure)


def _e1(args):
    graph, calibration, ctx, _ = _model_context(args, 'e1')
    base = _base(graph, calibration)
    ranges = [calibration[name] for name in ctx.act_tensors]
    evaluate_pending(args.out/'e1/rows.jsonl', _e1_items(args.assets_root, args.model, base),
                     lambda item: _measure_config(item, base, ranges, ctx))


def select_e1confirm(rows, pareto_keys):
    """Select the prescribed union, preserving source order and unique keys."""
    selected = []
    seen = set()
    for name in NAMES:
        group = [row for row in rows if row.get('model') == name]
        if not group:
            continue
        frozen = next(row for row in group if row['key'] == f'{name}/reference/r8e7')
        for row in group:
            feasible = row.get('feasible_like', False) and 'error' not in row
            if (row.get('kind') == 'reference' or feasible and (
                    row['latency_pred_us'] < frozen['latency_pred_us']
                    or row['key'] in pareto_keys.get(name, ())
                    or row.get('kind') == 'anchor')) and row['key'] not in seen:
                selected.append(row)
                seen.add(row['key'])
    return selected


def _e1confirm(args):
    source = _rows(args.source/'e1/rows.jsonl')
    summary = json.loads((args.source/'e1/summary.json').read_text())
    items = [row for row in select_e1confirm(
        source, {name: value['pareto_keys'] for name, value in summary['models'].items()})
             if row['model'] == args.model]
    graph, calibration, ctx, _ = _model_context(args, 'e1confirm')
    base = _base(graph, calibration)
    ranges = [calibration[name] for name in ctx.act_tensors]

    def measure(item):
        result = _measure_config(item, base, ranges, ctx)
        result.update(feasible_like_200=item['feasible_like'],
                      acc_drop_pp_200=item['acc_drop_pp'], n_diverged_200=item['n_diverged'],
                      robust_feasible=result['feasible_like'], input_count=len(ctx.val_images))
        return result

    evaluate_pending(args.out/'e1confirm/rows.jsonl', items, measure)


def _matched_pair(item, weights, source_path):
    from ..cost import estimate
    from ..dsl.parser import parse

    program = parse(Path(source_path).read_text())
    matched = estimate(program, weights)
    return dict(item, boots_matched=matched['boots_pred'],
                latency_matched=matched['latency_pred_us'],
                placement_matched=matched['placement_model'], source_path=str(source_path))


def _e6_resummarize(args):
    from ..models.zoo import get

    model = get(args.model, args.models_dir)
    graph = model.import_graph()
    existing = {item['key']: item for item in _existing_pairs(args.assets_root, args.model)}
    items = [row for row in _rows(args.source/'e6/rows.jsonl') if row.get('model') == args.model]

    def measure(item):
        if 'error' in item:
            return dict(item)
        path = (existing[item['key']]['source_path'] if item['key'].startswith('existing/')
                else Path(item['report_path']).parent/'program.ptdsl')
        return _matched_pair(item, graph.weights, path)

    evaluate_pending(args.out/'e6/rows.jsonl', items, measure)


def _compile_pair(args, item, graph, calibration, base, ranges):
    from .. import toolchain
    from ..cost import estimate
    from ..dsl.printer import print_program
    from ..run import _inside as run_inside

    program = _configuration_program(item, base, ranges)
    predicted = estimate(program, graph.weights, placement_model='depth-dp')
    lazy = estimate(program, graph.weights, placement_model='lazy')
    case = args.out/'e6/programs'/args.model/item['key'].split('/', 1)[1].replace('/', '-')
    case.mkdir(parents=True, exist_ok=True)
    report_path = case/'REPORT.json'
    current = json.loads(report_path.read_text()) if report_path.exists() else {}
    # A completed native call may precede the append checkpoint during interruption.
    if current.get('compile') is None:
        source = Path(item['source_path']).read_text() if item['kind'] == 'reference' else print_program(program)
        (case/'program.ptdsl').write_text(source)
        _write(case/'calibration.json', calibration)
        run_inside(SimpleNamespace(out=case, model=args.model, models_dir=args.models_dir, planner='rule',
                                   deg=None, margin=None, refresh='compiler', act_method='cheb',
                                   stages=['check', 'count', 'emit', 'compile'], final=False))
        current = json.loads(report_path.read_text())
    emit = current.get('emit') or {}
    if (current.get('compile') or {}).get('exit') == 0 and not current.get('plan') and emit.get('run_dir'):
        from ..cost import plan_latency

        directory = Path(emit['run_dir'])
        current.update(plan=toolchain.plan_stats(directory),
                       plan_latency=plan_latency(directory/'lowered._hecate_golden.runtime-plan.json'),
                       complete=True, stage='complete')
        _write(report_path, current)
    if emit.get('run_dir') and not current.get('pruned'):
        current['pruned'] = toolchain.prune(Path(emit['run_dir']))
        _write(report_path, current)
    if (current.get('compile') or {}).get('exit') != 0 or not current.get('plan'):
        raise RuntimeError(f"compile failed: {current.get('error')}; REPORT={report_path}")
    return _pair_row(item['key'], args.model, predicted, lazy,
                     current['plan']['boots'], current['plan_latency']['latency_us'],
                     program.header.refresh, 'dacapo-depth-dp', report_path)


def _pair_row(key, model, predicted, lazy, boots, latency, refresh, mode, path):
    return dict(key=key, model=model, kind='pair', program=key, boots_pred=predicted['boots_pred'],
                boots_lazy=lazy['boots_pred'], boots_plan=boots,
                latency_pred=predicted['latency_pred_us'], latency_lazy=lazy['latency_pred_us'],
                latency_plan=latency, source_refresh=refresh, compile_mode=mode, report_path=str(path))


def _existing_pairs(root, model):
    items = []
    if model == 'resnet20':
        path = root/RUNS['r6b']/'REPORT.json'
        for index, case in enumerate(json.loads(path.read_text())['cases']):
            items.append(dict(key=f'existing/r6b/{index}', model=model, source_path=case['program_path'],
                              report_path=str(path), case=index))
        for name in ('R6c', 'R7c-B'):
            directory = root/RUNS['e7']/'baselines'/name
            items.append(dict(key=f'existing/baseline/{name}', model=model,
                              source_path=str(directory/'program.ptdsl'), report_path=str(directory/'REPORT.json')))
    for label in ('e5', 'e7'):
        directory = root/RUNS[label]/model if label == 'e5' else _e7(root, model)
        items.append(dict(key=f'existing/{label}/{model}', model=model,
                          source_path=str(directory/'program.ptdsl'), report_path=str(directory/'REPORT.json')))
    return items


def _existing_pair(item, weights):
    from ..cost import estimate, plan_latency
    from ..dsl.parser import parse

    program = parse(Path(item['source_path']).read_text())
    predicted = estimate(program, weights, placement_model='depth-dp')
    lazy = estimate(program, weights, placement_model='lazy')
    path = Path(item['report_path'])
    report = json.loads(path.read_text())
    if 'case' in item:
        report = report['cases'][item['case']]
    elif report.get('baseline'):
        report = json.loads(Path(report['original_frozen']['report_path']).read_text())
    plan = report['plan']
    latency = report.get('plan_latency')
    if latency is None:
        directory = (report.get('emit') or {}).get('run_dir') or report['run_dir']
        latency = plan_latency(Path(directory)/'lowered._hecate_golden.runtime-plan.json')
    mode = 'pars' if program.header.refresh == 'agent' else 'dacapo-depth-dp'
    return _pair_row(item['key'], item['model'], predicted, lazy, plan['boots'], latency['latency_us'],
                     program.header.refresh, mode, path)


def _e6(args):
    if args.source is not None:
        return _e6_resummarize(args)
    import numpy as np
    from ..models.zoo import get

    model = get(args.model, args.models_dir)
    graph = model.import_graph()
    calibration = json.loads((_e7(args.assets_root, args.model)/'calibration.json').read_text())
    base = _base(graph, calibration)
    ranges = [calibration[node.inputs[0]] for node in graph.nodes if node.op == 'act']
    rows = [row for row in _rows(args.out/'e1/rows.jsonl')
            if row.get('model') == args.model and row.get('feasible_like') and 'error' not in row]
    selected = []
    if rows:
        targets = np.quantile([row['latency_pred_us'] for row in rows], np.linspace(0, 1, 8))
        for target in targets:
            nearest = min(rows, key=lambda row: (abs(row['latency_pred_us']-target), row['key']))
            if nearest['key'] not in {item['key'] for item in selected}:
                selected.append(nearest)
    else:
        print(f'NO_FEASIBLE_E1 {args.model}', flush=True)
    directory = args.out/'e6'
    evaluate_pending(directory/'rows.jsonl', selected,
                     lambda item: _compile_pair(args, item, graph, calibration, base, ranges))
    evaluate_pending(directory/'rows.jsonl', _existing_pairs(args.assets_root, args.model),
                     lambda item: _existing_pair(item, graph.weights))


def _summarize(out, experiment):
    import numpy as np

    directory = out/experiment
    rows = _rows(directory/'rows.jsonl')
    result = dict(experiment=experiment, rows=len(rows), errors=sum('error' in row for row in rows), models={})
    for name in NAMES:
        group = [row for row in rows if row.get('model') == name]
        good = [row for row in group if 'error' not in row]
        model = dict(rows=len(group), errors=len(group)-len(good))
        if experiment == 'e1':
            valid = [row for row in good if row.get('latency_pred_us') is not None]
            feasible = [row for row in valid if row['feasible_like']]
            latency = [row['latency_pred_us'] for row in valid]
            if latency:
                lo, med, hi = float(min(latency)), float(np.median(latency)), float(max(latency))
                model.update(latency_min=lo, latency_median=med, latency_max=hi,
                             max_min_ratio=hi/lo if lo else None)
            model.update(feasible_like_count=len(feasible), feasible_like_fraction=len(feasible)/len(group) if group else None,
                         feasible_min_latency=min((row['latency_pred_us'] for row in feasible), default=None),
                         anchors=[row for row in group if row.get('kind') == 'anchor'])
            reference = next((row for row in valid if row['key'] == f'{name}/reference/r8e7'), None)
            model['frozen_percentile'] = (100*sum(row['latency_pred_us'] <= reference['latency_pred_us'] for row in feasible)/len(feasible)
                                          if reference and feasible else None)
            model['pareto_keys'] = [row['key'] for row in valid if not any(
                other['latency_pred_us'] <= row['latency_pred_us'] and other['acc_drop_pp'] <= row['acc_drop_pp']
                and (other['latency_pred_us'] < row['latency_pred_us'] or other['acc_drop_pp'] < row['acc_drop_pp'])
                for other in valid)]
        elif experiment == 'e4':
            ranges = [row for row in good if row['kind'] == 'range']
            ratios = [row['ratio'] for row in ranges]
            model.update(ratio_max=max(ratios, default=None), ratio_median=float(np.median(ratios)) if ratios else None,
                         ratio_gt_1p1_fraction=sum(value > 1.1 for value in ratios)/len(ratios) if ratios else None,
                         margin_sweep=[row for row in group if row['kind'] == 'margin'])
        elif experiment == 'e1confirm':
            frozen = next((row for row in group if row['key'] == f'{name}/reference/r8e7'), None)
            robust = [row for row in good if row['robust_feasible']]
            fastest = min(robust, key=lambda row: (row['latency_pred_us'], row['key']), default=None)
            confusion = dict(both_feasible=0, only_200_feasible=0,
                             only_2000_feasible=0, neither_feasible=0)
            for row in good:
                before, after = row['feasible_like_200'], row['robust_feasible']
                key = ('both_feasible' if before and after else 'only_200_feasible' if before
                       else 'only_2000_feasible' if after else 'neither_feasible')
                confusion[key] += 1
            model.update(confirmed_count=len(group), robust_feasible_count=len(robust),
                         robust_feasible_fraction=len(robust)/len(group) if group else None,
                         fastest_latency_us=fastest['latency_pred_us'] if fastest else None,
                         fastest_key=fastest['key'] if fastest else None,
                         frozen_latency_us=frozen['latency_pred_us'] if frozen else None,
                         fastest_to_frozen_ratio=(fastest['latency_pred_us']/frozen['latency_pred_us']
                                                  if fastest and frozen else None),
                         frozen={key: (frozen.get(key) if 'error' not in frozen or key in ('key', 'error') else None)
                                 for key in ('key', 'acc_drop_pp', 'n_diverged',
                                             'robust_feasible', 'error')} if frozen else None,
                         confusion=confusion, confusion_evaluated=len(good))
        result['models'][name] = model
    if experiment == 'e4':
        _csv(directory/'ranges.csv', [row for row in rows if row['kind'] == 'range'],
             ('model', 'tensor', 'calibration_min', 'calibration_max', 'unseen_min', 'unseen_max', 'ratio_hi', 'ratio_lo', 'ratio', 'error'))
        _csv(directory/'margin_sweep.csv', [row for row in rows if row['kind'] == 'margin'],
             ('model', 'method', 'margin', 'top1', 'exact_top1', 'acc_drop_pp', 'n_diverged', 'max_u', 'error'))
    elif experiment == 'e6':
        pairs = [row for row in rows if 'error' not in row]
        for label, boots, latency in (('depth-dp', 'boots_pred', 'latency_pred'),
                                     ('lazy', 'boots_lazy', 'latency_lazy'),
                                     ('matched', 'boots_matched', 'latency_matched')):
            available = [row for row in pairs if boots in row and latency in row]
            relative = [abs(row[latency]-row['latency_plan'])/row['latency_plan'] for row in available]
            result[label] = dict(pairs=len(available), boots_exact_fraction=sum(row[boots] == row['boots_plan'] for row in available)/len(available) if available else None,
                                 boots_mae=float(np.mean([abs(row[boots]-row['boots_plan']) for row in available])) if available else None,
                                 latency_relative_error_median=float(np.median(relative)) if relative else None,
                                 latency_relative_error_max=max(relative, default=None))
        _csv(directory/'pairs.csv', [dict(row, program=row.get('program', row['key'])) for row in rows],
             ('program', 'boots_pred', 'boots_lazy', 'boots_matched', 'boots_plan',
              'latency_pred', 'latency_lazy', 'latency_matched', 'latency_plan', 'error'))
    _write(directory/'summary.json', result)


def _limits():
    _, current_hard = resource.getrlimit(resource.RLIMIT_AS)
    hard = 6*2**30 if current_hard == resource.RLIM_INFINITY else min(current_hard, 6*2**30)
    resource.setrlimit(resource.RLIMIT_AS, (min(4*2**30, hard), hard))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def _outer(args):
    from ..paths import REPO
    from ..registry import append, git_rev, new_run_id

    _limits()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    with (out/'run.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        (out/'runner.pid').write_text(str(os.getpid())+'\n')
        current = json.loads((out/'REPORT.json').read_text()) if (out/'REPORT.json').exists() else {}
        round_name = 'R9a2' if args.source is not None else 'R9a'
        report = dict(round=round_name, git_rev=git_rev(), complete=False, stage=args.command, api_calls=0,
                      models_dir=str(args.models_dir), assets_root=str(args.assets_root), tasks=[],
                      previous_attempts=current.get('previous_attempts', [])+([
                          {key: value for key, value in current.items() if key != 'previous_attempts'}] if current else []))
        if args.source is not None:
            report.update(source=str(args.source), compiles=0, report_set_reads=0,
                          input_split='test[8000:10000]' if args.command == 'e1confirm' else None)
        _write(out/'REPORT.json', report)
        start = time.monotonic()
        try:
            subprocess.run(['git', 'merge-base', '--is-ancestor', 'pt-agent-freeze-v1', 'HEAD'],
                           cwd=REPO, check=True)
            for experiment in EXPERIMENTS if args.command == 'all' else (args.command,):
                report['stage'] = experiment
                _write(out/'REPORT.json', report)
                if experiment == 'e3':
                    _e3(out, args.assets_root, REPO)
                    report['tasks'].append(dict(experiment=experiment, exit_code=0))
                else:
                    for model in NAMES:
                        case = out/experiment/model
                        case.mkdir(parents=True, exist_ok=True)
                        command = [sys.executable, '-B', '-m', 'pt_agent.tools.nixrun', '--seconds', str(INNER_SECONDS), '--',
                                   '-m', 'pt_agent.experiments.r9a_motivation', experiment, '--inside', '--out', str(out),
                                   '--model', model, '--assets-root', str(args.assets_root), '--models-dir', str(args.models_dir)]
                        if args.source is not None:
                            command += ['--source', str(args.source)]
                        index = len(list(case.glob('stdout-*.txt')))
                        with (case/f'stdout-{index}.txt').open('w') as stdout, (case/f'stderr-{index}.txt').open('w') as stderr:
                            result = subprocess.run(command, stdout=stdout, stderr=stderr,
                                                    timeout=INNER_SECONDS+120, preexec_fn=_limits)
                        report['tasks'].append(dict(experiment=experiment, model=model, exit_code=result.returncode))
                        _write(out/'REPORT.json', report)
                        if result.returncode:
                            raise RuntimeError(f'{experiment}/{model} worker exited {result.returncode}')
                print(f'COMPLETE {experiment}', flush=True)
            report.update(complete=True, stage='complete')
        except Exception as exc:
            report.update(complete=False, error=f'{type(exc).__name__}: {exc}')
        report.update(seconds=time.monotonic()-start, run_id=new_run_id(round_name, 'dev-models', args.command))
        _write(out/'REPORT.json', report)
        append(dict(run_id=report['run_id'], round=round_name, model='dev-models', mode=args.command,
                    git_rev=report['git_rev'], metrics=report, artifacts_dir=str(out), ts=datetime.now(timezone.utc).isoformat()))
        print(f'REPORT={out}/REPORT.json', flush=True)
        return 0 if report['complete'] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=(*EXPERIMENTS, 'e1confirm', 'all'))
    parser.add_argument('--out', type=Path)
    parser.add_argument('--inside', action='store_true')
    parser.add_argument('--model', choices=NAMES)
    parser.add_argument('--models-dir', type=Path)
    parser.add_argument('--assets-root', type=Path)
    parser.add_argument('--source', type=Path)
    args = parser.parse_args()
    if args.command == 'e1confirm' and args.source is None:
        parser.error('e1confirm requires --source')
    if args.source is not None:
        args.source = args.source.resolve()
        if args.command not in ('e1confirm', 'e6', 'e3'):
            parser.error('--source supports e1confirm and e6/e3 summary recomputation')
    if args.inside:
        if any(value is None for value in (args.out, args.model, args.models_dir, args.assets_root)) or args.command not in ('e1', 'e4', 'e6', 'e1confirm'):
            parser.error('inside requires e1/e4/e6/e1confirm and explicit out, model, models-dir and assets-root')
        if args.source is not None and args.out.resolve().is_relative_to(args.source):
            parser.error('output must be outside the source results directory')
        args.out.mkdir(parents=True, exist_ok=True)
        {'e1': _e1, 'e4': _e4, 'e6': _e6, 'e1confirm': _e1confirm}[args.command](args)
        _summarize(args.out, args.command)
        return 0
    from ..paths import RESULTS_DIR

    args.assets_root = (args.assets_root or Path(os.environ.get('PT_RESULTS', str(RESULTS_DIR)))).resolve()
    args.models_dir = (args.models_dir or args.assets_root/'models-v2').resolve()
    prefix = 'r9a2-' if args.source is not None else 'r9a-'
    args.out = (args.out or args.assets_root/(prefix+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S'))).resolve()
    if args.source is not None and args.out.is_relative_to(args.source):
        parser.error('output must be outside the source results directory')
    return _outer(args)


if __name__ == '__main__':
    sys.exit(main())
