"""Bounded, metadata-only Fig7 export of actual compiler operation intervals."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path

from .native_component_matrix import array_items


def export_plan(plan, output, *, model, system, param_group, expected_boots,
                expected_ops=None, interval=4096):
    """Retain unmapped labels when the exported IR has no source locations."""
    if type(interval) is not int or interval < 1:
        raise ValueError('Use a positive operation interval')
    plan, output = Path(plan), Path(output)
    levels = {row['id']: row['level'] for row in array_items(plan, 'values')}
    layers, sites, counts = [], [], Counter()
    current, total = None, 0
    for index, op in enumerate(array_items(plan, 'execution')):
        if index % interval == 0:
            if current is not None:
                layers.append(current)
            current = dict(label='unmapped', op_start=index, op_end=index,
                           ops_by_type={}, boots=0, level_min=None,
                           level_max=None, approx=None)
        kind = op['op']
        counts[kind] += 1
        current['ops_by_type'][kind] = current['ops_by_type'].get(kind, 0) + 1
        current['op_end'] = index + 1
        observed = [levels[v] for v in op.get('inputs', []) + [op['output']]]
        current['level_min'] = min(observed + [current['level_min']]
                                   if current['level_min'] is not None else observed)
        current['level_max'] = max(observed + [current['level_max']]
                                   if current['level_max'] is not None else observed)
        if kind == 'boot':
            current['boots'] += 1
            sites.append(dict(op_index=index, ordinal=op['ordinal'],
                              input_levels=[levels[v] for v in op['inputs']],
                              output_level=levels[op['output']]))
        total = index + 1
    if current is not None:
        layers.append(current)
    if counts['boot'] != expected_boots or (expected_ops is not None and total != expected_ops):
        raise ValueError('Actual RuntimePlan counts differ from their accepted source record')
    if sum(row['boots'] for row in layers) != expected_boots:
        raise ValueError('Intervals do not cover the actual bootstrap inventory')
    result = dict(model=model, system=system, param_group=param_group,
                  boots=counts['boot'], layers=layers, exec_ops=total,
                  ops_by_type=dict(counts), bootstrap_sites=sites,
                  source_runtime_plan=str(plan.absolute()),
                  op_interval_convention='Zero-based execution-array indices; op_end is exclusive; ordinal is retained separately',
                  source_mapping='RuntimePlan exports lack DSL statement locations. Fixed operation intervals remain unmapped; no inferred layer names.',
                  interval_size=interval, external_payloads_read=False,
                  model_or_CKKS_executed=False, integrity_checks=0)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError('Preserve an existing artifact publication: ' + str(output))
    temp = output.with_name(output.name + '.new')
    with temp.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write('\n')
    os.replace(temp, output)
    return dict(output=str(output), boots=counts['boot'], exec_ops=total,
                intervals=len(layers), mapped_source_labels=0)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('plan', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('model', 'system', 'param-group'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--expected-boots', required=True, type=int)
    parser.add_argument('--expected-ops', type=int)
    args = parser.parse_args(argv)
    print(json.dumps(export_plan(args.plan, args.out, model=args.model,
                     system=args.system, param_group=args.param_group,
                     expected_boots=args.expected_boots, expected_ops=args.expected_ops)))


if __name__ == '__main__':
    main()
