"""Extract actual original-DaCapo/Agent SSA neighborhoods from saved plans.

This is metadata reporting. Whole RuntimePlan JSON is parsed, including any
inline metadata payloads; external plaintext bundle files are never opened.
No numerical model, approximation or cryptographic operation is evaluated.
"""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import gc
import json
from pathlib import Path


META_FIELDS = ('id', 'kind', 'context', 'level', 'scale_log2', 'components', 'ntt')
MAX_PLAN_BYTES = 256 * 1024**2


def neighborhoods(plan, expected_boots):
    values = {v['id']: v for v in plan['values']}
    if len(values) != len(plan['values']):
        raise ValueError('Duplicate RuntimePlan value identity')
    defined = set(plan['external_inputs'])
    if len(defined) != 1 or len(plan['final_outputs']) != 1:
        raise ValueError('The original primary model uses one physical input and output')
    for item in plan['initialization']:
        if item['output'] in defined or item['output'] not in values:
            raise ValueError('Invalid initialization identity')
        defined.add(item['output'])
    boots = []
    producers = {}
    operations = plan['execution']
    for index, item in enumerate(operations):
        if (item['kind'] != 'compute' or item['output'] in defined
                or item['output'] not in values or any(i not in defined for i in item['inputs'])):
            raise ValueError('The saved execution must preserve original topological SSA')
        defined.add(item['output']);producers[item['output']] = index
        if item['op'] == 'boot':boots.append(index)
    if any(name not in defined for name in plan['final_outputs']):
        raise ValueError('Final output has no saved definition')
    if not boots or len(boots) != expected_boots:
        raise ValueError('The actual compiled bootstrap count differs from the primary inventory')
    windows = []
    for label, sequence in [('first', 1), ('middle', len(boots)//2+1), ('last', len(boots))]:
        center = boots[sequence-1]
        start, end = max(0, center-2), min(len(operations), center+3)
        rows = []
        edges = []
        for index in range(start, end):
            item = operations[index]
            rows.append(dict(execution_index=index, ordinal=item.get('ordinal'), op=item['op'],
                inputs=list(item['inputs']), output=item['output'], attrs=item.get('attrs', {}),
                input_metadata=[{k: values[name][k] for k in META_FIELDS if k in values[name]}
                    for name in item['inputs']],
                output_metadata={k: values[item['output']][k] for k in META_FIELDS if k in values[item['output']]}))
            for operand_index, name in enumerate(item['inputs']):
                producer = producers.get(name)
                if producer is not None and start <= producer < end:
                    edges.append(dict(producer_index=producer, consumer_index=index,
                        value=name, operand_index=operand_index))
        windows.append(dict(label=label, bootstrap_sequence_number=sequence,
            execution_index=center, rows=rows, actual_SSA_edges=edges,
            dependencies_outside_window_omitted=True))
    return dict(instruction_count=len(operations), value_count=len(values),
        op_counts=dict(Counter(i['op'] for i in operations)), bootstrap_count=len(boots),
        bootstrap_execution_indices=boots,
        input_metadata=[{k: values[n][k] for k in META_FIELDS if k in values[n]} for n in plan['external_inputs']],
        output_metadata=[{k: values[n][k] for k in META_FIELDS if k in values[n]} for n in plan['final_outputs']],
        windows=windows, semantic_layer_alignment=False)


def export(source_table, output):
    with Path(source_table).open() as stream:
        selected = [r for r in csv.DictReader(stream) if r['system'] in ('Agent', 'DaCapo-original')]
    if (len(selected) != 6 or {(r['model'], r['system']) for r in selected}
            != {(m, s) for m in ('ResNet20', 'VGG16', 'AlexNet') for s in ('Agent', 'DaCapo-original')}):
        raise ValueError('Six original DaCapo and primary Agent cells are required')
    result = []
    flat = []
    for row in selected:
        if (row['compiled_or_optimized_target_IR_ready'] != 'True'
                or row['model_specific_native_codegen_verified'] != 'True'):
            raise ValueError('Every selected original native compilation must be verified')
        source = Path(row['source_result']);record = json.loads(source.read_text())
        if row['system'] == 'Agent':
            path = Path(record['phases']['compile']['runtime_plan'])
        else:
            path = Path(record['plan']['path'])
        if not 0 < path.stat().st_size <= MAX_PLAN_BYTES:
            raise ValueError('Plan metadata exceeds this finite reporting allowance')
        plan = json.loads(path.read_text())
        details = neighborhoods(plan, int(row['bootstrap_inventory_count']))
        details.update(model=row['model'], system=row['system'], cell_id=row['cell_id'],
            source_result=str(source), source_completion=row['source_completion'],
            source_plan=str(path), source_plan_bytes=path.stat().st_size,
            actual_parameter_metadata=json.loads(row['parameter_metadata_json']))
        result.append(details)
        for window in details['windows']:
            for operation in window['rows']:
                flat.append(dict(model=row['model'], system=row['system'], window=window['label'],
                    bootstrap_sequence_number=window['bootstrap_sequence_number'],
                    total_compiled_bootstraps=details['bootstrap_count'],
                    execution_index=operation['execution_index'], ordinal=operation['ordinal'],
                    op=operation['op'], inputs_json=json.dumps(operation['inputs']), output=operation['output'],
                    input_metadata_json=json.dumps(operation['input_metadata']),
                    output_metadata_json=json.dumps(operation['output_metadata']),
                    source_plan=str(path), source_result=str(source)))
        del plan;gc.collect()
    report = dict(status='verified_actual_native_SSA_neighborhoods',
        created_utc=datetime.now(timezone.utc).isoformat(), source_table=str(Path(source_table)),
        scope='Each plan independently selects its first, floor(bootstrap_count/2)+1, and last '
            'bootstrap, with at most two adjacent execution instructions on either side. '
            'Actual local SSA edges retain operand multiplicity. Parameter groups and approximation '
            'semantics remain distinct; windows assert no semantic-layer alignment or common security '
            'or quality acceptance. Original DaCapo and internal fixed-Chebyshev controls are separate.',
        rows=result, models=3, native_plans=6, windows=18, exported_instruction_rows=len(flat),
        external_payload_reads=0, new_model_runs=0, new_compiler_runs=0, new_API_calls=0, new_CKKS_runs=0)
    output = Path(output);output.mkdir(exist_ok=True, parents=True)
    (output/'ORIGINAL_DACAPO_AGENT_NATIVE_NEIGHBORHOODS.json').write_text(json.dumps(report,
        indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    with (output/'ORIGINAL_DACAPO_AGENT_NATIVE_NEIGHBORHOODS.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-table', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args(argv)
    result = export(args.source_table, args.out)
    print(json.dumps({k: result[k] for k in ('status', 'native_plans', 'windows', 'exported_instruction_rows')}))
    return 0


if __name__ == '__main__':raise SystemExit(main())
