"""Export nonoverlapping phase attribution for the three closed primary runs.

Input tables retain their independently checked source clocks. This exporter
partitions recorded exclusive spans and separately accounts for setup and
completion outside the outer span. It creates no experimental process.
"""
import argparse
import csv
from datetime import datetime, timezone
from decimal import Decimal
import json
from pathlib import Path


CATEGORIES = (
    'intake_and_preflight', 'calibration', 'llm_requests',
    'candidate_parse_math_and_cost_tools', 'candidate_tuning',
    'independent_freeze_confirmation', 'strict_packed_arithmetic',
    'agent_context_and_session', 'placement', 'emit_and_constant_archive',
    'trace', 'native_compile_stage', 'outer_orchestration',
    'outside_root_span_setup_and_completion',
)
LABELS = {
    'source_preflight_and_snapshot': 'intake_and_preflight',
    'intake': 'intake_and_preflight', 'calibration': 'calibration',
    'llm_call': 'llm_requests',
    'agent_evaluate': 'candidate_parse_math_and_cost_tools',
    'agent_tune': 'candidate_tuning',
    'agent_confirm_select': 'independent_freeze_confirmation',
    'strict_packed_arithmetic': 'strict_packed_arithmetic',
    'agent': 'agent_context_and_session', 'placement': 'placement',
    'emit': 'emit_and_constant_archive',
    'archive_emit_constants': 'emit_and_constant_archive',
    'trace': 'trace', 'compile': 'native_compile_stage',
    'fresh_end_to_end': 'outer_orchestration',
}


def nanoseconds(seconds):
    value = Decimal(str(seconds)) * 10**9
    if not value.is_finite() or value < 0 or value != value.to_integral_value():
        raise ValueError('A nonnegative exact nanosecond clock is required')
    return int(value)


def partition(rows, cold_seconds):
    """Validate the linked cross-process tree before summing exclusive times."""
    by_id = {}
    for row in rows:
        if not row['event_id'] or row['event_id'] in by_id:
            raise ValueError('Every recorded stage needs a unique event ID')
        if row['status'] != 'ok':
            raise ValueError('Closed successful primary stages are required')
        item = dict(row)
        for key in ('inclusive_ns', 'exclusive_ns'):
            item[key] = int(row[key])
        if not 0 <= item['exclusive_ns'] <= item['inclusive_ns']:
            raise ValueError('Invalid recorded stage duration')
        by_id[row['event_id']] = item
    roots = [r for r in by_id.values() if not r['parent_id']]
    if len(roots) != 1 or roots[0]['stage_id'] != 'fresh_end_to_end':
        raise ValueError('One continuous fresh end-to-end root is required')
    root = roots[0]
    children = {key: [] for key in by_id}
    for row in by_id.values():
        if row['parent_id']:
            if row['parent_id'] not in by_id:
                raise ValueError('Recorded parent stage is missing')
            children[row['parent_id']].append(row)
    sums = dict.fromkeys(CATEGORIES, 0)
    classified = []
    for row in by_id.values():
        seen = set()
        current = row
        label = None
        while True:
            if current['event_id'] in seen:
                raise ValueError('Recorded stage ancestry contains a cycle')
            seen.add(current['event_id'])
            if label is None:
                label = LABELS.get(current['stage_id'])
            if not current['parent_id']:
                if current is not root:
                    raise ValueError('Recorded stage escapes the primary root')
                break
            current = by_id[current['parent_id']]
        if (row['exclusive_ns'] + sum(c['inclusive_ns'] for c in children[row['event_id']])
                != row['inclusive_ns']):
            raise ValueError('Exclusive children do not partition the recorded parent')
        sums[label] += row['exclusive_ns']
        classified.append(dict(event_id=row['event_id'], stage_id=row['stage_id'],
            parent_id=row['parent_id'], category=label, exclusive_ns=row['exclusive_ns']))
    if sum(sums.values()) != root['inclusive_ns']:
        raise ValueError('Recorded exclusive spans do not partition the root')
    cold_ns = nanoseconds(cold_seconds)
    if cold_ns < root['inclusive_ns']:
        raise ValueError('The completion clock precedes the measured outer span')
    sums['outside_root_span_setup_and_completion'] = cold_ns - root['inclusive_ns']
    return dict(cold_clock_ns=cold_ns, outer_span_ns=root['inclusive_ns'],
        stages=classified, category_ns=sums,
        attribution_sum_ns=sum(sums.values()),
        exact_partition_verified=sum(sums.values()) == cold_ns)


def export(delivery):
    delivery = Path(delivery)
    primary_path = delivery/'CURRENT_PRIMARY_TWELVE_CELL_COMPILE_DATA.csv'
    stage_path = delivery/'STAGE_ATTRIBUTION.csv'
    with primary_path.open() as stream:
        selected = [r for r in csv.DictReader(stream) if r['system'] == 'Agent']
    with stage_path.open() as stream:
        stages = list(csv.DictReader(stream))
    if len(selected) != 3 or {r['model'] for r in selected} != {'ResNet20', 'VGG16', 'AlexNet'}:
        raise ValueError('Exactly the three primary model entries are required')
    models = []
    flat = []
    for row in selected:
        if row['source_status'] != 'ok' or row['source_trace_reused'] != 'False':
            raise ValueError('Primary attribution requires a closed fresh run')
        case = Path(row['source_result']).parent
        spans = [r for r in stages if r['cell'] == row['cell_id'] and r['run'] == case.name]
        if not spans or any(Path(r['source_case']) != case for r in spans):
            raise ValueError('Stage and primary source identities differ')
        result = partition(spans, row['cold_intake_through_compile_complete_seconds'])
        llm = [r for r in spans if r['stage_id'] == 'llm_call']
        if len(llm) != int(row['API_calls_in_this_source']):
            raise ValueError('Recorded request spans and actual source API count differ')
        result.update(model=row['model'], cell_id=row['cell_id'], source_result=row['source_result'],
            source_completion=row['source_completion'], bootstrap_count=int(row['bootstrap_inventory_count']),
            api_calls=int(row['API_calls_in_this_source']),
            native_compiler_child_seconds=row['native_compiler_process_attempt_seconds'],
            details=[dict(event_id=r['event_id'], stage_id=r['stage_id'], inclusive_ns=int(r['inclusive_ns']))
                for r in spans if r['stage_id'] in ('llm_call', 'agent_evaluate', 'agent_tune',
                    'agent_confirm_select', 'frozen_program_placement', 'native_compile')])
        models.append(result)
        for category, duration in result['category_ns'].items():
            flat.append(dict(model=row['model'], cell_id=row['cell_id'], category=category,
                nanoseconds=duration, seconds=str(Decimal(duration)/10**9),
                fraction_of_cold=float(Decimal(duration)/result['cold_clock_ns']),
                cold_clock_seconds=row['cold_intake_through_compile_complete_seconds'],
                actual_compiled_bootstraps=result['bootstrap_count'], source_result=row['source_result']))
    record = dict(status='verified_recorded_exclusive_stage_partition',
        created_utc=datetime.now(timezone.utc).isoformat(),
        sources=dict(primary=str(primary_path), stages=str(stage_path)), models=models,
        scope='Continuous fresh intake through compilation and completion; queue waiting excluded. '
            'Nested exclusive spans are counted once. The outside-root category includes setup before '
            'the recorded outer span and completion after it. Candidate evaluation combines its recorded parse, '
            'math and cost work; their unrecorded subdivisions remain unknown. Native compiler child '
            'time is a separate detail, already contained in the compile-stage bar. Independent report '
            'testing is outside these cold clocks. No ciphertext/model/API execution performed.',
        full_model_ciphertext_measurements_added=0, new_api_calls=0)
    target = delivery/'PRIMARY_AGENT_COLD_STAGE_PARTITION.json'
    target.write_text(json.dumps(record, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    with (delivery/'PRIMARY_AGENT_COLD_STAGE_PARTITION.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
        writer.writeheader();writer.writerows(flat)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--delivery', required=True, type=Path)
    result = export(parser.parse_args(argv).delivery)
    print(json.dumps(dict(status=result['status'], models=len(result['models']),
        exact_partitions=[r['exact_partition_verified'] for r in result['models']])))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
