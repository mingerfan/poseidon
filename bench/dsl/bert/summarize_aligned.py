#!/usr/bin/env python3
"""Summarize completed native reports; never infer missing encoder latency."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


def load(path):
    path = Path(path)
    raw = path.read_bytes()
    return json.loads(raw), dict(path=str(path), sha256='sha256:' + hashlib.sha256(raw).hexdigest())


def runtime_summary(path):
    report, source = load(path)
    if not report.get('completed'):
        raise ValueError('native report did not complete: ' + str(path))
    runs = report.get('runs', [])
    if (report.get('instrumented_diagnostic') or any(r.get('instrumented_diagnostic') or
            r.get('formal_steady_sample') is False for r in runs)):
        raise ValueError('instrumented diagnostic is not a formal benchmark sample')
    measured = [r for r in runs if r['phase'] == 'measured']
    online = [r['online_execution_ms'] for r in measured]
    wall = [r['wall_ms'] for r in measured]
    if not online or any(not math.isfinite(x) or x <= 0 for x in online + wall):
        raise ValueError('missing/invalid completed timing samples')
    planned = report['native_boot_instructions_per_plan']
    if any(r['native_boot_calls'] != planned for r in report['runs']):
        raise ValueError('executed native Boot count differs from plan')
    measured_boots = sum(r['native_boot_calls'] for r in measured)
    all_boots = sum(r['native_boot_calls'] for r in report['runs'])
    if (measured_boots != report['native_boot_calls_measured'] or
            all_boots != report['native_boot_calls_total_including_warmup']):
        raise ValueError('aggregate native Boot counters disagree with completed runs')
    out = dict(source=source, completed=True, warmup=report['warmup'],
        measured_samples=len(measured),
        measurement_class=('single cold iteration' if report['warmup'] == 0 and len(measured) == 1
                           else 'warm measured iterations' if report['warmup'] else 'includes cold iteration'),
        online_ms=online, online_median_ms=statistics.median(online),
        runtime_wall_ms=wall, runtime_wall_median_ms=statistics.median(wall),
        native_boots_per_run=planned,
        native_boots_measured=measured_boots,
        native_boots_including_warmup=all_boots,
        model=report.get('model', {}),
        context=report['context'], native_profiles=report.get('native_boot_profiles', []),
        actual_q_moduli=report['actual_q_moduli'], actual_p_moduli=report['actual_p_moduli'],
        host_max_rss_kib=report.get('host_max_rss_kib'),
        tracked_device_measurements=report.get('device_measurements'),
        timing_scope=report['timing_scope'],
        finite_output_slots=sum(r['finite_slots'] for r in report['outputs']),
        nonfinite_output_slots=sum(r['nonfinite_slots'] for r in report['outputs']),
        max_reference_error=(max(r['max_abs_reference_error'] for r in report['outputs'])
                             if report['outputs'] and all(r.get('reference_error_valid')
                             for r in report['outputs']) else None),
        accuracy_acceptance_enforced=False,
        prepared_linear=report.get('prepared_linear'),
        prepared_linear_setup_ms=report.get('prepared_linear_setup_ms'))
    return out


def require_single_encoder(model):
    config = model.get('config', {}) if isinstance(model, dict) else {}
    expected = dict(tokens=128, hidden=768, heads=12, intermediate=3072, layers=1)
    if (not isinstance(model, dict) or not isinstance(config, dict) or
            model.get('scope') != 'encoder' or any(type(config.get(k)) is not int or
            config[k] != value for k, value in expected.items())):
        raise ValueError('report is not the full T128/H768 single-block encoder')


def add_encoder_variant(result, label, path):
    """Append only a completed, uninstrumented, full single-block variant."""
    if not label or label in result.get('encoder_variants', {}):
        raise ValueError('empty/duplicate encoder variant label')
    variant = runtime_summary(path)
    require_single_encoder(variant['model'])
    variant['latency_percent_of_cerium'] = (100 * variant['online_median_ms'] /
                                          result['cerium']['encoder_internal_median_ms'])
    result.setdefault('encoder_variants', {})[label] = variant


def summarize(old_root, new_root, encoder_report=None, baseline_report=None):
    old_root, new_root = Path(old_root), Path(new_root)
    cerium, cerium_source = load(old_root / 'cerium/RESULTS.json')
    audit, audit_source = load(new_root / 'cerium-audit.json')
    ffn = runtime_summary(new_root / 'compact-native-build/full-ffn-up/strict-report.json')
    old_encoder = runtime_summary(old_root / 'native-results/encoder-selected-host.json')
    old_ffn = runtime_summary(old_root / 'native-results/ffn-up-l5-fused-host.json')
    result = dict(format_version=1, workload=dict(tokens=128, hidden=768, heads=12,
        head_dim=64, intermediate=3072, encoder_blocks=1, synthetic_dense_weights=True),
        equal_parameter_comparison=False, task_accuracy_validated=False,
        comparison_limits=[
            'Different HE parameters, nonlinear approximations and CUDA Graph policies.',
            'New Poseidon preserves its earlier polynomial workload, not Cerium exact approximations.',
            'Cerium paired vec2 counts two independent CT streams but one bootstrap core.',
            'Runtime online excludes per-run setup/initialization; wall includes them.',
            'No speedup is inferred from Boot counts, incomplete attempts, or a sublayer.'],
        cerium=dict(source=cerium_source, audit_source=audit_source, refresh_counts=audit['totals'],
            encoder_internal_median_ms=cerium['fullblock']['steady_runtime_internal_median_ms'],
            encoder_h2d_run_d2h_median_ms=cerium['fullblock']['steady_h2d_run_d2h_median_ms'],
            ffn_program_run_wall_median_ms=cerium['ffn_up']['run_wall_median_ms']),
        previous_packed_encoder=old_encoder, previous_fused_ffn=old_ffn, compact_strict_ffn=ffn,
        ffn_shape_only_ratios=dict(
            previous_online_over_new_online=old_ffn['online_median_ms']/ffn['online_median_ms'],
            cerium_run_wall_over_new_online=cerium['ffn_up']['run_wall_median_ms']/ffn['online_median_ms']))
    cerium_ms = cerium['fullblock']['steady_runtime_internal_median_ms']
    result['latency_percent_of_cerium'] = dict(cerium=100.0,
        previous_packed=100 * old_encoder['online_median_ms'] / cerium_ms)
    if encoder_report is not None:
        encoder = runtime_summary(encoder_report)
        require_single_encoder(encoder['model'])
        result['compact_encoder'] = encoder
        result['latency_percent_of_cerium']['compact_encoder'] = 100 * encoder['online_median_ms'] / cerium_ms
        result['encoder_shape_only_ratios'] = dict(
            previous_online_over_new_online=old_encoder['online_median_ms']/encoder['online_median_ms'],
            new_online_over_cerium_internal=encoder['online_median_ms']/cerium['fullblock']['steady_runtime_internal_median_ms'])
    else:
        result['compact_encoder'] = dict(completed=False, online_median_ms=None)
    if baseline_report is not None:
        result['raw_prefix_baseline'] = runtime_summary(baseline_report)
        require_single_encoder(result['raw_prefix_baseline']['model'])
        result['raw_prefix_baseline']['steady_state_comparison'] = False
        result['latency_percent_of_cerium']['raw_prefix_cold'] = (
            100 * result['raw_prefix_baseline']['online_median_ms'] / cerium_ms)
        if encoder_report is not None:
            result['cold_to_warm_numeric_ratio_not_controlled_speedup'] = (
                result['raw_prefix_baseline']['online_median_ms'] / encoder['online_median_ms'])
    else:
        result['raw_prefix_baseline'] = dict(completed=False, online_median_ms=None,
            note='First attempt timed out; whole cold process bound is not online latency.')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--old-root', type=Path, default=Path('test-results/bert-a800-20261114'))
    parser.add_argument('--new-root', type=Path, default=Path('test-results/bert-a800-20261115'))
    parser.add_argument('--encoder-report', type=Path)
    parser.add_argument('--baseline-report', type=Path)
    parser.add_argument('--encoder-variant', action='append', default=[], metavar='LABEL=REPORT',
                        help='additional completed full single-block report; never a profiler/multi-plan sample')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.old_root, args.new_root, args.encoder_report, args.baseline_report)
    for value in args.encoder_variant:
        if '=' not in value:
            parser.error('--encoder-variant requires LABEL=REPORT')
        label, path = value.split('=', 1)
        try:
            add_encoder_variant(result, label, Path(path))
        except ValueError as error:
            parser.error(str(error))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(args.output)


if __name__ == '__main__':
    main()
