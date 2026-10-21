"""Summarize verified outputs; never report invalid calls as passing timings."""
from pathlib import Path
import json
import statistics

root = Path(__file__).resolve().parent
rows = []
reports = []
for scale in (40, 45):
    case = root / f'qwen30-s{scale}'
    report = json.loads((case / 'report.json').read_text())
    status = json.loads((case / 'status.json').read_text())
    reports.append(report)
    assert report['requested_eval_mod_log_scale'] == scale
    assert report['input_scale_log2'] == 40
    assert report['slots_checked_per_call'] == 32768
    assert report['degree'] == 65536
    assert report['q_bits'] == [30] * 50
    assert report['p_bits'] == [30] * 8
    assert report['tolerance'] == 0.003
    samples = report['samples']
    measured = [s for s in samples if s['kind'] == 'measured']
    passed = report['passed']
    if passed:
        assert status['exit_code'] == 0 and report['stage'] == 'complete'
        assert len(samples) == 13 and len(measured) == 10
        assert all(s['passed'] and s['nonfinite'] == 0 for s in samples)
        assert report['output_scale_log2'] == scale
    else:
        assert status['exit_code'] != 0
    rows.append(dict(
        requested_eval_mod_log_scale=scale,
        input_scale_log2=report['input_scale_log2'],
        output_scale_log2=report.get('output_scale_log2'),
        output_level=report.get('output_level'), passed=passed,
        calls=len(samples), measured_calls=len(measured),
        passed_calls=sum(bool(s['passed']) for s in samples),
        mean_ms=statistics.mean(s['wall_ms'] for s in measured) if passed else None,
        median_ms=statistics.median(s['wall_ms'] for s in measured) if passed else None,
        min_ms=min(s['wall_ms'] for s in measured) if passed else None,
        max_ms=max(s['wall_ms'] for s in measured) if passed else None,
        cold_ms=samples[0]['wall_ms'] if samples else None,
        max_complex_abs_error=max((s['max_complex_abs_error'] for s in samples), default=None),
        profile_setup_ms=report.get('native_profile_setup_ms'),
        tolerance=report['tolerance'], stage=report['stage'], error=report.get('error'),
    ))

assert reports[0]['actual_q_moduli'] == reports[1]['actual_q_moduli']
assert reports[0]['actual_p_moduli'] == reports[1]['actual_p_moduli']
summary = dict(
    scope='Single V100 GPU0, N65536 (32768 complex slots), Q50/P8 all30bit, input scale 2^40; native Boot only',
    independent_variable='GpuBootstrapProfileConfig.eval_mod_log_scale: 40 vs 45; full degree58/DA2 unchanged',
    polynomial_degree=65536, complex_slots=32768,
    identical_actual_moduli=True, cases=rows,
    pool='initial1GiB,maximum28GiB',
    measurement='One cold call, two warmups, ten measured; per-call actual GPU completion; validate every complex slot',
    limits='Same deterministic message and same ciphertext repeated within each case; new random keys/ciphertext between cases. Fresh full-level input, not a complete Qwen execution. Other GPUs occupied; GPU0 checked free before each case. No nsys or NVML querying during execution, no fixed clocks or affinity.',
)
(root / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
print(json.dumps(summary, indent=2, ensure_ascii=False))
