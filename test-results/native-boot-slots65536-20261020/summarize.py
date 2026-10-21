"""Check all slots and distinguish setup failures from numerical failures."""
from pathlib import Path
import json
import statistics
import re

root = Path(__file__).resolve().parent
reports, rows = [], []
for case_name, scale, cap in [('qwen30-s45',45,28672),
                              ('qwen30-s40',40,28672),
                              ('capacity32000-qwen30-s40',40,32000)]:
    case = root / case_name
    report = json.loads((case / 'report.json').read_text())
    status = json.loads((case / 'status.json').read_text())
    reports.append(report)
    assert report['degree'] == 131072
    assert report['slots_checked_per_call'] == 65536
    assert report['input_scale_log2'] == 40
    assert report['requested_eval_mod_log_scale'] == scale
    assert report['tolerance'] == 0.003
    assert report['q_bits'] == [30] * 50
    assert report['p_bits'] == [30] * 8
    samples = report['samples']
    measured = [s for s in samples if s['kind'] == 'measured']
    passed = report['passed']
    if passed:
        assert status['exit_code'] == 0 and report['stage'] == 'complete'
        assert len(samples) == 13 and len(measured) == 10
        assert report['output_scale_log2'] == scale
        assert all(s['passed'] and s['nonfinite'] == 0 for s in samples)
    else:
        assert status['exit_code'] != 0
    failures=[]
    for line in (case/'run.log').read_text().splitlines():
        if line.startswith('GPU_ALLOCATION_FAILURE '):
            failures.append({k:int(v) if v.isdecimal() else v
                for k,v in re.findall(r'(\w+)=(\S+)',line)})
    rows.append(dict(
        case=case_name,maximum_pool_mib=cap,
        eval_mod_log_scale=scale, output_scale_log2=report.get('output_scale_log2'),
        output_level=report.get('output_level'), passed=passed,
        calls=len(samples), measured_calls=len(measured),
        mean_ms=statistics.mean(s['wall_ms'] for s in measured) if passed else None,
        median_ms=statistics.median(s['wall_ms'] for s in measured) if passed else None,
        min_ms=min(s['wall_ms'] for s in measured) if passed else None,
        max_ms=max(s['wall_ms'] for s in measured) if passed else None,
        cold_ms=samples[0]['wall_ms'] if samples else None,
        max_complex_abs_error=max((s['max_complex_abs_error'] for s in samples), default=None),
        max_nonfinite=max((s['nonfinite'] for s in samples), default=None),
        profile_setup_ms=report.get('native_profile_setup_ms'),
        process_wall_seconds=status['wall_seconds'], exit_code=status['exit_code'],
        stage=report['stage'], error=report.get('error'),
        numerical_validation_started=bool(samples), allocation_failures=failures,
    ))

assert reports[0]['actual_q_moduli'] == reports[1]['actual_q_moduli']
assert reports[0]['actual_p_moduli'] == reports[1]['actual_p_moduli']
assert reports[0]['actual_q_moduli'] == reports[2]['actual_q_moduli']
assert reports[0]['actual_p_moduli'] == reports[2]['actual_p_moduli']
summary = dict(
    complex_slots=65536, polynomial_degree=131072,
    context='Q50/P8, all30bit; input scale=2^40; full degree58/DA2, no truncation',
    gpu='V100-SXM2-32GB, physical GPU0',
    pool='1GiB initial; first pair28GiB cap, target scale40 resource-only recheck31.25GiB cap',
    tolerance=0.003, identical_actual_moduli_between_cases=True, cases=rows,
    limits='Fresh full-level ciphertext, same deterministic message/ciphertext repeated within a case; independently generated random keys between cases. Not a complete Qwen or low-level-input validation. Other GPUs occupied, no fixed CPU affinity/GPU clocks. No nsys/NVML queries during execution.',
    timing='One cold call + two warmups + ten measured on passing cases. Wait for actual GPU completion; exclude setup/transfers/decryption validation. Stop after first numerical failure.',
    key_capacity_cpu_audit=json.loads((root/'key-capacity.json').read_text()),
)
(root / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
print(json.dumps(summary, indent=2, ensure_ascii=False))
