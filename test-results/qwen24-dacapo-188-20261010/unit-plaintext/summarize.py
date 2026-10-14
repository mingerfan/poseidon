"""Summarize independently verified full plans after unit plaintext reduction."""
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent / 'gpu-plans'
REMOTE = '/home/xuming/poseidon-qwen-dacapo-20261010-IUOMH1/artifacts/qwen24-unit-plaintext/gpu-plans'
records = []
for devices in (1, 4):
    directory = ROOT / f'gpu{devices}'
    def read(name):
        return json.loads((directory / name).read_text())
    compiler = read('compiler-report.json')
    validator = read('validation-report.json')
    validation = read('validation.json')
    unit_counts = read('unit-counts.json')
    assert sum(unit_counts.values()) == validation['full_unit_encode_count']
    assert compiler['returncode'] == validator['returncode'] == 0
    assert all(validation[name] for name in ('matches_complete_decoded_host_graph', 'hashes_verified',
        'topology_verified', 'physical_metadata_verified', 'release_reuse_verified', 'memory_report_independently_verified'))
    memory = read('qwen24._hecate_qwen25_24layer.memory.json')
    before = read('qwen24._hecate_qwen25_24layer.memory.before.json')
    previous = json.loads((OLD / f'gpu{devices}/validation.json').read_text())
    gpu = [row for row in memory['places'] if row['place']['kind'] == 'device']
    log = (directory / 'compile.log').read_text()
    times = {}
    for label in ('Parser', 'GreedyBootstrapPlacement', 'AssignPlacement', 'MaterializeCommunication',
                  'PlanRuntimeMemory', 'EmitRuntimePlan', 'Output'):
        match = re.search(r'^\s*([\d.]+)\s+\([^\n]*\)\s+' + label + r'$', log, re.M)
        assert match, label
        times[label] = float(match[1])
    reduction = {name: previous['instruction_counts'][name] - validation['instruction_counts'][name]
                 for name in ('encode', 'compute', 'transfer', 'release')}
    records.append({'devices': devices,
        'plan_remote': f'{REMOTE}/gpu{devices}/qwen24._hecate_qwen25_24layer.runtime-plan.json',
        'compiler': compiler, 'compiler_pass_seconds': times,
        'unit_counts': unit_counts,
        'validation_supervisor': validator, 'validation': validation,
        'instruction_reduction': reduction,
        'mul_cp_before': previous['op_counts']['mul_cp'], 'mul_cp_after': validation['op_counts']['mul_cp'],
        'gpu_peak_TiB': [row['peak_bytes'] / 2**40 for row in gpu],
        'aggregate_gpu_peak_TiB': validation['aggregate_gpu_peak']['bytes'] / 2**40,
        'aggregate_gpu_peak_before_TiB': previous['aggregate_gpu_peak']['bytes'] / 2**40,
        'memory': memory, 'memory_before_release': before})
reference = json.loads((ROOT / 'semantic-reference.json').read_text())
summary = {'status': 'passed', 'scope': 'Complete 24-layer Qwen + full LM head, two-token prefill',
    'weights': 'synthetic distinct-layer weights; no pretrained checkpoint',
    'provenance': json.loads((ROOT / 'provenance.json').read_text()),
    'full_unit_encodes_before': reference['full_unit_encode_count'],
    'full_unit_encodes_after': records[0]['validation']['full_unit_encode_count'],
    'full_one_encodes_before': sum(row['count'] for row in reference['full_unit_histogram'] if row['sign'] == 1),
    'full_one_encodes_after': records[0]['unit_counts']['full_one_encode_count'],
    'boot_count': 10835, 'plans': records,
    'memory_kind': 'Sequential RNS object estimates; not measured GPU/process peaks or parallel bounds',
    'assumption': records[0]['memory']['assumption'], 'excluded': records[0]['memory']['excluded'],
    'eager_plaintext_initialization': True, 'fits_32GiB_gpus': False,
    'decoded_arithmetic_matches_original': True,
    'full_model_ckks_rounding_or_noise_validated': False, 'gpu_execution_tested': False,
    'large_artifacts_in_tmp': False}
(ROOT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps({'unit_encodes': [summary['full_unit_encodes_before'], summary['full_unit_encodes_after']],
                 'plans': [{key: row[key] for key in ('devices', 'instruction_reduction', 'mul_cp_before',
                    'mul_cp_after', 'gpu_peak_TiB', 'aggregate_gpu_peak_TiB')} for row in records]}, indent=2))
