#!/usr/bin/env python3
"""Build comparison tables from retained compiler/Runtime/GPU evidence."""
import json
from pathlib import Path

base = Path(__file__).resolve().parent
evidence = base/'server'
read = lambda path: json.loads((evidence/path).read_text())


def compiled(path):
    p = read(path/'clear-execute.json')
    return dict(boots=p['boots'], rescale_q=p['rescale_q'], modswitch_q=p['modswitch_q'],
                total_deleted_q=p['rescale_q']+p['modswitch_q'], outputs=p['outputs'],
                verifier_passed=p['verifier_passed'], clear=p['clear_runtime'])


def full(policy, strategy):
    path = Path(f'qwen-full-{policy}')/strategy
    status = read(path/'status.json')
    out = dict(status=status)
    if (evidence/path/'audit.json').exists(): out['audit'] = read(path/'audit.json')
    if (evidence/path/'return-refresh.json').exists(): out['return_refresh'] = read(path/'return-refresh.json')
    return out


resnet = []
for strategy in ['greedy', 'depth-dp', 'optimized']:
    row = dict(strategy=strategy, after=compiled(Path(f'resnet20-runtime-{strategy}')))
    if strategy != 'optimized':
        row['before'] = compiled(Path(f'resnet20-runtime-before-{strategy}'))
        row['boot_reduction_percent'] = 100*(1-row['after']['boots']/row['before']['boots'])
    row['after_search'] = read(Path('resnet20-after')/strategy/'status.json')
    resnet.append(row)
prefix = []
for policy in ['before', 'after']:
    for strategy in ['greedy', 'depth-dp']:
        path = Path(f'qwen-prefix-{strategy}' if policy == 'after' else f'qwen-prefix-before-{strategy}')/'gpu1'
        gpu = read(path/'gpu-report.json')
        row = dict(policy=policy, strategy=strategy, plan=compiled(path),
                   gpu_passed=gpu['passed'], gpu_comparison=gpu['comparison'],
                   gpu_outputs=gpu['outputs'], runtime_ms=gpu['runtime_ms'])
        if policy == 'after': row['uniform_one_rounding_diagnostic'] = read(path/'clear-rounded.json')
        prefix.append(row)
projection = read(Path('qwen-projection-plan/gpu1/gpu-report.json'))
result = dict(
    resnet20=resnet, resnet_semantics=read(Path('semantics.json')),
    qwen_prefix=prefix,
    qwen_projection=dict(passed=projection['passed'], comparison=projection['comparison'],
                         runtime_ms=projection['runtime_ms'], native_boots=projection['native_boot_instructions']),
    qwen_full={s:dict(before=full('before',s), after=full('after',s)) for s in ['greedy','depth-dp']},
    optimized_prefix_time_limit=read(Path('qwen-prefix-optimized/gpu1/experiment-time-limit.json')),
    optimized_old_policy_rejection=read(Path('optimized-before-analysis/status.json')),
    gpu_physical_index=2, gpu_uuid='GPU-0768eaf5-15f4-6cca-f263-141df9bae9c6',
    arithmetic_policy_changed=False, addition_precision_compensation='deferred',
    regressions=['Full Qwen depth-dp Boot count: 42808 -> 130298 (+204.38%).',
                 'Normalized Qwen K/V prefix fails original GPU accuracy tolerance before and after.'],
    limits=[
        'ResNet20: real model and compiled FP64 replay; no full ResNet20 CKKS GPU execution.',
        'Full Qwen: logical compiler/Q audit only; no full-model placement or encrypted forward.',
        'Qwen normalized K/V prefix fails original CKKS tolerances before and after.',
        'Uniform encoded-one rounding explains an independent failure, not the entire GPU error.',
        'GPU numerical context N16384 uses sec_level_type::none; no security certification.',
        'Old compiler has only a Boot input/output contract shim; old scale/Q placement is preserved.',
        'Compiler cost tables are estimates; optimized search time is not GPU execution time.'])
(base/'validation-summary.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
print('ResNet20 Boot comparison:', [(r['strategy'],r.get('before',{}).get('boots'),r['after']['boots']) for r in resnet])
