#!/usr/bin/env python3
"""Summarize the two completed runs; preserve the full execution interval."""
import json
from pathlib import Path
import re
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
RUNS = ROOT/'runs-nvml'


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def interval(markers, name):
    selected = [m['monotonic'] for m in markers if m['text'] == name]
    assert len(selected) == 1, (name, selected)
    return selected[0]


def weighted(samples, begin, end, value):
    """Nearest-sample midpoint intervals, clipped to the marked execution."""
    times = [s['monotonic'] for s in samples]
    assert times[0] <= begin <= end <= times[-1]
    cuts = [times[0]] + [(a+b)/2 for a, b in zip(times, times[1:])] + [times[-1]]
    total = 0.0
    covered = 0.0
    for i, sample in enumerate(samples):
        seconds = max(0.0, min(end, cuts[i+1])-max(begin, cuts[i]))
        total += seconds * value(sample)
        covered += seconds
    assert abs(covered-(end-begin)) < 1e-6
    return total / covered


def summarize(name):
    root = RUNS / name
    status, report = read(root/'status.json'), read(root/'report.json')
    markers, samples = read(root/'markers.json'), rows(root/'samples.jsonl')
    begin = interval(markers, 'PREFIX_EXECUTION_BEGIN')
    end = interval(markers, 'PREFIX_EXECUTION_END')
    assert end > begin
    selected = [s for s in samples if begin <= s['monotonic'] <= end]
    foreign = sorted({a['pid'] for s in selected for a in s['applications']
                      if a['pid'] != status['pid']})
    devices = []
    for index in range(4):
        util = weighted(samples, begin, end, lambda s: s['gpu'][index]['gpu_util'])
        devices.append(dict(index=index, time_weighted_gpu_util_percent=util,
            sampled_peak_gpu_util_percent=max(s['gpu'][index]['gpu_util'] for s in selected),
            sampled_peak_gpu_memory_mib=max(s['gpu'][index]['memory_mib'] for s in selected)))
    result = dict(variant=name, window=status['window'], status=status,
        execution_begin_monotonic=begin, execution_end_monotonic=end,
        execution_seconds=end-begin, cold_process_seconds=status['wall_seconds'],
        marked_forward_seconds=next(m['monotonic'] for m in markers if m['text'].startswith('FORWARD complete'))
                              - interval(markers, 'FORWARD begin'),
        mean_four_gpu_util_percent=statistics.mean(d['time_weighted_gpu_util_percent'] for d in devices),
        devices=devices, execution_samples=len(selected),
        median_query_sample_interval_seconds=statistics.median(b['monotonic']-a['monotonic']
                                                             for a,b in zip(selected,selected[1:])),
        median_query_seconds=statistics.median(s['query_seconds'] for s in selected),
        maximum_query_seconds=max(s['query_seconds'] for s in selected),
        observed_foreign_compute_pids=foreign, runtime_timing=report['runtime_timing'],
        preparation_measurements=report['preparation_measurements'],
        device_measurements=report['device_measurements'], comparison=report['comparison'],
        checkpoint_count=report['output_count'], checkpoint_pass_count=sum(o['passed'] for o in report['outputs']),
        boot_implementation=report['boot_implementation'],
        pinned_transfer_live_after_validation_bytes=report['pinned_transfer_live_after_validation_bytes'],
        bindings=read(root/'bindings.json'))
    log = (root/'run.log').read_text()
    peak_rows = re.findall(r'waterline GPU(\d+) front_peak=(\d+) plain_peak=(\d+) remaining_front=(\d+) remaining_plain=(\d+)',log)
    assert len(peak_rows) == 4
    result['waterline_ledger_peaks'] = [dict(zip(
        ('device','front_peak_bytes','plain_peak_bytes','remaining_front_bytes','remaining_plain_bytes'),
        map(int,row))) for row in peak_rows]
    shared = re.findall(r'waterline shared host_peak=(\d+) pinned_peak=(\d+) raw_peak=(\d+)',log)
    assert len(shared) == 1
    result['waterline_shared_peaks'] = dict(zip(('host_bytes','pinned_bytes','raw_bytes'),map(int,shared[0])))
    assert report['output_count'] == 16
    assert report['runtime_timing']['compute_calls'] == 508394
    assert report['runtime_timing']['encode_calls'] == 143955
    assert (report['preparation_measurements']['cpu_prepare_calls']
            + sum(d['gpu_encode_calls'] for d in report['device_measurements'])) == 143955
    assert not report['model']['complete_qwen']
    assert not report['model']['full_model_ckks_validated']
    assert report['tolerances'] == dict(atol=.001, rtol=.0001)
    assert not status['telemetry_errors'], status
    return result, samples


def main():
    results, traces = {}, {}
    for name in ('previous','updated'):
        results[name], traces[name] = summarize(name)
    old, new = results['previous'], results['updated']
    reports = [read(RUNS/n/'report.json') for n in ('previous','updated')]
    for key in ('actual_q_moduli','actual_p_moduli','context'):
        assert reports[0][key] == reports[1][key], key
    assert [d['uuid'] for d in reports[0]['devices']] == [d['uuid'] for d in reports[1]['devices']]
    for key in ('plan_sha256','spec_sha256','manifest_sha256','bundle_data'):
        assert old['bindings'][key] == new['bindings'][key], key
    fixtures = [read(RUNS/n/'fixture.json') for n in ('previous','updated')]
    for fixture in fixtures:
        del fixture['waterline']['lookahead_groups']
    assert fixtures[0] == fixtures[1]
    results['comparison'] = dict(execution_speedup=old['execution_seconds']/new['execution_seconds'],
        execution_reduction_percent=100*(1-new['execution_seconds']/old['execution_seconds']),
        gpu_util_increase_percentage_points=new['mean_four_gpu_util_percent']-old['mean_four_gpu_util_percent'],
        samples_per_variant=1,
        order=['updated','previous'],
        numerical_validation_passed_for_both=old['comparison']['passed'] and new['comparison']['passed'],
        gpu_util_metric='Time-weighted NVIDIA GPU busy %, midpoint sample intervals clipped to execution markers; not FLOP efficiency.',
        scope='Same bounded Qwen 4-GPU plan, previous Auto/8 vs updated Auto/128; combined production changes, not an isolated window experiment.')
    results['measurement'] = dict(valid_run_folder='runs-nvml',excluded_run_folder='runs',
        observer='Persistent NVML; initialized once; 2-second queries; no SMI/pmon subprocesses in valid runs.',
        excluded_reason='Repeated SMI query startup materially interfered; see observer-query-aba-diagnostic.json.')
    (ROOT/'summary.json').write_text(json.dumps(results,indent=2,allow_nan=False)+'\n')
    fig, axes = plt.subplots(3,1,figsize=(12,8),gridspec_kw={'height_ratios':[2,2,1.3]},constrained_layout=True)
    colors = ['#376bc5','#e0872e','#2a9d78','#bf5f97']
    for ax, name in zip(axes,('previous','updated')):
        result, trace = results[name], traces[name]
        start, end = result['execution_begin_monotonic'], result['execution_end_monotonic']
        selected = [s for s in trace if start <= s['monotonic'] <= end]
        for d in range(4):
            ax.plot([s['monotonic']-start for s in selected],
                    [s['gpu'][d]['gpu_util'] for s in selected],lw=.8,alpha=.8,color=colors[d],label=f'GPU {d}')
        ax.set(xlim=(0,max(old['execution_seconds'],new['execution_seconds'])),ylim=(-1,101),
               ylabel='GPU busy (%)',title=f'{name.capitalize()} Auto/{result["window"]}: {result["execution_seconds"]:.1f}s, mean {result["mean_four_gpu_util_percent"]:.2f}%')
        ax.grid(alpha=.2)
        ax.legend(loc='upper right',ncol=4,fontsize=8)
    axes[1].set_xlabel('Seconds from execution begin (includes admission/setup inside the execution call)')
    axes[2].barh(['Updated','Previous'],[new['execution_seconds'],old['execution_seconds']],color=['#2a9d78','#376bc5'])
    for y, name in enumerate(('updated','previous')):
        axes[2].text(results[name]['execution_seconds'],y,f'  {results[name]["execution_seconds"]:.1f}s',va='center')
    axes[2].set_xlabel('Execution wall time (seconds)')
    axes[2].set_xlim(0,1.2*max(old['execution_seconds'],new['execution_seconds']))
    axes[2].grid(axis='x',alpha=.2)
    fig.suptitle('Qwen3-0.6B: 2,000,000 execution instructions on four V100s (one run per variant)',fontsize=13)
    fig.savefig(ROOT/'gpu-utilization.png',dpi=160)
    plt.close(fig)
    lines = ['Qwen3-0.6B 四 V100，200 万条执行记录，单次对照', '',
             '指标 | previous Auto/8 | updated Auto/128',
             f'执行段秒数 | {old["execution_seconds"]:.3f} | {new["execution_seconds"]:.3f}',
             f'完整 runtime.run 秒数 | {read(RUNS/"previous/report.json")["runtime_ms"]/1000:.3f} | {read(RUNS/"updated/report.json")["runtime_ms"]/1000:.3f}',
             f'进程墙钟秒数 | {old["cold_process_seconds"]:.3f} | {new["cold_process_seconds"]:.3f}',
             f'四卡平均 GPU busy % | {old["mean_four_gpu_util_percent"]:.3f} | {new["mean_four_gpu_util_percent"]:.3f}']
    for d in range(4):
        lines.append(f'GPU{d} busy % | {old["devices"][d]["time_weighted_gpu_util_percent"]:.3f} | {new["devices"][d]["time_weighted_gpu_util_percent"]:.3f}')
    lines.extend([f'通过检查点 | {old["checkpoint_pass_count"]}/16 | {new["checkpoint_pass_count"]}/16',
        f'最大绝对误差 | {old["comparison"]["max_abs_error"]:.9g} | {new["comparison"]["max_abs_error"]:.9g}',
        f'最大虚部 | {old["comparison"]["max_imaginary"]:.9g} | {new["comparison"]["max_imaginary"]:.9g}',
        '',f'执行段速度比 old/new = {results["comparison"]["execution_speedup"]:.4f}。',
        f'GPU busy 增量 = {results["comparison"]["gpu_util_increase_percentage_points"]:.3f} 个百分点。',
        '', '以下是跨线程可重叠的累计值，不可相加当作墙钟时间：',
        '指标 | previous | updated'])
    for key in ('encode_submission_ms','bundle_read_ms','compute_submission_including_boot_ms',
                'host_boot_ms','communication_post_ms','communication_wait_ms'):
        lines.append(f'{key} | {old["runtime_timing"][key]:.3f} | {new["runtime_timing"][key]:.3f}')
    for label, metric in [('CPU Prepare 次数',lambda r:r['preparation_measurements']['cpu_prepare_calls']),
                          ('CPU Encode 累计 ms',lambda r:r['preparation_measurements']['cpu_encode_ms']),
                          ('CPU 明文 H2D bytes',lambda r:r['preparation_measurements']['cpu_plain_h2d_bytes']),
                          ('Host cipher 与 CPU 明文 H2D bytes',lambda r:r['preparation_measurements']['transport_h2d_bytes']),
                          ('GPU Encode 次数',lambda r:sum(d['gpu_encode_calls'] for d in r['device_measurements'])),
                          ('GPU Encode 累计 ms',lambda r:sum(d['gpu_encode_ms'] for d in r['device_measurements'])),
                          ('pinned 校验后存活 bytes',lambda r:r['pinned_transfer_live_after_validation_bytes'])]:
        lines.append(f'{label} | {metric(old)} | {metric(new)}')
    lines.extend(['', '每组只运行一次；前后使用相同当前 GPU 后端，仅 Runtime 和窗口不同。',
                  '这是截断图及 16 个 K/V 检查点的验证，不是完整 Qwen 输出验证。',
                  '含 1692 次依赖 secret key 的 Host decrypt-reencrypt refresh。',
                  '完整方法见 method.txt，有效报告和遥测见 runs-nvml/；runs/ 是已排除的 SMI 监测干扰诊断轮。',
                  'source-inspection.txt 是未验证线索；其驱动等待快照来自干扰轮，不能用来证明生产 Runtime 的瓶颈。'])
    (ROOT/'results.zh-CN.txt').write_text('\n'.join(lines)+'\n')
    print(json.dumps(results['comparison'],indent=2))
    for name in ('previous','updated'):
        r=results[name]
        print(name, 'seconds',r['execution_seconds'],'GPU mean',r['mean_four_gpu_util_percent'],
              'passed',r['comparison']['passed'],'foreign PIDs',r['observed_foreign_compute_pids'])


if __name__ == '__main__':
    main()
