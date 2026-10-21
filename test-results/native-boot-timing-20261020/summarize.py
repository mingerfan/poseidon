#!/usr/bin/env python3
"""Keep validated native timings distinct from failed profiles and old builds."""
import json
from pathlib import Path
import re
import statistics

root=Path(__file__).resolve().parent
summary={}
for name in ('runtime-mixed','qwen30','full59-runtime-mixed','full59-qwen30',
             'full58-runtime-mixed','full58-qwen30'):
    report=json.loads((root/name/'report.json').read_text())
    status=json.loads((root/name/'status.json').read_text())
    result=dict(report=report,status=status)
    samples=[s for s in report['samples'] if s['kind']=='measured']
    if report['passed']:
        assert status['exit_code']==0 and len(samples)==10 and len(report['samples'])==13
        assert all(s['passed'] and s['nonfinite']==0 for s in report['samples'])
        assert report['tolerance']==.003
        wall=[s['wall_ms'] for s in samples]
        result['validated_statistics']=dict(count=len(wall),mean_wall_ms=statistics.mean(wall),
            median_wall_ms=statistics.median(wall),minimum_wall_ms=min(wall),maximum_wall_ms=max(wall),
            mean_gpu_stream_span_ms=statistics.mean(s['gpu_stream_span_ms'] for s in samples),
            cold_wall_ms=report['samples'][0]['wall_ms'],
            max_complex_abs_error=max(s['max_complex_abs_error'] for s in report['samples']))
    else:
        assert status['exit_code']!=0
    summary[name]=result
text=(root/'existing-dynamic32/run.log').read_text()
native_ms=float(re.search(r'\| full_bootstrap_high_precision\s*\|\s*SKIP\s*\|\s*([0-9.]+)',text).group(1))
native_error=float(re.search(r'\| Full bootstrap\s*\| GPU/source decoded\s*\|\s*([0-9.eE+-]+)',text).group(1))
assert json.loads((root/'existing-dynamic32/status.json').read_text())['exit_code']==0
assert '[OK] One GPU bootstrap data set passed' in text
summary['existing-dynamic32']=dict(mean_wall_ms=native_ms,max_abs_error=native_error,
    scope='Separate existing standalone binary and shared library; not pooled with the current-backend measurements.')
(root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
lines=['原生 GPU Boot 单次耗时：单 V100，N=16384，8192 个复数槽',
    '当前后端与前次 Qwen 四卡复测共享相同数学/GPU 库；此次仅用物理 GPU0。','',
    '配置 | 预热后平均 ms | 中位 ms | 最小～最大 ms | 冷启动 ms | 最大复数绝对误差 | 精度']
for name,label in [('full58-runtime-mixed','Q34/P9 mixed-chain'),('full58-qwen30','Q50/P8，全30-bit，Qwen参数形状')]:
    s=summary[name]['validated_statistics'];r=summary[name]['report']
    lines.append(f'{label} | {s["mean_wall_ms"]:.3f} | {s["median_wall_ms"]:.3f} | {s["minimum_wall_ms"]:.3f}～{s["maximum_wall_ms"]:.3f} | {s["cold_wall_ms"]:.3f} | {s["max_complex_abs_error"]:.9g} | 13/13次通过')
    lines.append(f'  profile setup（含评估密钥/矩阵构建上传）{r["native_profile_setup_ms"]/1000:.3f}s；输出level={r["output_level"]}，scale=2^{r["output_scale_log2"]}。')
lines.extend(['','每组先测一次冷调用，再预热2次，最后10次计入平均；每次验证全部8192个复数槽。',
    '保留原有 native E2E 容差0.003，不放宽；同一组重复使用同一个加密输入，未验证完整Qwen。',
    '计时包含 Native GPU Boot API 调用和GPU实际完成等待；不含密钥生成、矩阵构建上传、输入上传、输出下载和解密校验。',
    'CUDA event给出同一执行流跨度，包含提交间隙；它不是各kernel执行时长之和。',
    '未使用nsys；计时期间不运行NVML/nvidia-smi查询。未固定CPU亲和性或GPU时钟。','',
    '通过的配置：生成请求阶数59的完整离散余弦多项式（实际阶数58），不截断，double-angle=2。',
    '默认配置：生成59阶后截断到22阶，double-angle=3；在两个上下文均数值失败，原样保存。',
    '这次同时改变多项式和double-angle配置，没有隔离哪一项导致默认配置的误差。',
    '显式配置为59阶时遇到metadata校验拒绝，因为生成器的实际多项式是58阶；按实际阶数配置后通过。','',
    '失败路径的冷调用时间仅作诊断，不是通过校验的Boot基准：'])
for name in ('runtime-mixed','qwen30'):
    s=summary[name]['report']['samples'][0]
    lines.append(f'{name}: {s["wall_ms"]:.3f}ms，最大误差{s["max_complex_abs_error"]:.9g}，未通过。')
lines.extend(['',f'另一个既有独立构建 dynamic32：预热2次、10次平均{native_ms:.3f}ms，源消息最大误差{native_error:.9g}，其容差为0.002。',
    '该二进制/共享库与当前Qwen后端不同，只作独立参考；版本和哈希见provenance.json。',
    '尝试以当前后端编译大型standalone测试时，先遇RMM include路径兼容问题，修正私有副本后又遇已缺失的CPU eval_mod_high_precision接口，未为绕过它改动生产代码。','',
    '与Host refresh的比较边界：前次Qwen Host refresh平均145.497ms，但它输出level49、scale=2^120。',
    '本次Q50/P8 Native输出level22、scale=2^45，剩余计算预算不同，不能按单次时间比推算整模型加速。',
    'Qwen runner和Waterline Native Boot的资源/workspace计费仍未接入；此次仅验证单个Native Boot及重复调用。',
    '所有配置均为security_level=none的数值验证上下文，不是安全参数认证。',
    '生产源码及共享库未改，精度通过来自显式的实验配置；结束后显存归零、ECC=0。',
    '源码、命令、环境、每次耗时和校验值均保留；未提交实验文件。'])
(root/'results.zh-CN.txt').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines[:11]))
