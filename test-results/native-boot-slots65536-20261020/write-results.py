from pathlib import Path
import json

root=Path(__file__).resolve().parent
summary=json.loads((root/'summary.json').read_text())
lines=[
    '65536个CKKS复数槽测试，对应N=131072',
    '服务器V100-SXM2-32GB，物理GPU0。Q50/P8、全30-bit、输入scale=2^40；完整实际58阶、多项式不截断、DA2。',
    '沿用scale40/45小尺寸实验的同一个二进制及同版本数学/GPU库；生产代码未改。',
    '先以28GiB显存池对照scale45和40，再仅对目标40把池上限提高到32000MiB（31.25GiB），排除28GiB容量限制。初始池均1GiB。',
    '',
    '配置 | pool上限GiB | 进程墙钟s | 结果 | 已完成校验调用数',
]
for row in summary['cases']:
    outcome='数值通过' if row['passed'] else ('数值失败' if row['stage']=='numerical_failure' else '初始化/资源错误，不能判断精度')
    lines.append(f"scale{row['eval_mod_log_scale']} | {row['maximum_pool_mib']/1024:.2f} | {row['process_wall_seconds']:.3f} | {outcome} | {row['calls']}")
    if row['error']:lines.append('  error: '+row['error'])
    for event in row['allocation_failures']:
        lines.append('  allocation: requested={:.3f}MiB, live={:.6f}GiB, reserved={:.6f}GiB, cap={:.6f}GiB, driver_free={:.3f}MiB'.format(
            event['requested_bytes']/2**20,event['allocated_live']/2**30,
            event['pool_reserved']/2**30,event['pool_cap']/2**30,event['driver_free']/2**20))
    if row['calls']:
        lines.append(f"  max_complex_abs_error={row['max_complex_abs_error']}, tolerance=0.003, output_level={row['output_level']}")
    if row['passed']:lines.append(f"  valid measured mean={row['mean_ms']:.6f}ms, median={row['median_ms']:.6f}ms")

audit=summary['key_capacity_cpu_audit']
lines.extend([
    '',
    f"CPU结构审计：默认三段C2S/S2C的旋转集合共{audit['galois_key_count']}把Galois密钥；实际生成一把密钥后按GPU uint32布局计量，每把{audit['packed_bytes_per_galois_key']}B。",
    f"全部Galois和Relin密钥的GPU backing下界为{audit['packed_key_backing_lower_bound_gib']:.8f}GiB；不含DFT矩阵、EvalMod常量、context参数、指针视图及workspace。",
    '对应CPU审计不执行GPU运算，不把这个下界当作完整profile显存峰值。',
    '',
    '资源失败不能算作数值失败。若calls=0，则既没有解密误差，也没有有效的单次Boot运行时间；上表进程墙钟只是到错误发生并清理结束的总时间。',
    '每组按原计划先冷调用1次、预热2次、测量10次，逐次验证全部65536复数槽，容差0.003；初始化失败时没有进入这段校验。',
    '不运行nsys，执行期间不轮询NVML/nvidia-smi。GPU0开始时无其他compute进程；其他卡有任务，CPU/PCIe共享且时钟、亲和性未固定。',
    '这是独立native Boot测试，不是完整Qwen验证，也没有覆盖真实Qwen低level输入；security_level=none。',
    '证据：summary.json，各case目录report/status/run.log，provenance.json与binary.json；CPU审计源码和key-capacity.json。',
])
(root/'results.zh-CN.txt').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))
