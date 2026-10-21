from pathlib import Path
import json

root=Path(__file__).resolve().parent
summary=json.loads((root/'summary.json').read_text())
rows=summary['cases']
lines=[
    'N=65536的原生GPU Boot对照（32768个复数槽）',
    '按用户澄清的多项式阶数N=65536测试；此前N131072/65536复数槽的OOM记录独立保留，不作本轮结果。',
    '服务器V100-SXM2-32GB GPU0；Q50/P8、全30-bit，输入scale=2^40；完整实际58阶、生成请求59、不截断、DA2。',
    '同一个实验二进制、同版本数学/GPU库，两组实际Q/P质数一致，只改变eval_mod_log_scale=40或45。生产代码未改。',
    '',
    '配置 | 输出scale(log2) | 输出level | 最大复数绝对误差 | 通过/调用数 | 有效平均ms',
]
for row in rows:
    mean=f"{row['mean_ms']:.6f}" if row['mean_ms'] is not None else '无'
    error=f"{row['max_complex_abs_error']:.12g}" if row['max_complex_abs_error'] is not None else '未取得'
    lines.append(f"{row['requested_eval_mod_log_scale']} | {row['output_scale_log2']} | {row['output_level']} | {error} | {row['passed_calls']}/{row['calls']} | {mean}")
    lines.append(f"  stage={row['stage']},profile_setup_ms={row['profile_setup_ms']},cold_ms={row['cold_ms']}")
    if row['error']:lines.append('  error='+row['error'])
lines.extend([
    '',
    '原生E2E容差保持0.003，每次检查全部32768个复数槽。通过组：1次冷调用、2次预热、10次计时；数值失败即停止该组，不将失败路径时间当作有效性能。',
    '单次Boot计时覆盖API调用和实际GPU完成等待，不含资源构建上传、输入上传、输出下载与解密校验。没有nsys或执行期间GPU监测查询。',
    '每组重复同一确定性消息及同一个新加密的完整level输入；组间随机密钥/密文不同。没有验证完整Qwen、真实低level输入或其他消息分布。',
    '初始pool1GiB、上限28GiB；GPU0开始时无其他compute进程。其他卡有任务；CPU亲和性和GPU时钟未固定。security_level=none。',
    '证据：summary.json、各case的report/status/command/environment/run.log、provenance.json、binary.json；源与二进制SHA绑定先前实验。',
])
(root/'results.zh-CN.txt').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))
