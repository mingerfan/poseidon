#!/usr/bin/env python3
import json
from pathlib import Path
base=Path(__file__).resolve().parent
rows=[]
for name,hardware in [('gpu1','4060 Laptop ×1'),('gpu2','V100 ×2 (physical 0,3)'),('gpu1-n65536','V100 ×1 (physical 0)')]:
    r=json.loads((base/name/'report.json').read_text())
    assert r['stage']=='complete' and r['runtime_timing']['boot_calls']==4
    assert r['host_boot_instructions']==0 and r['native_boot_instructions']==4
    assert not r['secret_key_assisted_refresh']
    assert r['comparison']['slots_checked']==r['context']['poly_degree']
    assert all(p['output_level']==27 and p['output_scale_log2']==40 for p in r['native_boot_profiles'])
    rows.append(dict(case=name,hardware=hardware,degree=r['context']['poly_degree'],devices=r['device_worker_count'],native_boots=4,profile_setup_ms=sum(p['setup_ms'] for p in r['native_boot_profiles']),runtime_ms=r['runtime_ms'],online_ms=r['runtime_timing']['online_execution_ms'],completed=True,numerical_passed=r['passed'],slots_checked=r['comparison']['slots_checked'],max_abs_error=r['comparison']['max_abs_error']))
for count in [1,4]:
    status=json.loads((base/'compiler-script-smoke-final'/f'gpu{count}'/'status.json').read_text())
    assert status['returncode']==0
limit=json.loads((base/'gpu1/limit-report.json').read_text())
assert limit['stage']=='execute_complete_plan' and 'exceeds configured hard limit' in limit['error']
summary=dict(synthetic_graph_not_qwen=True,api_regressions_passed=27,compiler_regressions_passed=5,production_compile_script_devices=[1,4],cases=rows,workspace_limit_negative_case=limit['error'])
(base/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
lines=['scale40 编译器/runtime 接入验证（2026-10-21）','',
'这是工程链路验证，不是完整 Qwen 推理或精度验收。未修改 Boot 数学算法，未放宽 atol=0.001、rtol=0.0001。',
'测试通过正式 hecate-opt --dacapo、RuntimePlan binary、Auto 水位加载及 poseidon_runtime_gpu_qwen_e2e 执行。',
'图为两条独立的九次 MulCP 链，系数 1.01，输入 0.25 / -0.125；共四次原生 Boot，无 Host refresh。',
'Q50/P8 均为 30-bit；工作 scale40；Boot 使用完整 degree58（生成请求59，无截断）、DA2、内部/输出 scale40。',
'输入和 Boot 结果均按可恢复的物理 level27 编译，四次逻辑降尺度为 27→22→17→12→7。',
'两个输出各检查 N/2 槽，尾部零槽和虚部也检查；N=65536 代表每密文32768槽。','',
'硬件 | N | Boot profile 初始化(秒，设备依次构建合计) | runtime.run(毫秒) | online(毫秒) | 数值校验',
'--- | --- | --- | --- | --- | ---']
for r in rows:
    lines.append(f"{r['hardware']} | {r['degree']} | {r['profile_setup_ms']/1000:.3f} | {r['runtime_ms']:.3f} | {r['online_ms']:.3f} | {'通过' if r['numerical_passed'] else '失败，退出码1'}")
lines += ['',
'初始化包含 Boot 的 CPU plans/keys/constants 生成和 GPU 上传，与执行时间分开。runtime.run 包括其 setup、initialization 和 online；不含 profile 构建及最终解密校验。',
'单次运行，无统计重复；不同硬件与 GPU 数量的数据不能作为完整 Qwen 加速比。重复错误 Boot 的数值误差不能与之前单次 Boot 的误差直接对比。','',
'验证：27 项 GPU API 回归通过；5 项编译器回归通过。正式 compile_plans.py 在服务器对小图生成单卡、四卡计划成功。',
'资源回归将所有种类的 Boot 常量填入后，比较实际 allocator 字节；两 profile 共享一份 key，reserve 恰好够时通过、少1字节时拒绝。',
'真实 GPU 将 Boot workspace 上限设为1字节时，首次请求262144字节即报错；没有 Host 回退。',
'Boot 临时空间由线程局部分配器硬限制；全额容量计入固定 reserve，N16384 默认1GiB，N65536默认4GiB。输出复制到普通图存储。',
'N65536 本次配置每卡 M=31GiB、reserve=24GiB；日志测得 fixed reserve 需要21201045176字节（约19.75GiB），包括常量/keys/参数、workspace容量和余量。','',
'服务器隔离目录：/home/xuming/poseidon-native40-integration-20261021',
'该目录包含更新后的 src、build/bin/poseidon_runtime_gpu_qwen_e2e、toolchain/bin/hecate-opt、profiles 和 qwen3_dsl/compile_plans.py。',
'测试没有覆盖旧服务器源码/二进制或旧 Qwen artifacts；toolchain/lib 和 python 复用旧工具链依赖。',
'源码改动在本地 framework 工作区及其 Dacapo 子模块内；本轮未做 commit/push。',
'local-provenance.json 与 server-provenance.sha256 记录当前最终源文件/二进制哈希。二卡样例早于最后的分配器对齐/计数/溢出检查调整；N65536 和最终本地测试使用最终实现。',
'prepare-smoke.py 生成样例与编译命令，run-local.py 重跑本地正例及上限反例；生产编译脚本的命令和状态在 compiler-script-smoke-final/。','']
(base/'results.zh-CN.txt').write_text('\n'.join(lines))
print(json.dumps(summary,ensure_ascii=False,indent=2))
