#!/usr/bin/env python3
"""Validate and summarize the twelve real compiler outputs."""
import datetime
import gzip
import hashlib
import json
from pathlib import Path
import subprocess


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root=Path(__file__).resolve().parent
    repo=root.parents[1]
    compiler=repo/'third_party/ckks-runtime/third_party/dacapo'
    rows=[]
    for mode in ['none','double','double-align','all']:
        for strategy in ['greedy','depth-dp','optimized']:
            case=root/mode/strategy
            result=json.loads((case/'audit.json').read_text())
            status=json.loads((case/'status.json').read_text())
            assert status['returncode']==0 and result['state_audit_passed']
            assert hashlib.sha256(gzip.decompress((case/'logical.mlir.gz').read_bytes())).hexdigest()==result['logical_sha256']
            rows.append(dict(mode=mode,strategy=strategy,**status,
                **{k:result[k] for k in ['boot_count','rescale_q','modswitch_q',
                                       'multiply_by_two','multiply_by_one','output_states','operations']}))
    assert len({r['compiler_sha256'] for r in rows})==1
    checks=(root/'compiler-tests.log').read_text()
    assert '100% tests passed out of 17' in checks
    source=repo/'build-relu-20261024/traced/ResNet20.mlirbc'
    model=repo/'bench/resnet20_s2c_first/support/data/resnet20'
    provenance=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        root_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
        compiler_base_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=compiler,text=True).strip(),
        compiler_sha256=rows[0]['compiler_sha256'],
        source=dict(path=str(source),bytes=source.stat().st_size,sha256=sha(source)),
        frontend_sha256=sha(repo/'test-results/model-regression-20261021/source/ResNet20.py'),
        relu_sha256=sha(model/'relu_param/d13.txt'),
        weights={p.name:sha(p) for p in sorted((model/'pretrained_parameters/resnet20_new').glob('*.txt'))},
        compiler_profile_sha256=sha(root/'compiler-profile.json'),
        scope='Local CPU compilation and small compiler semantic checks only; no GPU, encrypted model run, or GPU timing',
        defaults=dict(polynomial_optimizations='double-align',coefficient_min_scale=30,coefficient_precision_bits=24),
        settings=dict(N=65536,slots=32768,initial_scale=40,rescale_bits=30,
                      boot_output_level=22,boot_output_scale=45,minimum_level=4,usable_q=18))
    (root/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    (root/'summary.json').write_text(json.dumps(dict(cases=rows,compiler_tests_passed=17,
        gpu_executed=False,encrypted_accuracy_validated=False),indent=2)+'\n')
    text=['ResNet20 ReLU 编译优化对比（2026-10-24）','',
          '全部在本地使用同一份真实 ResNet20 Earth bytecode 和同一套 profile 编译。',
          'N65536、32768槽、初始scale40、30bit Rescale、Boot输出level22/scale45、最低level4，预算18Q。',
          '保留原d13多项式：15/15/27，未修改系数、近似函数或Boot实现。',
          '没有GPU运行、完整模型加密精度或性能测量；小图的编译/理想算术与物理scale检查通过。','',
          'mode | greedy Boot | depth-dp Boot | optimized Boot']
    for mode in ['none','double','double-align','all']:
        counts=[str(next(r['boot_count'] for r in rows if r['mode']==mode and r['strategy']==strategy))
                for strategy in ['greedy','depth-dp','optimized']]
        text.append(mode+' | '+' | '.join(counts))
    text += ['',
        'none：原编译规则；double：全槽精确常量2乘法改为自加。',
        'double-align：再把单消费者系数乘法后的对齐吸收到编码，允许穿过Rescale/ModSwitch；默认启用。',
        'all：再降低全槽均匀实数系数的编码scale；降低时要求至少30bit scale、非零编码系数至少24bit幅值精度；实验可选。',
        '原scale40已不能满足该局部精度条件的系数保持原编码，本轮不提高或补偿它们。',
        '该精度条件仅约束标量编码，不证明高阶复合多项式或整网加密误差。',
        '系数scale策略可能减少累计Rescale而增加Boot，不自动按模型挑选策略或回退。',
        '对齐吸收保持原Q消耗与最终scale/level；单独收益主要是少一次编码乘1。',
        '只处理全槽常量2；短常量/掩码保留。共享和sm_fixed中间值保留原状态。','',
        'mode / strategy | Rescale Q | ModSwitch Q | 普通乘2 | 乘1']
    for r in rows:
        text.append(f"{r['mode']} / {r['strategy']} | {r['rescale_q']} | {r['modswitch_q']} | "
                    f"{r['multiply_by_two']} | {r['multiply_by_one']}")
    text += ['',
        'Q消耗是应用图指令累计，不是单条路径预算，也不含Boot内部。',
        '计数来自最终unplaced CKKS IR，并与编译器打印的Boot数量核对；每个密文的Q/scale状态审计通过。',
        '同一二进制显式指定四种模式；默认仅前两项，第三项需--polynomial-optimizations=all。',
        'greedy/depth-dp/optimized都使用同一系数scale选择函数；DaCapo候选估价也包含对齐吸收。',
        '不同编译任务并行运行，elapsed_seconds不作为编译器性能基准。',
        'logical.mlir.gz省略大常量，只供审计；重编译需使用保存来源的真实bytecode。',
        'compiler.patch及新增测试保留本轮实现；提交和推送信息以Git历史为准。']
    sweep=root/'coefficient-floor-sweep/summary.json'
    if sweep.exists():
        text += ['', '附加编码scale下限扫描（幅值精度固定24bit）：', '下限 | greedy Boot | depth-dp Boot']
        records=json.loads(sweep.read_text())
        for floor in sorted({r['min_scale'] for r in records}):
            text.append(str(floor)+' | '+' | '.join(str(next(r['boot_count'] for r in records
                if r['min_scale']==floor and r['strategy']==strategy)) for strategy in ['greedy','depth-dp']))
    (root/'results.zh-CN.txt').write_text('\n'.join(text)+'\n')
    print('\n'.join(text))


if __name__=='__main__':main()
