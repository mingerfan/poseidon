# v5 opt-in polynomial reassociation（CPU验证＋v6组合GPU实测）

单独v5消融尚未GPU计时。与prepared AV组合的v6已完成同runner同窗1warmup＋3测：
control-B8.236990537s→v6**6.303876115s**（Cerium104.04%），耗时降低23.47%。
不能将组合收益归因本pass；不同进程keys/profile物料也不支持误差因果归因。
全槽finite但max参考误差37.94，精度未通过。详情见[PREPARED_AV.md](PREPARED_AV.md)。

`--polynomial-reassociation` 独立于默认关闭的其余实验开关；
`polynomial_rewrite.py`只匹配原始power DAG与**全slot**公开系数。
没有降低次数、删项或改HE metadata；三轮reciprocal、所有dense权重与Boot标签保留。

- exp4：把末两项`x³/6+x⁴/24`先相加，允许同物理level的两个pending CP共享Rescale。
- GELU4：`.5*x+c*x²-c/6*x⁴ → .5*x+x²*(c-c/6*x²)`，相同浮点公开系数、仍2个CC。
- rsqrt3：`1-.5*u+.375*u²-.3125*u³ → ((-.3125*u+.375)*u-.5)*u+1`，仍2个CC。

这是实数多项式等价；浮点重排、CKKS Rescale与keyswitch的噪声顺序不同，
**不是ciphertext/解码结果逐bit等价**。

共享子图有保守guard：外部使用旧power/系数或rsqrt二次部分和时跳过相应改写，
避免保留旧平方又增加Horner CC。独立review发现的共享部分和P2已修复，并补了直接输出
和间接消费者的reachable/physical CC=2回归。

## T128一完整block的CPU计划（基于相同v3输入与全部参数SHA）

| 普通算子/opaque | v4 algebra＋shared＋QK | 再加v5 polynomial |
|---|---:|---:|
| Native Boot计划数 | 23 | 23 |
| MulCC/Relin | 856/856 | 856/856 |
| MulCP | 3287 | 3269 |
| Rescale调用 | 2619 | 2595 |
| Rotate | 2026 | 2026 |
| prepared Linear/QK calls | 27/6 | 27/6 |

Opaque内部算术仍须另加，不能把这张普通算子表当全部HE工作量。
匹配6个exp、12个GELU、2个rsqrt；参数/输入真实绑定，28,339,200B dense binary逐字节一致。
标准PlanVerifier通过。完整目录104项CPU测试通过，默认CLI七主产物再次逐字节复现v3。

- `test-results/bert-a800-20261115/polynomial-v5-cpu/`：早期物理计划研究，**没有fixture，不可直接GPU运行**。
- `.../polynomial-v5-bound/`：显式CLI生成的完整绑定候选，有fixture及SHA。
- 独立v6测量包含本pass＋AV；未从少18CP/24Rescale估算毫秒收益，单独v5 GPU时间仍缺。

生成示例：

```bash
python3 bench/bert_dsl/generate_aligned.py --output-dir NEW_V5 \
  --data-from-plan GOLD_V3 --periodic-encoding --algebra-rewrite \
  --shared-rotation-rescale --prepared-qk --polynomial-reassociation \
  --profile-report PROFILE_R256.json
```
