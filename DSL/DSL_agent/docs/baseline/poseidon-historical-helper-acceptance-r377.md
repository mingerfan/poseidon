# 历史 Agent helper 与分块迁移验收

2026-10-01，ARM Ubuntu，`lhy-agent-dsl`。

**13 份历史真实 Agent 原 DSL 在 Poseidon CPU 重新编译、密态执行、逐项比较和程序包重载全部通过。** 模型、公开权重、候选、验收输入和独立 reference 保持原值；原响应、旧报告和源码绑定保留。

## 兼容性结果

| 原程序 / 模型 | 最大绝对误差 | 包重载 |
|---|---:|---|
| helper-v4 / conv_1_2_2_1_1 | 3.06188e-10 | 通过 |
| helper-v9 / chunk_helper_4 | 5.76471e-10 | 通过 |
| helper-v3 / concat_then_linear | 4.2141e-09 | 通过 |
| helper-v8 / bench_helper_0008 | 1.0441e-09 | 通过 |
| helper-v10 / fixed_Poly_Leaf2_1 | 5.44868e-08 | 通过 |
| helper-v5 / bench_helper_0053 | 9.213e-09 | 通过 |
| helper-v2 / linear_then_bn | 2.08304e-08 | 通过 |
| helper-v6 / bench_helper_0090 | 7.06216e-11 | 通过 |
| helper-v7 / ds_then_square | 8.55145e-09 | 通过 |
| chunked-linear-tail13 / chunk_helper_1 | 8.15395e-08 | 通过 |
| chunked-multi-input-output / chunk_helper_7 | 5.41871e-08 | 通过 |
| packed-tail-32 / bench_helper_0062 | 9.37009e-09 | 通过 |
| packed-tail-128 / bench_boundary_0014 | 6.89624e-09 | 通过 |

首轮52组/572值，max `8.153948267e-8`；13包重载另计52组/572值，max `1.187915575e-7`。重载生成与编译为0。9个helper配置加2个额外分块helper任务，以及P32/P128尾块补验。

实际调用含 HE_Conv、HE_BN、HE_Concat、HE_MPBN、HE_Avg/HE_Pool、HE_DwConv、HE_DS、Poly_Leaf2 与分块 HE_Linear。与 r371 合为27份历史程序案例，首轮108组/772值，重载另计相同数量；不是27个去重数学模型，不计为当前新生成。

## helper 输出贡献

13份交付来源审计通过。其中11 helper任务做三个公开探针的返回干预，分别改变单次调用与同名调用全部返回，确认输出响应；补充证据匹配原真实 Hecate trace 的 caller、callee、span、源码哈希。

原 trace 的 `output_contribution_proven=false` 保持原样，补充要求与原调用分别绑定。干预是有限探针上的影响检查，不是所有输入的证明；该审计未新增编译或密态，不读取私有 reference。

## 绑定与记录

r375绑定 `75fd171d47647dea62f6d60e70ab8ebd42ae1e0caf5a9da7a831b9d15649c545`，执行575/应用518文件；r376审计绑定 `4190809556503fd551c8433f2fd91b64404963a07eb42df01b4e82180c55f240`。容量扩展的579/522文件另有验收，不覆盖原结论。

原始证据：`/home/lhohy/poseidon-work/platforms/aarch64-linux/results/poseidon-historical-helper-replay-r375`、`/home/lhohy/poseidon-work/platforms/aarch64-linux/results/poseidon-historical-helper-audit-r376`。

每例 historical/ 保留原生成，delivery.json连接历史来源与新包。新包 agent_generated=false 表示本次没有生成，历史来源由独立封装表达。

13 passed / 0 failed / 0 blocked / 0 not_run / 0 skipped。门限 `1e-5+1e-4*abs(reference)` 不变。耗时约222秒、主批产物约87.7MiB。付费、安装、下载、commit、push为0。

[机器可读报告](poseidon-historical-helper-acceptance-r377.json) · [核心迁移 r373](poseidon-migration-status-r373.md)
