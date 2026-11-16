# 固定输入/权重A/B：不能只固定seed

本轮发现一个真实跨环境差异：本机NumPy2.5.4/glibc2.44与冻结12层远端
NumPy2.1.2/glibc2.35，`model.py`完全同SHA、相同seed/config，仍有45个大矩阵字段SHA不同。
独立检查`layer0.v_weight`589,824个float32：**只有1个float差1ULP**，绝对差
7.450580596923828e-09；远端原环境重新生成首层则逐bit匹配冻结资产。

这不是已发现的compiler rewrite改权重，也不能单独归因于NumPy或libm。
证据：`test-results/bert-a800-20261115/weight-environment-audit/`。
**本地未绑定的12层v4 CPU计划不是与冻结12层逐bit同权重的A/B输入。**
其结构/SSA/276计划Boot验证仍有效，不能据相同seed宣称权重逐字节相同。

## 精确前N层诊断

显式加`--data-prefix-layers --layers N`可从已有完整模型绑定前N层（N不超过source layers）。
仅layers可以不同，其他完整config与seed仍须相同；整个dense资产SHA、所有source矩阵
描述符/范围/集合仍检查，选中参数和原始输入逐字段SHA仍验证。未选层参数不重算，
metadata以`source_layers`/`selected_prefix_layers`/`all_source_parameter_hashes_verified`区分。
不加flag绝不允许隐式截层，也不把prefix输出称为完整12层输出。

T128两/四层CPU计划已验证54/108 Linear、12/24 QK及12/24 AV、46/92**计划**Boot，
每层Q13/直接SSA；这些本地结构测试绑定local12数据，不能冒充远端原始12层GPU数据。
正式新多层测试将从真实远端冻结12层资产绑定，而不跨环境重新抽样。

## 显式 `--data-from-plan GOLD_PLAN_DIR`

`weight_binding.py`：

1. 校验相同完整config和seed，记录原fixture/manifest SHA；按manifest中的SHA验证完整
   dense资产。基准调度方仍须用冻结外部清单确认GOLD目录本身。
2. 从真正的`linear-weights.bin`加载全部六组dense W/bias，不在新环境重新抽样它们。
3. **再校验所有参数字段SHA**。未单独存为raw dense的LN gamma/beta仍由原factory生成，
   必须与原fixture逐bit吻合，否则fail closed；不声称已导出不存在的raw LN数据。
4. 从原fixture的`inputs`载入**最初的清晰客户端输入**、核对config和原始f64 input SHA；
   不读取`expected_outputs`作为推理输入，不读取任何HE输出，不是decrypt–reencrypt。
5. 输入读取还须匹配参数绑定时的同一fixture SHA，防止中途替换混合快照；畸形schema
   失败为明确校验错误。生成器仍只做CPU DSL/lowering并记录来源SHA；另获GPU授权后，
   runner才正常加密原始输入并执行完整HE推理。

默认不开启，不改变v3旧CLI行为。若LN字段也出现跨环境差异，须使用原数值环境或另行
正式归档全部raw参数，不能跳过hash检查。

```bash
python3 bench/bert_dsl/generate_aligned.py --output-dir NEW_BOUND_PLAN \
  --data-from-plan GOLD_PLAN_DIR --layers 12 --scope encoder \
  --periodic-encoding --algebra-rewrite --shared-rotation-rescale --prepared-qk \
  --profile-report PROFILE.json
```

真实T128单层已验证绑定：`algebra-shared-qk-v4-bound/`。输入及全部16个参数字段SHA
与冻结v3单层一致，28,339,200B dense asset逐字节相同，标准PlanVerifier通过。
该绑定模式已用于冻结v4真实GPU A/B；当前活跃源码最近132项Python测试通过。
用户现已长期授权自主继续，不需逐项审批，仍有有界资源准入/清理。
