# Poseidon CPU：归一化与批量矩阵乘

12 个补充模型完成真实 frontend tracing、编译、RuntimePlan/bundle 检查、密态执行、解密及逐项数值比较。它们来自已有算子注册与分解实现；没有修改编译器、Runtime、参数或近似配置，也没有调用模型 API。

## 已验证范围

| 计算 | 三个上下文 |
| --- | --- |
| LayerNorm | 输入 [4]、[2,3]、[1,2,3]；分别归一化 [4]、[3]、[2,3] |
| GroupNorm | [1,2,3]/1组、[1,4,2]/2组、[1,2,2,2]/2组 |
| InstanceNorm | [1,2,3]、[2,2,2]、[1,2,2,2]；按通道归一化，无运行统计量 |
| 批量 matmul | [2,2,3]×[2,3,2]；[1,2,3]×[2,3,2]；[2,3]×[2,3,2] |

两个独立底层 reference 与高层数学 reference 的 192 组输入检查通过。最终密态验收为 48 组输入、336 个输出值，最大绝对误差 7.56172e-7，门限保持 1e-5 + 1e-4*abs(reference)。候选是规则 self-test 或人工等价表达，真实 Agent 成功数为 0。固定 1200 模型集保持原样；这些补充模型与旧集的结构重叠尚未统一审计，不直接增加原 Benchmark 规模。

## 归一化表达

最初默认规则基线：matmul 3/3 通过，归一化 9/9 在编译阶段失败。失败点为最后的公开增益乘法，输入类型为 ci<75 *11> 与 pl<45 *11>；当前剩余模数不足以满足对应检查。

等价表达把增益及输出 mask 提前，提取各组均值、方差，复用已有倒平方根迭代，再乘倒平方根并加偏置。修正后的九项源程序通过静态检查和独立参考，随后全部通过 Poseidon CPU。首次探针使用了未开放 rotation step 和尾块未声明 mask，静态拒绝记录保留；后续只改表达，沿用已绑定 mask 并组合合法 rotation key，没有放宽契约。

默认规则转换器的九项失败仍保留；替代表达是开发 fixture，尚未接入默认 lowering，也没有据此宣称真实 Agent 已生成这些程序。

## 近似精度

本批使用已有显式配置：epsilon=0.2、倒平方根区间 [0.1,4]、两轮 Newton。密态比较验证的是该近似计算。它相对原始归一化函数的最大误差为 0.632650，尚不适合目标网络精度验收。需要网络精度时，应先校准和冻结新的近似配置，再单独验证，不能用 CKKS 小误差替代这一项。

## 程序包

LayerNorm 多轴案例和广播批量 matmul 均导出 v4 RuntimePlan 程序包，并完成直接重载数值验收。重载共 56 个输出值，最大绝对误差 2.30258e-7；生成与编译调用均为 0。

第一个包已实际重载通过，但初版开发汇总器读取了错误的 manifest 字段名，随后中止。r319 日志与不完整汇总保留；r321 按相同源码绑定重新审计该次成功重载，只新执行第二个包，不覆盖旧记录。

程序包路径：

- LayerNorm：/home/lhohy/poseidon-work/platforms/aarch64-linux/results/poseidon-normalization-packages-r319/layer_norm_2/program
- matmul：/home/lhohy/poseidon-work/platforms/aarch64-linux/results/poseidon-normalization-packages-r321/batched_matmul_1/program

## 状态与复现

应用核心 514 文件仍匹配 r286。当前已确认上述有限 shape 与近似配置的 CPU 能力；新上下文真实 Agent、默认归一化 lowering 修复、网络近似精度和大张量布局分别待验收。原 12 个高阶多项式失败及 bootstrap 阻塞未消失，见 [深度检查](poseidon-polynomial-depth-audit-r312.md)。

matmul 的三个现有成功路径可从尚不存在的结果目录重现：

```sh
POSEIDON_PLATFORM=aarch64-linux POSEIDON_WORK_ROOT=/home/lhohy/poseidon-work \
python3 -B scripts/baseline/check_operator_regression.py \
  --manifest scripts/baseline/cases/operator-extension-v5/manifest.json \
  --case-id batched_matmul_0 --case-id batched_matmul_1 --case-id batched_matmul_2 \
  --compiler-configuration poseidon-cpu-eva-w45-v1 \
  --output /home/lhohy/poseidon-work/platforms/aarch64-linux/results/matmul-next --run-local
```

归一化探针入口为 scripts/baseline/benchmarks/tools/check_poseidon_normalization.py，使用保留的 r316 preparation/report.json 及新的 --output；它属于开发验收工具。宿主应用继续使用现有组件接口。

本批 passed：12 个最终密态案例、2 个程序包；failed：保留的 9 个默认编译路径及开发工具历史错误；skipped=0，最终选定路线 not_run=0。无付费、安装、下载、SDK 重建、提交或推送。CUDA、x86 原生、空白安装继续暂缓。

机器记录：[r322 JSON](poseidon-normalization-matmul-acceptance-r322.json)。
