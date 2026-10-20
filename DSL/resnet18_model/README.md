# ResNet18 模型级 DSL（整网加密验证通过）

本包的模型级 Python DSL 已通过 image0 单样本的整网明文、GPU 加密和最终独立审计。
结果见 [最终审计](precision-combined/evidence/final-audit.json)。这不代表数据集准确率。

实际执行路线：

```text
模型级 Python → Hecate/Earth → 完整 Dacapo depth-dp → RuntimePlan
              → 正常 SequentialRuntime → Poseidon GPU / Native NoHoistBsgs
```

`ResNet18.py` 不指定 bootstrap、rescale 或物理 level，不读取旧 native trace/HEVM。

## 模型与配置

保留已验证原生模型的 7×7 stride2 stem、BN、原 d13 `[15,15,27]` 多项式 ReLU、
3×3 stride2 **平均池化**、2/2/2/2 BasicBlock、global average pool 和1000类FC。
边界归一化20；严格加载102个文本参数，拒绝缺失、未知、shape错误或非有限权重。

N65536 / 32768 slots；输入按行列奇偶拆成四组，每组两个密文，共8个输入。
布局显式屏蔽补零区域；最后对有效7×7平均，FC采用矩形对角线和baby/giant rotations。

最终有效配置是 [precision-combined/profiles-combined](precision-combined/profiles-combined)：
Q50/P25、H192、context scale51、内部 EvalMod scale60、ratio128、degree58/generation59、
NoHoistBsgs=4、实数投影。Native组件返回level15/scale60，完整Dacapo自动重新规划。
旧scale52/ratio32整网虽完成，但16个logit超差；它不是最终验收配置。

## 验证结果

| 检查 | 结果 |
| --- | --- |
| 槽布局 / 独立Torch FP64 | PASS |
| 前端1000 logits与dense最大差异 | 8.881784197001252e-15 |
| 最终整网明文Runtime与Python最大差异 | 0 |
| 最终编译计划逻辑模型一致性 | PASS |
| GPU整网Native Boot | 4705次，完整执行 |
| GPU最大绝对误差 / RMS | 0.008928450882893735 / 0.002623180777834832 |
| GPU最大虚部 | 9.927378136936084e-5 |
| 预测 / 标签 | 65 / 65 |
| 最终独立审计 | PASS |

1000个输出均满足原门槛 `abs(error) <= 0.1 + 0.006 * abs(reference)`，虚部门槛0.01。
最大误差比原失败配置0.178400884020407改善约19.98倍；未放宽门槛或固定FHE随机种子。
在线运行11292.59秒，GPU进程总耗时3:11:15；不用于跨机器性能比较。

## 源码与复现材料

- `ResNet18.py`、`imagenet_layout.py`：模型与槽布局。
- `check_layout.py`、`check_semantics.py`：独立明文校验；[语义报告](evidence/semantics-final.json)。
- `compiler-support/optimizer.cpp`：可选按需明文编码导出，避免全部权重同时驻留显存。
  `check_deferred_plan.py` 核验原始/按需计划的算术、常量和CKKS类型不变。
- `runtime-support/`：通用测试入口的8输入/1000输出支持；顺序Runtime单rank每256次传输
  回收已完成handle，已有40000次传输压力检查；GPU描述符查询索引。没有模型专用执行器。
- `precision-combined/`：实际成功的配置、GPU入口、编译/运行/独立审计脚本和证据。
  GPU入口仅增加读取 `bootstrap_ratio`；生产Runtime和密码学库复用。
- 顶层 `run_validation*.py`、`run_pool31.py` 是旧配置的服务器实验记录；最终成功流程是
  `precision-combined/run_precision.py`，原配置失败不能用作最终验收。

这些脚本保留验证时的服务器目录约定，并非开箱即用的仓库根目录命令。
基础目录为 `/home/hanshubin/no-hoist-dsl-work/resnet18-model-v2`，修正版为
`/home/hanshubin/no-hoist-dsl-work/resnet18-precision-combined-v1`；框架依赖为同级
`resnet20-framework-v1`。源码构建入口、原命令、源文件/二进制/配置SHA都随证据保存。
复现需要Linux/CUDA框架、匹配权重/图像/d13及原Hecate环境；调整路径时须重新生成溯源记录。
组件探测/比较脚本依赖服务器保留的历史实验；实际组件C++源码也已随包提供。

约25GiB Earth bytecode、约26GB权重bundle、二进制和完整trace保留服务器，不纳入Git。
没有中间解密重加密refresh、手写FHE计划、旧HEVM或trace重放。
历史native-logits是加密输出，单独记录其差异，不替代独立dense方程参考。
