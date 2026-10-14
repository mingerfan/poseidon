# Poseidon DSL 迁移验收

本批修复 RuntimePlan 编译入口，并补验定向 DSL 构造。Dacapo、Runtime 和 Poseidon 库源码与二进制保持原样；模型、reference、参数与数值门限保持原值。

## 结果

| 范围 | 结果 |
|---|---|
| 组件、配置及程序包测试 | 76 passed，0 failed，3 项历史证据测试 skipped |
| native 定向构造 | 129 个原任务获得通过证据，43 个分区，各 3 个不同拓扑上下文 |
| public 定向构造 | 48/48，16 个分区，各 3 个不同拓扑上下文 |
| 独立参考与密态比较 | 2,832 组双参考检查；708 组密态输入、2,164 个输出值 |
| 最大绝对误差 | 7.8923775e-07 |
| 应用与程序包 | MLP、RMSNorm、Attention、native 返回组合的数值等级，以及 MLP 的编译等级，5/5 |
| 重载与缓存 | 各包重载通过；重载生成/编译调用 0，缓存 Provider 调用 0 |
| 真实 Agent 生成 | 本批未运行，付费调用 0 |

全部数值比较使用 `1e-5 + 1e-4*abs(reference)`。native 中 9 项为空返回或空参数展开的结构验收，不计其数值贡献。其余构造核验实际 tracing、返回依赖和有限干预。

native 的 108 个任务使用 v3 配置，21 个保留 r247 的 v2 绑定；本表汇总各绑定的范围证据，未把历史记录改标为当前源码通过。public 48 个任务与应用 5 项使用最终配置及实现。

## 编译入口修复

原批次 39/48 通过，9 个编译失败：`native.call` 的独立公开参数 helper 触发 ciphertext modswitch 验证；`native.return` 和 `native.starred` 的空返回 helper 被 RuntimePlan 导出器拒绝。真实 frontend 已把调用展开到 `_hecate_golden`，编译器仍处理了未使用的独立定义。

新入口保留完整原始 IR、helper tracing 和构造证据；只将没有符号引用的非入口函数标为 private，随后调用现有 `symbol-dce`，再执行 EVA 和 RuntimePlan 导出。函数体、常量、类型及入口均保持原字节。符号引用、歧义声明或入口缺失会明确拒绝。

9 个原失败任务在新入口下全部真实密态通过。ModswitchC 验证器保持原样，候选 DSL 中的 helper 保留。

编译配置升级为 schema 3，绑定入口策略；schema 1/2 的历史请求可精确重建。程序包保存原始/入口 IR、选择记录和编译命令，重载重新核对它们。

## 覆盖缺口

[覆盖叠加账本](poseidon-semantic-migration-overlay-r257.json)保留原 401 项清单，映射 59/209 个构造分区和 45/153 个数学分区。其余 150 个 public 分区对应 450 个原任务尚缺本轮 Poseidon 证据。compiler 的旧 HEVM 证据协议、helper 原任务映射和拒绝任务映射仍待迁移。

这是离线工程验收。自由生成和定向真实 Agent 的 Poseidon 覆盖单独计分；本批没有增加 Agent 成功数。

r233 bootstrap 批准已执行为 r234：有符号输入首次刷新误差 1.0398e-4 超限，按方案停止。当前 Agent 仍不开放 bootstrap；r235 ratio128 试验需另行批准。x86 原生、空白安装与 CUDA 继续暂缓。

## 使用

开发工具运行一个最多 48 项的定向分片：

```sh
python -B scripts/baseline/benchmarks/tools/run_poseidon_directed_acceptance.py \
  --profile public --requirement <原账本分区ID> --output <当前平台results内的新目录>
```

增加 `--plan` 只准备和验证明文参考；默认不调用模型 API。该工具复用正常单例验证适配器，应用请求继续使用原组件接口。

完整记录：[JSON](poseidon-dsl-migration-r257.json)。当前付费提案：[三个小模型](poseidon-whole-models-live-proposal-r256.json)。旧 r243 保留原绑定，付费命令未执行。
