# 张量轴生成规则与反馈验收

本轮增加 opt-in `explicit-v12`，已完成离线工程与代表性 Poseidon CPU 验收。此前两项真实 Agent 失败仍保留；本轮没有新的付费生成，也未将人工或规则程序记为 Agent 新生成成功。

## 改动

- 按声明图的实际算子补充分组归约、Linear 收缩轴、广播、reshape/transpose/permute 与分块输入的逻辑坐标。记录公开索引关系，候选自行选择中间密文布局。
- 静态、编译、产物及数值反馈使用相同的公开规则。`fixed-earth-mul-diagnosis-v2` 的受控 scale/budget 约束已进入结构化反馈；无稳定位置的编译提示仍不能触发“重复同一错误”终止。
- Controller 单独保存 `provider-transport.json`，保留缺少 DONE、无效事件等白名单诊断及重试决定。Provider 重试策略、旧生成记录格式及调用额度保持。
- 原有默认选择与 explicit-v1–11 保持。新版本同时挂载到通用和 Poseidon 专用只读沙箱。

## 验收

| 范围 | 结果 |
|---|---:|
| 单元与组件回归 | 308 passed，0 failed/errors/skipped |
| 默认及 v1–11 请求与旧实现比较 | 12/12 字节相同并可重建 |
| r1330 两份冻结请求 | 原始文件及请求哈希保持 |
| 代表性候选 | 3/3 数值验收通过 |
| 真实编译 / Poseidon 密态调用 | 3 / 6 |
| 固定与新输入 | 24 组，128 个输出值 |
| 最大绝对误差 | 1.45560083042e-07 |
| 程序包重载与新输入执行 | 3/3，通过；生成和重新编译均 0 |

门限为 `abs(actual-reference) <= 1e-5 + 1e-4*abs(reference)`。每例先以独立 PyTorch 与数学解释器核验 8 组输入，再完成四组资格验收与四组程序包新输入执行。

| 模型 | 来源 | 内容 |
|---|---|---|
| group_mean_center | 规则基线 | 分行 mean 与减法广播 |
| leading_linear_transpose_sum | 规则基线 | 有前导维度的 Linear、transpose 与分组 sum |
| historical_masked_mha | 历史真实 Agent 原源码 | 多输入 Attention、公开 mask/padding 与命名输出组合；源码 SHA 保持 |

历史 MHA 源码为 `292bff0b21f31c77036cd063c8e641df458058b9599cba4d5096b3fce98077da`。本次只重绑定生成规则与当前依赖，模型、公开常量、布局、packing v9、编译配置及独立 reference 保持。交付包来源为历史源码重放，并非当前 Agent 新生成。

## 失败记录与边界

首轮真实 tracing 因 Poseidon 专用沙箱漏挂新 Agent 模块停止；修复后两项通用模型通过。历史 MHA 验收启动器随后纠正了 packing 版本及组件选项，原失败均保留。短窗口到时的任务通过现有 recover 接口终止为 service_error，没有自动恢复或重放调用；只对剩余历史模型开启独立 5 分钟离线复核，45.64 秒完成。没有重复执行已通过的两项模型。

这些记录属于接线或验收启动器问题，未被删除或归因于模型不可表达。[机器报告](agent-tensor-axis-guidance-r1332.json) 保留每轮报告路径、哈希、终止和恢复记录。

## 状态与下一步

**已确认事实：**三项验收都有真实 tracing、Earth/CKKS IR、RuntimePlan、OperatorSpec、bundle、Poseidon 执行、解密和逐项比较；654 项后端/参考保护文件保持，代码只在 ARM Ubuntu 修改。编译和链接上限、4 GiB/150 秒 native 限制及全部安全检查保持。x86 配置保留，本轮未在 x86 执行。

**基于证据的判断：**v12 的公开逻辑轴关系针对 r1331 self 候选分头与槽位混合问题；跨通用模型的回归说明接线可用，不能据此推断新生成成功率提升。

**尚未验证：**v12 新生成质量、此前 self 与 cross 的新 Agent 重测、原非线性函数应用精度；bootstrap/CUDA 与更大整模型范围保持原状态。此前 cross 的失败子类型已在旧记录中过滤，不能由新增记录接口倒推出原因。

下一步先冻结当前模型、请求、源码和预算后申请真实 Agent 对比评测；继续通用接入、组合、容量和精度工作。本轮不新增任何付费授权。

## 使用

在 `/home/lhohy/Code/Poseidon` 准备新版本的公开请求：

```bash
python3 -B scripts/agent.py --backend local candidate -- \
  --case model.json --prepare --unified-guidance explicit-v12 \
  --compiler-configuration poseidon-cpu-eva-w45-v1
```

控制测试与一次性验收启动器位于 `scripts/baseline/benchmarks/tools/tensor_axis_activation_v1`。资格验收和程序包目录、依赖/产物哈希见机器报告；新验收使用独立日志和结果目录。

本轮安装、下载、SDK 重建、付费 API、后端库/参数/模型/reference 修改及 commit/push/PR 均为 0。[此前真实 Agent 结果](agent-mha-feedback-live-results-r1331.md)。
