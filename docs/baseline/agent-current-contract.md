# Agent 接口与模块

用户入口见 [Agent README](../../scripts/README.md)，模型格式见[输入说明](supported-inputs-models-configurations.md)。

## 调用链

```text
模型 → 类型/形状/布局检查 → 生成请求 → Provider
     → DSL 检查 → Hecate tracing → Dacapo → HEVM/CST
     → SEAL CPU → 解密比较 → 程序包
```

输入是静态图和公开权重。Provider 只接收公开模型、布局、DSL 规则和受控反馈，不接收参考答案、测试数组、凭据或私钥。候选响应只包含 schema、request_id、hecate_source。

## 模块

| 模块 | 职责 |
|---|---|
| application_component.py | 任务状态、预算、取消、终态、复用 |
| application_policy.py | 能力描述、上下文选择和修复决策 |
| component_contract.py | 请求准备、候选检查、生成接口 |
| run_candidate.py / run_feedback_loop | 单例 CLI 与生成修复循环 |
| validation_adapter.py | 静态、编译、产物和数值验收 |
| component_backend.py | 对接现有 SEAL 验证运行器 |
| deepseek_provider.py | DeepSeek 协议和受控传输重试 |
| candidate_bundle.py | DSL 程序包导出、完整性和重放 |
| benchmark_graph.py | 通用图类型与 shape 检查 |
| model_decomposition.py / restricted_model_python.py | 高层算子分解、受限 Python 导入 |

文件均在 [scripts/baseline](../../scripts/baseline/) 下。`benchmark_graph` 不读取实验批次，模型准入不依赖模型名称。

## 运行约定

默认要求真实密态执行并逐项比较；只有应用组件的显式 `compiled` 等级允许只编译。数值门限为 `1e-5 + 1e-4*abs(reference)`。

模型、权重、布局、规则与配置参与请求身份；候选不能更改它们。生成和修复共用同一 Provider，额度由宿主控制。重复失败、不可自动修复错误、取消和超时均返回终态。

应用 worker 与实验工具分开。宿主接口详见[应用组件](application-agent-component-v1.md)。验证后端仍有旧 CLI 兼容桥和全局工作环境依赖，需独立进程执行。

## DSL 配置

统一图默认 `native` 构造；`--unified-profile public-v1` 选择公开函数、容器和控制流的受限归一化。两者并非完整 Python，也不能任意混用旧构造开关。

上游 helper 使用 `--unified-helpers` 显式选择受信适配器；定向构造和 helper 组合由契约检查。配置清单见 [DSL 选项](dsl-options.md)。可用值以 `candidate -- --help` 为准，绑定实现在 [upstream_candidate_helpers.py](../../scripts/baseline/upstream_candidate_helpers.py)。

对象数组的一个 Expr 单元代表完整密文表达式，不是一个 slot。公开循环只展开公开计算，不能根据密文数据分支。当前 SEAL 链路不执行真实 bootstrap。

Benchmark、人工探针和批次修复入口见[开发测试说明](semantic-benchmark-v1.md)，不属于应用请求流程。
