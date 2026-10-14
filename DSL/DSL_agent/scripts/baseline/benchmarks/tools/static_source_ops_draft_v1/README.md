# Static source operation draft

Agent 侧的公开除法和 functional MaxPool 输入适配草稿。

入口：

- `source_ops_draft.source_call()`：验证静态调用元数据，生成数学图及来源绑定。
- `fx_source_ops_draft.from_fx_call()`：在可信捕获进程中映射固定 Torch FX 调用。
- `audit.py --output <fresh-results-directory>`：在现有锁定 Nix Python 环境运行签名、拒绝、FX 与独立参考检查。

依赖当前的 Python 3.10.14、NumPy 1.25.2、Torch 2.0.1。沿用已注册的 broadcast 和 MaxPool 分解，以及既有资源配置。

接受 float64 加密输入元数据和公开非零有限分母。MaxPool 使用 N/C 布局、静态空间参数和显式绝对值近似配置，返回 floor-mode 的值输出。

当前完成明文参考验收。生产输入版本、Hecate 编译、物理布局、密态执行及 Agent 生成需要分别验收。归档子目录保留了开发中的原失败与已验证源码。

详见 `docs/baseline/agent-static-source-ops-draft-acceptance-r1057.md`。
