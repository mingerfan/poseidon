# MultiheadAttention 输入草案

这是 Agent 自有的输入适配草案，用于核对固定 Torch 2.0.1 的模块调用与声明式图语义。生产入口保持当前版本。

## 接口

- `snapshot_module(module)`：读取原生、eval、CPU float64 的 MultiheadAttention 公开配置和权重。
- `public_mask(tensor)`：将公开布尔或 float64 mask 转成数据；负无穷用独立阻断标记表示。
- `from_fx_call(...)`：按固定调用签名绑定已知本地 FX 节点。
- `plan(...)`：输出原始请求、数学图、分解图、输出结构和来源绑定。
- `verify(...)`：结合外部固定的请求哈希核验内容和实现身份。

## 已检验范围

12 个小型上下文覆盖自注意力、交叉注意力、多头、不同 Q/K/V 特征宽度、两种 batch 布局、无 batch、公开 mask、padding mask、额外 bias/zero token、因果提示及 attention weights。

输入为 rank 2 或 3。输出保持 `(attn_output, attn_weights)` 结构；`need_weights=False` 时第二项是静态 `None`。MHA 的布尔 mask 中 `True` 表示阻断。

调用方提供 exp 和 reciprocal 近似方法及有效区间。原函数参考、多项式参考和分解后结果分别记录。

## 验收

使用现有锁定的 pure Nix Python 环境运行 `audit.py --output <新结果目录>`。结果目录位于当前 work root 的 `results` 下。

r1063：25 项测试通过，12 个上下文、192 组明文差分、1,920 个输出值。原函数独立参考与 Torch 最大差异约 `5.55e-17`；分解最大差异约 `6.66e-16`。

当前状态是数学草案验收。生产接入、物理布局、真实编译、密态执行和真实 Agent 生成分别待验收。应用精度也需在实际输入域上认证。

详见 [验收报告](../../../../../docs/baseline/agent-mha-source-draft-acceptance-r1064.md)。
