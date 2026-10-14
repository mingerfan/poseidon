# 静态广播规范化草稿 v2

独立检验 Tensor.expand_as 和 Tensor.broadcast_to 转为已有 expand 的静态语义。补充 FX immutable_list、列表与关键字形状、沿已有维度的 -1；检查原图、规范化图和独立 NumPy 参考、dtype、stride、共享存储。保持 v1 工具和结果不变。

生产入口尚未接入，当前仅已保存草稿，明文、编译、密态和真实 Agent 验收未运行。当前原样回放批次结束后，在既有 Poseidon Nix 环境运行 python3 -B scripts/baseline/benchmarks/tools/static_broadcast_source_v2/verify.py 。
