# PyTorch 函数式输入验收

ARM Ubuntu `/home/lhohy/Code/Poseidon`，分支 `lhy-agent-dsl`。本批扩展 Agent 输入映射，沿用现有分解规则及 Poseidon CPU 后端。

| 输入写法 | 范围 |
|---|---|
| `F.layer_norm` | 静态归一化后缀；公开 gain/bias 或默认单位 gain、零 bias |
| `F.group_norm` | rank 3–4，静态分组；公开 gain/bias 或默认值 |
| `torch.var`、`Tensor.var` | 总体方差，`correction=0` 或 `unbiased=False`；静态轴和 keepdim |

归一化需要显式 rsqrt 近似。模块式 LayerNorm／GroupNorm 的既有 affine 契约保持原样。方差的 correction 使用锁定 Torch 2.0.1 接受的整数类型。

254 项组件、PyTorch、旧入口及程序包测试通过，0 failed/errors/skipped。初次 r668 的两个错误及修复前源码保留：FX 静态列表类型处理，以及浮点 correction 与固定 Torch API 不匹配。

三个算式各三个上下文，共9个模型通过真实 tracing、Dacapo 编译、RuntimePlan／bundle 检查和 Poseidon CPU 数值验收。三份代表程序包通过重载、三个新输入各重复执行两次，以及缓存复用。

| 验收 | 输入组 | 输出值 |
|---|---:|---:|
| 首次密态验收 | 36 | 152 |
| 程序包重载 | 12 | 56 |
| 新输入重复执行 | 18 | 84 |
| 合计 | 66 | 292 |

最大绝对误差 **3.0301137352873937e-7**，门限保持 `1e-5 + 1e-4*abs(reference)`。45组不同输入；重复执行另计。每例另有16组隔离的明文参考检查。

这9份源码来自规则候选。新增真实 Agent 生成、HTTP 和付费调用为0。r620原授权范围已完成14/14，此次未重复。

归一化示例使用domain=[1,5]、一次Newton迭代的显式近似；16组参考探针的原函数近似误差最高 **0.2249618675234475**。密态通过说明执行了该近似计算，原模型应用精度尚未认证。方差为总体方差，独立原函数参考误差在浮点微差范围内。

20项既有模型／请求哈希一致：14项保留模型和6项重新隔离捕获的外部状态模型。新的源码、来源和验收绑定单独保存。r635未执行提案保留原文件及绑定，后续采用[r675新提案](agent-pytorch-functional-live-proposal-r675.md)。

545个后端C++／构建配置文件及5个参考／门限模块核验未改。安装、下载、SDK重建、参数变更、commit、push、PR均为0。x86配置保留，本批实际执行平台为ARM64。

开发批次入口：

~~~bash
POSEIDON_PLATFORM=aarch64-linux POSEIDON_WORK_ROOT="$HOME/poseidon-work" \
python3 -B scripts/baseline/benchmarks/tools/check_pytorch_sources.py \
  --index scripts/baseline/cases/trusted_pytorch_functional_v1/cases.json \
  --trust-local-code --output "$HOME/poseidon-work/platforms/aarch64-linux/results/functional-next"
~~~

`--capture-only` 完成输入捕获和独立明文检查。完整模式复用Application，提交规则候选并验收；入口独立于历史付费提案。输出使用新目录，源码或依赖变化建立新绑定。

原始证据位于 `/home/lhohy/poseidon-work/platforms/aarch64-linux/results/agent-functional-offline-r673`，及同级 `agent-pytorch-compatibility-r671.log`、`agent-request-compatibility-r674.json`、`agent-state-capture-r674`。

[机器报告](agent-pytorch-functional-acceptance-r678.json)。后续推进常见模块与函数写法对齐、近似选择及应用精度、资源估计与更大组合；新功能真实Agent测试使用单独批准的冻结范围。bootstrap、密文分区和CUDA状态保持独立。
