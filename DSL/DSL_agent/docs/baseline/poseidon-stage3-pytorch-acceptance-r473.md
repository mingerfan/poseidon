# PyTorch 张量输入与程序包验收

开发：ARM Ubuntu，lhy-agent-dsl。执行后端：Poseidon CPU。当前源码绑定包含 598 个执行文件。

## 本轮改动

- FX 捕获新增 torch.sum、Tensor.sum、torch.mean、Tensor.mean、cat／concat／concatenate 和 stack。支持静态轴、轴元组、keepdim，以及 float64；最终标量输出用 reshape(1) 声明。
- 中型程序包来源审计、分解与独立参考检查沿用模型声明的 resource_profile。
- v5 请求的重复绑定元数据使用有界列式编码，展开后恢复同一逻辑请求和哈希。原 128 KiB 请求／tracing 限制、256 个公开向量及 4096 个 AST 节点限制保留。
- 相同原生 Linear helper 的完整函数体复用，独立输入仍各自调用。原 Linear 融合候选 AST 从 5257 降到 3264。
- 开发批处理将单例布局容量阻塞单列，继续验收其他独立案例。

## 验收结果

最终组件、来源、FX、分解、传输和空间算子测试：179 passed，0 failed，0 skipped。

三个原 PyTorch 模块均在独立沙箱内完成 16 组参考检查，共 48 组。检查原 nn.Module、高层图，以及两套分解图参考。

| 模型 | 输入元素 | 输出元素 | 当前结果 |
|---|---:|---:|---|
| reduction_fusion：cat→sum／mean→stack | 516 | 137 | 真实 tracing、编译、密态执行、比较和程序包重载通过 |
| spatial_residual：深度卷积→平方残差→池化→归约／连接 | 578 | 164 | 捕获与参考通过；公开向量注册超过 256，编译前阻塞 |
| linear_fusion：共享 Linear、多输入、stack／归约／连接 | 260 | 34 | tracing、编译与产物检查通过；执行出现 std::bad_alloc，随后命中原有超时，数值验收失败 |

数值通过的模型比较 4 组输入、548 个输出值，最大绝对误差 1.1847337034742722e-6。程序包重载另比较 548 个值，最大误差 7.096782644233834e-7；重载生成／编译调用为 0。全部保持 1e-5 + 1e-4*abs(reference)。

本轮候选为规则基线，新增真实 Agent 生成和付费调用为 0。三个模型均保留，不将阻塞或失败计为通过。

MLP、RMSNorm 的两份原真实 Agent 源码另按当前绑定重新编译、密态验收及重载通过；模型、权重、源码和 reference 保持不变。初次比较 24 值，最大误差 4.066955250348059e-8；重载 24 值，最大误差 8.066592130373529e-9。这是原 Agent 源码复核，新增生成为 0。

## 原失败与定位

- r459：归约融合 tracing 总载荷超限；空间模型公开向量超限；第三例尚未运行。
- r463：归约融合密态通过，导出仍按 legacy 容量审计而失败；空间模型阻塞；Linear 融合超 AST。
- r470：两项工程问题修复后，归约融合完成程序包验收；空间阻塞保留；Linear 融合暴露执行分配／超时问题。
- 早期系统 Python 测试的 NumPy 缺失保留在 r456/static.log；最终测试采用现有锁定 Nix 环境。

空间布局诊断定位到 17 宽输入的步幅池化路由；卷积本身占 44 个公开向量，新增池化路由使总量超过 256。Linear 内存问题需要进一步核验 RuntimePlan 寄存器及中间密文占用；当前日志不足以直接认定 Runtime bug。

## 后续

1. 保持原空间模型，压缩奇数宽度池化的公开路由。
2. 保持原 Linear 模型，减少布局展开产生的中间密文；若需 Runtime 核心或资源参数修改，单独提出方案。
3. 为新增 PyTorch／容量入口和 Attention 修复反馈冻结真实 Agent 测试。
4. 继续重复 bootstrap 精度、ResNet20 布局及整网近似精度工作。

本轮未安装、下载、调整参数、修改编译器／Runtime 核心、提交或推送。x86 原生执行、空白安装、CUDA继续暂缓。

## 重现

以下命令在已配置的 Ubuntu checkout 执行；输出目录使用新名称：

~~~sh
export POSEIDON_PLATFORM=aarch64-linux
export POSEIDON_WORK_ROOT="$HOME/poseidon-work"

python3 -B scripts/baseline/benchmarks/tools/qualify_poseidon_pytorch.py \
  --manifest scripts/baseline/cases/trusted-pytorch-medium-v1/acceptance.json \
  --output "$POSEIDON_WORK_ROOT/platforms/$POSEIDON_PLATFORM/results/pytorch-medium-next"
~~~

追加 --execute 执行这三个冻结案例。当前结果预期包含一个通过、一个容量阻塞和一个执行失败；此命令属于开发验收工具。

[机器报告](poseidon-stage3-pytorch-acceptance-r473.json) · [三个整模型的真实 Agent 结果](poseidon-whole-models-agent-acceptance-r453.md)。
