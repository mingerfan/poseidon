# PyTorch 输入写法审计与离线工具调整

2026-10-04 · ARM Ubuntu · `lhy-agent-dsl`

本轮复核 24 种常见 PyTorch 写法，并加快离线验收工具的身份检查。应用组件和后端源码保持原样，r675 的 15 项待批准请求绑定仍有效。

## 输入检查结果

13 种通过隔离捕获及 16 组明文参考检查：nn.Linear、F.linear、nn.Conv1d、nn.AvgPool1d、推理 nn.BatchNorm1d、F.layer_norm、F.group_norm、shape 参与 reshape、固定次数公开循环、标量 reshape 后输出、tuple 输出、dict 输出、显式 float32→float64 转换。

以下 8 种写法在接入层受阻，尚未进入编译器：

| 写法 | 当前原因 | Agent 层改进 |
|---|---|---|
| `x @ weight.T + bias` | 未识别 Tensor.T | 按真实 rank 映射 transpose／permute |
| F.conv1d | 缺函数调用映射 | 复用卷积规则，校验公开权重和参数 |
| F.avg_pool1d | 缺函数调用映射 | 复用池化规则，明确 padding、除数和输出 |
| 推理 F.batch_norm | 缺函数调用映射 | 绑定固定统计量，拒绝训练语义 |
| nn.LayerNorm，关闭 affine | 显式要求 affine | 生成数学默认 gain=1、bias=0 |
| nn.GroupNorm，关闭 affine | 显式要求 affine | 复用函数式默认常量规则 |
| `range(x.size(1))` | FX tracing 时 size 仍为 Proxy | 增加按静态签名工作的 Agent tracer |
| 原生标量 Tensor 输出 | 要求先 reshape(1) | 版本化内部标量规范化和输出恢复 |

另有 3 项符合当前契约的拒绝：加密数据分支、strict-float64 下的 float32 状态、默认样本方差。已开放的 variance 是总体方差，参数需为 correction=0 或 unbiased=False。

初次 float32 转换探针的输入 shape 清单有误；仅修正该探针后重测，通过。r680 原记录与 r681 复核记录分别保留。这次修正不涉及已有冻结模型、reference 或 Agent 候选。

这些结果证明特定输入写法的捕获与参考一致性；新增写法的编译、密态和真实 Agent 验收分别安排。

## 工具与组件边界

`check_pytorch_sources.py` 将 worker 检查统一委托给 ContextBackend.identity，减少一次相同的快照检查。每次仍读取当前源码、依赖、平台及环境；发现变化即拒绝。

- 6 项边界测试通过，0 failed／skipped：源码变化、依赖变化、环境变化、不同 worker、无检查后端及正常路径。
- 固定环境各测两次，身份检查中位耗时由 14.27 秒降至 7.17 秒。这是该检查的耗时，尚未测量全批次加速比。
- 单例隔离捕获回归通过。旧工具版本已按原 SHA 保存在原始证据中。

程序包的新输入打包接受了超出 LayerNorm 近似域的有限值；独立数学参考按已声明区间拒绝。该探针未执行密态计算。当前新输入返回也没有声明数值或应用精度认证；仍需在加密执行前增加明确的近似域预检及失败码。

## 后续实施顺序

1. 函数式 Conv／Pool／推理 BatchNorm、无 affine 模块、Tensor.T 写法对齐。
2. 版本化标量输出与静态 shape 循环接入。
3. 逐节点／模块路径近似选择及程序包新输入域预检。
4. 每组追加独立参考、代表性 Poseidon CPU 验收；真实 Agent 测试按新绑定单独授权。

以上优先改动均在 Agent 层。密文分区交接、Runtime 内存生命周期、真实 bootstrap 和 CUDA 实机仍分别保留现有缺口。

## 批次与权限

r620 的 14 项授权已执行完：14/14 通过，16 次生成、16 次 HTTP。r675 的新 15 项未调用 API，等待授权，绑定：

`c089b9c4ce2d49e8c7ed7b8ff23218c91c10d5245779f889f26f6072b109d982`

本轮 API、DSL 编译、SDK 重建、密态执行、新安装及下载均为 0。545 个后端 C++／配置文件、5 个 reference 模块和共享应用源码哈希未变。未提交或推送。

## 复核入口

在 Ubuntu checkout 运行，输出目录采用新名称：

~~~bash
POSEIDON_PLATFORM=aarch64-linux POSEIDON_WORK_ROOT=/home/lhohy/poseidon-work \
python3 -B scripts/baseline/benchmarks/tools/audit_pytorch_input_forms.py \
  --output /home/lhohy/poseidon-work/platforms/aarch64-linux/results/input-forms-next
~~~

`--case NAME` 可只检查指定原案例。明文模型源和清单在 `scripts/baseline/cases/pytorch_form_audit_v1/`。

[机器报告](agent-input-form-audit-r685.json)。原始证据位于 work root 的 `agent-pytorch-input-forms-r680`、`r681`、`agent-pytorch-guard-domain-r682` 和 `agent-pytorch-guard-capture-r684`。
