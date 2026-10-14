# PyTorch 组合验收 r864

本批补充 8 组、24 个小模型上下文，未重建原 1,200 模型 Benchmark。21 个 PyTorch 来源完成真实捕获；3 个 torch.split 来源保留捕获失败。规则候选及人工等价 JSON 图合计 16 个通过 Poseidon CPU，8 个停在编译层。新增真实 Agent 生成和付费调用为 0。

## 范围

每组包含三个形状及 Tensor／tuple／dict 输出上下文，公开状态以 87 个 NPY 数组绑定。输入总元素最多 24，输出最多 12，原图最多 15 节点。忽略名称与权重值后有 24 个图／形状／参数签名、18 个拓扑组。Attention 核心涵盖点积、声明近似 Softmax 和值融合；整层投影、RoPE、因果掩码及 KV 缓存不属于本批。

| 组合 | PyTorch 捕获 | 编译、密态与数值 |
|---|---:|---:|
| Attention 核心：点积→近似 Softmax→值融合 | 3/3 | 0/3 |
| 双输入广播→InstanceNorm→平方→全局池化→Linear | 3/3 | 0/3 |
| Conv→GroupNorm→AvgPool→Linear | 3/3 | 3/3 |
| 公开索引→批量矩阵乘→平方与残差 | 3/3 | 3/3 |
| 1–3 层平方 MLP→残差→LayerNorm→Linear | 3/3 | 1/3 |
| 公开批量矩阵融合→转置→总体方差 | 3/3 | 3/3 |
| Linear 残差→LayerNorm→Linear | 3/3 | 3/3 |
| 静态 split→分支→stack→permute→Linear | 0/3 | 等价 JSON 3/3 |

21 个捕获来源及 3 个手工等价图共通过 384 组独立数学／PyTorch／高层图参考。手工 split 图保留原权重和原源码，另有 48 组原模块差分。21 个来源重复分解一致，33 次输入干预影响输出。

## 真实后端结果

13 个 PyTorch 规则程序、3 个人工等价 JSON 规则程序通过真实 Hecate tracing、Earth／CKKS、RuntimePlan／OperatorSpec／常量 bundle、Poseidon CPU、解密和逐项比较。AST 检查 24 项通过；来源身份拒绝检查 42 项通过。数值统计仅计密态通过及程序包复用结果：84 组、478 值，maximum absolute error 1.5275907316e-7，门限保持 1e-5 + 1e-4*abs(reference)。失败、未运行和 skipped 均未计通过；当前计划无未运行或 skipped。

cnn_norm_head_1 和 indexed_batched_matmul_1 完成程序包重载、三组新输入各执行两次及缓存复用。687 个包文件和 655 个编译产物哈希核验。执行身份为 Poseidon CPU、N=32768、Q13/P1、tc128；执行器无私钥。

## 保留的缺口

- torch.split 三项缺少 FX 映射。等价 JSON split 图通过说明这三个静态配置已有后端路径；Python 来源仍记失败。下一步在新源码绑定下补静态 sections、axis 与 tuple 结果映射。
- attention_core_0–2、channel_broadcast_mix_0–2、norm_depth_branches_1–2 共八项当前规则表达超过固定编译预算。第一组累计 scale 780，广播组合为 750，其余以机器报告逐项记录，预算均为 720。这里的 level 为 Earth 已消耗层数。
- attention_core_0、channel_broadcast_mix_0、norm_depth_branches_1 改用现有 period-32 后仍在相同编译约束处失败，模型、权重、编译配置与安全参数保持。三项新绑定保留，原失败记录不变。
- 优先检查 Agent 层公开乘法合并、合法浅表达和布局指导。当前拒绝只证明这些表达失败，尚不能证明所有等价表达都无法编译，也不作为编译器 bug 或必须 bootstrap 的结论。

声明近似相对原函数在本批固定探针上的最大误差为 0.1874585152；其与 CKKS 执行误差分开。原模型应用精度未认证。本批没有 bootstrap、CUDA、x86 原生或空白安装验收。

## 使用

来源、清单与权重位于 [组合模型](../../scripts/baseline/cases/trusted-pytorch-composition-v1/)；[生成器](../../scripts/baseline/benchmarks/tools/generate_composition_sources.py)可复现 137 个文件。

~~~bash
results="$POSEIDON_WORK_ROOT/platforms/$POSEIDON_PLATFORM/results"
python3 -B scripts/prepare_model.py \
  --trusted-pytorch-manifest scripts/baseline/cases/trusted-pytorch-composition-v1/cnn_norm_head_1/manifest.json \
  --trust-local-code --resource-profile general-tensors-v1 --write \
  --output "$results/imported-composition"
~~~

导入后沿用 app.submit_import、app.run 和程序包接口；当前 v7 图选择 v9 公共布局。生成评测按独立批准的冻结请求执行。

## 绑定与权限

本批绑定 7aef9760e92dfa3b3dfdc8a38202cbcd1ce2a4a5c483a658524eacfef6272793；period-32 对照绑定 847ac8dc0ab76c46307094b966c6a6572b7bfa0ed7d63eebca0d613cae1e0008。671 Agent 源文件、648 后端文件、6 reference 文件及 10 份历史／提案记录保持。r848 的 671 源文件与 220 案例文件未改变，仍待对应付费批准。本次 r690 回复不用于扩展 r848 授权；r690 对应续验已有 r695 完成记录，24/24、复用 18 项、续验 6 项，本批重跑 0。

开发位于 ARM Ubuntu /home/lhohy/Code/Poseidon、lhy-agent-dsl。安装、下载、SDK 重建、后端／参数／reference／门限修改、付费调用及 Git 发布均 0。编译≤2／链接≤1／密态串行；本批截至最终快照约 31.1 分钟；新增产物 188.7 MiB，全局 24.6 GiB，磁盘余量 45.44 GiB。全部在原一小时／4 GiB 新增／32 GiB 全局边界内。

[机器报告](agent-pytorch-composition-acceptance-r864.json) · [原续验](agent-pytorch-spelling-acceptance-r695.md) · [当前付费提案](agent-pytorch-general-input-live-proposal-r848.md)。
