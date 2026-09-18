# 较大输入与batch的Conv/average-pool

本轮在schema5 / periodic-packed-v1上增加Conv1D/2D和average-pool，不改变旧四元素接口、
安全参数、默认编译配置或数值门限。后端仍是Dacapo自带的SEAL CPU，不是Poseidon GPU。

## 语义与边界

- Conv1D支持`[C,L]`或`[N,C,L]`，Conv2D支持`[C,H,W]`或`[N,C,H,W]`。
- 整个输入最多256元素；Conv/Pool整个输出最多16个scalar-neuron密文，包括所有batch和通道。
- 显式整数kernel/stride/zero-padding/dilation/groups；Conv是PyTorch的cross-correlation，
  不翻转kernel。支持分组、depthwise和depthwise multiplier。
- Average-pool使用floor输出几何，支持`count_include_pad=True/False`；边界窗口的除数
  分别包含padding或只计有效样本。禁止ceil_mode、自定义divisor、非零padding模式。
- 不引入ReLU近似、max-pool、动态shape、Conv3D、额外bootstrap或安全参数变化。
- 原数据flow仍有明确shape、packing与算子预算限制；这不是任意模型的完整支持声明。

## 实现位置

- `spatial_ops.py`：原有窗口与矩阵降级实现，新增显式可选资源上限；默认仍是4输入/4输出。
- `packed_spatial.py`：新接口的batch/channel几何，按batch生成互不交叉的块对角权重。
  reference逐个原始样本直接遍历卷积窗口，不读取lowering矩阵。
- `packed_model.py`：检查JSON图的spatial字段、权重shape、分组和输出预算。
- `model_graph.py`：原始PyTorch functional模型和独立窗口reference。
- `fx_to_hecate.py`：将空间运算降级到现有密态Linear/rotation/add路径。
  显式新接口也接受嵌套`nn.Conv1d/2d`、`nn.AvgPool1d/2d`模块；模块与functional
  路径已经通过相同源码/常量/layout的离线对照，并不冒称额外的真实Agent案例。
- `test_packed_spatial.py`、`test_packed_spatial_evidence.py`：语义及密态产物审计。

仍未开放Agent直接调用上游`poly.HE_Conv/HE_Pool`等高层helper；这里生成的是已验证的
Hecate基础密文表达式。算子支持与某个同名高层helper的支持不是一回事。

## 确定性密态证据

基础14模型：`packed-spatial-goldens-hcvi8_gd/report.json`。
SHA256：`1eaa22ae87e00d406c0cb5e270cecdf289b6ac6c0035e6a9a68c8a56570bf8e7`。
14/14正确程序通过；另2个合法语法、错误语义的程序被数值比较拒绝，分别误用Conv分组输入
和错误地复用另一pool窗口。正确程序448个输出值，最大绝对误差`3.8523333134588e-10`。

256输入/16输出边界：`packed-spatial-max-goldens-9nu5sl9n/report.json`。
SHA256：`84e298173a10c16558867ad18c335f290cab858fed601e4cb28ee2d0da0594c0`。
两例分别为`[2,1,8,16]` batched Conv2D和`[2,8,16]` depthwise Conv2D。
2/2通过，128个输出值，最大绝对误差`1.0660744509394249e-10`。

以上路径均位于`/home/lhy/poseidon-work/results/`。共16个正确模型、64组密态输入、
576个输出值；它们是确定性规则程序，不是Agent。基线临时密钥已自动清理，报告、模型和产物保留。

```bash
cd '/mnt/d/Code Space/Poseidon'
timeout -k 5s 3650s python3 scripts/baseline/run_packed_model_goldens.py --spatial
timeout -k 5s 1250s python3 scripts/baseline/run_packed_model_goldens.py --spatial-max
```

## 真实Agent入口

固定16例清单为`cases/packed-spatial-16-manifest.json`；原14例清单也保留，不回写旧实验。
包含Conv/grouped/depthwise/dilated/batched、带padding的两种pool除数、小型
Conv—square—pool—Linear、双卷积分支残差以及零卷积权重行。

下列入口会付费，仅在目标内授权后运行；凭据统一在本地`.env`加载。

```bash
timeout -k 10s 11000s python3 scripts/baseline/run_agent_batch.py \
  --deepseek --case-manifest scripts/baseline/cases/packed-spatial-16-manifest.json \
  --provider deepseek --model deepseek-flash --reasoning-effort high \
  --jobs 10 --api-timeout 1200 --max-tokens 384000 --provider-retries 3 \
  --stream --loopback-proxy-port 6478 --compiler-configuration seal-cpu-eva-w45-v1
```

每例最多3轮修复、每次生成最多3次网络重试，即最多64次生成/256次HTTP尝试。
只发送公开合成模型、常量、layout、DSL规则和受限诊断；不发送测试输入、reference、密钥或规则答案。

## 真实Agent结果

报告：`/home/lhy/poseidon-work/results/agent-batch-x47lnr8p/report.json`。
SHA256：`ed8515adc7a7f16be623460a7b304ba8c9dce178ead220881dd59b80ff95aeba`。

- 16/16首轮通过parse/check/compile/execute/数值比较；0修复、0网络重试。
- 64组真实密态输入、576个输出值；最大绝对误差`5.33774524580366e-10`，
  按输出元素数加权MAE `3.735481081420492e-11`。
- 数值门限保持`1e-5 + 1e-4*abs(reference)`；没有降低安全参数或改写模型语义。
- 16次DeepSeek `deepseek-flash/high`调用，输入83137、输出167993、总计251130 tokens。
  每次都有usage回执；未读取或估算人民币账单金额。
- Agent自行生成Hecate基础密文表达式，没有复用规则程序答案；这些结果不证明所有输入上的等价性。

| 案例 | 输入shape | 最大绝对误差 |
|---|---|---|
| Conv1D | `[1,17]` | `7.42e-11` |
| grouped Conv1D | `[2,9]` | `9.15e-11` |
| dilated Conv1D | `[1,17]` | `3.28e-11` |
| dilated Conv2D | `[1,5,5]` | `4.60e-11` |
| depthwise multiplier | `[2,4,4]` | `8.68e-11` |
| batched Conv1D | `[2,1,9]` | `5.48e-11` |
| batched Conv2D | `[2,1,4,4]` | `8.63e-11` |
| AvgPool1D包含padding | `[1,9]` | `1.37e-10` |
| AvgPool1D排除padding | `[1,9]` | `2.50e-10` |
| batched AvgPool2D包含padding | `[2,1,4,4]` | `6.45e-11` |
| Conv—square—pool—Linear | `[1,5,5]` | `5.34e-10` |
| 双卷积分支残差 | `[1,4,4]` | `1.03e-10` |
| 零卷积权重行 | `[1,7]` | `2.54e-10` |
| batched AvgPool2D排除padding | `[2,1,4,4]` | `1.85e-10` |
| 256元素batched Conv2D | `[2,1,8,16]` | `9.02e-11` |
| 256元素depthwise Conv2D | `[2,8,16]` | `1.45e-10` |

独立只读审计：`agent-lineage-audit-lr3sx5l2/report.json`，16/16覆盖、0次新API/密态执行、
0个遗留密钥目录。重新读取原始响应、固定模型/reference、编译产物和解密结果，
仍标记`poseidon_gpu_validated=false`、`all_goal_requirements_complete=false`。

```bash
timeout -k 5s 230s python3 scripts/baseline/audit_agent_lineage.py \
  /home/lhy/poseidon-work/results/agent-batch-x47lnr8p/report.json
```

本批自动清理`9432319019`字节临时密钥；报告、原始响应、源码、IR/HEVM/CST及数值证据保留。
加上两个规则基线，共清理`19936325410`字节，约18.57 GiB。原随机密钥不可恢复，
但可重跑实验生成新密钥；未删除用户源码、依赖或历史实验报告。

网络探针`deepseek-tls-probe-hk64bb7p`：默认路由3/3连接超时，6478 CONNECT代理3/3
通过TLS证书校验并得到无凭据请求预期的401。真实批次显式使用6478且没有传输失败。
此结果不保证未来网络永远稳定；超时和重试仍保留硬上限。

Windows Git的`submodule status`只读检查因其shell工具路径缺少`basename/sed/git-sh-setup`失败；
改用已有WSL Git检查成功，Dacapo仍是固定commit。未因此修改系统环境或安装软件。

## 回归与本地交付

164项回归全部通过、0跳过，包含本轮16例真实Agent的只读证据重算及历史native-function、
chunked/packed/tensor、空间算子和零密文证据。15个涉及文件的Python/JSON语法及定向空白检查通过。
回归没有新增付费调用；临时密钥清理后WSL文件系统约190 GiB可用。

主仓库仍为`feat/agent-dsl-correctness`，HEAD `4995e7cadedf2bfb9104658b5638662ecf6a1d0a`。
现有C++/CMake/mGPU和Dacapo修改保留；本轮只改本地baseline代码、测试、清单和本文档。
未commit、push或创建PR。将来获准提交时，建议按“输入/FX语义扩展”“确定性与付费证据回归”
“覆盖清单与文档”拆分检查，不把此前无关修改一并提交。

下一门禁仍是剩余DSL构造及其与新packing接口的组合验证、高层helper和真实bootstrap语义，
以及独立的Poseidon GPU编译产物执行对接。当前结果不将这些未完成项标记为通过。
