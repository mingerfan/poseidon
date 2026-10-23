# PyTorch v10 接入验收

2026-10-06，ARM Ubuntu `/home/lhohy/Code/Poseidon`，`lhy-agent-dsl`，HEAD `7588e48093631eaf8385506e3db058e7ca3cfe7e`。

## 已接入

可信PyTorch新增显式v10来源、参考和来源包格式，保留v1–v9的规则与默认值。补齐AdaptiveAvgPool、InstanceNorm及padding的结果形状查询，并将函数式AdaptiveAvgPool接入现有算子分解。模型、算子数学、公开权重、资源上限、reference和后端保持。

生产改动五个文件：`pytorch_static_trace.py`、`pytorch_model_capture.py`、`pytorch_model_intake.py`、`application_capabilities.py`、`application-source-catalog-v1.json`。新增一个测试模块。原文件完整备份在本批工作目录；原有修改保留。

## 检查结果

| 项目 | 结果 |
|---|---:|
| 实际CLI捕获，39份写法、12个数学模型组 | 39/39 |
| 模块与独立明文参考 | 624组 |
| 等价声明式图对照 | 624组 |
| 明文最大绝对差异 | 1.1102230246251565e-16 |
| 实际Poseidon适配器应用提交 | 39/39，排队后取消 |
| 来源、策略、版本或模型绑定篡改 | 195次拒绝 |
| 旧v9兼容成功来源 | 6/6 |
| 旧v9原拒绝行为 | 18/18保持 |
| v10及旧v8公开拒绝检查 | 7/7 |
| 组件与旧输入回归 | 183项独立测试通过；首轮182/183，单项修正后1/1复验 |

测试首轮的一项失败是测试夹具漏设静态捕获的输出结构前置参数。原失败日志与原测试源码保留，修正后只复验该项；两轮间未修改生产逻辑。错误、skip及剩余测试失败为0。明文比较使用现有 `atol=rtol=1e-12`。

本次应用仅执行提交、状态查询和取消，未生成或验收DSL程序包。输入捕获与Hecate DSL tracing分开计数。当前新增范围尚未获得Poseidon编译、密态、解密或真实Agent生成的通过结论。

## 后端范围与身份

新离线范围已准备112项：原103项全部保留，新增九份v10来源；15份成功程序包另查重载、新输入重复执行和缓存复用。原103项模型、公开权重、reference图、请求和规则源码逐项保持，46份受实现身份变化影响的PyTorch参考在新目录重新捕获。旧八项编译失败和原记录保留。

684份源码、3171份案例/证据及四份独立驱动已冻结。648份后端文件、6份独立reference文件和三组历史记录均匹配。编译SDK身份保持，无重建。

旧r976提案及四驱动保持原文件，但其源码身份已过期。新r991绑定为 `32ade2c4bb66fc021f283e1264e79dad1fb12f68fb16e971bb99afb41ed3025d`。本批未创建授权收据或后端执行目录，未授权启动检查按预期拒绝。

## 执行范围

本批Hecate tracing、编译、密态执行、解密、新Agent生成、付费调用、安装、下载、SDK重建、后端/reference/参数修改、commit和push均为0。冻结数值门限保持 `1e-5 + 1e-4*abs(reference)`。分支及x86配置保持；新增功能的x86实际执行仍待验证。

本次只扩展输入适配。物理容量、bootstrap精度与接线、密文交接/分区、应用精度、ResNet20、CUDA实机和空白安装继续单列。现有1,200模型Benchmark及历史结果保持。

## 使用与下一步

来源清单选择 `"format": "poseidon-trusted-pytorch-v10"`，通过既有 `prepare_model.py` 捕获，再用 `app.submit_import(directory)` 提交。[输入说明](operator-decomposition-and-python.md#pytorch-v10-算子结果与形状查询)。

下一步取得[r991离线范围](agent-operator-shape-offline-proposal-r991.md)的批准，再运行冻结入口。真实Agent新功能评测在离线验收后另行冻结、申请批准，不沿用历史付费额度。

## 记录

本批工作目录：`/home/lhohy/poseidon-work/platforms/aarch64-linux/results/agent-pytorch-operator-shape-integration-r985`。

- `tests-r986.json/log`：首轮183项。
- `source-test-r986.py`、`tests-r987.json/log`：测试前置条件修正与单项复验。
- `agent-operator-shape-public-v10-r988/report.json`：实际CLI来源与独立参考。
- `agent-operator-shape-application-v10-r989/report.json`：应用提交、篡改拒绝和旧接口。
- `prepared-backend-r990.json`、`preflight-r993.json`：112项准备、SDK及冻结核验。

两个 `agent-operator-shape-*` 路径位于同一results根目录。机器记录为 `agent-pytorch-operator-shape-v10-acceptance-r994.json`。
