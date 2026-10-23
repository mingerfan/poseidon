# 源码准备接口与 Poseidon 验收

新增 scripts/agent_calibration.py 和 SourcePreparation 应用适配器。源码校准和近似配置冻结可在生成 DSL 前执行；成功后返回配置清单，现有 Application.submit_source 接收该清单。

## 命令行

在配置好的 Ubuntu 项目中执行：

~~~sh
python3 scripts/agent_calibration.py inspect \
  --request /absolute/path/inspect.json \
  --output /absolute/work/platforms/aarch64-linux/results/inspect \
  --trust-local-code

python3 scripts/agent_calibration.py observe \
  --request /absolute/path/observe.json \
  --output /absolute/work/platforms/aarch64-linux/results/observe \
  --trust-local-code

python3 scripts/agent_calibration.py freeze \
  --request /absolute/path/freeze.json \
  --output /absolute/work/platforms/aarch64-linux/results/freeze \
  --trust-local-code
~~~

普通 Python 入口自动使用现有锁定 Nix／venv；--timeout 为本次命令期限，默认 240 秒，可设置 1–300 秒。平台和工作目录沿用 POSEIDON_PLATFORM／POSEIDON_WORK_ROOT。每个动作显式调用，使用新的输出目录。

inspect 请求：

~~~json
{
  "format": "poseidon-source-preparation-request-v1",
  "action": "inspect",
  "manifest": {
    "path": "/absolute/path/manifest.json",
    "sha256": "<manifest hash>"
  },
  "source_signature": {
    "format": "poseidon-source-calibration-signature-v1",
    "outputs": [{"name": "out", "shape": [2]}]
  },
  "samples": [{"x": [0.25, -0.25]}],
  "input_dtypes": {"x": "float64"},
  "output_structure": {"kind": "tensor", "name": "out"},
  "resource_profile": "general-tensors-v1"
}
~~~

observe 在此基础上使用 action=observe，并增加[绑定图和节点的 selection](agent-fx-calibration-r1633.md)。freeze 保留 format、action、manifest、source_signature，另提供 calibration_evidence 和 configuration，格式见[配置冻结](agent-approximation-freeze-r1642.md)。

CLI 返回状态、绑定、图节点清单或统计；原输入数组、源码、reference 答案不进入命令输出。详细记录保存在调用者指定的工作目录。

## 应用调用

~~~python
from source_preparation_service import SourcePreparation

prepared = SourcePreparation().run(
    request, output_directory, trust_local_code=True
)
# freeze 成功后：
manifest_path = prepared["configured_manifest"]
~~~

状态包括 described、observed、frozen、unsupported、validation_failed、service_error、timed_out。适配器只执行指定动作；生成、预算、取消与程序包仍由现有应用控制层管理。

## 本批结果

| 项目 | 结果 |
|---|---|
| 单元／接口／兼容检查 | 119 passed，0 failed／error／skip |
| 普通 Python → 锁定环境 | 实际 inspect 通过 |
| 完整接口链路 | GELU 仿射、平方后 Sigmoid、RMS 分母 rsqrt，3/3 |
| 实际编译与密态调用 | 3 次编译、9 次 Poseidon 密态执行 |
| 初次四组输入验收 | 12 组、48 值；最大误差 2.985e-9 |
| 新输入和重复执行 | 12 组、48 值；最大误差 3.723e-9 |
| 包导出／重载／复用 | 3/3 |
| 付费 API | 0 |

候选为规则基线，不计为真实 Agent 新生成。应用实际采用了被冻结的配置，数值门限保持 1e-5＋1e-4×abs(reference)。重复执行分别与同一独立 reference 比较，允许 CKKS 随机误差。

初始运行器多余导入、审计逐位比较及输入数组格式问题单列保留；审计修正复用了已完成产物，无新增执行。未修改后端、参数、reference、门限、网络或 x86 配置，安装、下载、提交及推送为 0。

[机器报告](agent-source-preparation-component-r1644.json)。下一步冻结真实 Agent 生成案例，扩展模块／复合 helper 位置映射，并分别检查原函数精度。容量、通用算子、Bootstrap、CUDA 和整模型目标继续推进。
