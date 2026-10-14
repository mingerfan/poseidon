# PyTorch 有序操作与常见调用组合

此目录是 Agent 输入层的独立开发草案。活动入口继续使用 V12；
草案增加 V13，组合已登记的常见调用和有序张量操作。

## 支持范围

- `torch.add/subtract` 与 Tensor 方法的公开 `alpha` 参数。
- 推理模式 Dropout，保留别名和原地更新的先后顺序。
- 静态 scaled dot product attention，支持公开布尔、有限加性和因果掩码。
- 与 view、transpose、clone、静态索引、归约及残差组合。
- 原始 NPY 权重绑定、派生常量校验、命名输出和来源包重载。

Attention 使用逐节点声明的 exp／reciprocal 近似配置。捕获、分解和原函数精度分别验收。

## 文件

| 文件 | 用途 |
|---|---|
| `targets/` | 待接入的 Agent 模块 |
| `originals/` | 现有模块的恢复副本 |
| `plan.json` | 原文件及目标文件哈希、验收阶段 |
| `common_models.py` | 固定源码模型与工厂 |
| `test_composition.py` | 有序操作、独立参考和常量篡改检查 |
| `test_source_package.py` | 真实源码 worker、来源包与旧版本拒绝检查 |
| `run_isolated.py` | 使用现有锁定 SDK 的只读隔离验收入口 |

## 验收入口

在 Ubuntu checkout 中执行，结果名应为一个新的 `agent-ordered-common-tests-r数字`：

~~~sh
python3 -B scripts/baseline/benchmarks/tools/staged_ordered_common_v1/run_isolated.py agent-ordered-common-tests-r1250 reference
python3 -B scripts/baseline/benchmarks/tools/staged_ordered_common_v1/run_isolated.py agent-ordered-common-tests-r1251 source
python3 -B scripts/baseline/benchmarks/tools/staged_ordered_common_v1/run_isolated.py agent-ordered-common-tests-r1252 retained
python3 -B scripts/baseline/benchmarks/tools/staged_ordered_common_v1/run_isolated.py agent-ordered-common-tests-r1253 component
~~~

每组保持网络隔离、清空环境和资源限制。使用现有 Python 3.10.14、
NumPy 1.25.2、Torch 2.0.1 ARM CPU 环境；输出保存到平台 results。

## 接入顺序

来源及组件验收 → 安全切换边界 → 代表性 Poseidon 编译／密态回归 → 新功能真实 Agent 验收。

当前生产文件保持 r1232 提案中的冻结身份。详细阶段结果见
`docs/baseline/agent-ordered-common-progress-r1244.md`。

## Poseidon 联合验收

runtime_acceptance.py 为草案来源提供只读叠加的真实后端验收入口，使用同一 Application 和 Poseidon 验证服务。
三例范围：alpha、dropout、boolean。每例依次检查数值资格、新输入执行和缓存复用。

r1261 已完成三例来源捕获与reference校验，在沙箱预检的共享锁目录处失败；后端执行尚待续验。
29项控制测试通过。r1263追加30分钟的无收费提案待批准，生产源码保持V12绑定。
启动器使用现有 Nix SDK、参数FD与有限只读挂载页；可信主机仅写共享资源锁和本批产物。

~~~sh
POSEIDON_PLATFORM=aarch64-linux POSEIDON_WORK_ROOT=/home/lhohy/poseidon-work python3 -B scripts/baseline/benchmarks/tools/staged_ordered_common_v1/runtime_acceptance.py --plan
~~~
