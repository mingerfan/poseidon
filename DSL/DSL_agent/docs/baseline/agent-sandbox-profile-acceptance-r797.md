# Poseidon 沙箱清单验收 r797

Poseidon tracing／执行入口已使用独立文件清单，进程资源限制提取为公共接口，并补齐清单身份检查。应用、CLI、请求和程序包接口保持。

5份未修改的历史Agent DSL完成真实编译、Poseidon CPU密态执行、解密、逐项比较及程序包重载；每包另3组新输入重复执行两次和缓存复用。含Linear CLI，共 **74组输入、344个输出值**，最大绝对误差 **9.5405380318e-8**。

本批是历史真实Agent源码离线重放。新生成、API和HTTP为0；五例不增加唯一模型数，也不代表全部DSL语义的新版本验收。

## 实现

| 文件 | 调整 |
|---|---|
| `process_resource_limits.py` | 公共进程限制，函数AST与修改前一致 |
| `poseidon_sandbox_policy.py` | Poseidon专用83项只读挂载表 |
| `sandbox_policy.py` | 历史清单保留原项目，增加公共限制模块；原限制入口为兼容别名 |
| `poseidon_candidate_sandbox.py` | 改用Poseidon清单和公共限制 |
| `application_identity.py` | 两个清单的唯一声明、字面tuple、非空、相对路径及去重检查；全部挂载文件必需读取和哈希 |
| `candidate_bundle.py` | 当前实施身份绑定Poseidon清单与公共限制 |
| `test_poseidon_sandbox_policy.py` | 实际子进程限制、命令隔离、导入、清单完整性和漂移回归 |

当前清单移除人工工具`upstream_helper_probe.py`和旧清单挂载。旧人工工具、`candidate_sandbox.py`及`candidate_worker.py`逐字保留。

地址空间4GiB、CPU150／155秒、单文件16MiB、描述符128、core0保持。实际bubblewrap预检确认网络／PID隔离、清空环境、请求只读及私有文件不可见。身份分析以模块标识判别清单，避免判别字符串制造旧清单依赖；非法清单、缺失文件和间接路径仍拒绝。

## 静态与兼容测试

316项：**313 passed、0 failed、0 errors、3 skipped**。新增清单9项通过，同时运行组件、Provider、旧接口、身份、native与helper回归。

三项跳过为旧SEAL条件证据测试，没有当前所需历史证据，未计通过：

- `CompilerConfigurationEvidenceTests.test_new_requests_real_trace_compile_execute_and_old_failure_preserved`
- `CompilerConfigurationGuardEvidenceTests.test_worker_rejected_before_keys_or_hevm_execution`
- `CompilerLegacyCompatibilityEvidenceTests.test_legacy_request_unchanged_and_explicit40_has_distinct_identity`

## 真实Poseidon CPU验收

| 原案例 | 数学模型／构造 | 结果 |
|---|---|---|
| `helper-v2` | Linear→上游BatchNorm | 编译、密态、数值、重载及新输入通过 |
| `helper-v3` | 双输入上游Concat→Linear，含rotation | 同上 |
| `helper-v10` | 上游Chebyshev Poly_Leaf2及输入分支相加 | 同上 |
| `unified-matrix` | native函数返回二维Expr对象数组 | 同上 |
| `unified-loop` | 公开循环及增强赋值 | 同上 |

每例16组独立数学解释器／CPU float64 Torch参考，共80组。当前qualification及重载各用原四组输入，再用三组新输入各执行两次。新输入由独立审计比较；本地执行接口自身不宣称新输入已通过独立reference验收。

三项helper保留真实frontend调用、返回依赖及有限干预证据。native数组、函数和增强赋值有真实tracing事件。没有替代上游helper。

Earth／CKKS IR、RuntimePlan、OperatorSpec、bundle和产物哈希核验通过。沿用原请求w45／w40，`tc128`、N=32768、Q13／P1；执行器没有私钥，本批没有bootstrap或中间解密再加密。

Linear CLI原响应解析错误及故意数值错误被拒绝，最终正确候选通过。故意失败的4组／8值单列，未加入74组通过数。

门限保持`abs(actual-reference) <= 1e-5 + 1e-4*abs(reference)`。这是有限输入数值验证，没有原函数应用准确率或形式化等价验收。

## 绑定与资源

新绑定`450f2a57daab208fb652ad3169d17ea1c1f8ee9afe004ee908d20390827bfdd1`。666个Agent／配置源码、137个历史案例文件、648个后端文件、6个reference／门限／检查器及8个既有报告／工具核验一致。原模型、权重、候选及reference保持。130个程序包文件、132个编译产物哈希通过。原始生成响应及历史绑定留在原目录。

worker均终态；新增约74.05MiB、记录worker235.64秒。全局约22.76GiB、磁盘余量约45.98GiB。原1386个修改／未跟踪路径均保留，Git diff check通过。

安装、下载、SDK重建、后端／参数／reference变更及Git发布0。r791的审计键查询错误和r795的Git status格式错误已保留；检查修正后通过，没有重复密态任务。

## 依赖边界及下一步

实际Poseidon沙箱闭包550个文件，无旧清单、人工helper探针或Provider。`seal_artifact_gate.py`公共检查函数和部分历史配置模块仍为兼容依赖，没有SEAL执行选项或回退。

应用保守闭包仍为生成659／其他657、共享657。剩余路径是`trusted_pytorch_import`仅为限制函数导入旧`candidate_sandbox`，间接引入历史工具。下一批使用公共限制入口，并验收真实隔离PyTorch捕获和程序包兼容；当前沙箱边界不等于全部应用阶段独立。

新输入真实Agent、原函数应用精度、深Attention、bootstrap、密文分区、CUDA、x86原生／空白安装和ResNet20继续分别待验收。

## 证据

- [机器报告](agent-sandbox-profile-acceptance-r797.json)
- 预检／测试／审计：`/home/lhohy/poseidon-work/platforms/aarch64-linux/results/agent-sandbox-profile-preflight-r790`
- 真实qualification／程序包：`/home/lhohy/poseidon-work/platforms/aarch64-linux/results/agent-sandbox-profile-backend-r793`
- [应用接口](application-agent-component-v1.md#poseidon-沙箱清单)
