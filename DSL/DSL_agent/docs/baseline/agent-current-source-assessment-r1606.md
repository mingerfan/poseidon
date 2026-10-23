# 原始 PyTorch 精度与程序包复用验收

ARM Ubuntu /home/lhohy/Code/Poseidon，lhy-agent-dsl，2026-10-14。

本批复用已有 ApplicationAssessment 和 agent_assessment.py，在三份真实 Agent 程序包上检查原始 PyTorch reference、Poseidon CPU 执行和数据集重载。前一份进度中提出另建 application.evaluate 接口，是检索范围不完整造成的判断；现有接口已具备这些职责。两份指标草稿继续仅作为开发工具。

## 修正内容

Top-1 下降恰好一个百分点时，96/100 与 95/100 的浮点相减曾得到 0.010000000000000009，导致边界误拒绝。现在使用正确预测数之差／总数的精确分数，与显式 max_drop 比较。单片及跨片汇总使用相同规则；max_drop=0.01、返回字段、CKKS 门限和 reference 保持。

修改范围为 task_numerical_assessment.py、task_dataset_assessment.py 及相关测试。生成、编译、执行和验证的 SDK 身份保持；精度评估使用新的独立身份，旧回执保留原绑定。

## 验收结果

| 检查 | 结果 |
|---|---|
| 受影响控制测试 | 69 个不同测试通过；失败、错误、skip 为 0 |
| 去重 | 移除 62 个重复 unittest ID；不同测试保持 |
| 当前三份程序包 | 3/3，通过；6 组新输入，16 个输出值 |
| 新增调用 | 3 次真实 Poseidon CPU，3 个原生参考 worker |
| 生成与编译 | 0 次 API，0 次重新编译 |
| 重载与复用 | 终态读取、控制对象重建和再次推进保持同一回执；原任务及包字节保持 |

| 程序 | 原始 dtype | CKKS 最大绝对误差 | 对原始源码最大绝对误差 |
|---|---|---:|---:|
| 矩阵 contraction | float32 | 1.26546995105e-9 | 1.38311733681e-9 |
| 负轴 contraction | float64 | 1.45641559030e-8 | 1.45641559004e-8 |
| 标量中间值 | float64 | 3.50545576009e-8 | 3.50545576009e-8 |

CKKS 逐项门限仍为 1e-5 + 1e-4*abs(reference)。本批另行声明的原函数指标为 MaxAbs ≤ 1e-5；六组输入均通过。float32 原始 forward 与 float64 数学参考存在正常舍入差异，相关诊断保留，未改成相同参考。

较宽的控制集合在 240 秒窗口内没有完成，不计为全量通过。首版只读审计误把句柄当作自绑定记录，修正为核对其指向的计划绑定后通过；失败脚本和记录保留。审计独立核对 NPY 解密数据、数学公式、原始源码结果、调用账本、冻结脚本及历史程序包字节，没有增加执行调用。

## 使用入口

数据集入口仍为 ApplicationAssessment(app, store) 的 submit、status、advance、result 和 cancel。原生来源精度显式选择 bound-source-native-dtype-finite-v1，指定 input_dtypes 与 trust_local_code。

命令入口：

~~~sh
python3 scripts/agent_assessment.py --help
~~~

[接口示例](agent-application-assessment-r1326.md) · [CLI 参数](agent-assessment-cli-r1327.md) · [机器报告](agent-current-source-assessment-r1606.json)

## 后续工作

本批只证明声明数据上的数值和应用误差，未认证整个原模型精度。继续核验完整数据集的原函数／近似函数比较及校准域接口，保持与小规模密态验收分离；容量、模型组合、bootstrap、CUDA 与完整 ResNet20 仍按总计划推进。

总产物上限 48 GiB，每例／每批 4 GiB，磁盘余量至少 4 GiB。后端、参数、网络、模型、reference 和数值门限保持；新安装、下载、commit、push 均为 0。
