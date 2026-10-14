# PyTorch 空间计算入口 v27

可信本地来源清单新增 `poseidon-trusted-pytorch-v27`，接入 Upsample/interpolate、ConvTranspose1d/2d 的模块及函数写法。输入调用沿用 v26 的字典、嵌套容器、关键字参数与公开默认参数，输出保留原容器结构。

## 实现与检查

- 复用现有来源沙箱、公共状态绑定、独立参考、应用提交和程序包接口。
- 34 个明文上下文通过，544 组／7,824 值，最大差异 `6.106226635438361e-16`。
- 128 项生产单元／兼容检查 passed；failed、errors、skipped 均 0。
- 3 个生产应用提交通过真实来源沙箱、独立参考和请求准备；API、编译、密态调用均 0。
- 随后冻结 18 个代表请求，全部通过真实来源沙箱／图导入、独立参考和规则候选静态检查；14 项执行控制检查通过。

34 个上下文包含各空间调用的三个场景，以及别名修改、多输入、字典输入、公开权重转换和命名输出。来源沙箱与应用检查复用其中部分模型，数量分别记录。 分阶段明文报告保留当时绑定；能力描述随后完善，生产检查与 18 项来源准备使用最终绑定。

ConvTranspose 模块重追踪时，元数据读取曾触发 FX 添加多余 get_attr。元数据改从已登记的公开参数表读取；权重内容、数学运算与后端保持。中间失败和原绑定保留。原来源默认 v18、latest=v21 及旧版本、x86 配置保持。

## 验收状态

| 层级 | 当前结果 |
|---|---|
| 来源捕获、明文参考、应用请求准备 | 通过上述范围 |
| v27 的 Hecate 编译与 Poseidon CPU 密态 | 待 r1405 代表批次 |
| v27 新功能的真实 Agent 生成 | 待单独冻结与批准 |
| 旧 JSON 空间计算 | [r1403](agent-spatial-offline-results-r1403.md) 45 项 CPU 通过，绑定保持 |

Poseidon、CKKS Runtime、Dacapo/Hecate、参数及误差门限保持；无安装、下载、网络变更、commit 或 push。

[输入说明](operator-decomposition-and-python.md#pytorch-v27-空间计算) · [示例](../../scripts/baseline/cases/pytorch-spatial-v27/catalog.json) · [机器记录](agent-spatial-source-component-checkpoint-r1404.json)。
