# 程序包静态返回结构修复

Agent 的程序包执行入口现在使用 `pytorch_ordered_contract.allows_none_output()`，与来源捕获和原生参考采用同一份版本规则。允许静态 None 的 v16–v23 可恢复字典／列表／tuple 中的 None，v15、未知版本和未来版本保持拒绝。

## 验证

32 项相关控制与兼容测试通过，failed/errors/skipped 为 0。范围包括 v16 的原策略、继承版本、未登记版本、名称与结构、缺失张量、输入不变性、原程序 I/O 及原生精度接口。

用原 failed 调用保存的 3 组、36 个解密值核对实际程序包元数据，新规则正确恢复 `{"primary": y, "saved": [x, None]}`。这是历史结果的结构恢复检查；修复后没有新增编译或密态调用。

当前程序包依赖检查确认旧执行身份被拒绝，未绕过或修改旧 manifest。新 SDK 为 `e2b3e63c27e8ed49f6ddca34b2c5a5d8be366eefa9cdb1ab67d9b953b8a64992`；共享模块的保守依赖闭包使四阶段身份重新绑定。编译器、Runtime、Poseidon 库与后端身份均未改变。

[原批次结果](agent-public-expression-offline-results-r1370.md)保持 17 完整通过／1 失败；18 个模型的首次数值验收通过单独记录。当前修复仍须 [1 项补验](agent-public-expression-output-offline-proposal-r1372.md) 才能确认新绑定的完整执行。

## 修改范围

只改 `scripts/baseline/program_execution.py` 的版本判断，并增加 7 项专门测试，连同 25 项旧检查共 32 项。前一版源码在 Agent 开发工具目录保存。没有改变 DSL、模型、权重、reference、数值门限、后端库或网络。

修复报告绑定：`c25e0a4cb6136d60e56fadc417339eb1d04bed972777fc2257846a0d3907386d`。机器记录与 diff 见同名 JSON。API／HTTP、安装下载、提交推送为 0。
