# PyTorch v23 Poseidon CPU 离线结果

18 份来源配置全部完成真实 tracing、编译、产物检查、密态执行和四组逐项数值比较。完整流程为 **17 passed、1 failed、0 skipped、0 not run**。

## 验收范围

| 层级 | 结果 |
|---|---|
| 来源配置 | 18 份；规范化请求去重后 14 份 |
| 真实 tracing／编译 | 18 / 18 |
| RuntimePlan／OperatorSpec／bundle 检查 | 18 / 18 |
| 四组输入密态数值验收 | 18 / 18 |
| 程序包文件哈希与重载检查 | 18 / 18 |
| 新输入＋重复执行完整通过 | permute_1、bool_cast_1 |
| 命名输出程序包 | 首次验收通过，新输入的返回结构恢复失败，重复执行未运行 |

案例采用原固定规则候选，真实 Agent 生成数为 0。这 18 份来源配置用于公开变换分区验收，不计为 18 个新增 Benchmark 模型。

合计 23 次真实 Poseidon CPU 调用、83 组输入、512 个输出值，最大绝对误差 `7.243918576e-09`。门限保持 `1e-5 + 1e-4*abs(reference)`。执行器记录私钥访问为 false；bootstrap 与 CUDA 未执行。

## 唯一失败

named 模型返回 `{"primary": y, "saved": [x, None]}`。首次四组数值验收通过，新输入的 3 组真实密态运算也已完成，解密数组保留。失败发生在 Agent 的 `restore_outputs`：程序包只为 v16 启用静态 None 返回，v23 没有继承该标志。

在同一组已保存结果上，默认恢复准确重现 `Explicit static None output`；使用已登记的 None 规则可恢复完整结构。独立数学比较也通过。原应用结果仍保持 `validation_failed`，未将这项改写为通过；原重复执行没有发生。

修复与新身份见 [r1371](agent-public-expression-output-repair-r1371.md)。当前绑定的完整补验另见 [r1372](agent-public-expression-output-offline-proposal-r1372.md)。

## 资源和绑定

实际运行 427.93 / 900 秒。新增 results 68.71 MiB / 512 MiB；全局 28.381 GiB / 32 GiB，余量 41.38 GiB。
编译≤2、链接≤1、密态串行，native 4 GiB／CPU150秒保持。

原 SDK：`b41fe9599da34b14b013ce1aad5237969897132f0f0130643c795fc780841d5a`  
审计绑定：`43f485c321dce9572e9d4abbd5b61b5122d9fb40af282f5574232b18d174145f`  
原运行报告绑定：`17093989bc49b162a81352e23ebeb86a8cc9519215ca0596cbc64e1cd04df1e4`

源代码、模型、权重、reference 和原结果按冻结身份核验；原始失败与历史报告保持。API／HTTP、安装下载、网络修改、后端与参数变更、commit/push 均为 0。
详细路径、真实运行记录与文件哈希见同名 JSON。
