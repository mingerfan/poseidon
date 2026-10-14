# 命名输出补验结果 r1373

原 named 案例在修复后的执行身份下通过。字典中的 Tensor、列表中的 Tensor 与静态 None 均可正确返回；程序包完成重载、新输入执行和重复执行。

| 验收 | 输入组 | 输出值 | 最大绝对误差 | 结果 |
|---|---:|---:|---:|---|
| 首次编译及数值验收 | 4 | 48 | 8.1459362763e-9 | 通过 |
| 程序包重载后的新输入 | 3 | 36 | 3.4263070159e-9 | 通过 |
| 同一输入重复执行 | 1 | 12 | 4.2746665044e-9 | 通过 |
| 合计 | 8 | 96 | 8.1459362763e-9 | 通过 |

本批 1 passed、0 failed、0 skipped、0 not_run。实际 1 次编译、3 次 Poseidon CPU 密态调用，完成 Hecate tracing、Earth/CKKS、RuntimePlan、OperatorSpec、plaintext bundle、解密与逐项比较。冻结门限为 `1e-5 + 1e-4*abs(reference)`。

此前的 32 项控制与兼容检查来自 [r1371](agent-public-expression-output-repair-r1371.md)，本批未重复运行。独立只读审计另以标准库按固定公开常量复算 96 个数值，核对程序包 20 个文件的哈希、执行元数据及旧记录。

## 修改与来源

执行入口以共同 `allows_none_output` 登记选择静态 None 策略，覆盖 v16–v23；v15、未知及未来版本仍拒绝。该修改仅位于 Agent 的程序包返回适配，后端库、编译器和参数保持原样。

模型、公开权重、原候选源码、独立 reference 及 request_id 全部保持。候选来源为原规则基线，API/HTTP 调用均为 0，未进行新的真实 Agent 生成。

旧 [r1370](agent-public-expression-offline-results-r1370.md) 保持 17 完整通过／1 返回失败。此次按新绑定补齐 named，未改写旧失败，也未将其余 17 项重新标为当前 SDK 全量通过。已完成的 18 份来源配置与本次复用验证属于明确版本下的离线后端证据。

## 环境与预算

- 开发／执行：ARM Ubuntu `/home/lhohy/Code/Poseidon`，`lhy-agent-dsl`，HEAD `7588e48093631eaf8385506e3db058e7ca3cfe7e`。
- Poseidon CPU，N=32768，Q13／P1，tc128；执行器私钥权限为 false，未使用 bootstrap 或 CUDA。
- 用时 47.98／240 秒，新增 results 6.90／128 MiB。
- results 总量 28.389／32 GiB，磁盘余量 41.36 GiB，最低要求 4 GiB。
- 网络、安装、下载、后端、参数、reference、门限及 Git 提交／推送修改均为 0。

## 绑定与产物

- 批准范围：r1372，binding `0e0d796df8da1070f56c39209036bf1239490b1c8acbd0425e18d0042fc0e63a`。
- 当前 SDK：`e2b3e63c27e8ed49f6ddca34b2c5a5d8be366eefa9cdb1ab67d9b953b8a64992`。
- 原 request_id：`61e277c6e9d3062e24c98c14cadb54702edd201785b4f8c3b780c50257e6372d`。
- 当前任务：`5a0a79ee63264a8aa277a589d55729bb`。
- 程序包：`e638a18c7f17186a2f66cae4adf7e205869b56244c6a1f8fc6af35a087a3d951`。
- 审计绑定：`201c6b9107416a7040d76bd14af777838914e34dc32bc255257c665b737385cb`。
- 本报告绑定：`74480e40be808c014fa45c2cbe98d28f1ca3e63d70a5a469e475a9c92d983801`。

完整机器可读记录在 [r1373 JSON](agent-public-expression-output-results-r1373.json)；原始 report、qualification、新输入／重复结果、比较和 audit 均保留在 Ubuntu `/home/lhohy/poseidon-work/platforms/aarch64-linux/results/agent-public-expression-output-tree-v23-v1`。

## 后续

v23 新功能的真实 Agent 生成仍需单独冻结案例与预算后验收。容量、算子组合、真实 bootstrap、CUDA 实机和整模型任务继续分别推进，本项通过不代替这些里程碑。
