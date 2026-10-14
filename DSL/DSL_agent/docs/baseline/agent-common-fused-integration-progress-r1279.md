# 联合接入准备（r1279）

有序张量、common-call 和融合矩阵草案已整理为一份 10 文件的接入包。
覆盖顺序与已验收的隔离环境一致。8 个现有文件保存恢复副本；
两个新增模块为 `pytorch_common_constant_provenance.py` 和
`pytorch_fused_matrix_normalization.py`。

生产源码继续保持当前 704 文件身份；706 文件身份是待接入版本。
来源、生成身份与编译执行身份分别保留，未将草案记录改写为生产通过。

## 已完成

- 来源包验收、九份生成请求与权重描述符的证据已绑定到接入包。
- 只读守卫核对最终源码、覆盖顺序、恢复副本、分支、真实进程和后端/reference。
- 22 项控制层测试通过，失败／错误／跳过为 0。
- 修改原文件、变动前置覆盖层、损坏恢复副本、普通或悬空链接文件冲突、
  验收记录变动、丢失联合模块、未执行后端验收却标为通过等情况被拒绝。

测试使用系统 Python 3.10.12。隔离的公开源码副本用于拒绝路径检查，
这些测试不是 frontend tracing、编译或密态执行证据。
当前真实进程检查为空，守卫返回 awaiting_backend_qualification。

## 下一步与批准边界

本轮编译、密态、解密和付费调用均为 0，没有安装、下载、后端／参数／
reference 改动、提交或推送。原有修改、x86 配置、旧响应、报告和源码绑定保留。

下一步需要按已准备的 r1263 有界离线方案恢复三例 common-call Poseidon 验收。
此前执行窗口已结束，新增窗口尚未收到批准。r1232 三例付费测试独立等待批准；
当前接入包不启动 API。现有待批方案和绑定保持，切换生产时另建编译执行绑定。

联合版本还须真实 frontend tracing、编译、产物检查、Poseidon 数值、
程序包重载和新输入执行。后续算子、容量、近似／bootstrap、整模型和
CUDA 目标仍属于原计划；本阶段不代表整个目标完成。

[接入包说明](../../scripts/baseline/benchmarks/tools/staged_common_fused_integration_v1/README.md)
· [机器可读记录](agent-common-fused-integration-progress-r1279.json)
· [控制测试](../../scripts/baseline/benchmarks/tools/staged_common_fused_integration_v1/acceptance.json)

接入身份：`53eb5a4baaac35f680dd19a8bd57fba6533dbe1348f37bac0ddb3f008ba1ee7d`。
控制测试：`3542b34d11b0962cf1301009a00096cff1e83fe0c3115875443d78a55bb33e92`。
