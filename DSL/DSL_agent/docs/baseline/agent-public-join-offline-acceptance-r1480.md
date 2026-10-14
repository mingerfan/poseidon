# PyTorch v34 拼接离线验收

公开与加密混合拼接、全公开拼接及其组合共21项通过，含18个新上下文和3个旧版本兼容。
实际21次编译、27次Poseidon CPU密态调用，API为0；失败、跳过、未运行均为0。

独立数值审计为96组输入、720个输出值，最大绝对误差1.039295491278569e-6。
门限保持1e-5 + 1e-4*abs(reference)。21份程序包重载和缓存复用通过；
mixed_cat_2、mixed_stack_2、combination_5另完成新输入和重复执行。

325项组件控制、12项执行控制通过。来源隔离捕获21/21，独立明文参考336组/2448值。
RuntimePlan、OperatorSpec、常量bundle、tc128、模型和依赖身份已核验。
本报告为规则候选验收；真实Agent18项已启动，尚未结算，另行计分。

默认来源v18/latest v28保持，显式v34/explicit-v14；后端库、参数、reference、门限、网络保持。
源码继续冻结。任意Python、最大物理容量、bootstrap、CUDA实机和完整ResNet20仍待验证。
[完整离线证据](agent-public-join-offline-acceptance-r1480.json)。
