# Poseidon Agent 测试进度

记录时间：2026-10-11 20:30:34 UTC。

空间算子的34个来源案例均已有真实Agent通过记录。原批次33通过、1失败；唯一失败项经新的真实Agent生成和自动修复后通过。原失败响应、源码、报告与绑定保留。

v23/v24的51个来源完成当前隔离捕获和独立参考检查，按26个精确公开请求去重。68项启动器、预算和来源控制检查通过；25个来源回放的16项规划及源码控制检查通过。以上控制检查本身未进行密态计算。

26项真实Agent测试当前状态：running，已结算10通过、0失败、16尚未完成；已结算生成10次、HTTP10次。owner实际仍在运行：True。进行中案例可能还有尚未结算调用，不能将这些数字当作最终总计。

每个通过案例检查真实编译、Poseidon CPU密态执行、解密与独立reference逐项比较，以及程序包重载、三组新输入、一组重复输入和缓存。数值门限为1e-5+1e-4*abs(reference)。

26项终态后统一审计；余25个来源原样重放对应真实Agent DSL并分别验收，不再生成、不冒用跨来源缓存、不计为新的Agent生成。

执行采用已批准的计划内测试总授权，继续使用冻结案例、版本和预算。Poseidon、Runtime、Dacapo、参数及网络保持不变；未安装、下载、提交或推送。Bootstrap、CUDA及通用化剩余工作仍有独立待验事项。

源码位于ARM Ubuntu /home/lhohy/Code/Poseidon，分支lhy-agent-dsl。完整动态进度在/home/lhohy/poseidon-work/platforms/aarch64-linux/results/agent-normalized-source-live-v1/report.json；本页为记录时的快照。

下一步：核验当前owner与终态账本，审计26项结果，完成25个来源的独立回放，再推进下一项输入扩展。
