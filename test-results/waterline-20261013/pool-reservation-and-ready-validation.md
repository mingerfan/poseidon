# 物理池预留与完成回收

完整 Q50/逻辑40 的旧 GPU-only 前向在 2:29:19 后因 RMM 分配失败退出1，未产生完整输出。请求为1,966,080B；尾部、线程顺序与接收密文大小指向 GPU3→GPU0 的接收分配，但没有分配时设备/池统计，不能认定唯一根因。事实与推断分开保存于 [失败记录](188-q50-lazy40-failed-tail-interpretation.json)。

本次修正不改 HEFT、逻辑 scale40、Rescale 位置或数学路径：

- RMM 默认在建池时一次实际预留已核准的最大池。显式 `POSEIDON_GPU_RUNTIME_INITIAL_POOL_MB` 仍可设置较小初始值；Waterline只使用预检时已reserved的容量，扣除固定分配、保留workspace及64MiB余量，不再承诺未来可能扩张的容量。
- GPU API `is_ready(true)` 先清理该完成点保活的临时资源，随后执行器才返还额度；回收锁忙时返回false。目标Ready第一次已确认完成后，用同一完成点身份清理，避免第二次非阻塞查询的竞争。固定缓存仍按固定额度保留，直到drain。
- 分配失败只追加CUDA设备、实际当前设备、请求大小/stream、allocated live/peak、pool reserved/cap、driver free/total及查询状态，保留原异常，不重试或换后端。

[本地API绑定与结果](reserved-ready-api-validation.json)为23/23通过。默认256MiB池确实全部reserved；显式initial128/max256时graph capacity为57,119,656B。Rescale与非缓存GPU Prepare在真实CUDA gate后，仅外部stream同步、直接调用is_ready，分别回收1,310,720B与245,760B临时GPU对象，Prepare pinned源归零；输出保留并核对全部2048复数槽和元信息。仅恢复 [旧Ready](reserved-ready-old-is-ready.log) 或 [旧最大容量公式](reserved-ready-old-unreserved-capacity.log) 的私有负控均在对应断言失败，未改生产源码/库。

[独立RMM机制对照](rmm-pool-fragmentation-regression-result.json)三项通过：initial4/max16MiB的多块池即使总free10MiB仍不能申请8MiB；仅将initial改为16MiB后同序列成功；上游拒绝6MiB增长后，RMM按原策略回退4MiB成功。这说明池块边界的风险和RMM已有回退行为，**不是模型OOM重放或确切根因证明**。

[真实GPU Prepare故障测试](reserved-ready-oom-test.log)仍在第二次实际RMM分配注入异常：消费者0次、无换后端/隐式重试、自动drain一次、动态GPU及pinned归零，同runtime显式再跑2048复数槽通过。其initial从64调整到128MiB，以先覆盖固定分配与64MiB余量；原64MiB在188已被新预检拒绝，见 [首次检查](188-reserved-pool-initial-oom-fixture-rejected.log)，不能把该拒绝当成完成故障注入。

188新版[完整检查exit0](188-reserved-pool-stage-exit-status)，CPU9/GPU API23/scalar16、真实Prepare故障、三压力配置及q_proj三后端均通过，[build绑定](188-reserved-pool-build.status)。两份四卡压力的每后端均3个peer实际先于慢GPU0完成，HostStaged每后端20/20 posts观察到pinned；front/plain及pinned收尾归零。q_proj Auto实际361份CPU、664份GPU Prepare。新完整前向已启动但尚无数值结果，局部通过不能替代。物理预留池与实际活跃对象峰值必须分别报告；预留也不能消除池内全部碎片或替代正确的workspace上界。
