# Waterline 反例覆盖审计

更新于2026-10-14。依据两份当前中文计划及本地保存的源码/日志，核对完整组准入、真实 Ready、共享预算、生命周期、失败收敛和 HEFT 保持。当前已推送引用为 Poseidon `3d4cff66`、ckks-runtime `369bdc5`、DaCapo `6e75be2`；Rescale预乘修复本地及188正控通过。最新补测结果分别绑定其日志，不混用不同版本的墙钟。**完整 Qwen 数值验收尚未完成。**

最新 [core逐项日志](bundle-read-core-tests.log) 为8/8通过（9.62 s），包含 Waterline、Release及失败清理；[编译器日志](compiler-heft-preservation-tests.log) 为10/10通过（2.64 s）。较早 `core-tests-failure-multigpu.log` 保留 Mock 逐项输出，[前缀反例日志](core-prefix-audit-tests.log) 保留共享预算专门结构。失败清理版188 [单卡API](188-failure-cleanup-gpu-api-single.log)、[全部四卡](188-failure-cleanup-waterline-all.log)、[全部四卡HostStaged](188-failure-cleanup-waterline-host-staged.log) 及 [真实GPU Prepare分配失败](188-failure-cleanup-gpu-prepare-oom.log) 的监督 [exit0](188-failure-cleanup-exit-status)，实际二进制/源码见 [hash](188-failure-cleanup-build.status)。这些是小图正确性、故障及资源检查，不是完整模型或独占性能对照。

表中分别标出 Mock、真实 CUDA 和编译器图回归的范围；较早失败日志保留，不能仅凭文件名含“failure”判断当前结果。

| 文档反例或性质 | 源码和已有通过证据 | 覆盖范围及未覆盖部分 |
| --- | --- | --- |
| lazy scale40的Rescale数值与level/scale一致 | 完整Q50计划在系数后乘API中遇到160→40/drop5，原生先除使scale约降至10，后乘策略的精度保护据此拒绝；[失败报告](188-q50-scale40-preflight-model-report.json) 和 [日志](188-q50-scale40-preflight-model.log) 保存计算提交前的真实预检失败。新修复先乘全部RNS系数并同步实际scale，再原生除模，保持原逻辑scale40、target level和既有lazy/Boot图。CPU预乘版9/9；54个scale40 dense组合各检查全部2048复数槽，maxabs=5.0261e-8，含CP/CC、split/mixed链及独立AddCC/AddCP，见 [CPU日志](cpu-rescale-prescale-tests.log)。普通 [GPU API日志](gpu-rescale-prescale-api-final.log) 20/20通过。 | **CPU/GPU/Auto的40与120组合图已在本地和188通过**，见 [本地49152槽结果](lazy-rescale-composite-regression.json) 和 [188 scale40 GPU报告](188-prescale-composite-scale40-gpu-report.json)。较早scale120 q_proj通过未覆盖此缺陷；提高逻辑waterline会改变规划，不能替代API顺序修复。原Q50/scale40编译计划保留可复用，完整模型三后端均未完成验收。 |
| 下一组需两份明文，空间只够一份；不能先拿半组 | `tests/runtime_waterline_tests.cpp::main` 构造已占 front=400、plain=300 的账本，确认单份 plain=100 可放、完整组 plain=200/front=200/Host=100 不可放；失败 reserve 后计数不变，回收后整组可放。上述 Mock 详细日志最后有 `PASS complete atomic group...`，新 core suite 同测试通过。 | **账本级覆盖**。当前合法 Compute 签名至多一个 plaintext；Verifier 规定 CommAction 必须一个 input/source，Transfer 还必须一个 output，所以不能用双输入 CommPost 构造合法双 Prepare 消费者。没有为了凑测试扩展协议或引入非法计划。合法 executor 的完整多资源阻塞反例见下行。 |
| 输入能放，消费者输出或 workspace 不够 | `test_initialization_preflight` 用 M=450 拒绝输入+输出+workspace=500 的完整基线，确认 initialization Encode/Transfer 和 Prepare 均未提交；`test_allocator_capacity` 对 CPU/GPU/Auto 在 allocator=550 拒绝，600/650 接受并滚动回收。编译器 `check-waterline.py` 用 `workspace-bytes-per-op=8192` 单独触发 single-group 失败。Mock 详细日志及当前 core/compiler suite 通过。 | 输出和 workspace 被计入准入的反例已覆盖。真实 CUDA 的 `test_cpu_prepare_host_budget` 另外验证 Host 边界，不能替代单独的 GPU 输出/workspace 人工延迟反例。 |
| 共享明文首次消费后继续保活，最后使用完成才释放 | `check-waterline.py` 的同一 Prepare 供 200 AddCP，检查只生成一次且无重编码/Fence；真实 `test_waterline_runtime` 每卡一份 Prepare 供五轮 AddCP，最后一轮才 Release。188 单卡、全部卡 CPU/GPU/Auto 日志验证所有复杂槽、`encode_calls==device_count`、最后 plain=0、pinned_live=0。 | 编译器共享及真实多消费者保活已覆盖；四卡 CPU Prepare 是每目标设备分别准备，不是同一 Host Encode 结果的多目标异步上传。后者原有 Mock Release fanout/split-send 已通过，当前 188 压力日志没有独立证明共享 Host Encode 的四卡上传寿命。 |
| 单组/保留对象加下一组超过水位，明确失败或等回收再推进 | `test_small_waterline` 的 20 个组记录每个消费者完成先于下一 Prepare，CPU 与 Auto 均恢复全部组且无 Fence；`main` 分别降低 fraction/M/Host/pinned/raw 或关闭后端，确认无 Prepare 提交即失败。`test_fixed_prepare_buffers` 跨滚动完成保留固定 Host/pinned credit；当前 core suite 通过。真实 CPU Host 负边界日志显示 `cpu_prepare_calls_delta=0`，正边界完成并 pinned=0。 | 已覆盖小水位暂停/回收及不可行配置的加载前拒绝。未找到在执行中直接修改 waterline 比例的用例；现有测试更改的是各次运行的配置。 |
| 多卡 GPU/Host/pinned 等预算整体准入，避免部分占用互等 | 账本失败 reserve 后包括 Host 的所有计数不变；`test_four_device_progress_and_shared_budgets` 分别令共享 Host 或 pinned 只能容纳两项，断言 peak_prepare=2、四卡均回收、GPU1/2/3 先于慢 GPU0 完成。新增 `PrefixBudgetPrepareApi` 在四个合法 Prepare/AddCP 分支中令第一 Prepare 延迟 300 ms，第二组设备预算可放但共享预算只剩 200、需要 400，后面两组各需 100 本来可放。检查运行时确实在第一 Ready 前查询受阻第二组，所有后组 Prepare 都在第一完成后才开始；最终四路数值正确、每卡 front/plain 返回 0、没有重试/换后端/Fence。新逐项日志两个版本均 PASS。188 HostStaged 四卡每后端 20 次 ring posts，pinned 实体观察为 20 次，数值及回收通过。 | **实际 executor 的原子多资源等待及拓扑前缀覆盖**：不能先启动共享预算不足的组，也不能跳过它启动后面可放的组；真实完成后四组整体推进，没有遗留部分设备额度。Host 和 pinned 分别作为唯一限制。没有故意绕过正常 API 让卡先取得非法部分额度，亦未宣称执行过这种内部状态破坏的死锁负控。 |
| 唯一 CPU worker 的消费者不能阻塞必需 CPU Prepare | `test_host_waiter` 将依赖 Device AddCP 的 Host Negate 放在 CPU 队列，检查 Prepare 与 Host consumer 为同一线程、Prepare 事件在先，最终值正确且无 Fence。Mock 详细日志有对应 PASS，当前 core suite 通过。 | 直接覆盖唯一 CPU worker 的 producer 优先反例；此专门结构是 Mock 测试。真实小图中 Host Boot/CPU Prepare 各自通过，但未找到包含此人工排队反例的真实 CKKS 用例。 |
| 未完成真实 Ready 不能被消费者使用；Release/reuse 不能提前回收 | 当前 `188-rescale-gpu-api-single.log` 及本地 `gpu-normalized-rescale-api.log` 通过 upload completion、unfinished compute/transfer Release、非最终输出依赖 drain、reuse 等公开 API 测试。`stream-ready-validation.md` 引用同一生产 stream 的六个 80 ms CUDA gate 案例及 AddCP c1 的逐 word 正负控；正控 Ready 后全字正确，恢复旧默认流 copy 的隔离负控稳定失败。原始日志和源代码保留。 | 有真实 CUDA 事件及有判别力的负控；80 ms gate 是本地单卡，不能当作 188 完整模型 trace。188 四卡不均衡检查对 peers 调用 synchronize 后才计完成，真实三张 peer 均在慢 GPU0 前完成。 |
| 不需要批末 Fence，显式 Fence 保留整体边界语义 | 编译器检查 200 计算无任何自动 Fence；Mock 20 组滚动 `fence_calls==0`，`test_prepare_across_fence_and_unused` 保留一条显式 Fence、跨界注册的 Prepare 能被后续消费、未使用 Prepare 不提交。真实 `test_runtime_explicit_encode_fence` 在两种执行模式执行显式 Encode/Fence 数值对照，当前 188 单卡 API 日志 PASS。 | 自动批次路径的替换与显式 Fence 的单卡语义已覆盖。该显式 Prepare 跨界结构的专门用例是 Mock；188 全部卡 Waterline ring 无 Fence。 |
| 读取/后端/分配失败后，错误可见、依赖唤醒、在途工作安全结束 | Mock `test_prepare_failure_and_explicit_restart` 对固定 CPU/GPU 的第二次 Prepare 注入异常，错误可见、无换后端或自动重跑，另一次显式调用同runtime完成。新增正式 `poseidon_gpu_prepare_oom_test.cpp` 通过公开RMM hook，让真实CUDA Prepare输出分配及stream先在途，再在第二次Scratch分配请求注入 `std::bad_alloc`。188日志确认消费者0次、GPU尝试1次、CPU尝试0次；`run()`返回前等待gate结束、自动drain1次、dynamic/pinned live=0，同runtime保持存活并显式再跑成功，2048复数槽maxabs=4.2764e-8。当前Release suite也验证失败send的源保活至drain后释放所有权。 | **生产CUDA分配路径的故障收敛已补齐**，不是整卡显存耗尽实验。本地 [修复后正控](gpu-prepare-oom-local-v5-after-cleanup-fix.log) 通过；[旧header负控](gpu-prepare-oom-local-v5-beforefix-negative-control.log) 确实失败，返回时drain=0、dynamic=229376B、pinned=16384B，判别出缺陷。新增 [真实缺失blob测试](bundle-read-worker-tests.log) 在CPU/GPU Prepare worker内失败，失败消费者无提交、无换后端/重试，源保活至drain结束并在返回后释放；显式修复后同runtime/紧预算20组完成。[core8/8](bundle-read-core-tests.log)通过。这条测试不声称错误瞬间另有pending Prepare，异步在途失败由前述用例覆盖。 |
| HEFT 前台设备与相对顺序保持 | 新增 `check_heft_preservation` 从未placement的共享权重fanout/Rotate/reduction/Host Boot图实际执行四卡HEFT及通信物化一次，再从同一placed IR分别导出eager和Waterline。逐项比较initialization/execution的前台密文子序列（仅去除绝对ordinal）、密文descriptor、external/final output ID；65条密文计算、37条密文通信的设备/参数/ID/相对顺序一致，8个Prepare输出均被复用。[专门日志](compiler-waterline-heft-preservation.log) 和当前10/10 suite通过；原200 AddCP回归保留。 | **实际HEFT多卡图的后期改写保持已补齐**，含真实编译生成的跨卡边及Host边。该测试不是CUDA执行，也不是完整Qwen图。**完整Qwen同配置审计现已通过**：同一pass调用两侧直接捕获IR，4,870,497条CT计算、3,420,801条CT传输及描述/ID/设备/参数/相对顺序逐项一致，[完整摘要](188-full-qwen-heft-audit-canonical-summary.json)；新旧binary SHA256完全一致，[绑定](188-full-qwen-heft-audit-runtime-plan-binding.json)。这是独立全图保持审计，不能当作数值执行证据。 |

## HEFT 源码核对

本次重新通过 `git show <revision>:<path>` 核对SHA-256。两份源码在基线 `39c1fb2`、`ad35c87`、新增物理level校正的 `d5ed984`、仅加测试的 `6e75be2` 内容完全相同：

| DaCapo 文件 | 共同 SHA-256 |
| --- | --- |
| `lib/Dialect/CKKS/Transforms/AssignPlacement.cpp` | `6c0738775b77e3131376e8c49467f8dc6f4b8fa5aef666c81d08e3eb28ae5e46` |
| `lib/Dialect/Earth/Transforms/LatencyEstimator.cpp` | `848b5fab24e69321f21abbf151d74214a673a177cc61e3d8e2c34ca3b3222e19` |

这补充 [旧hash记录](heft-source-sha256.json) 原来只到 `ad35c87` 的范围。实际物理参数/profile已改变，源码相同不保证新Q50与旧Q40产生相同placement；本次已对完整模型**同一配置、同一pass调用的前后IR**完成该比较；未声称新Q50与旧Q40 placement相同。

## core 前缀反例的复现

2026-10-13的前缀反例补测仅在独立目录构建，当时未改执行器、Verifier、协议或编译器，修改文件只有core的 `tests/runtime_waterline_tests.cpp`；其后失败清理修复另有上述最终日志。前缀补测命令：

```sh
cmake -S third_party/ckks-runtime -B build-waterline-prefix-audit \
  -DCKKS_RUNTIME_BUILD_TESTS=ON -DCKKS_RUNTIME_BUILD_EXPERIMENTS=OFF \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-waterline-prefix-audit -j2
ctest --test-dir build-waterline-prefix-audit -V
```

实际被测测试源码 SHA-256 为 `eef77e8dac64d591705554fbf97cf81691eba8fa5b227b83352f3e63f1f9221e`，独立 `runtime_waterline_tests` binary 为 `5d50201a7db5f0efd49f7bc5cfd3c04def59cc62744cab794fd7a2717c40521d`。新两个反例全部卡 front_peak=500/plain_peak=150、Host_peak=1000，Host 受限版 pinned_peak=150，pinned 受限版 pinned_peak=600；所有卡 remaining_front/plain 都为 0。峰值在完整测试日志内是对应 PASS 紧接的 stderr 输出；输出次序受缓冲影响。

以上缺项应按其专门结构补证据，或者在交付中明确验证范围；不能将有实现代码、CTest 总通过或有限小图数值通过扩大成尚未执行的反例及完整模型验收。
