# GPU Boot 三项优化移植与 V100 ResNet20 对照

基点：`framework` / `b6e184cb`，保留上一轮 CPU hoist、通用 QP 压缩接线和当前物理 Q-prefix 密钥裁剪。定向来源为 `origin/integration/resnet20-manual-opt` 的 `54fee8af`，没有整支覆盖或替换 runtime 子模块。

## 移植范围

1. **主机 key-view 缓存**：`gpu_key.h/.cpp` 按 active Q 缓存非拥有视图，evaluator 的 11 处调用借用 const 引用。保留 `storage_q_count` 和原始 P 偏移；移动跟随原 GPU allocation，深拷贝重建视图并重基 Galois 指针表。公开 metadata/layout/allocation 在修改前必须显式失效；`make_view()` 自动失效，不支持边计算边改布局。
2. **底层算术/NTT**：96-bit lazy ModUp/P→Q、ModDown four-step、N65536 任意 P 的 QP four-step。保留 P9 专用分派与原 single-prime/tail-digit 路径、执行流和 cropped-key storage stride。96-bit 累加限定 uint32 residues，并检查项数边界。
3. **QP-c0 double hoist**：非 identity giant groups 只对 c1 做内层 ModDown，c0 和 identity group 留在 QP，最终统一 ModDown。它改变舍入，**仍默认关闭**。新增分量计数，不把 group 次数误当实际组件工作量：有 identity 的 G 个 giant groups，从 2G 个内层组件降至 G−1 个。

未移植 SSE、degree22/DA3/K12、Boot recipe/boot_step、跨指令 rotation hoist、lazy-linear、pinned/allocation cache 或运行时 fused rescale，也没有改变当前 native59 Boot 参数。

### 开关

| 环境变量 | 默认 | 用途 |
| --- | --- | --- |
| `POSEIDON_GPU_KEY_VIEW_CACHE` | 开 | `0` 每次重建视图；仅用于有界 A/B，稳定引用快照会保留到显式失效/owner 销毁，主机 metadata 可增长。 |
| `POSEIDON_GPU_LAZY_BASECONV` | 开 | 96-bit lazy basis conversion；`0` 回旧路径。 |
| `POSEIDON_GPU_MODDOWN_FOURSTEP` | 开 | lazy、P>9、N65536 下的 ModDown NTT；`0` 单独禁用。 |
| `POSEIDON_GPU_QP_FOURSTEP_ANY_P` | 开 | N65536 的通用 P QP NTT；不取代原 P9 独立开关。 |
| `POSEIDON_DOUBLE_HOIST_QP_C0` | **关** | `1` 显式启用 QP-c0；还须 direct giant accumulation 和相应预旋转布局。 |

本轮性能运行显式打开 QP-c0，前三项显式设为 1（与新默认一致）。**全驻留和 direct 普通旋转键保持 runner 默认**，没有用 binary/组合键、Waterline、UVM、权重流式或秘密密钥辅助刷新规避显存需求。

## 测试与额外修复

新增 `poseidon_gpu_key_view_cache_tests`，覆盖多 Q 缓存、并发只读查找、失效、深拷贝、move/self-move、trimmed storage/P 偏移及指针表重基。

新增 `poseidon_gpu_boot_optimization_tests`：

- N4096/N65536，P=2/9/10/25，完整 Q 与 Q3、29–32-bit 混合素数；普通/预旋转 rotation 和 relin 在同输入同密钥下切换旧/新路径，逐 RNS 比较。低层夹具使用 scale44，生产 ResNet20 的 scale40 没有改变。
- QP-c0 用独立明文 diagonal oracle，覆盖 full/compressed、identity-only、无 identity、2/3/4 个 giant groups、奇数 selected c1=1/3、跨 Q 和布局保留 scratch、重复执行。
- 显式 nonblocking execution stream；延迟 producer 后读回，验证下载排序。
- 算术/NTT 重排预期 bit-exact；QP-c0 按解密误差对照，不宣称逐密文字节相同。

扩展的非阻塞流测试发现一个**既有 uploader 缺陷**：offset D2H 使用默认流 `cudaMemcpy`，并不等待 nonblocking producer，因而连原始 QP-c0=0 路径也会偶发读到未完成数据。仅在下载前等待 producer stream 即能消除失败，`CUDA_LAUNCH_BLOCKING=1` 也会掩盖它。现将 offset H2D/D2H 与现有 DeviceVector helper 统一为 `cudaMemcpyAsync(..., gpu_execution_stream())` 后同步该流；不改变裁剪、offset 或 count。测试不靠 device-wide barrier 通过。

RTX 4060 Laptop 上修复后的完整低层矩阵通过 Compute Sanitizer memcheck，`ERROR SUMMARY: 0 errors`。本地完整构建及 key-view/低层优化/API/key-policy/scalar/NTT-stream/OOM 七个回归通过。V100 上 CPU hoist、低层矩阵、key-view、GPU API、普通 Boot、压缩 Boot、QP-c0 Boot **7/7 通过**，总计 358.86 s。早期本地 N16384 Boot 进程曾被系统终止、无数值诊断，不计为通过，也不把它混同为 V100 的数值测试结果。

旧 helper 的 Nsight 诊断也与竞态一致：7346 个 kernel 全在 producer stream 13，而拷贝同时出现于 stream 7 和 13；在测试中只等待 producer stream 即通过。最终回归移除了临时等待，由 uploader 自身保证排序。

```bash
cmake --build build-runtime-gpu-api-release --target \
  poseidon_gpu_boot_optimization_tests poseidon_gpu_key_view_cache_tests \
  poseidon_runtime_gpu_bootstrap_e2e -j2
ctest --test-dir build-runtime-gpu-api-release \
  -R '^poseidon_(gpu_(boot_optimization|key_view_cache)_tests|runtime_gpu_bootstrap.*e2e)$' \
  --output-on-failure
```

`workspace_peak_bytes` 仍是历史估算，不是严格 retained/峰值上界；本轮容量比较使用 allocator 实测，不把 NVML 的预留池大小当有效 payload。

## V100 运行口径

- 服务器 SSH alias：`188Server`；物理 GPU1，Tesla V100-SXM2-32GB，UUID `GPU-5aaf51d1-a1db-e8f5-19bd-db86599054c5`。启动前检查空闲，不干预其他进程；对照串行运行。
- 新源码/完整构建：`/home/xuming/poseidon-boot-port-20261110T140206Z/{src,build}`。CUDA12.2、GCC11、CMake3.31.6、SM70；不链接旧 ABI 对象。
- 重用冻结编译器生成的完整 ResNet20 RuntimePlan，不使用手写整网，也不使用另会话正在开发的 Cerium DSL bridge。模型、Boot 位置、权重和输入不为本轮性能改写。
- 计划来源：`/home/xuming/poseidon-resnet20-resident-20261108`；编译器 SHA256 `c6f8d9e3bc838fc039e02419d9914f763c42adf869eabed70fd611015ae9d836`。
- RuntimePlan SHA256：`ba84089c1db2bd0e105c0a33c4ef19887ec55d549a8db71409b9a6d94407d9cf`；OperatorSpec：`f37403b1cdbdbbf9633dfa032e66d43ebab6674c18edb9ba535cfd5027a70bb2`；GPU fixture：`8cea868e4b481950566bce57955e1f5f811c38cf1200ec5cd9191beb6e9b5c0e`。
- N65536、Q50/P25、H192、scale40；18 次 native degree59/DA2 Boot，Q6/s40→Q31/s40。未启用稀疏秘密切换。研究参数禁用安全级别校验，不构成安全认证。
- 全部设备权重在 initialization 编码上传并保留至 forward 结束；在线无 Encode/Prepare/通信。普通旋转键策略环境变量不设置，以验证默认 direct，并核对实际 236 个普通键。
- 30 GiB 预留池；CPU taskset 0–3、OMP/BLAS 4 threads，关闭逐指令 trace。主指标为 `runtime_timing.online_execution_ms`，不含 context、密钥、Boot 常量、权重初始化和输出解密；它不是整个进程启动耗时。

## 结果与比较边界

测量结果、严格校验和分类结果分别报告。所有在线时间均排除密钥/权重准备及解密，不是进程启动至输出的总时间。

| 本轮同 GPU1 实测 | 图像0在线 / s | 三个不同图像均值 / s | 样本说明 |
| --- | ---: | ---: | --- |
| Poseidon 编译器版，full-resident / default direct / 三项优化 | **6.113565428** | — | 单进程首次运行，图像0一次，无额外预热 |
| Cerium 原生 Graph on | **2.073** | **2.067** | 图像0/1/2：2.073 / 2.064 / 2.064 |
| Cerium 原生 Graph off | **2.163** | **2.075666667** | 图像0/1/2：2.163 / 2.032 / 2.032 |

Poseidon 耗时约为 Graph-on 图像0的 **2.949×**（相对三图均值 **2.958×**），目前仍明显慢于 Cerium。Graph-off 首图与后两图不同，不据这三个样本断言关闭 Graph 的稳定态收益。

原先 b6e184cb 同卡 direct/full-resident 的 **8.744290087 s** 仅是历史参考；本轮观察到 **1.430×** 的时间比、耗时降低 **30.09%**，但这不是本轮成对同密钥/同进程 A/B，也不能把全部变化逐项归因到某个 kernel。

### Poseidon 容量、初始化与数值结果

- 实际 ordinary direct keys：**236**；native Boot：**18**；在线无编码/准备/通信，设备权重 payload **6.449951 GiB**。
- Managed allocator 实测峰值 **26,668,921,824 bytes = 24.837369 GiB**；预留池/容量 **30 GiB**。后者不是有效分配峰值，更不是 UVM。
- context **10.238 s**；普通密钥生成 **214.591 s**；GPU API 准备 **299.297 s**（其中 native profile **290.156 s**，不能重复相加）。
- Runtime setup **5.596 s**，权重/输入初始化 **68.993 s**，online **6.114 s**，整个 `Runtime.run` **80.719 s**；全进程 `/usr/bin/time real` **609.82 s**。Cerium Graph on/off 的三图全进程分别 **160.29 / 158.42 s**，也不能把它们与单图 online 混比。
- 分类 **3 / 参考3**，前10 logits 最大误差 **0.011261767**，原应用 `.1` 门限通过。
- **严格全槽校验仍失败**：`atol=.001, rtol=.0001`，32768 槽中 **100** 个实部超差；最大虚部 `2.962e-5`，无虚部超差、无非有限数。原始 `passed=false` 和进程 exit=1 均保留，没有扩大门限或修正输出。
- Cerium Graph on/off 均为三张图 top1 **3/3 正确**，不构成完整 logits 或全槽精度通过。

源文件传输/构建均在新目录完成；最后的 host-callback gate 异常安全/握手补强也已在 V100 单独重编并重跑低层矩阵通过。本地/部署的 **7136 个源码/依赖文件 SHA256 全部一致**，二进制和链接库摘要见归档 `remote/provenance-final.json`。实验结束后 GPU1 已释放至 0 MiB/0%；没有干预其他 GPU 的后来任务。原有子模块 pin 和嵌套 DaCapo 未提交修改没有被本任务改动。

Cerium 部署基于上游 `6b487041`，沿用已有 CUDA12/SM70 等兼容修改；“原生”指 FusedKernelProgram 执行路径，不表示零修改的官方论文构建。Cerium 沿用隔离 runner 的 OMP/BLAS=1 配置，Poseidon 为 4；两边均限 CPU0–3，但不宣称其主机执行策略完全相同。Cerium 使用新的隔离 batch：`/home/xuming/resnet20-matrix-20261109/cerium/batches/20261110T134659.407763Z`，原生 FusedKernelProgram，Graph on/off 分别运行图像0/1/2，UVM=0、单 V100、CPU0–3、NUM_ITERS=1。不覆盖已有 batch，不把论文数字当本轮实测。

**跨系统比较不是等价算法实验：** Cerium 为 degree30/DA3、20 Boot/19 ReLU，Poseidon 为 degree59/DA2、18 Boot；Cerium 的 FC 未使用原模型非零 bias。两边参数、精度门限和统计样本也不同，不能仅由速度宣称模型/精度/安全等价。Poseidon 必须保留严格 `atol=.001, rtol=.0001` 的全槽状态，分类一致或原应用 logits 容差 `.1` 通过不能替代严格通过。

完整原始日志/报告和重跑脚本置于忽略的 `test-results/bootstrap-gpu-port-20261110/` 及上述新远端目录；不是源码仓库内附带的预编译复现包。
