# Bootstrap 分支优化核对与定向集成（第一轮）

后续已继续移植 key-view 缓存、lazy/four-step 和 QP-c0；第二轮实现、测试及 V100/Cerium 对照见 [GPU Boot 三项优化移植](bootstrap-gpu-port-v100.md)。下文“暂缓”等状态记录第一轮结束时的决策。

核对基点：`framework` 的 `b6e184cb`。已执行 `git fetch origin --prune`，检查本地分支、远程分支和仍可访问的历史 lfy 提交；不根据提交祖先关系单独判断代码是否缺失。

## 本次选择

### CPU single hoist：`28513634`（`origin/gs/feat-hoist`）

移植 `CpuHoistWorkspace`、HYBRID 批量旋转及两个 CPU BSGS baby-rotation 入口。同一输入只分解/ModUp/NTT 一次；每个直接旋转仍分别乘钥和 ModDown。这不是 GPU double hoist，也不会自动融合 runtime 的独立 Rot 指令。

保留原普通旋转回退：缺直接密钥时用 NAF，BV/GHS/硬件后端不走 HYBRID hoist。调用者可跨批复用 workspace；BSGS 内 scratch 仍为局部对象。连同上游测试及三个正确性修复一起移植：

- 稀疏秘密采样的索引上界改为 `N-1`，避免写越界。
- 单 Q prime 向 P 升基时比较 `modulus_p[i]`，而不是错误地索引 `modulus_q[i]`。
- 初始化 `using_fast_plain_lift_`。

`KSwitchBase` 新增虚接口，GPU profile 配置结构也新增字段，旧二进制依赖必须全量重新编译，不能混用旧头文件/对象文件和新库。

### GPU QP 对角明文无损压缩：摘取 `54fee8af` 的独立配置接线

新增 `GpuBootstrapProfileConfig::compress_linear_plaintexts`（默认 `false`），让通用 DoubleHoistBsgs profile 使用已经存在的 exact periodic QP uploader/kernel。仅改变存储布局，不修改 EvalMod、模数链、秘密分布或输出契约；无周期的对角线不保证节省空间。手写 ResNet20 builder 原本已启用此压缩，本次不改变它。

增加压缩 bootstrap e2e 用例，同时检查压缩计划、逐残基精确重建标记、存储大小和解密后的输出误差。两个 generic e2e 使用显式非零 H=5 秘密并拒绝非有限值；命名压缩用例固定 N16384/DoubleHoistBsgs 并清理 manual/output 覆盖环境，避免意外测试其他路径。这是功能夹具，不是安全参数认证。ClassicBsgs 不使用 QP 压缩路径。

## 未重复合入或暂缓的候选

| 来源 | 核对结果 / 决策 |
| --- | --- |
| lfy 的 `16c266c5`、`c5d235a8`、`d7a12f4a`、`c8480c42` 等 | 精确自举、双 prime rescale、GPU double hoist 和主要融合 kernel 已由 `656674c5` 等同步；不再重复移植。`90da4763` 与 `656674c5` 的 advance、GPU evaluator/keyswitch/double-hoist 和 kernels 完全一致。 |
| `6a71bda7` P9 digit-batched ModUp/NTT | 尚缺。仅 N65536/P9 等特定形状，历史约 1% 收益、增加约 86 MiB 临时区；需要独立同密钥 A/B 和当前执行流适配，暂缓。 |
| `6a71bda7` 最终实投影的共轭 workspace 复用 | 当前正确性等价，但局部 scratch 仍可复用；不覆盖当前 S2C-first 主分支，暂缓。 |
| `6a71bda7` 按层缩小 P | 不等同现有 Q-prefix 裁剪。需要新密钥/上下文、资源预算和并发所有权设计，不能直接搬旧回调。 |
| `f3fc01d1` “自举同时进行relu” | 实际只有冻结多项式求值包装，并无 Boot/ReLU 融合；CMake 还引用分支内缺失的 `Trident/resnet20`，不合入。 |
| `origin/hsb-resnet20-bootstrap-expansion` / `4ef4513d` | 明确是 native template pending 的结构性 checkpoint，缺完整 native 模板、dispatch 和数值验收，不作为可用加速合入。 |
| `origin/gs/feat-application` / `42482f5e` | 独立 Q70/P35、H192、K25/DA5 的 degree22 应用，不等同当前通用 profile；需单独移植和验证，不能仅改 degree 常数。 |
| `origin/integration/resnet20-manual-opt` / `54fee8af`、`a0ca0a89` | 真实的 manual degree22/DA3/K12/full-coefficients/h32 封装及大量 GPU/runtime 优化；约 40 文件，runtime 分叉且冲突当前 per-rotation Q-prefix 裁剪。除上述压缩接线外，暂不整合。其 full-Q50/P25 dense→sparse 切换密钥明确不安全，仅适于性能实验，不能静默替换框架默认。 |

注意：当前通用 profile 的 degree22（从 degree59 截断）并不是准确性已验证的证明；当前手写 Q50/P25 S2C-first profile 仍为 degree59/DA2。旧截断实验、Q70 的 DA5 profile 和稀疏秘密 DA3 profile 是三种不同方案，不可混用精度或性能结论。

## 复现

CPU 回归无需 CUDA 或 runtime；构建也可关闭可选依赖：

```bash
cmake -S . -B build-bootstrap-cpu \
  -DPOSEIDON_BUILD_CPU_TESTS=ON -DPOSEIDON_BUILD_EXAMPLES=ON \
  -DPOSEIDON_BUILD_DEPS=OFF -DPOSEIDON_USE_ZLIB=OFF \
  -DPOSEIDON_USE_ZSTD=OFF -DPOSEIDON_USE_MSGSL=OFF
cmake --build build-bootstrap-cpu --target poseidon_cpu_hoist_tests test_ckks_bootstrap_cf_real -j2
ctest --test-dir build-bootstrap-cpu -R '^poseidon_cpu_hoist_tests$' --output-on-failure
build-bootstrap-cpu/bin/poseidon_cpu_hoist_tests --benchmark
build-bootstrap-cpu/bin/test_ckks_bootstrap_cf_real --log-n 13 --h 5 --full-coefficients
```

CPU 测试覆盖多种 Q/P、逐级降链、普通旋转对照、NAF fallback、正负/重复/零旋转、别名、异常输出保持、工作区复用、普通和 streamed BSGS 以及上述越界修复。内置 benchmark 比较相同密钥/输入下 16 个普通旋转与 hoisted batch 的中位数；不等同于端到端 bootstrap 加速比。CF-real 示例使用单 P 的 BV 回退，可用于原有 bootstrap 语义回归，不能拿它证明 HYBRID hoist 加速。

在已启用 `POSEIDON_BUILD_CKKS_RUNTIME_API` 和 GPU tests 的完整构建中：

```bash
cmake --build build-runtime-gpu-api-release --target poseidon_runtime_gpu_bootstrap_e2e -j2
ctest --test-dir build-runtime-gpu-api-release -R '^poseidon_runtime_gpu_bootstrap(_compressed)?_e2e$' --output-on-failure
```

压缩用例通过 `POSEIDON_RUNTIME_BOOTSTRAP_COMPRESS_LINEAR=1` 启用；其余 profile 参数与原 e2e 一致。没有引入稀疏秘密切换，也没有改变当前默认 profile。

## 本次验证结果

GPU 为 RTX 4060 Laptop 8 GiB；Release 构建，测试设置 `OMP_NUM_THREADS=2`。

- CPU hoist 回归通过：Q/P=5/2、8/3、3/5、4/1、3/3；ASan 回归也通过（`ASAN_OPTIONS=detect_leaks=0`，不含泄漏检测）。
- 同输入/同密钥、N8192/Q10/P3、16 个旋转、5 轮中位数：普通 265.175 ms，hoisted 142.328 ms，约 **1.86×**；保留 workspace 6,684,672 bytes。这是单次本机旋转基准，不是整 Boot 或整网收益保证。
- CPU CF-real 完整系数、N8192/H5、六类输入，单/双路共 12 次自举均通过；最大误差约 `4.017e-4`（常量输入的已有近似偏差），输出 level=5/scale=2^40。
- 完整 GPU/runtime 构建通过。CPU runtime API、GPU runtime API、普通 GPU Boot、压缩 GPU Boot 四个 CTest 用例，以及 rotation-key policy、GPU scalar、NTT stream、prepare OOM 四个回归用例均通过。GPU Boot 使用原 e2e 的 degree58/DA2/scale45 参数，并非未验收的默认截断 degree22。
- 最终两个 GPU Boot 使用非零 H=5 秘密，均输出 level=22/scale=2^45，最大解密误差分别 `1.84755e-4` / `2.03760e-4`（门限 `1e-3`）。两进程生成不同随机密钥，不据此要求输出密文逐字节相同。此前零秘密夹具的结果不作为非零秘密正确性的证据。
- 压缩用例对角明文由 140,148,736 words 降到 49,244,832 words，即 **534.625 → 187.854 MiB，减少 64.86%**；逐残基精确重建检查通过。这不是整个进程/GPU 常驻内存降幅，也未将包含 setup 的 CTest 耗时作为 kernel 加速证据。

最初直接运行原有构建中的 GPU Boot 曾因 `free(): invalid pointer` 中止；重新配置、重编库及测试后上述用例通过，未把这个旧构建现象单独归因于某一项修复。

原有 `third_party/ckks-runtime/third_party/dacapo` 未提交修改保持原样；未替换任何子模块指针。历史分支保存的 V100/4090 性能数字不是本次硬件复验结果。
