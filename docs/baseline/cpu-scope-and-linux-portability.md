# CPU 验证路线的代码边界与双 Linux 平台移植方案

日期：2026-09-17。状态：设计与只读代码核查记录，尚未实施代码拆分、删除或跨平台改造。

## 1. 结论与适用范围

1. 如果不做 Poseidon GPU 路线，可以使 Poseidon adapter、GPU schedule 和 CUDA 执行对接的增量退出 Agent 的必需依赖，但不能把 Dacapo 的修改一起全部删除。
2. 当前需要保留的是：Agent 与验证框架、其所依赖的 Hecate/Dacapo 语义修复、Dacapo/SEAL CPU 编译执行环境。
3. 路径和架构硬编码可以改造成同一份源码支持 Linux x86_64 与 Linux aarch64；依赖锁、编译产物、Python wheel 需要按平台区分。
4. 这里的 Mac 目标指 Mac 上的 Linux 虚拟机，不是原生 macOS。尚无本项目 ARM Linux 端到端验证结果。
5. 本文不是删除清单、安装授权或执行成功报告。后续修改前必须重新检查主线程工作进度和 Git 差异，避免覆盖并行工作。

证据分级：

- **Confirmed fact**：本次通过当前工作树、构建配置和源码确认。
- **Evidence-based inference**：由当前依赖关系支持的设计判断，仍需构建或回归验证。
- **Unconfirmed**：尚未在目标环境执行验证，不作为已完成能力。

## 2. 核查基线

| 项目 | 核查结果 |
|---|---|
| Windows 源码位置 | `D:\Code Space\Poseidon` |
| 当前 WSL 源码位置 | `/mnt/d/Code Space/Poseidon` |
| 分支 | `feat/agent-dsl-correctness` |
| 主仓库 HEAD | `4995e7cadedf2bfb9104658b5638662ecf6a1d0a` |
| 配置的 upstream | `origin/gs/feat-application`，未在本次刷新远端 |
| Dacapo HEAD/gitlink 基线 | `4616402710f39df3e5f5bd7930a6c036025aaac3` |
| 工作树 | 主仓库和 Dacapo 均有未提交修改；另有大量未跟踪新增文件 |

以上提交号不能单独复原当前项目：未提交修改和未跟踪文件也是当前实现的一部分。Dacapo 补丁必须单独保存；主仓库补丁不会自动包含 submodule 内的源码差异。

## 3. 问题一：不做 GPU，哪些内容还需要？

### 3.1 实际 CPU 链路

```text
明文模型描述与独立 reference
    → Agent 生成 Hecate Python
    → AST、类型、布局与语义约束检查
    → Hecate frontend
    → Earth / CKKS 编译
    → HEVM / CST
    → Dacapo 的 SEAL HEVM CPU 执行器
    → 解密与数值差分
```

**Confirmed fact**：`scripts/baseline/run_candidate.py` 将后端记录为 `upstream_SEAL_HEVM_CPU`，记录 `libSEAL_HEVM.so` 的指纹，并明确标记 `poseidon_gpu_validated=False`。该执行路线不通过 Poseidon GPU schedule/backend。

源码位于 Poseidon 仓库，不意味着每个实验都使用 Poseidon evaluator。SEAL CPU 路线不是 Poseidon CPU 后端，也不是 GPU 执行的模拟通过结果。

### 3.2 修改分类

| 类别 | 代表位置 | CPU 主线处理建议 |
|---|---|---|
| Agent、provider、模型/reference、检查器、修复闭环、报告 | `scripts/baseline/` 中相应模块 | 保留；该目录也包含历史/GPU 辅助工具，不应整体无差别归类 |
| Golden、正反例、语义规则、审计与回归文档 | `scripts/baseline/`、`docs/baseline/` | 保留与 CPU 支持集合有关的内容和历史证据 |
| Dacapo frontend/compiler/helper 修复 | 下文四个 Dacapo 文件 | 对当前受支持功能属于必要依赖，不能整体回退 |
| 编译器和 SEAL 环境配置 | `src/poseidon/tools/dacapo/` 的 Nix、lock 文件及相应构建脚本 | 保留并通用化，不因目录名含 Poseidon 就删除 |
| Poseidon HEVM 常量编码与执行计划适配增量 | `src/poseidon/frontends/dacapo/hevm_plaintext_encoding.*`、`hevm_static_execution_plan.*` | 不属于 SEAL CPU 运行必需路径，可单独归档或保留为可选模块 |
| DropModulus、GPU schedule/backend 对接增量 | `src/poseidon/mgpu/` 中相应差异 | 可退出 CPU 主线；不等于删除原分支已有 mgpu 代码 |
| Poseidon 专用测试和例程增量 | `examples/ckks/`、Poseidon frontend/mgpu 测试 | 分离为后端专项验证，不计入 Agent CPU 必需依赖 |

“退出主线”优先指不构建、不调用、不阻塞 CPU 验证，不要求立即物理删除。最终可回退的 hunk 清单需要依赖搜索与回归验证，不能只按文件路径决定。

### 3.3 不能误删的 Dacapo 修改

| 文件（相对于 `third_party/dacapo`） | 当前差异的作用 | 回退风险 |
|---|---|---|
| `python/hecate/hecate/expr.py` | 修正增强赋值方向、容器一元运算写回，增加装饰函数调用与返回容器处理、trace 状态检查 | 如 `x -= y` 恢复为反向操作，或函数/容器构造无法按当前约定 trace |
| `tools/frontend.cpp` | 区分明密文参数；修正公开常量减法；保存嵌套 insertion point；静态内联函数体 | 正确 DSL 可能形成错误类型/运算的 IR，或丢失当前函数调用支持 |
| `lib/Dialect/Earth/Transforms/Common.cpp` | scale 调整后同步最终 block argument 类型与函数签名 | 函数类型和 scale 元数据可能不一致 |
| `python/poly/poly/MPCB.py` | concat mask、尾部 carry 和分支 packing 兼容性检查修复 | 相应高层 helper 的布局或数值结果可能错误 |

这些是 frontend/compiler/helper 层的修复或功能扩展，不是 Poseidon GPU adapter 修复。并非每个简单模型都用到每项，但当前完整支持集合可能依赖它们。

如果坚持使用完全未修改的 upstream Dacapo，必须缩小允许的 DSL 子集，或提供经过验证的替代 lowering，并重跑受影响的测试。不能恢复原码后继续沿用此前全部覆盖声明。

### 3.4 拆分原则与验收

建议将增量逻辑分为：

1. Agent/语义规则/验证与实验框架。
2. Dacapo 必要修复及对应最小回归。
3. 双 Linux 平台环境配置。
4. 可选 Poseidon 后端对接与专用测试。

这只是建议的变更组织方式，不授权创建、切换分支或提交。

真正回退 GPU 增量之前，应：

- 保存精确基线、tracked 差异、未跟踪文件及 submodule 内补丁；不得仅依赖 `git diff` 或 `git bundle`。
- 检查 CPU 入口和导入链是否意外引用拟移除内容，包括构建配置、文档引用和测试发现入口。
- 在不调用 Poseidon adapter/backend 的条件下运行 Hecate trace、编译、真实 SEAL 密态执行与差分回归。
- 使用正确与刻意错误的 DSL 案例，确认比较器不仅能通过正确程序，也能拒绝错误程序。
- 保留原始误差门限、输入、reference、安全参数，不以模拟、降低参数或放宽门限替代验证。

CPU 路线成立的结论应限定为“已支持模型和 DSL 子集，在指定编译器及 SEAL CPU 后端上通过测试”，不代表形式化等价证明、全部 DSL 覆盖、Poseidon GPU 通过或真实 bootstrap 通过。

## 4. 问题二：路径和架构障碍能否通用化？

### 4.1 目标与非目标

目标：一份源码、两份平台配置，分别在 WSL x86_64 Linux 和 Mac 上的 aarch64 Linux 虚拟机运行。Intel Mac 对应 x86_64 Linux 配置。

非目标：复用跨架构二进制；在原生 macOS 直接运行 Linux 沙箱；为迁移更换算法、安全参数或 reference；把旧实验的绝对路径和哈希改写成新机器的值。

### 4.2 已确认障碍与处理方式

| 障碍 | 源码证据 | 建议改造 |
|---|---|---|
| 源码和工作目录固定 | `scripts/baseline/continue_dacapo_cpp.py` 的 `ROOT`、`WORK` | 源码根由模块位置推导；工作目录统一配置，可显式覆盖 |
| 启动器/cache 路径固定用户名 | `scripts/baseline/nix_portable.sh` | 统一路径解析并完整引用带空格路径，不修改历史证据 |
| 平台锁定为 x86 | `dependency-lock.json` 的 `system` | 公共源码锁与平台配置分离，只允许已定义平台，未知平台明确失败 |
| LLVM 仅构建 X86 target | `dacapo-dependencies.nix` | 按宿主平台及实际代码生成需求选择 X86/AArch64；LLVM target 不等于宿主 CPU 架构，不能机械替换 |
| Python 二进制发行物为 x86 | `python-wheels.lock.json` 的 `cp310-linux-x86_64` | 按架构锁定 wheel 版本、来源和 SHA-256；缺失 ARM 发行物时显式阻断并评估源码构建/版本方案 |
| Nix launcher 指纹固定 x86 资产 | `scripts/baseline/nix_portable.sh` | 为每个平台选择经核验的 launcher 或标准 Nix 路径，不复用错误架构的哈希 |
| 本机 Nix store 源码位置固定 | `dependency-lock.json` 的 `bundled_path` | 从验证过的依赖定位结果或显式配置取路径，保留 revision/content hash 校验 |
| 构建与运行库绑定本机 | CMake cache、Python 环境、`.so` 文件 | 按平台、版本和补丁标识独立构建；不复制旧 venv/二进制作为 ARM 环境 |
| 沙箱路径与依赖绑定 | `scripts/baseline/candidate_sandbox.py` 等 | Linux 隔离机制保留；受控解析实际工具和库路径，重新验证隔离 |

上述是已发现的主要障碍，不是对所有源码、汇编和第三方依赖的完整 ARM 审计。实施时还需搜索 CPU intrinsic、架构判断、ABI 假设和绝对路径引用。

### 4.3 建议的配置接口（尚未实现）

可以集中定义 `PROJECT_ROOT`、`WORK_ROOT`、`DACAPO_ROOT`、`BUILD_ROOT`、`RESULTS_ROOT` 和平台标识，供所有入口共用，而不是每个脚本各写一套推导。

- 配置优先级和默认值必须明确；旧环境默认仍能定位当前 checkout 和原有工作目录。
- 构建目录应包含平台/配置标识，禁止 x86 与 ARM 共用 CMake cache、锁文件和动态库输出目录。
- 运行报告记录实际平台、工具版本、源码/补丁指纹、参数和结果，不依赖目录名推断可信状态。
- 继续从单一本地 `.env` 管理 provider 凭据，禁止写入报告、日志、Git 或公共备份。
- Linux 进程隔离不为跨平台方便而关闭；沙箱不可用则 fail closed。
- 代理不固定为某台 Windows 的 gateway 或端口；连接配置与程序语义分离。

建议继续固定 Dacapo 基线、LLVM/MLIR 18.1.2 和 SEAL 4.0.0，先尝试保持 Python 3.10 与现有 Python 包版本。ARM 资产的可用性尚待核验；任何版本调整都应形成明确变更和回归证据。

### 4.4 分阶段实施与退出条件

| 阶段 | 内容 | 通过条件 |
|---|---|---|
| P0 | 只整理边界与路径清单 | 无源码删除、无基线变化；本文属于该阶段 |
| P1 | 集中路径解析，保留 x86 默认行为 | 路径含空格、不同用户名、显式覆盖等单测通过；现有 CPU 回归不退化 |
| P2 | 公共依赖锁与平台锁分离 | x86 配置仍可复现；ARM 依赖来源、版本、校验和及下载预算明确 |
| P3 | 在 ARM Linux 构建 compiler/runtime 与 Python 环境 | 工具版本、动态库加载、sandbox、最小 trace/compile 通过 |
| P4 | ARM 真实密态差分 | 加法/仿射、Linear 跨元素归约、短多项式 MLP 及错误反例通过相应门禁 |
| P5 | 复用已保存 Agent 程序进行回归 | 所有适用案例有明确分母和结果；不支持项独立报告 |
| P6 | 后续真实 Agent 生成测试 | 前述环境门禁通过且另获付费授权，不用付费批次诊断基础安装问题 |

大型 LLVM/MLIR 构建最多 `-j2`，内存不足时降低并发。API 请求并发与密态执行并发分别控制。

原有逐项数值门限保持 `abs(actual-reference) <= 1e-5 + 1e-4 * abs(reference)`，若具体既有案例有单独冻结门限则遵从其记录，不在迁移过程中擅自放宽。记录 MAE、最大绝对误差、适用的相对误差和逐项结果。随机加密及跨架构数值差异不要求密文或结果逐 bit 相同。

## 5. 迁移证据与授权边界

- 未跟踪的代码、文档、测试，以及 Dacapo 的未提交修改都必须纳入迁移清单。
- 保存付费 Agent 响应、修复历史、固定权重/输入、编译产物指纹、解密结果和报告；它们不是可以随便删除的缓存。
- 保留历史报告原件，通过路径映射或新运行记录解释新位置，不批量重写旧报告。
- 不复制旧 x86 build/venv 充当新 ARM 环境；可再生依赖与不可丢失证据分开备份。
- 安装、sudo、大型下载、代码回退和付费测试需各自符合用户授权；文档整理不隐含这些授权。
- 不自动 commit、push、创建 PR、切换分支、关闭 sparse checkout 或修改现有后台任务。

## 6. 本次交付状态

**已完成**：只读核查与本文新增；将 CPU 主线、Dacapo 必要修改、可选 Poseidon 后端增量和平台移植事项分开说明。

**未完成/未执行**：代码删除或回退、路径重构、ARM 依赖锁生成、Mac 安装、ARM 构建、跨平台端到端测试、付费 API 调用。

因此，本文是后续决策与实施验收依据，不是跨平台功能已经完成的声明。
