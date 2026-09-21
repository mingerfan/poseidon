# Mac / Ubuntu 本地迁移接续

## 最新接续：2026-09-21

本次本地提交包含双平台 CPU 适配、固定补丁及统一依赖构建入口；
提交在 Ubuntu 唯一开发 checkout 的 lhy-agent-dsl 分支进行，远程推送需单独授权。
下文验收过程中的“未提交”描述是当时状态，不代表提交后的 Git 状态。

新增统一依赖准备入口 `scripts/setup_agent.py`，仅在 Ubuntu checkout 开发。
使用 `--platform auto/x86_64-linux/aarch64-linux --plan` 检查计划；
显式批准预算后才能 `--apply`。用法及验证边界见 [依赖准备说明](agent-setup.md)。
本次只做入口实现与离线验证，没有新安装、构建、密态执行或付费调用。

ARM64 原生 Ubuntu CPU 离线验收及旧模拟 VM 清理已完成。当前使用说明见
[Linux 双平台入口](linux-platforms.md)，验收见 [完整摘要](arm64-acceptance-summary.json)，
删除清单与归档位置见 [旧 VM 清理记录](x86-retirement.json)。下文保留历史过程，
其中旧 PID、运行中/待批准/待验收描述不能作为当前状态。

- 2026-09-21 用户指定唯一开发 checkout：ARM Ubuntu `/home/lhohy/Code/Poseidon`。
  后续源码、测试、文档修改及编译验收均在此完成；Mac 通过 SSH 操作客体。
  Mac `/Users/lhohy/Projects/Poseidon` 仅保留为历史副本，不再开发，不自动双向同步或覆盖客体。
  切换前核验两端 51 个修改/未跟踪项目文件的 SHA-256 一致，客体无活动构建/验收批次。
  客体分支 `lhy-agent-dsl`，跟踪 `origin/lhy-agent-dsl`，HEAD 为
  `534eae4bca450a2d4101abc9abd40d8feb50650f`；全部已有修改保留。
  基础 work root：`/home/lhohy/poseidon-work`，ARM 有效目录追加 `platforms/aarch64-linux`。
  commit、push、PR 分别需要用户授权；付费调用及新安装仍需另行批准。
- 入口显式 `--platform aarch64-linux` 或 `POSEIDON_PLATFORM=aarch64-linux`；原 x86 默认保留。
  固定 GCC 13.2.0 / LLVM/MLIR 18.1.2 / SEAL 4.0.0 / GSL 3.1.0 /
  Python 3.10.14 / NumPy 1.25.2 / ARM Torch 2.0.1 已实际加载验证。
- 最终配置与 Nix 测试 63 passed / 0 failed / 0 skipped；真实 CST 回归 1 passed。
  11 个不同密态正例、44 组输入、808 个输出值通过；5 个数值反例和 1 个格式反例正确拒绝。
  最大绝对误差 `1.1502960717280075e-7`；冻结门限未变。没有付费 API、commit、push 或 PR。
- 两个最小模型不足以代表全部能力；额外验证了 packed/native 六项、对象数组、双输入、
  分块输入和相关反例。真实 DeepSeek 生成、全部 Agent 语法/模型、完整 Clang profile、GPU 未验收。
  x86 本轮仅通过默认配置/求值/门禁回归，未完成旧模拟器的新二进制 E2E。
- Dacapo 仍固定原 gitlink。现在按 patches/README.md 顺序应用 11 个补丁；新增内容有
  GCC/精简链接兼容和跨函数 CST 路径修复。干净重放与实际 14 个上游文件一致。
- 所有批次已结束。修复期间仅在失败停止边界更新源码，增量复用已完成 SDK；失败日志保留。
  旧 x86 guest 的 148 项记录/源码差异已归档并逐文件核验，不含凭据；约 18.2 GiB 旧内容已清理。
- 下一条命令：通过 `Poseidon-arm64-access/ssh_config` 登录后，在执行源码目录运行
  `python3 -B scripts/agent.py --backend local --platform aarch64-linux doctor`。
  下一次真实 DeepSeek Linear 仍需单独确认当前接口、外发内容和费用上限；不能自动收费。

---

更新：2026-09-20。本文记录新机实测状态，不替代当前输入契约或历史实验原始证据。

## 路径和环境

- Mac checkout：`/Users/lhohy/Projects/Poseidon`。
- 执行端 checkout：`/home/lhohy/Code/Poseidon`，位于客体原生 ext4 文件系统。
- 执行端 work root：`/home/lhohy/poseidon-work`；已创建 `deps`、`build-poseidon`、`build-dacapo`、`venvs`、`cache`、`results`。
- 主机：Apple Silicon M5、24 GiB 内存、macOS 27.0；UTM 4.7.5。
- 路线 A：QEMU x86_64 模拟；不是 ARM64 原生客体或 Rosetta。
- 客体：Ubuntu Server 22.04.5 x86_64，内核 `5.15.0-119-generic`，1 vCPU、8 GiB 内存、80 GiB 动态磁盘、4 GiB swap。安装后根分区约 68 GiB 可用，空间会变化。
- 已验证移除活动安装介质后从 `/dev/vda2` 启动、SSH 密钥登录、AppArmor 服务和用户命名空间探针。SSH 转发仅绑定主机 `127.0.0.1:22222`；无宿主目录共享。
- 系统工具：Python 3.10.12、Git 2.34.1。系统 Python 不是已经恢复的锁定 tracing 环境。

## 源码同步与 Git

- 两端分支均为 `lhy-agent-dsl`，HEAD 为 `534eae4bca450a2d4101abc9abd40d8feb50650f`，跟踪 `origin/lhy-agent-dsl`。
- origin：`https://github.com/mingerfan/poseidon.git`；fetch refspec 仅包含该开发分支。
- 客体由 Mac checkout 的完整分支 Git bundle 克隆；未重新获取 GitHub 远程 HEAD。该 SHA 表示本次同步的源码状态，不保证未来远程不变。
- bundle 大小 21,489,259 字节，SHA-256：`163ceec8b1539929298ecd968f320d78e0428ef051c7404851853b4e61adf1c2`，客体校验通过。
- Dacapo gitlink：`4616402710f39df3e5f5bd7930a6c036025aaac3`。客体已初始化并按顺序应用六补丁；Mac checkout 的子模块仍未初始化。主仓库 gitlink 和 `.gitmodules` 未改。
- 未复制 `.env`、私钥、旧 venv、旧构建或旧实验 results。用户确认持有 API 密钥，并明确决定不保留旧 `poseidon-work/results` 原始实验记录。仓库中的旧报告只能作为历史报告引用，不能声称已在新机复核其原始证据。用户选择不保留旧机未提交的 GPU 修改，本次未操作或删除旧机文件。
- 本文是未提交的接续记录；未执行 commit、push、PR。

## 本机验证

2026-09-20 10:25 UTC，在 Ubuntu checkout、`POSEIDON_WORK_ROOT=/home/lhohy/poseidon-work` 下运行：

```sh
python3 -B -m unittest discover -s scripts/baseline -p test_portability.py -v
python3 -B scripts/agent.py --backend local --work-root /home/lhohy/poseidon-work doctor
```

- portability：12 passed、0 failed、0 skipped，退出码 0；unittest 报告 0.929 秒。覆盖路径、启动参数与门禁逻辑，包含 mock/平台模拟；不构成真实 SSH、编译或密态执行证据。真实 SSH 另行验证通过。
- doctor：退出码 2。客体平台和路径正确；缺少 Dacapo frontend、nix-portable、tracing venv、hecate-opt、SEAL metadata 动态库及 `bwrap`。这是依赖层未恢复，不是数值正确性失败。
- 原始 stdout、stderr 与 `checks.json` 保存在客体 `results/migration-2026-09-20/`（相对于 work root）。
- 本轮无编译、无密态执行或解密、无付费 API、无依赖安装。不得把上述检查计入历史真实 Agent 成功数。

后续只读准备：`test_environment_gate` 与 `test_nix_dependency_plan.DependencyLockTests` 共 11 passed、0 failed、0 skipped（0.028 秒），验证依赖检查逻辑与锁文件，包含 mock，不是 Nix 实际求值或原生库加载。输出保存在同一 results 目录的 `dependency-guards.*`。尚不具备 Nix 条件的测试类没有执行。

## 已批准并完成的基础环境步骤

用户于 2026-09-20 单独批准此阶段，以下步骤已完成；不代表完整编译链恢复：

- 在客体初始化固定 Dacapo gitlink，使用 HTTPS，保持主仓库 gitlink 和 `.gitmodules` 不变；按既定顺序逐项 `apply --check` 后应用六个补丁，发现已有修改即停止检查。
- 下载并校验 nix-portable v012 x86_64 到 work root 的 `deps/nix-portable-v012/`。官方 release 元数据大小为 68,062,412 字节；必须符合现有 wrapper 的 SHA-256 `b409c55904c909ac3aeda3fb1253319f86a89ddd1ba31a5dec33d4a06414c72a`，不接受自动更新哈希。
- 仅在 Ubuntu 客体以 sudo 安装 bubblewrap 0.6.1-1 到系统包位置。现有 Ubuntu 包索引记录包大小 46,028 字节、安装大小 129 KiB；只读安装模拟为新增 1 包、升级 0 包。正式安装前仍需核验版本和包哈希，来源不可用时不静默换版。
- 本阶段批准下载上限 512 MiB、客体磁盘预留 2 GiB。可见成功/失败下载载荷约 384 MiB；Git 协议开销未独立精确计量，预算按约 410 MiB 保守记账。Dacapo 的 GitHub size 字段不是实际网络传输量。未下载 LLVM 或 Torch。
- 验证固定提交与补丁、bubblewrap 隔离、离线 Nix 版本/路径探针、启用 sandbox 的最小构建，然后只做固定编译依赖 dry-run。完整工具链预计需要较长跨架构模拟构建；其预算必须依据新机 dry-run 另行批准。
- 不安装 Mac 系统包，不改全局 shell/代理配置，不初始化其他子模块，不运行付费 API。

### 新机实际结果与故障记录

- Dacapo：固定提交与完整对象校验通过；六次 `apply --check` 和应用均成功。仅四个上游文件变更，共 180 行新增、33 行删除，`git diff --check` 通过。差异与逐补丁哈希记录在 `dacapo-applied.diff`、`dacapo-patches.json`。
- 下载修复：客体 GitHub 传输发生低速超时/TLS 失败（接口接收 68,969,873 字节）；Mac 备用下载又因本次设置过于保守的 200 MiB 单文件保护中止。保留的完整 Git 对象被复用，仅补齐剩余对象，未接受缺失对象或改变提交。Mac 打包附带的 120 个 AppleDouble `._` 文件（19,560 字节）导致客体首次 fsck 失败；这些文件已移入 cache 下的 `macos-transfer-metadata` 保留，随后 fsck 通过。没有删除真实 Git 对象或忽略校验错误。
- 本地克隆外层等待超过 180 秒时，原 Git 子进程仍运行；继续观察同一操作直至完成，没有另起重复克隆。此后核验 checkout 干净，再应用补丁。
- bubblewrap 0.6.1-1：在客体 sudo 安装已校验包，未升级其他包；无网络命名空间和只读根目录探针通过。
- nix-portable：断点续传后完整大小及锁定 SHA-256 通过；离线 Nix 2.20.6、x86_64-linux、bwrap、虚拟 store 与工作路径探针通过。`sandbox=true` 的最小离线构建通过，输出字节核对正确；这不是 Hecate 编译或 FHE 执行。
- 首次依赖求值与 Git 重操作并行时触发旧脚本 90 秒上限。等待 Git 完成后，以仍然有界的 300 秒上限串行重试：元数据求值 76.06 秒、dry-run 17.15 秒，均退出 0。原有脚本和安全门禁未改。
- `test_nix_dependency_plan` 与 `PythonWheelLockTests`：14 passed、0 failed、1 skipped，25.049 秒。跳过的是尚未构建完整依赖的 shell 加载测试；不计为通过。报告为 `bootstrap-validation.*`。
- 新 doctor 仍退出 2，但缺项已缩减为 tracing venv、hecate-opt、SEAL metadata 动态库。报告为 `doctor-after-bootstrap.*`。
- 客体统计的依赖、迁移缓存、Dacapo checkout 与 Git 对象共约 1.63 GiB，在本阶段 2 GiB 预留内；根分区约 66 GiB 可用。原始用量记录为 `bootstrap-disk-usage.txt`。
- 客体主仓库当前差异为 Dacapo 的四文件补丁和本文；没有提交、推送、付费调用、Hecate 编译、密态执行或解密。

## 已批准的完整 CPU 编译链与离线验收（执行中）

用户已单独批准本阶段的下载、构建与离线验收；不包含付费 API。

### 当前构建范围：Hecate 专用 SDK

用户追加要求尽快简化 5,791 步的默认构建。已主动取消旧宽范围构建
（退出码 1，日志明确为用户中断，不是编译失败），保留原日志和
`cache/nix-portable-tmp/nix-build-poseidon-llvm-mlir-clang-18.1.2.drv-0`。
SEAL 安装产物、源码缓存和已校验 Python wheel 均复用。

- 默认 Nix `toolchainProfile` 改为 `hecate`，通过上游 distribution 目标只构建、安装
  `hecate-toolchain-components.json` 中的开发 SDK。`full` 配置保留供另行需要时使用。
- 初次精简保持 LLVM/MLIR/Clang 18.1.2；随后经用户批准改用已有 Nix GCC 13.2.0，
  仅构建 LLVM/MLIR 18.1.2 SDK。SEAL 4.0.0、全部内容哈希、X86 target、沙箱及编译 2 / 链接 1 保持。
  去掉无关工具、上游测试构建目标、Clang 静态分析器和 ARC 迁移工具；项目的实际
  编译、tracing、密态计算、解密及数值验收不减少。
- 额外补丁 `dacapo-minimal-mlir-link.patch` 已逐项 check 后应用，只修改
  `tools/CMakeLists.txt`，让已注册的 Earth/CKKS、Func/Tensor 路线使用明确的 MLIR 库。
  原六个语义补丁保持不变。Hecate 启动构建显式选择该链接模式。
- 使用实际必需的 `mlir-tblgen` 替代通用 `mlir-opt` 的版本探针，仍要求精确 18.1.2。
  不把版本探针作为真实编译或密态执行证据。
- 新配置已完成 CMake configure/generate，实际以 `ninja -j2 distribution` 开始编译，
  总任务数为 **3,330**，较原 5,791 步减少约 **42%**。这已取代此前 3,544 步的估计；
  步数减少不等于耗时按相同比例减少，也不代表编译、安装或密态验收完成。
- 新 Nix 离线求值和 dry-run 通过；无新增下载计划，仅有 LLVM SDK 及 wrapper 两项构建。
  环境/锁文件检查 11 passed、0 failed、0 skipped（含 mock，非密态测试）。
  证据为 `lean-toolchain-metadata.*`、`lean-toolchain-dry-run.*`、`lean-environment-guards.*`。
- 精简构建使用 v3 supervisor，沿用原下载/磁盘基线与原 LLVM 阶段的 12 小时截止时间，
  不因更换配置重置预算或时限。没有直接复制旧对象到新 Nix 输出路径。
- 第一次精简配置的 CMake configure/generate 已成功，但自定义 buildPhase 导致 Nix
  选用 Makefiles，随后 Ninja 找不到构建文件，阶段退出 1，未进入 C++ 编译。
  已改回 Nix 原有 Ninja hooks，仅通过 `ninjaFlags` 和 `installTargets` 选择 distribution；
  新求值与 dry-run 通过，无新增下载。失败现场保留在同名 `drv-1` 目录。
  修复后已在真实 CMake 进程中核对 `-GNinja`，并成功进入精简构建；证据为
  `lean-ninja-active-configure.json` 和 `lean-toolchain-build-plan-verified.json`。
- 用户在 GCC 方案确认问题之后回复“继续减少构建步数”，批准复用已有 Nix GCC 13.2.0。
  已取消保留 Clang 的 3,330 步构建并保留现场，默认 SDK 不再构建 Clang。
  新组合仍须通过 C++ 链接探针、Hecate 编译及完整密态数值验收。新配置实际
  Ninja 总任务数已确认 **1,587**，相对原 5,791 步减少约 **73%**；已开始 C++ 编译，
  未完成安装或密态验收。没有新增下载授权或重置原预算、截止时间。
  `gcc-lean-build-plan-verified.json` 记录实际 Ninja 进程、CMakeCache 哈希、88 个
  SDK 组件、MLIR-only/X86/Release 配置与编译 2 / 链接 1 的真实规则。
- GCC 路线离线检查：17 passed、0 failed、1 skipped（SDK 未建成，跳过 shell 加载）；
  已有 Nix GCC 的实际版本输出为 13.2.0。最终 dry-run 仅需 1 项 LLVM/MLIR SDK 构建，
  不新增下载。显式选择 `compiler.out`，避免请求 GCC 的未安装手册输出。
  证据：`gcc-lean-tests.*`、`gcc-compiler-version.*`、`gcc-lean-out-dry-run.*`。
  GCC 方案已由 supervisor 52856 启动，原阶段截止 epoch 1789951254 保持。
- 当前生成的 CMake 导出含 84 个目标，LLVM/MLIR 链接依赖及 SDK 根目标无缺项；
  证据 `gcc-lean-export-closure-verified.json`，仅为导出依赖检查，不能代替实际链接。
- 环境检查跟随 Nix shell 实际选择的 GCC/Clang 及 wrapper 路径，拒绝未知编译器，
  精确版本分别为 13.2.0 / 18.1.2。相关 13 项检查通过、无失败或跳过（含 mock），
  证据 `selected-compiler-gate-tests.*`。该修复未改变或重启正在运行的 SDK 构建。

以下 dry-run 与 5,791 步记录是精简前的迁移过程记录，不能当作当前精简构建已完成的证据。

新机 dry-run：148 个缓存路径，232.34 MiB 下载、969.38 MiB 解包；另有 9 个 derivation 需构建，包括锁定 LLVM/SEAL 源码。加源码 126.53 MiB，C++ 阶段预计约 358.87 MiB。记录在 `dependency-metadata.*`、`dependency-dry-run.*`；这次是新机实测计划，不是沿用旧机报告。

- 版本：LLVM/MLIR/Clang 18.1.2、SEAL 4.0.0、Microsoft GSL 3.1.0、CMake 3.28.3、Ninja 1.11.1、Python 3.10.14，及固定 Nix 闭包。
- Python tracing：9 个 hash-locked wheels 共 222,195,856 字节（211.90 MiB），包括 NumPy 1.25.2、Torch 2.0.1+cpu。只在 `venvs/hecate-2.0.1-cpu` 创建新环境。
- 已批准新增总下载上限 1 GiB（预计有效载荷约 571 MiB，失败传输计入）、客体新增磁盘预算 40 GiB。下载或空间超范围即停止重新评估；不自动扩盘或删除旧证据。
- Nix store 继续位于 `deps/nix-portable-v012`，Dacapo/helper 构建位于 `build-dacapo`，缓存及日志位于 `cache`、`results`。不向 Mac 或 Ubuntu 系统目录安装编译工具链。
- Nix 同时构建最多 1 项，编译最多 2 jobs、链接 1 job，可按内存进一步降低；每个大型构建阶段最长 12 小时，触发上限后保留诊断并重新评估，不承诺模拟环境内一定可完成。
- 验收包括实际工具版本/库加载、Hecate frontend tracing、doctor、无收费 Linear self-test 与 Linear → square → Linear golden，检查完整 IR/HEVM/CST、真实 SEAL 密态执行、解密和冻结门限。
- 此计划不包含真实 DeepSeek 调用、GPU/CUDA、commit/push/PR。

### 批量继续方式（已启动）

用户要求批量进行，已将后续已批准的离线命令串成顺序批次；不逐条等待聊天确认。
当前批次控制进程为 65502，等待原工具链监督进程 52856 成功结束后，依次执行：
Nix 产物完整性检查 → shell → Hecate → 锁定 Python 环境 → SEAL helpers →
doctor → Linear self-test → 单个 `mlp4x4x2` golden（真正的 Linear → square → Linear）。

- 继续使用同一下载/磁盘预算台账、编译 2 / 链接 1 限制和各阶段硬超时；
  不重启当前工具链，不调用付费 API，不提交或推送。
- 任一阶段失败、预算超限或已固定源码/补丁发生变化即停止，保留日志，不自动重试。
- 客体 `results/migration-2026-09-20/approved-cpu-batch-plan-v2.json` 固定 793 个相关源码文件哈希、
  Dacapo 补丁差异哈希及监督器哈希；`approved-cpu-batch-status.json` 记录阶段结果。
- 可再生成的本机执行辅助脚本位于 work root 的 `cache/migration-bootstrap/`，
  名为 `approved_cpu_batch_v2.py` 与 `approved_cpu_supervisor_v5.py`；它们串联现有项目入口。
- 批次最后只记录 `commands_completed_pending_evidence_audit`，仍需审查真实 tracing、IR、
  HEVM/CST、密态执行、解密数值、skip 与版本证据，再依既有策略处理本批次临时密钥。
- 恢复会话时先核验控制进程及当前阶段进程的实际身份；批次运行时不要重复手工启动阶段。

## 下一步与批准边界

1. 保留已初始化且打补丁的客体 Dacapo，不重打补丁，不清理现有差异。
2. 按已批准的固定依赖方案安装、构建和验收；版本、哈希与隔离不得静默替换。
3. 恢复后重跑上面的 doctor，再运行 Linear `--self-test`，随后验收 Linear → square → Linear。当前尚不具备运行条件。
4. 付费 Linear 和其他真实 Agent 批次须另行批准；历史测试结果尚未在新机复现。

当前执行记录：

- 源码阶段第一次退出 1：LLVM 下载触及 600 秒上限，保留 71,382,336 / 132,060,436 字节；本阶段当时网络总接收 93,823,540 字节。确认原进程退出后续传，36 秒完成，完整大小及原锁定 SHA-256 均通过。Nix 导入和源码阶段复核退出 0；证据为 `llvm-source-verified.json`、`llvm-source-nix-import-result.json` 和 `approved-cpu-sources-*-result.json`。
- SEAL 阶段已完成 GSL 3.1.0：上游 CTest 16 passed、0 failed、0 skipped，安装版本文件核对通过，证据为 `gsl-build-tests-verified.json`。Zstd 1.5.5 的锁定 Nix 配方指定 `playTests` 也已通过（1/1，171.16 秒）；已核实静态库 `libzstd.a` 存在，证据为 `zstd-build-check-verified.json`。其他上游测试没有被该配方选择，不能称为全部 Zstd 测试通过。SEAL 4.0.0 本体已完成全部 40 步、安装检查与 `nix-store --verify-path`，阶段退出 0，累计耗时 3072 秒（含 GSL/Zstd）。静态库 SHA-256 为 `cc4a3eba9f08a76b18c8aa8190afb29881e0ad375a8040b4ce0e88a85b319314`；证据为 `seal-build-verified.json`。LLVM 阶段最新 dry-run 尚需 15 个缓存路径（31.09 MiB 下载、92.49 MiB 解包）及两项构建；已启动固定 LLVM/MLIR/Clang 18.1.2 构建。使用 1 core 保证链接并发不超过 1；后续 LLVM 仍是编译 2 / 链接 1。Nix 显式启用 sandbox；尚无 Hecate/FHE 验收结果。
- 9 个 Python wheel 已缓存并逐一通过大小及 SHA-256 检查，总计 222,195,856 字节；证据为 `python-wheel-cache-verified.json`。下载耗时 152 秒；尚未创建 venv、安装包或执行 tracing。
- LLVM 配置与生成已完成，进入实际 C++ 编译（Ninja 共 5,791 步）。实际 CMakeCache 确认为 Release、`clang;mlir`、X86 target、编译并发 2 / 链接并发 1；这只是正在构建的配置证据，尚未通过安装和工具版本验收。该阶段开始约 11 分钟时，累计网络接收约 501 MiB、磁盘增长约 3.1 GiB，均在批准预算内。
- 状态和日志位于客体 `results/migration-2026-09-20/`，使用 `approved-cpu-*` 及 `approved-python-wheels-*` 文件。恢复会话时核验真实 supervisor/子进程，不凭状态文件重复启动。
- 全阶段共用网络接收和磁盘增长基线，失败传输及本地 SSH 开销保守计入。距 1 GiB 下载上限预留 32 MiB 即停，距 40 GiB 磁盘增长上限预留 1 GiB 即停。源码阶段上限 20 分钟、续传单次 600 秒、SEAL 及后续大型阶段至多 12 小时；超限保留现场重新评估。
- SEAL 的历史 45 分钟监督限制已在本次批准的 12 小时范围内调整，保持同一个构建进程；累计时间从原阶段开始计算，不重新计时。交接记录为 `approved-seal-supervision-handoff.json`。该交接已结束：扩展 guardian 正常恢复原 supervisor 并取得退出码 0，未重启构建；结果记录在 `approved-seal-extended-guardian-result.json`。当前 LLVM 由常规 v2 supervisor 监督。
- 监督器覆盖另起进程组的后代进程；实测清理两个不同进程组中的临时进程，通过且无存活子进程。这是监督器检查，不是编译或密态测试。

只读 doctor 在依赖恢复前预计仍返回 2。不要运行历史 `continue_dacapo_cpp.py --approved` 充当新机安装器，不修改平台门禁、哈希、沙箱或冻结误差门限。

## 从 Mac 发起客体检查

Mac 系统 Python 当前为 3.9.6，不用它建立 tracing 环境。直接通过已验证的专用 SSH 配置调用客体 Python：

```sh
ssh -F '/Users/lhohy/Virtual Machines/Poseidon-x86_64-access/ssh_config' poseidon-local-x86 \
  'cd /home/lhohy/Code/Poseidon && python3 -B scripts/agent.py --backend local --work-root /home/lhohy/poseidon-work --timeout 60 doctor'
```

主机与客体是两份 checkout。现有 Agent 启动器不自动上传源码；修改代码后，应核对两端差异再同步所需源码文件，不覆盖 `.env`、已有修改或 work root。完整工具链和离线验收仍在进行，当前不能把 doctor 或依赖构建当作密态执行成功。


## 恢复后的覆盖工作

- 六项定向 packed/native 定义及贡献检查已在当前提交中：`pn-array-view`、`pn-loop`、`pn-star`、`pn-matrix`、`pn-zero-item`、`pn-scalar`。单候选路径已接入构造检查与真实 tracing 事件验证；`run_agent_batch.py` 尚无对应的专用六项批次参数。它们与已有自由生成 `packed-native-6-manifest.json` 不是同一批测试。
- 当前相关报告没有给出这六项定向构造的同范围真实 Agent 付费通过证据；native-array 十项报告也明确记为付费批次待执行。两者均未在新机运行，不能从人工 fixture 或自由生成六例结果推断覆盖完成。
- 先完成本次 Linear 和 Linear → square → Linear 离线验收，再单独审批真实 Agent 最小恢复；后续覆盖批次另列范围和费用，不自动重跑历史批次。

## 最新范围缩减：仅保留恢复必需的验收

用户再次要求删除非必需内容。原等待批次控制进程 63354 已停止，SDK 监督进程
52856 保持运行，避免丢弃已完成的编译。精简后的 v2 批次（控制进程 65502）已核验存活，
状态为 `waiting_for_toolchain`，完成后自动衔接。

- 人工 golden 从 extended 五例缩为既有 `mlp4x4x2` 一例；新增受允许列表限制的
  `seal_cpu_golden.py --case mlp4x4x2` 入口，仍使用原模型、四组输入、reference、
  参数、opcode 检查及冻结误差门限。
- 当前 SEAL helper 只构建 `seal_golden_keys` 和 `seal_golden_metadata`；省去
  precision probe 及两个 packed helper。后续 schema 5 验收前必须另行构建其 helper，
  本轮不宣称 packed 路线已恢复。
- 省去批次中重复的整套 Nix 计划单测；保留产物完整性、实际工具版本、C++ 链接/库加载、
  doctor 和两项真实密态验收。
- 新补丁 `dacapo-unused-mlir-dependencies.patch` 已在当前修改基础上先 check 再应用：
  去掉三个未使用的 `MLIRLLVMCommonConversion` 直接链接、一个 EmitC 直接链接以及
  frontend/optimizer 的未使用大范围头文件。尚需后续实际 C++ 编译验证。
- 只删这些 SDK 根目标，旧依赖图估计从 1,587 变为 1,572，仅少 15 步；不为这点差异
  重启当前 SDK。Tensor 本身间接依赖 Complex、LLVM IR、分析及调试库；上游 dialect
  库还默认依赖整套 MLIR 头文件生成。进一步砍掉这些需要单独裁剪上游编译器并验证，
  不能把它们直接当作可关闭的 GPU/测试功能。当前未修改 LLVM/MLIR 上游源码。

- 本轮用例选择检查：4 passed、0 failed、0 skipped；这仅验证精简入口，尚未完成真实编译或密态执行。


## 2026-09-21：双架构适配进行中

最新方向：用户明确选择增加原生 ARM64 Ubuntu，同时保留 x86 配置；ARM 新环境需另行批准。
宿主 checkout 是当前开发端；x86 客体 checkout 保持冻结，不同步本轮修改。初次核验其真实
控制进程/SDK 监督进程存活，793 项源码及 Dacapo diff 哈希均匹配，不重启构建。

当前接口：`--platform aarch64-linux` 或 `POSEIDON_PLATFORM=aarch64-linux`；默认仍是 x86。
ARM 有效工作目录为基础 work root 的 `platforms/aarch64-linux` 子目录。配置、wheel 锁、
Nix system、精确 Torch 版本、ELF/HEVM ABI 门禁及使用方法见 [双平台说明](linux-platforms.md)。
ARM launcher hash 暂未建立，门禁保持拒绝执行，不冒充完整支持。没有新增 VM/安装/完整 wheel 下载。

宿主使用已有 Codex Python 3.12.14，仅运行配置及离线单测，不充当锁定 tracing Python。
最新批次 55 passed、0 failed、0 skipped，含 mock/合成 ELF 与 ABI 数据；首次出现一个
macOS `/var`→`/private/var` 路径规范化的测试预期错误，修正测试后通过。`git diff --check` 通过。
ARM 编译、Nix 求值、bubblewrap、实际库加载、密态执行、解密全部未验证。

下一条无安装命令：
`POSEIDON_PLATFORM=aarch64-linux python3 -B scripts/baseline/hecate_python_env.py --plan`

下一执行批次需批准新 ARM VM 和固定依赖恢复的版本/位置/预算；先核验平台与锁，再实际构建、
Linear、mlp4x4x2 和按改动选择的代表回归。未调用付费 API，未 commit/push/PR。
用户要求 ARM 验收完成后清理旧模拟 VM 和相关可再生成内容；当前未满足条件，未删除任何内容。


最新运行状态复核：2026-09-21 本轮末 SSH 22222 连接被拒绝；随后只读 `utmctl list`
确认 `Poseidon-x86_64` 为 **stopped**，宿主进程清单仅见 UTM、无 QEMU 执行进程。
停止原因未知，本轮没有发送停止/删除操作。此前 PID 与 running 状态文件现已过期，不能据此
声称构建仍运行。未重启、未挂载磁盘、未更改客体源码；日后恢复需重新核查构建现场和阶段记录，
不得直接重启旧自动批次或覆盖源文件。

ARM 环境待批准方案：复用 UTM 4.7.5，ARM64 QEMU/HVF 硬件虚拟化，2 vCPU、8 GiB RAM、
80 GiB 动态盘；候选位置 `/Users/lhohy/Virtual Machines/Poseidon-arm64.utm`（已查不存在），
相邻 `Poseidon-arm64-bootstrap` 和 `Poseidon-arm64-access` 也不存在。客体源码计划
`/home/lhohy/Code/Poseidon`，基础工作目录 `/home/lhohy/poseidon-work`，有效 ARM 目录再追加
`platforms/aarch64-linux`。这些路径需本批次批准后才创建。

镜像为 Ubuntu 22.04 LTS 官方 release-20260913 ARM64 cloud image，704,972,800 bytes，
SHA-256 `ab5fcc80611a98bf999018045119d87b3a0e7c78f3b43b254b93d5c22bae3ff6`。
镜像、nix-portable 和 wheels 已知合计 876,398,740 bytes；其余固定 Nix 闭包和必要系统包须
先 dry-run 核算。申请新增网络总上限 3 GiB、客体新增实际磁盘用量 40 GiB（虚拟盘上限 80 GiB），
单个大型阶段最多 12 小时，编译最多 2 / 链接 1。超限保留现场，不自动扩预算。
先核验完整镜像 hash、HVF/aarch64、源码/子模块补丁、启动器 fingerprint 和 bwrap 隔离，
再求值固定依赖并分阶段构建验收。Ubuntu bubblewrap 候选为官方 ARM64 0.6.1-1ubuntu0.3，
与旧客体 0.6.1-1 的差别须在批准和验收中明确记录；未安装。


## ARM 环境批准及首次启动

用户已批准新 ARM 环境、固定依赖与离线验收，下载总上限 3 GiB、客体实际新增用量 40 GiB；
阶段 12 小时、编译 2 / 链接 1 限制保持。清理旧 VM 仍以 ARM 验收完成为条件。
新 VM UUID `7DC8A801-3630-4780-944B-A0306DFCC8B9`，位置为上面的 Poseidon-arm64.utm。
实际 Ubuntu 22.04.5、aarch64、kernel 5.15.0-191、2 CPUs、约 8 GiB RAM，根为 /dev/vda1 ext4，
80 GiB 动态盘已扩容。UTM 启动参数含 HVF；不是 x86 模拟或 Rosetta。
SSH 配置：`/Users/lhohy/Virtual Machines/Poseidon-arm64-access/ssh_config`，别名
`poseidon-local-arm64`，127.0.0.1:22223；专用 known_hosts 指纹与新 VM 串口日志匹配。

Ubuntu 镜像完整官方 hash 已通过。nix-portable v012 ARM 资产完整大小及本地 hash 已写入平台清单，
值为 `af41d8defdb9fa17ee361220ee05a0c758d3e6231384a3f969a314f9133744ea`；它是自解压 Bash，
不能套用普通 ELF 头检查，实际内部 Nix 平台尚待下一阶段核验。
直连镜像较慢，保留约 53 MiB 后经当前 Mac 代理 127.0.0.1:7892 续传成功；没有修改系统代理，
全部传输计入专用 bootstrap/budget.json。新访问配置仅 SSH 密钥登录，不写入项目 .env。
旧 x86 VM 仍停止且磁盘保留。本阶段尚未构建 Hecate，也未执行任何密态计算。

## ARM 原生依赖批次（2026-09-21 01:07 CST）

已在新客体 `/home/lhohy/Code/Poseidon` 从本地 bundle + 工作区 diff + 显式未跟踪文件恢复同一
HEAD/分支，43 个改动文件传输哈希一致；没有同步 `.env`。Dacapo 固定 gitlink 已初始化，
六个语义补丁和两个精简链接补丁均先检查再应用。Ubuntu bubblewrap 0.6.1-1ubuntu0.3、
Nix 2.20.6/aarch64-linux、固定 nixpkgs NAR 与依赖求值已验证。GCC 13.2.0、LLVM/MLIR
18.1.2、SEAL 4.0.0、GSL 3.1.0、Python 3.10.14 是求值结果，不能冒充编译和加载成功。

客体单测 56 passed / 0 failed / 0 skipped，包含合成与 mock；仓库证据为
`docs/baseline/arm64-native-config-tests.json`。首次两处默认 x86 测试受 ARM 环境污染，
在安全停止边界修正测试隔离后重跑通过。首次依赖下载因 Nix cache partial-file 失败；
保留有效缓存和失败日志，连接数改为 2 后做一次有界重试，已开始真实依赖构建。

当前批次目录：`/home/lhohy/poseidon-work/platforms/aarch64-linux/results/arm-build-batch-20260920T170432`。
机器专用控制器：`/home/lhohy/arm-bootstrap/poseidon-arm-build-batch.py`；冻结 8,490 个源码文件，
每阶段重新校验，实时统计宿主已记账下载 + 全部客体接收字节（含本地传输，保守计量）及磁盘用量。
步骤已串联为：配置回归 → 固定依赖 → Hecate/ABI/动态库 → wheels → helper → doctor →
Linear self-test → mlp4x4x2，失败或达到预算即停止。最小链路之后仍需代表性离线覆盖。
不要再次启动相同批次，也不要向运行中的客体同步新源码。控制器 PID 5013 仅为当时观察，
下一条动作是通过专用 SSH 配置读取该目录 `state.json`、对应阶段日志，并用实际进程核对。
宿主文档可以更新；客体执行快照保持冻结。ARM 验收未完成，旧 x86 VM 未删除。
