# Linux CPU 后端：x86 与 ARM

2026-09-21：ARM64 Ubuntu 原生离线 CPU 后端已通过本页范围的验收。
x86 配置及默认入口保留；本次没有完成新机 x86 原生密态复验。
真实 DeepSeek 生成、完整 Agent 能力覆盖及 Poseidon GPU 均不在本次成功声明内。

## 平台选择

| 项目 | x86（原默认） | ARM（显式选择） |
|---|---|---|
| 平台 / Nix system | `x86_64-linux` | `aarch64-linux` |
| Linux machine | x86_64 / amd64 | aarch64 / arm64 |
| 精简工具链 | GCC 13.2.0、LLVM/MLIR 18.1.2 | 相同版本，已原生构建 |
| LLVM 代码生成目标 | X86 | hecate 为空；full 为 AArch64 |
| SEAL / GSL | 4.0.0 / 3.1.0 | 4.0.0 / 3.1.0 |
| Python / NumPy | 3.10.14 / 1.25.2 | 3.10.14 / 1.25.2 |
| Torch | 精确 `2.0.1+cpu` | 精确 `2.0.1`，实际 cuda/hip 均为 None |
| wheel lock | `python-wheels.lock.json` | `python-wheels-aarch64.lock.json` |
| 有效工作目录 | 基础 work root | 基础 work root 下 `platforms/aarch64-linux` |

`scripts/baseline/platform-profiles.json` 定义两个平台；未配置架构、系统/架构不匹配、
非小端或非 64 位进程均拒绝。实际 ARM 客体已验证接受 ARM、拒绝 x86 误选及未知架构。
`--work-root` / `POSEIDON_WORK_ROOT` 始终指基础目录；不要重复添加平台后缀。
每个平台分别使用 deps、build-dacapo、build-poseidon、venvs、cache、results。
源码和纯 Python 包可以相同，二进制、CMakeCache、venv、动态库不能跨架构复用。

`toolchainProfile=hecate/full` 与平台选择独立。hecate 使用固定 GCC 和精简 SDK，
不重建 Clang。LLVM_TARGETS_TO_BUILD 控制代码生成后端，不决定工具自身的运行架构；
ARM Hecate→HEVM 在空目标列表下已实际编译、链接、运行。full 配置未在 ARM 验收。
公共 dependency-lock.json 的 system/python_stage 保留 x86 历史默认；有效平台和 wheel
锁由平台清单选择，Nix 显式传入 `--argstr platform`，固定源码哈希不变。

## 当前路径与启动

唯一开发及执行 checkout：ARM Ubuntu `/home/lhohy/Code/Poseidon`（ext4），
分支 `lhy-agent-dsl`。源码、测试和文档直接在客体修改，编译验收也在客体完成。
Mac `/Users/lhohy/Projects/Poseidon` 仅为保留的历史副本，不再日常开发；
禁止自动双向同步或以 Mac 副本覆盖客体已有修改。Mac 通过下方 SSH 入口操作。
基础 work root 为
`/home/lhohy/poseidon-work`。UTM 4.7.5 / QEMU HVF，Ubuntu 22.04.5 aarch64，
内核 5.15.0-191-generic，2 vCPU、8 GiB、80 GiB 动态盘；这是 ARM 硬件虚拟化。

从 Mac 进入客体：

```sh
ssh -F '/Users/lhohy/Virtual Machines/Poseidon-arm64-access/ssh_config' poseidon-local-arm64
```

然后在 Ubuntu 内运行：

```sh
cd /home/lhohy/Code/Poseidon
export POSEIDON_WORK_ROOT="$HOME/poseidon-work"
export POSEIDON_PLATFORM=aarch64-linux
python3 -B scripts/agent.py --backend local doctor
python3 -B scripts/agent.py --backend local candidate -- \
  --case scripts/baseline/cases/linear-example.json --self-test
python3 -B scripts/baseline/seal_cpu_golden.py --case mlp4x4x2
```

也可在启动器 command 前加 `--platform aarch64-linux`。x86 Ubuntu 不设平台时仍采用原默认。
本项目当前通过 macOS 的 SSH 连接开发客体；macOS 不能当作 Linux 执行环境。doctor 是前置检查；self-test
使用脚本化响应，二者都不能称为真实 LLM 生成。密钥只放执行端 `.env`，本次未复制或使用密钥。

## 构建与依赖

新用户入口：`python3 -B scripts/setup_agent.py --platform auto --plan`。
按架构选择锁定依赖、requirements 导出、批准后的准备与构建见
[依赖准备说明](agent-setup.md)。新入口目前通过离线测试，尚未从空白环境完整执行。

ARM launcher 完整大小 74,671,948 bytes，SHA-256
`af41d8defdb9fa17ee361220ee05a0c758d3e6231384a3f969a314f9133744ea`；
实际 Nix 2.20.6 / aarch64-linux 已验证。此 hash 是 HTTPS 发布资产的本地完整指纹，
不是发布者签名。固定 nixpkgs revision/NAR hash 已通过实际求值校验。

ARM 9 个 wheels 共 96,753,992 bytes，完整哈希、离线解析安装、pip check、精确依赖集合、
精确导入版本及实际映射库已通过。Torch ARM 2.0.1 的官方来源与 x86 `2.0.1+cpu` 分开锁定；
没有宽泛匹配版本。bubblewrap 为 Ubuntu 0.6.1-1ubuntu0.3。

Dacapo 保持 gitlink `4616402710f39df3e5f5bd7930a6c036025aaac3`。
按 [补丁说明](../../scripts/baseline/patches/README.md) 应用原六补丁和五个构建/兼容修复，
每个先 `apply --check`。最终 11 补丁从干净固定版本重放后与实际 14 个修改文件逐字节一致。
新增修复仅涉及必要依赖/声明、SEAL consumer 的异常选项、四处 LLVM 名字限定，以及
ElideConstant 跨函数输出路径污染；未更改模型 reference、安全参数或数值门限。
编译最多 2 jobs，链接 1 job；LLVM SDK 无需因这些修复重建。

## 本轮实际验收

详见 [验收摘要及证据索引](arm64-acceptance-summary.json)。只计最终选定的一次结果，
不将修复前失败或重跑累加为新案例。

- 最终配置/平台/Nix 离线测试：63 passed、0 failed、0 skipped，包含 mock。
- 真实 CST pass 重用回归：1 passed；原二进制稳定失败于文件名串接，修复后通过。
- GSL 上游测试：16 passed。HEVM C++ ABI、LLVM 消费者链接、实际动态库加载通过。
- 真实密态正例：11 个不同案例、44 组输入、808 个输出值；最大绝对误差
  `1.1502960717280075e-7`，加权 MAE `9.852036505334457e-10`。
- 覆盖 Linear self-test、mlp4x4x2、native 数组转置、双输入 Linear、分块 1×7 Linear、
  packed/native 六项（P8/P16/P32/P64/P128/P256、广播/循环/数组/starred/归约/旋转/卷积代表组合）。
- 5 个数值错误反例在数值层拒绝，另有 1 个格式反例按预期拒绝。
- 真实候选沙箱检查通过：网络/PID 隔离、只读请求、隐藏源码/密钥、清空环境；实际
  RLIMIT_AS=4 GiB、CPU=150/155 s、文件=16 MiB、NOFILE=128、CORE=0 已在沙箱内读回。
- 每个正例都检查 Earth/CKKS、HEVM/CST、真实 SEAL 密态执行、解密和四组输入；始终使用
  `abs(actual-reference) <= 1e-5 + 1e-4*abs(reference)`。原始证据保留在 ARM results。

HEVM probe 使用真实上游头文件，验证大小 24/40/8、对齐 8/8/2、字段偏移、已知字节序列，
并与 Python `<IIQQ`、`<5Q`、`<4H` 对照；库加载前校验 ELF64 小端和 e_machine。
这不是所有输入上的形式化等价证明。x86 本轮验证了配置求值、默认行为和错误架构拒绝，
未在旧模拟 VM 完成新编译器的密态复验。实时结果的配置哈希可能早于最后的验收标记更新；
最终 Linear 复验使用最新清单并核验资源上限。历史记录不改写。

## 接续边界

完整迁移过程及失败记录见 [迁移接续](mac-migration-handoff.md)。没有 commit、push、PR 或付费调用。
后续真实 DeepSeek 生成仍需单独预算批准；不能将这批人工/确定性离线成功称为 Agent 在线成功。
旧 x86 VM、专用访问目录和两个迁移缓存已按用户授权删除，清理约 18.2 GiB 旧内容；
源码差异及迁移记录已归档并核验，见 [清理记录](x86-retirement.json)。
ARM VM 和 x86 平台支持源码保留。不要将 CPU 结果外推为 GPU 或原生 x86 性能。
