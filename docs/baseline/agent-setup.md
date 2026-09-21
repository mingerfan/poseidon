# 按架构准备 Agent CPU 依赖与构建

入口用于 Ubuntu Linux 上的 Hecate → Dacapo → SEAL HEVM CPU。
它不构建 Poseidon GPU，也不使用主库旧交叉编译开关。
x86 Ubuntu 原生构建 x86，ARM Ubuntu 原生构建 ARM；Mac 用户先进入 Ubuntu VM。

## 依赖清单

统一入口是 `scripts/setup_agent.py`，总清单是
[agent-requirements.json](../../scripts/baseline/agent-requirements.json)。
它引用已有锁文件，不另建一套不同版本的依赖：

| 层 | 清单与版本 |
|---|---|
| 平台及启动器 | platform-profiles.json；分别锁定 Nix 启动器 URL、大小、SHA-256 |
| C++ | dependency-lock.json + Nix；GCC 13.2.0、LLVM/MLIR 18.1.2、SEAL 4.0.0、GSL 3.1.0 |
| Python | Nix Python 3.10.14；9 个 wheel，锁定平台标签、来源、大小、SHA-256 |
| x86 wheel | python-wheels.lock.json；NumPy 1.25.2、Torch 精确 2.0.1+cpu |
| ARM wheel | python-wheels-aarch64.lock.json；NumPy 1.25.2、Torch 精确 2.0.1，验证 CPU-only |
| Dacapo | 固定 gitlink + 清单顺序中的 11 个补丁 |

LLVM、SEAL、bubblewrap 不是 pip 包，单独 requirements.txt 不能恢复整个后端。
该入口把 Python 清单与原生构建脚本组合起来。

## 新用户步骤

先获取包含这些改动的分支，在 **Ubuntu 项目根目录**操作。
新用户须使用包含 scripts/setup_agent.py 的提交；旧 Agent 发布提交尚不包含此入口。
本地提交与远程推送是独立步骤，需确认远程分支已包含这些文件。

前置条件：Python >=3.10，以及 bash、git、curl、GNU timeout、flock、
sha256sum、Linux bubblewrap。wrapper 要求 /usr/bin/git 与 /usr/bin/bwrap。
Ubuntu 22.04 是已验证环境。系统 Python 只运行启动器，tracing 使用固定 Python 3.10.14。
入口会报告缺项，**不会自动 sudo、apt 或修改系统包**。
bubblewrap 必须支持用户、网络等命名空间；执行阶段先做真实隔离探针。

先查看计划，不创建工作目录、不运行 Nix、不联网：

```sh
python3 -B scripts/setup_agent.py --platform auto --plan
```

也可明确选择 `--platform aarch64-linux` 或 `--platform x86_64-linux`。
自动选择只在这个新入口中生效，原 scripts/agent.py 的 x86 默认不变。
已设置 POSEIDON_PLATFORM 时默认采用该值，显式参数优先。
可以查看另一个平台的计划，但执行必须匹配本机 Linux 架构。
加 `--json` 查看完整下载来源、哈希、版本及阶段。

确认下载和磁盘预算后，再准备依赖并构建，例如：

```sh
python3 -B scripts/setup_agent.py --platform auto \
  --work-root "$HOME/poseidon-work" \
  --apply --init-submodule \
  --download-budget-mib 3072 --disk-budget-gib 40
```

3 GiB / 40 GiB 是示例上限，**不是保证够用，也不代替用户批准**。
--apply 授权所列依赖及构建阶段；--init-submodule 另行授权只初始化固定 Dacapo。
已有子模块可省略后者。入口不 clone 主仓库、不切分支、不提交或推送。
架构与 hecate/full 是独立概念；当前入口固定使用已验证的 hecate 精简 GCC 配置，
未开放 full 配置，不重新构建完整 Clang。

如同一批次也需离线密态验收，在上述命令后加 `--self-test`。
它运行脚本化 Linear 和既有 mlp4x4x2 golden，保留冻结门限，不调用 DeepSeek。
默认只准备和构建环境；doctor 成功不会被计为密态成功。

完成后，运行 Agent 时仍明确选择平台：

```sh
export POSEIDON_WORK_ROOT="$HOME/poseidon-work"
export POSEIDON_PLATFORM=aarch64-linux   # x86 使用 x86_64-linux
python3 -B scripts/agent.py --backend local doctor
```

只导出 Python requirements：

```sh
python3 -B scripts/setup_agent.py --platform aarch64-linux \
  --requirements > requirements-arm64.txt
```

输出固定 wheel URL 与 SHA-256，包含架构和 CPython 3.10 ABI。
它不包含 C++ 依赖，也不应直接安装到系统 Python；统一入口使用固定 Nix Python、
校验 wheel、离线解析依赖并安装到独立 venv。

## 重复执行与安全边界

- work root 使用 --work-root / POSEIDON_WORK_ROOT，默认 ~/poseidon-work。
  x86 保持原目录，ARM 使用其下 platforms/aarch64-linux。
  work root 必须与源码分离，不能互相包含。
- 验证 gitlink 后，在临时目录重放补丁，只接受干净状态或精确的补丁前缀。
  每个待应用补丁先 apply --check；全部已应用时不重打。
  额外修改、冲突、暂存修改或错误版本会停止，不 reset、不覆盖。
- Nix 启动器先校验大小与哈希；固定 nixpkgs 来自对应 v012 bundle，
  求值再次核验 NAR hash。固定源码不可用时停止，不换 revision/hash。
- 记录 Nix metadata/dry-run，依次构建源归档、SEAL、精简 SDK、shell、
  Hecate 和 CPU 密钥/观测辅助库。保留已完成 Nix 输出与 Ninja 增量成果。
- 已有 venv 只验证、不覆盖；不完整时停止。CMake cache 必须匹配原生 Linux
  架构，已有 frontend 路径链接必须正确。不使用历史 checkpoint 续建入口。
- 再次执行会重新核验并复用缓存，不信任阶段状态文件跳过验证。
  每阶段硬超时最多 12 小时；编译 2 jobs、链接 1 job。
  同一 work root 的 setup 有互斥锁；执行时不要另起手工构建或修改共用源码。
  主仓库输入和 Dacapo 源码在阶段边界核对哈希，变化即停止。
- 下载预算保守计入本次执行期间整个客体的非 loopback 接收字节，
  其他程序下载也占预算；磁盘预算计入所在文件系统的使用增量，保留至少 2 GiB 空闲。
  每 0.25 秒采样，达到预算就终止当前阶段；接口计数器变化/重置也停止。
  **这是带采样延迟的停止门禁，不是逐字节硬配额**，可能有在途超量。
  严格硬配额需额外系统级限制；示例预算不能外推到所有机器。
- 失败保留日志、部分下载和构建现场，不自动重试或删除失败产物。
  重新执行开始新的本次预算；失败后先扣除已用预算再批准重试。
  partial 下载需要人工检查，不会盲目覆盖或自动重下。
- 日志与报告在平台工作目录 results/agent-setup-*/。
  不读取 .env、不向子进程传凭据、不调用付费 API。
  只有本批次成功 golden 的中间密钥按已有策略清理；IR、报告、解密数组保留，
  不扫描删除其他批次证据。

## 验证范围

相关离线测试通过；只读核验了现有 ARM 客体的 11 补丁状态与 CMake 架构。
结果见 [入口验证记录](agent-setup-validation.json)。

**本轮没有在空白机器执行 --apply，没有重新下载、安装或编译。**
编排测试使用替身，不等于新安装成功，也不是新增密态执行证据。
现有 ARM 后端历史实机验收见 [双平台说明](linux-platforms.md)。
从空白 Ubuntu 使用这一入口完成全流程，仍需单独有预算验收。
