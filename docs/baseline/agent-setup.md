# Agent 依赖与构建

本入口为 Ubuntu x86_64 / ARM64 构建 Hecate、Dacapo 和 SEAL CPU 后端。运行示例见 [Agent README](../../scripts/README.md)。

## 前置环境

Ubuntu 22.04、Python >=3.10、bash、git、curl、GNU timeout、flock、sha256sum、bubblewrap。Git 和 bubblewrap 需位于 `/usr/bin/git`、`/usr/bin/bwrap`。bubblewrap 必须允许创建用户和网络命名空间。

安装器只检查系统依赖，不运行 sudo 或 apt。

## 锁定依赖

| 组件 | 版本 / 清单 |
|---|---|
| Nix 启动器、平台 | [platform-profiles.json](../../scripts/baseline/platform-profiles.json) |
| GCC / LLVM / MLIR | 13.2.0 / 18.1.2 / 18.1.2 |
| SEAL / GSL | 4.0.0 / 3.1.0 |
| Python / NumPy | 3.10.14 / 1.25.2 |
| x86 Torch | 2.0.1+cpu |
| ARM Torch | 2.0.1，CPU 版本 |
| Dacapo | 固定 gitlink，按顺序应用 11 个[项目补丁](../../scripts/baseline/patches/README.md) |

总清单为 [agent-requirements.json](../../scripts/baseline/agent-requirements.json)，它引用各平台的 wheel 和原生依赖锁文件。普通 `requirements.txt` 只能覆盖 Python 包，不能构建 LLVM、SEAL 和 Hecate。

默认采用 `hecate` 精简工具链，不另外构建完整 Clang。CPU 架构选择与工具链裁剪是独立配置。

## 命令

从仓库根目录查看计划：

```bash
python3 -B scripts/setup_agent.py --platform auto --plan
python3 -B scripts/setup_agent.py --platform auto --plan --json
```

`auto` 检测 Linux 架构，也可指定 `x86_64-linux` 或 `aarch64-linux`。可以查看另一平台的计划，实际构建必须匹配本机架构。

执行构建并运行 Linear / MLP 离线自测：

```bash
python3 -B scripts/setup_agent.py --platform auto \
  --work-root "$HOME/poseidon-work" \
  --apply --init-submodule \
  --download-budget-mib 3072 --disk-budget-gib 40 --self-test
```

预算数值是示例，按计划与可用空间调整。`--apply` 开始依赖准备；`--init-submodule` 允许初始化固定 Dacapo；`--self-test` 运行不收费的密态自测。

只导出 Python 依赖：

```bash
python3 -B scripts/setup_agent.py --platform aarch64-linux \
  --requirements > requirements-arm64.txt
```

导出内容包含固定 wheel URL 和 SHA-256，针对 CPython 3.10。不要将它直接安装到其他版本的系统 Python。上游 poly helper 的可选 einops 依赖另见 [poly 锁文件](../../src/poseidon/tools/dacapo/poly-python-wheels.lock.json)，不包含在基础 9-wheel 环境中。

## 目录与重复执行

- 基础工作目录默认 `~/poseidon-work`，可用 `--work-root` 或 `POSEIDON_WORK_ROOT` 修改。
- x86 使用基础目录，ARM 使用其下的 `platforms/aarch64-linux`。
- 源码与工作目录必须分开；不同架构不能共用 venv、CMakeCache 或二进制。
- 安装器核验并复用已有缓存。补丁只接受干净状态或已知补丁前缀，不覆盖其他修改。
- 编译最多 2 jobs、链接 1 job；每阶段最多 12 小时。同一工作目录只运行一个安装器。
- 下载预算包含执行期间客体的其他网络接收量；磁盘预算按文件系统增量统计，并保留至少 2 GiB 空闲。预算按采样检测，存在在途超量。
- 出错后保留日志和现场，不自动删除、重试或扩充预算。重跑前检查原因及已用空间。
- 日志位于平台工作目录的 `results/agent-setup-*/`。

现有 ARM 环境已完成实际编译和密态验收。安装器通过了离线测试，但从空白机器运行 `--apply` 的完整安装流程尚未验收。安装器不会读取 API 密钥或调用 DeepSeek。
