# Linux 平台

Agent CPU 后端支持以下配置：

| 项目 | x86_64 | ARM64 |
|---|---|---|
| 平台名 | `x86_64-linux` | `aarch64-linux` |
| 环境 | Ubuntu x86_64 | Ubuntu ARM64 |
| 工作目录 | 基础 work root | 基础 work root 下的 `platforms/aarch64-linux` |
| Python / NumPy | 3.10.14 / 1.25.2 | 3.10.14 / 1.25.2 |
| Torch | 2.0.1+cpu | 2.0.1 CPU |
| 原生依赖 | GCC 13.2.0、LLVM/MLIR 18.1.2、SEAL 4.0.0、GSL 3.1.0 | 相同版本 |

## 选择平台

```bash
export POSEIDON_PLATFORM=aarch64-linux
export POSEIDON_WORK_ROOT="$HOME/poseidon-work"
python3 -B scripts/agent.py --backend local doctor
```

x86 使用 `x86_64-linux`；未指定平台时旧运行器仍默认 x86。也可传入 `--platform`，放在 `candidate` / `batch` 之前。

安装器单独支持 `--platform auto`，见[安装说明](agent-setup.md)。

`POSEIDON_WORK_ROOT` 始终是基础目录，ARM 不要手动追加平台后缀。每个平台有独立的 deps、build-dacapo、venvs、cache 和 results。源码可以共享，二进制和构建目录不可跨架构复用。

## Mac 与虚拟机

macOS 不直接运行这套 Linux 后端。Apple Silicon 使用 ARM64 Ubuntu 硬件虚拟化；Intel Mac 使用 x86_64 Ubuntu。源码和构建建议放在客体原生文件系统。

ARM 环境已完成原生编译、沙箱、动态库加载和代表性密态测试。x86 配置保留；近期改动没有重新进行 x86 全量执行验收。GPU、CUDA 和 Metal 不属于这里的 CPU 后端。

## 配置来源

[platform-profiles.json](../../scripts/baseline/platform-profiles.json) 选择平台、Nix 启动器和 wheel 锁文件。系统、架构、字节序、64 位进程和 ELF 动态库均受检查。

`hecate/full` 表示工具链组件范围，不是运行架构。默认 `hecate` 使用 GCC；ARM 不需要通过构建所有 LLVM 代码生成后端来运行 Hecate→HEVM。

历史平台验收数据保留在 [arm64-acceptance-summary.json](arm64-acceptance-summary.json)。
