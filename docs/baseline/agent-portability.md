# Ubuntu Agent 路径与环境迁移

原 x86 默认行为保留，新增显式 aarch64 配置；ARM 已通过限定范围的原生离线 CPU 验收，见[双平台说明](linux-platforms.md)。本文件补充路径、依赖和结果迁移约束；启动命令与接受的输入见 [Agent README](../../scripts/README.md)。

## 源码与工作目录

`workspace_paths.py` 根据源码文件位置确定项目根目录，不依赖用户名或固定 checkout 路径。更换 Ubuntu 用户或源码目录不需要修改 Agent 源码。

- 源码、脚本和待提交修改保存在项目 checkout。
- 可再生依赖、构建和结果默认放在当前用户的 `~/poseidon-work`。
- 其下使用 `deps`、`build-dacapo`、`build-poseidon`、`venvs`、`cache`、`results`。
- ARM 在基础 work root 下追加 `platforms/aarch64-linux` 后使用上述分区，避免与原 x86 路径混用。
- 可通过 `POSEIDON_WORK_ROOT` 或启动器的 `--work-root` 指定另一绝对路径；该选项不自动搬移、下载或安装任何内容。
- 不接受相对工作目录、文件系统根目录或用户主目录本身作为工作根目录；源码和工作路径可以包含空格。

命令从实际项目根目录运行：

```bash
python3 scripts/agent.py --backend local doctor
python3 scripts/agent.py --backend local --work-root '/data/poseidon work' doctor
```

`candidate` / `batch` 后 `--` 后面的相对路径按照执行端项目根目录解析。使用 `--dry-run` 可以只查看命令，不执行程序。完整任务期限由 `--timeout` 控制，API 请求期限由 `--api-timeout` 控制，两者不同。

## 环境与凭据

当前配置支持 Linux x86_64 和 aarch64，分别锁定启动器与 wheels，使用 LLVM/MLIR 18.1.2、SEAL 4.0.0、隔离 Python 环境及 bubblewrap。架构检查、固定版本、沙箱隔离和完整性检查必须保留。

DeepSeek 密钥保存在项目根目录的单一 `.env` 中，只需 `DEEPSEEK_API_KEY`；不要把路径配置写入 `.env`，不要公开或提交此文件。启动器不读取或复制凭据。

`doctor` 只检查架构、路径和工具存在性，不验证编译器或密态执行。某些 Nix 符号链接只在隔离环境中可解析；真正的库加载、tracing、编译、沙箱和数值检查仍需单独通过。

迁移到新的 Ubuntu 环境时：

1. 保留当前 Agent 源码、本地修改和固定版本的 Dacapo 子模块。Agent 主体已发布在 `lhy-agent-dsl`；本轮未提交的双平台适配仍需单独保存，不能仅依赖重新 clone。
2. 按固定依赖重新配置环境，不直接复用其他目录或架构的 Python venv、CMake cache 和二进制。
3. 安装、sudo、子模块初始化及大型下载前先确认版本、位置和规模并取得批准。
4. `continue_dacapo_cpp.py --approved` 是带历史 checkpoint 门禁的续建操作，不是新 checkout 的通用安装器。
5. 先检查环境，再运行离线 Linear 自测，通过后再单独批准真实 API 调用。

```bash
python3 -B -m unittest discover -s scripts/baseline -p test_portability.py -v
python3 scripts/agent.py --backend local candidate -- \
  --case scripts/baseline/cases/linear-example.json --self-test
```

第一条是离线路径/启动器回归测试。第二条使用脚本化候选运行现有隔离编译和 SEAL CPU 差分检查，不调用付费 API，不构成真实 Agent 生成证据。

## 结果迁移与历史证据

新结果写入所选工作目录。缺失的历史报告不得视为本机新跑通过；旧报告中的绝对来源路径和冻结哈希不应静默改写。移动完整证据档案仍需独立的迁移清单及审计，单纯复制文件夹不保证旧审计入口全部有效。

原有 Ubuntu 离线 Linear 证据为 `RESULTS/candidate-replay-nalu1fgc/report.json`，SHA-256：
`8fe07539c5696bdfca9d6667a580956b540a70b4ed03e0f0bce5f6e2c31da319`。
通过的候选比较了 4 组输入共 8 个值，MAE 为 `2.9526754626377216e-9`，最大绝对误差为 `6.879170566520543e-9`。这些是保留的历史记录，不是本次文档修改后新执行的密态实验，也不代表 Poseidon GPU 链路已通过。
