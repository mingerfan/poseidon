# 实验产物与可维护代码的目录边界

`test-results/` 是本地/外部实验归档目录，**整个目录不再由 Git 跟踪**，没有源码、README 或 JSON 白名单。日志、报告、采样、trace、图表、编译计划、权重包，以及放在历史归档里的实验包装脚本和源码快照都留在归档中，不参与默认构建。

这不是停用测试。正式测试代码仍保留在 `src/poseidon/tests/`、CKKS Runtime 的 `tests/` 和 DaCapo 的 `test/`；不因名字包含 test 就取消跟踪。可维护的实验入口放在 `bench/` 或 `scripts/`，确实需要的固定输入放在专门的 fixtures 目录。

## 当前维护的入口和输入

- `scripts/benchmark_waterline.py`：Waterline benchmark；固定 MLP 输入已迁到 [`bench/fixtures/waterline/`](../bench/fixtures/waterline/README.md)，不再从 `test-results/` 加载。
- [`bench/resnet20_runtime/`](../bench/resnet20_runtime/README.md)：独立 resident runner 源码；运行所需的匹配模型、计划、bundle 和 GPU fixture 仍须按入口文档准备，不把历史生成物当作已附带的完整测试输入。
- Cerium 是外部项目，不是 Poseidon 的 submodule 或运行依赖。历史 Cerium 对照工具与解析器留在本地/外部实验归档，不纳入 Poseidon 代码。

新增可维护工具应显式区分源码、固定输入与每轮输出；输出可放进新的 `test-results/` 子目录，不要再让正式 benchmark 隐式依赖该目录里的历史文件。

## 历史文件与恢复

取消跟踪只改变 Git 索引，**不删除当前机器上的归档文件，也不重写历史**。此整理之前，已跟踪的历史归档可从提交 `5ef32ba8da97742e4942b99dd6d3a0b735e29f8e` 按需恢复单个文件：

```bash
git restore --source=5ef32ba8da97742e4942b99dd6d3a0b735e29f8e --worktree -- test-results/PATH
```

恢复后的文件仍被忽略。该提交只包含当时已经入库的文件，**不包含后来仍未跟踪的新 ResNet20/Cerium 实验**；这些需要依靠原有本地或远端归档，不能把上述命令当作其恢复保证。历史说明里的 `test-results/` 链接可能指向本地归档，而不是当前 Git 树中的文件。

忽略不是备份。重要结果需另存；不要在未备份时运行 `git clean -fdx`，它会删除被忽略的本地文件。
