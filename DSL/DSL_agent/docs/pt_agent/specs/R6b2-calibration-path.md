# R6b2：内层不读 registry，由外层传入校准路径，跑完 r6b_refresh

R6b 被阻塞的原因：`experiments/r6b_refresh.py` 的内层（在 Nix 中运行）调用 `registry.load()` 查找 R6c 的校准来源。Nix 的 `--pure` 环境里没有 `PT_RESULTS`，于是抛出 ValueError。

## 1. 修改：只改 `pt_agent/experiments/r6b_refresh.py`

1. 外层新增参数 `--calibration`，默认值为 `/home/lhohy/pt_agent_results/run-resnet20-rule-20261020T165817/calibration.json`，即 R6c 最终运行使用的校准文件。外层先检查这个文件存在，再把绝对路径作为 `--calibration <路径>` 传给内层。
2. 内层直接用 `json.load` 读取这个文件，删除对 `registry.load` 的调用（第 35 行附近）。内层不得 import 或调用 registry、config.load，也不得读取 PT_RESULTS。
3. 图仍然用 `models.resnet20.load_graph()`。
4. 其余步骤完全按 R6b spec §5 执行。

## 2. 测试

在 `tests/test_r6b_cost.py` 中加一个测试：在内层入口函数的模块里搜索字符串 `registry`，如果出现在可执行代码里（注释除外）就判为失败。也可以改为在测试中清空 `PT_RESULTS` 环境变量后调用内层的加载函数，确认不报错。两种方式任选其一。R1–R6b 的测试依然全部通过。

## 3. 运行与完成标准

- 重跑 `python3 -m pt_agent.experiments.r6b_refresh`，完成 R6b §5 的全部内容，最后调用 prune。
- 汇报内容同 R6b §8：4 行汇总表、boots_pred 与 plan 中实际 boots 的对照，以及 mock_equal。
- STATUS.md 中 R6b 改为完成，新增 R6b2 一行，并删除"阻塞"栏中的对应条目。
