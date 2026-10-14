# Model provenance serialization

正式出口位于 `candidate_bundle.serialize_model_provenance`。模型来源 JSON 保留字典声明顺序，canonical 身份与请求绑定继续使用既有规则。编码前执行完整来源检查，并分别检查 canonical 大小和实际导出字节数；重载使用原 `check_bundle`。

`source-before/` 保存本轮改动前的正式模块。`test_serialization.py` 包括三个保留来源的完整 roundtrip 与拒绝测试；常规控制另加入 `test_candidate_bundle.py`。`acceptance_window.py` 按新源码绑定重新编译固定规则 DSL、执行四组数值验收，随后用正式出口导出，再执行新输入和重复输入。测试不调用 Provider。

最新证据：`results/agent-formal-provenance-export-v1/report.json`（ARM work root）。
106 项控制通过；一个原布尔 mask/padding/残差模型完成真实编译、数值验收、正式程序包重载和重复执行。MHA v16 来源仍为独立 staging，生产接入状态另计；规则候选来源保持为 deterministic_rule_baseline。

只读复核：

~~~bash
python3 -B scripts/baseline/benchmarks/tools/model_provenance_serialization_v1/inspect_current.py
~~~

运行入口接收新的结果目录；已有目录和历史证据保留。配置继续使用冻结 Poseidon CPU w45、原 4 GiB/native CPU 边界，额外重测仅选受影响范围。
