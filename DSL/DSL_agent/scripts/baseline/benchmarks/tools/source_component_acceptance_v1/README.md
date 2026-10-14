# 来源到应用提交验收

复用已捕获的 V14 融合矩阵来源和请求，检查联合 Agent 代码的实际 Poseidon
后端身份、规范化图提交、源码与权重快照、来源绑定和取消。使用真实
`Application` 与 `PoseidonBackend`，无身份替身。

此工具只检查任务接纳和绑定。任务保持排队或取消，验收等级仍为 numerical；
没有编译、密态执行、数值验收、程序包缓存命中或 Agent 生成。

```sh
POSEIDON_PLATFORM=aarch64-linux POSEIDON_WORK_ROOT=/home/lhohy/poseidon-work \
python3 -B scripts/baseline/benchmarks/tools/source_component_acceptance_v1/run.py agent-source-component-admission-rNEW
```

运行于已配置的 ARM Ubuntu 环境；用新的结果名保存日志和源码快照。
联合目标采用只读覆盖，生产文件及原始来源证据保持。
