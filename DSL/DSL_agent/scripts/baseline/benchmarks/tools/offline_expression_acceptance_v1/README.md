# 单例规则表达的离线验收入口

该开发工具只执行 r1286 冻结的布尔掩码 Attention 表达。复用
`component_backend.qualify`、程序包导出及只读重载，模型、权重、参数、
reference 和数值门限保持原值。候选来源为人工选择的规则变体。

入口核验独立批准收据、原提案和控制源码；每份批准最多启动一次。
结果目录存在时停止，历史失败和绑定保持。正常应用请求不依赖此工具。

```sh
python3 -B scripts/baseline/benchmarks/tools/offline_expression_acceptance_v1/run.py --plan
```

收到 r1286 对应的人类批准并记录原范围及当前控制源码哈希后，在现有 ARM 环境执行：

```sh
POSEIDON_PLATFORM=aarch64-linux POSEIDON_WORK_ROOT=/home/lhohy/poseidon-work \
python3 -B scripts/baseline/benchmarks/tools/offline_expression_acceptance_v1/run.py --run
```

批次 15 分钟，资格阶段 180 秒，一次编译、一次密态执行、原四组输入；
随后导出并只读重载，不执行新输入或缓存命中。新产物 512 MiB、总量
32 GiB、磁盘余量 4 GiB。API、安装、下载和后端修改均为 0。

控制测试使用临时合成收据验证拒绝行为，不生成项目批准文件或启动后端。
