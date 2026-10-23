# Poseidon 批量入口迁移

## 结果

通用 Benchmark 的 baseline/free/directed 和扩展算子回归已采用 Poseidon CPU 配置及对应 Nix 环境。新增 CPU w40/w45 选择；SEAL 配置在入口拒绝。应用组件、Provider、模型和数值门限保持原样。

| 验收 | 结果 |
|---|---|
| 静态与旧接口测试 | 28 passed、0 failed、3 skipped；skip 为历史实验记录测试 |
| 通用 Benchmark 代表例 | 3/3 真实编译、密态执行与数值审计通过 |
| 扩展算子代表例 | 6/6 通过：广播、公开 gather、自适应平均池化、LeakyReLU、PReLU、LogSoftmax |
| 续跑 | 原三个结果的绑定、报告及证据哈希一致；复用已完成结果 |
| 新程序包 | 2/2 导出与数值重载通过；重载生成和编译调用均为 0 |

9 例共 144 组独立双参考、36 组密态输入、1196 个输出值；最大绝对误差 1.17222e-7。非线性算子验收的是原先冻结的显式近似计算。候选来自规则基线，不计真实 Agent 成功。门限保持 `1e-5 + 1e-4*abs(reference)`。

## 绑定与兼容

修改前的 512 文件完整快照在 `poseidon-benchmark-boundary-r272/core-source-before.zip`。修改发生在无活动批次的安全边界。

冻结模型集未再生成，原 index、shard、模型签名、划分和覆盖账本哈希继续校验。旧生成代码与当前执行代码分别记录；跨版本运行须显式提供 `--frozen-suite-index-sha256`。每个选中模型在执行前重新通过两套独立 reference。当前 Dacapo-modified frontend、运行代码和配置参与新的批次绑定。

原算子分解绑定不改写。先检查原绑定及模型哈希，再要求当前分解图逐字段等于冻结图，并核对来源、能力和近似配置；只有注册版本与实现身份可重新绑定。本轮所有六例均通过这一检查。

数值成功须来自 Poseidon CPU、numerical 等级和真实执行/比较记录；只编译及历史 SEAL 结果不能计入新通过数。旧 summary 仍可读取原格式，保留历史身份。

## 使用

~~~bash
python3 -B scripts/benchmark.py baseline \
  --suite scripts/baseline/benchmarks/semantic-v2-ledger-r38 \
  --frozen-suite-index-sha256 ee7a0868cd5ea68e7ada18e43a52249b6d1a474a4c13e006fb7285cbe506a6a2 \
  --limit 3 --shard-index 0
~~~

默认显示计划；加 `--execute --output <新的结果目录>` 执行规则候选验收。相同绑定可用 `--resume` 复用证据。

~~~bash
python3 -B scripts/baseline/check_operator_regression.py \
  --case-id broadcast_0 --case-id public_gather_0 \
  --output "$POSEIDON_WORK_ROOT/platforms/aarch64-linux/results/new-operator-batch"
~~~

加 `--run-local` 执行密态验收。默认 CPU w45，w40 可通过 `--compiler-configuration poseidon-cpu-eva-w40-v1` 选择。分片上限仍为 48，原资源限制保留。

## 范围与后续

此次计划对完整 1200 模型做当前规则预检：1155 ready、45 blocked_rule_preflight；实际只执行上述三个代表例。未选中的 1197 例不计通过，45 项预检受阻也不能直接认定为 DSL 不可表达。累计数学分区数仍沿用 [r271](poseidon-dsl-migration-r271.md)。

历史 helper-directed 与 compiler-evidence 封装仍指向旧实验工具，是下一项迁移工作；它们不参与应用的正常请求链路。没有为这两个入口宣称新 Poseidon 验收完成。

三个整模型的未执行提案 r268 因源码变化失效，现替代为 [r274](poseidon-whole-models-live-proposal-r274.json)。模型、公开请求和后端二进制未变；新源码共 513 文件，binding 为 `b130e29ba8b4e799435bd90977bf5b94534d243260079787e099e8eb952675ea`。仍待单独付费批准，最多 12 次生成／48 次 HTTP。

剩余 12 个数学模型的编译容量失败、真实 bootstrap 的误差失败保持未解决。r235 独立试验尚未执行。x86 原生、空白安装和 CUDA 继续暂缓。

[机器可读报告](poseidon-benchmark-migration-r273.json)。原始结果位于 `/home/lhohy/poseidon-work/platforms/aarch64-linux/results/poseidon-benchmark-entries-r273`。本轮没有安装、下载、付费调用、commit 或 push。
