# 有序张量输入 v12

Agent 输入层开发草案。显式格式为 poseidon-trusted-pytorch-v12，接入原 pytorch_model_intake 和来源包接口；生产接线状态见 integration.json。

## 输入

静态签名、Python 文件清单与哈希、工厂参数、完整公开 NPY 状态权重、输出声明及近似配置。捕获采用 CPU float64 参考。输出支持 Tensor、静态 tuple/list 和字符串键字典。

## 覆盖

- 静态 reshape、transpose、permute、索引及 split 的 view 关系。
- Tensor 别名、快照及顺序 add/sub/mul/copy/square 写入。
- 已登记的 Linear、Conv→Pool、残差、归一化及矩阵组合。
- 原始公开权重与 transpose/split 派生常量的独立绑定。
- 原始图、发射图、规范化图、分解图及节点来源。

公共状态修改、重叠写入、动态分支及未登记调用返回明确错误。v12 当前为有序张量子集；v11 common-call 写法继续使用旧入口。

## 验收

agent-ordered-intake-tests-r1187：106 项通过；10 个来源模型、16 组输入／模型，共 1280 值，三份来源包重载通过。

agent-ordered-intake-boundary-r1186：正常 facade 捕获与来源包重载 3/3，错误架构拒绝 1 项。复用现有 portable Nix store，Nix 启动次数为 0。

输入层之外，agent-joint-post-cutover-offline-r1229 已完成 Linear、LayerNorm 近似、公开权重 split→Linear 三例规则基线的真实 Hecate tracing、编译及 Poseidon CPU 密态验收；程序包重载、新输入执行与缓存复用均通过，21 组输入、119 个输出值，最大绝对误差 2.190414178926403e-7。该范围的独立审计见 agent-joint-post-cutover-audit-r1230。

新入口的真实 Agent 生成评测见 docs/baseline/agent-v12-live-proposal-r1232.json，目前待批准。integration.json 保存目标哈希及恢复副本；其余 V12 组合仍按独立状态验收。

## 复现

在 ARM Ubuntu checkout 使用新结果目录：

~~~sh
python3 -B scripts/baseline/benchmarks/tools/staged_ordered_intake_v1/run_isolated.py agent-ordered-intake-tests-r1190
python3 -B scripts/baseline/benchmarks/tools/staged_ordered_intake_v1/run_capture_boundary.py agent-ordered-intake-boundary-r1191
~~~

原 76 项已关闭并结算调用账本。27 项 Agent 文件已完成联合切换，三个代表模型的真实后端验收通过。旧 V1–V11 入口和身份配方保留；源码更新建立新身份，历史通过记录保持原绑定。
