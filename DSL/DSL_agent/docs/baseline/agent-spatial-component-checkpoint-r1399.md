# 空间算子接入：r1399

JSON 高层图 v4 已接入统一分解、能力检查、应用提交和来源导入。新增 resize、ConvTranspose1d/2d；目前完成明文与静态验收，Poseidon 编译和密态验收待执行。

| 检查 | 结果 |
|---|---|
| 单元及兼容检查 | 72 passed，0 failed，0 skipped |
| 原图、独立数学参考、Torch 参考及分解图比较 | 42 上下文，672 组，13,120 输出值 |
| 最大明文差异 | 4.440892098500626e-16 |
| 应用 submit_source → prepare_source | 3/3 通过，来源与冻结请求一致 |
| 冻结规则候选静态检查 | 45/45 通过：42 新上下文＋3 旧版本兼容 |
| 本轮编译／密态／API | 0／0／0 |

另外 8 项启动前检查通过，覆盖源码与后端身份、模型／候选哈希、外发调用为零和缺少授权时拒绝执行；[检查记录](agent-spatial-prelaunch-checks-r1400.json)。6 个生产文件语法和本轮 diff 检查通过。

## 接入范围

- resize：rank 3 的 nearest、nearest-exact、linear；rank 4 的 nearest、nearest-exact、bilinear。指定静态 size 或 scale_factor；linear/bilinear 使用明确 align_corners。语义对应 Torch 2.0.1 CPU Float64 连续布局，antialias=false。
- 转置卷积：1D/2D，固定公开权重与可选 bias；分组、步长、padding、dilation、output_padding、静态 output_size 按对应形状规则检查。
- 分解与已有矩阵、公开量算术、分支及命名输出组合；原图、权重、节点来源和分解身份分别绑定。

转置卷积的小型映射使用精确公开 Linear；矩阵超出既有常量预算时，采用公开索引、零掩码与已有 Conv 的分解。两条路径都有边界与独立参考检查，按资源判断选择，未加入模型名称分派。

原 r1396 有 5 项预检阻塞，r1398 有 3 项；r1400 在原模型上用等价分解与合法 packing 全部解除。43 项采用 P4，2 项采用 P8。原模型、公开权重和参考保持；旧失败记录、候选与绑定保留。这说明静态准备可行，实际编译深度、scale 和数值表现仍须执行验收。

## 版本与证据

开发 checkout：`/home/lhohy/Code/Poseidon`；分支 `lhy-agent-dsl`；HEAD `7588e48093631eaf8385506e3db058e7ca3cfe7e`。

新 SDK：`388e891a44863d95e6a8ce7e83c4206799434e08e122c63ff58afa2b62452337`。明文结果绑定：`06151aaac85345e42d73c718a48c877581ef74d62856c0e66192e7b9fb387fb4`。72 项控制检查绑定：`a6c91c13e9c9b9e823ebc17726783c61ce73fda229be996417af814f44ed9758`。

固定环境为 Python 3.10.14、NumPy 1.25.2、Torch 2.0.1；Poseidon、CKKS Runtime、Dacapo、参数、依赖锁、网络和 x86 配置保持。未安装、下载、提交或推送。

开发验收中曾修正测试的容量夹具与资源元数据断言；应用入口检查的初始任务目录位于源码 checkout，被安全检查正确拒绝，随后改用既有 work root。原日志均保留，生产安全检查未放宽。

r1393 的 12 项真实 Agent 成功仍属于当时 SDK；本次新 SDK 不能沿用其验收结论。冻结的 1,200 模型保持，42 个新增语义上下文单列。

## 下一批

[r1400 精确方案](agent-spatial-offline-proposal-r1400.md)包含 45 个规则候选及 3 份程序包的新输入、重复执行和缓存检查。批准前仅完成准备；真实 Agent 新生成、PyTorch 空间入口和 CUDA 验收仍为后续任务。

[机器记录](agent-spatial-component-checkpoint-r1399.json) · [输入用法](operator-decomposition-and-python.md#json-空间计算图)
