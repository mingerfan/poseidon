# PyTorch 算术结果 shape 接入

已增加 `poseidon-trusted-pytorch-v9`。Tensor 加减乘结果可以接静态 `shape`、`size`、`ndim`、`dim`、`numel` 和 `len` 查询，继续使用统一图、布局和组件接口。

| 检查 | 结果 |
|---|---|
| 最终相关测试 | 173 passed，0 failed/errors/skipped |
| 公开 CLI 合法来源 | 33/33；三组数学计算，写法变体另计 |
| 独立参考 | 528 组原模块／规范化／数学／Torch；528 组等价 JSON 对照，最大误差 0 |
| 新版本拒绝边界 | 加密数据分支、原地修改、out 写入三项均拒绝 |
| 来源与策略篡改 | 198 项均拒绝 |
| 旧 v8 兼容 | 六份成功来源图保持；九个方法结果查询继续按旧版本拒绝 |
| 应用提交 | 33 份进入数值验收任务队列，随后取消；没有运行生成和后端 |

24 份旧成功来源的数学图逐字保持；原九份失败、原报告与绑定保留。V9 记录首次 tracing 的独立 shape 策略；规范化后的计算仍走既有 INDEX 捕获。旧 V1–V8 格式和默认选择保持。新增支持是明确登记的静态元数据规则，输入元素、节点、权重及物理限制沿用所选配置。

## 使用

来源 manifest 选择 `poseidon-trusted-pytorch-v9`，其余字段沿用 v8。示例：

~~~python
y = x.add(public)
return y.reshape(y.numel())
~~~

~~~bash
python3 -B scripts/prepare_model.py   --trusted-pytorch-manifest scripts/baseline/cases/trusted-pytorch-shape-composition-v9/shape_add_method_result_numel/manifest.json   --trust-local-code --resource-profile general-tensors-v1 --write   --output "$POSEIDON_WORK_ROOT/platforms/$POSEIDON_PLATFORM/results/shape-v9-next"
~~~

宿主沿用 `app.submit_import(directory, validation_level="numerical", options=...)`。源码与外部权重在隔离 worker 中捕获；来源校验在宿主端只读取声明数据。

## 密态验收状态

本批没有 Hecate tracing、Dacapo 编译、密态执行、解密或真实 Agent 生成。上述通过是接入、组件提交及有限明文参考的一致性。CKKS 门限保持 `1e-5 + 1e-4*abs(reference)`。

当前新提案保留原 94 项，并补入九份修复来源，共 103 项、三片 48/48/7，待明确授权。刷新了受影响的 37 份 PyTorch 捕获身份，原模型、公开权重、数学图、请求和规则源码均保持；其余 57 份 JSON 来源复用。准备过程中发现的工具导入错误和 packing 名称错误保留原日志；没有重跑已完成的 37 份捕获。

[离线提案](agent-arithmetic-shape-offline-proposal-r976.md) · [机器记录](agent-pytorch-arithmetic-shape-acceptance-r974.json)。

## 保留范围

648 份后端、六份 reference／门限模块与历史记录核验保持；2,264 原工作区状态路径保留。实际 SDK 版本、参数和 x86 配置保持。无安装、下载、API、SDK 重建或 Git 发布。

原八项组合编译失败、新入口真实 Agent、bootstrap、密文分区、物理容量、原模型应用精度、ResNet20、CUDA 实机、x86 原生和空白安装仍各自待验收。
