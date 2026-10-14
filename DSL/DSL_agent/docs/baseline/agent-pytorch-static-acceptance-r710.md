# PyTorch 静态形状与标量输出验收

2026-10-04，ARM Ubuntu。来源为规则候选。

新增 PyTorch v4 捕获契约：按输入签名及登记的形状规则展开公开控制流，保存直接标量及嵌套输出。旧 v1–v3 保留默认策略、拒绝规则和模型／请求哈希接口。

| 模型 | 真实编译及密态数值 | 程序包复用 | 初次最大绝对误差 |
|---|---|---|---|
| loop_shape | 通过 | 通过；3 新输入 × 2；缓存复用 | 4.2006e-09 |
| loop_size | 通过 | 未安排 | 4.4870e-09 |
| loop_branch | 通过 | 未安排 | 5.8990e-09 |
| scalar_sum | 通过 | 通过；3 新输入 × 2；缓存复用 | 6.3990e-09 |
| scalar_mean | 通过 | 未安排 | 2.0603e-09 |
| scalar_dot | 通过 | 通过；3 新输入 × 2；缓存复用 | 6.6473e-09 |

- 静态、组件、兼容及隔离捕获测试：192 passed，0 failed/errors/skipped。
- 六个来源包各完成 16 组双参考及原模块比较，共 96 组。
- tracing、Earth/CKKS 编译、RuntimePlan／OperatorSpec／bundle、Poseidon CPU 执行及解密均通过。
- 合计 54 组密态执行、190 个输出值，maximum absolute error 9.724804683752808e-09。门限保持 1e-5 + 1e-4*abs(reference)。
- 三份包重载，三组新输入分别执行两次，缓存复用无生成／编译调用；独立检查 66 个结构化输出值及标量、容器结构。120 个包文件哈希通过。
- 648 个后端文件及 6 个参考／检查器文件保持原哈希。当前 640 个 Agent 文件绑定：98303d07b27938874754ce91c4b2e87d94e5d56af897cfc31de4a5ff13a39640。
- 本轮新增 API／HTTP、安装、下载、SDK 重建、参数／后端／reference 改动、commit／push／PR 均为 0。

执行约 2307.14 秒，测试累计约 63.30 秒；新增产物约 24.15 MiB，全局约 22.39 GiB，余量约 46.38 GiB。执行串行，原沙箱和资源限制保留。

初次联合测试的两个错误已保留并修复：FX 元数据读取碰到参数 Proxy，以及测试调用 prepare_task 的接口写法错误。最终只读审计首次按 JSON 字典键顺序比较失败；改为比较键集合并单独检查来源树顺序后通过。原报告、脚本与源码快照保留；没有重跑已完成的密态案例。

## 使用

格式与命令见[输入说明](operator-decomposition-and-python.md#静态形状与标量输出)。当前成果属于规则候选和真实后端验收；新功能的真实 Agent 生成需另行冻结案例与授权。

## 后续

逐节点／模块近似选择、更多 PyTorch 算子及静态索引写法、权重去重、身份检查开销与组件依赖边界继续推进。原 Attention 编译容量失败保留。Bootstrap、CUDA、x86 原生、空白安装及 ResNet20 分别验收。

[机器报告](agent-pytorch-static-acceptance-r710.json)。原始产物：

~~~text
/home/lhohy/poseidon-work/platforms/aarch64-linux/results/agent-static-pytorch-preflight-r707
/home/lhohy/poseidon-work/platforms/aarch64-linux/results/agent-static-pytorch-offline-r708
~~~
