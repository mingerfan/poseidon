# PyTorch 源码精度参考

`TaskSourceReference` 复用现有应用对象，执行任务中保存的可信 PyTorch 源码和固定权重，返回原生 dtype 下的 forward 输出。模型在独立 bubblewrap 进程中加载，调用端保留任务和程序包。

```python
from task_source_reference import TaskSourceReference

reference = TaskSourceReference(app).reference(
    task_id,
    samples,
    evidence_directory,
    expected_package_binding=package_binding,
    input_dtypes={"q": "float64", "k": "float64", "v": "float64"},
    trust_local_code=True,
)
```

每次接受 1–16 组符合程序包签名的输入。`input_dtypes` 为每个输入指定 float32 或 float64；参考输出记录原始参数、公开整数／布尔 buffer 和输出的 dtype，同时记录输入转换误差。Tensor、静态 tuple/list、字符串键字典、标量输出和 v16 的静态 None 延续既有输出约定。输入使用独立存储，源码中的合法原地操作可以执行。

返回 `evaluated` 表示源码 forward 已完成。生成、编译、CKKS 数值等级和应用精度各自保留：该结果用于核对源码、规范化和近似误差，应用精度通过明确指标另行判断。JSON 模型继续采用登记的数学参考；存在 PyTorch 源码的已交付任务通过此接口取得实际 forward。

低层 `source_precision_reference.run()` 校验源包、工厂参数、无 pickle 权重、输入签名和输出结构。worker 保留原始 dtype，在 CPU eval／no_grad 下执行，每组输入重复两次，检查 registered state、RNG、结果一致性及输入存储隔离。内存4GiB、CPU150秒、单文件16MiB、FD128及隔离网络 namespace 均实际核验；单次 wall 上限180秒。

## 验收

控制范围共有48个不同测试：原生参考19、任务边界7、旧数值组件22，全部通过、无skip。原生参考包含9份成功的真实 PyTorch forward 报告；组件状态测试使用注入后端。19项记录来自26项联合报告的通过子集，后续29项报告重测修正后的7项组件和旧22项数值组件。源码实现哈希保持匹配，旧失败报告完整保留。

真实应用入口复用历史MHA任务：3组输入、30个值，原生源码与独立数学参考最大差异 1.110223024625157e-16。返回的 source／task reference 分别绑定源码、权重、runtime、输入和程序包。

复用r1321已经保存的解密值，近似计算相对原生源码误差 0.7657156493384427，密态总误差 0.7657156472095206；原CKKS执行误差 5.218970255960187e-9。这定位了该案例的误差来源：源码与数学图一致，显著误差来自声明的近似计算。原模型应用精度尚未认证。

本轮新增编译、密态执行、API调用、安装、下载及后端修改均为0。旧源码包、模型、权重、参数、reference、数值门限、程序包及生成记录保持；691冻结源和654保护文件匹配。原两项MHA付费重测仍待已有范围授权。

早期测试的近似配置字段、JSON路径参数和worker平台选择问题已修复；NaN数据构造及queued状态断言也已修正。最初报告中的source_forward_executed为未证实标记，单独勘误确认成功原生报告为0；原报告未覆盖。

机器报告：`docs/baseline/agent-source-precision-reference-r1322.json`。

Ubuntu应用验收：`/home/lhohy/poseidon-work/platforms/aarch64-linux/results/agent-source-precision-acceptance-v1/report.json`。

## 接续

将绑定的原生源码输出接入显式应用指标，再补数据集与批次精度报告。当前接口解决原始dtype的参考执行，源码接入、生成、CKKS和模型应用精度继续分别验收。
