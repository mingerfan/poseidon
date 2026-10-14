# 可信 PyTorch 捕获依赖验收 r805

可信 PyTorch 捕获已直接使用公共进程限制，并把该文件纳入捕获源码身份。原捕获行为、隔离、转换、模型格式和数值规则保持。

编译、执行和验证的保守源码闭包已移除 Provider、生成适配器、旧沙箱、人工 helper 探针和实验运行器。三阶段从657个文件降为593个；生成阶段597个。各阶段仍共享模型语义和依赖校验模块，不能解释为完整阶段独立。

## 改动

`trusted_pytorch_import.run`直接导入`process_resource_limits.limits`。
`IMPLEMENTATION`增加该文件，使捕获期间的冻结、来源报告和后续核验绑定限制函数的实际字节。

对照修改前AST，确认仅此导入和清单成员改变；公共限制函数逐字保持。历史沙箱、人工工具、报告及旧程序包未改写。

旧格式v1–v6继续接受。旧捕获报告若与当前源码身份不一致，仍明确拒绝；重新捕获应保留原记录，不能直接修改旧绑定。

## 测试

367项：**364 passed、0 failed、0 errors、3 skipped**。包含v1–v6的捕获／导入、权重、静态形状、近似、索引、组件、程序包和旧接口测试。部分组件测试使用mock，真实密态范围单列如下。

新增4项检查公共限制哈希绑定、修改后的新鲜读取、缺失／漂移拒绝、实际捕获准备的导入边界和显式信任要求。

三项跳过均为旧SEAL条件证据缺失，未计通过：

- `CompilerConfigurationEvidenceTests.test_new_requests_real_trace_compile_execute_and_old_failure_preserved`
- `CompilerConfigurationGuardEvidenceTests.test_worker_rejected_before_keys_or_hevm_execution`
- `CompilerLegacyCompatibilityEvidenceTests.test_legacy_request_unchanged_and_explicit40_has_distinct_identity`

## 真实捕获和后端

| 模型 | 来源 | 已有DSL |
|---|---|---|
| 外部权重平方激活MLP | 可信PyTorch v3 NPY状态包 | r651原规则候选 |
| 功能式逐节点近似组合 | 可信PyTorch v5 | r720原规则候选 |
| 静态轴操作与输出组合 | 可信PyTorch v6 | r786原规则候选 |

三个模型各在当前沙箱重新捕获，完成16组原Module／纯函数／独立数学和float64 Torch检查。原图、分解图、权重、请求和已有DSL逐项匹配，未改模型或候选；不是新Agent生成。

每例完成真实Hecate tracing、Earth／CKKS IR、RuntimePlan／OperatorSpec／bundle核验、Poseidon CPU密态执行、解密和逐项比较。随后包重载原四组输入，再用三组新输入重复执行两次；缓存复用不生成源码。

包含Linear CLI正例，共 **46组输入、176个输出值**，最大绝对误差 **3.2887875917e-7**。48组独立参考、6项来源限制哈希拒绝通过。77个包文件和71个编译产物哈希核验通过。

Linear CLI的原响应解析和故意数值失败仍被拒绝。数值反例4组／8值单列，不混入通过数。

固定门限：`abs(actual-reference) <= 1e-5 + 1e-4*abs(reference)`。
新输入由独立审计比较；本地执行接口自身不声明新的独立reference验收。原float32应用精度、非线性原函数精度和所有输入形式化等价未在本批认证。

## 安全与绑定

原限制保持：地址空间4GiB、CPU150／155秒、单文件16MiB、描述符128、core0；网络／PID隔离、环境清理、只读请求和私有文件不可见均有实际候选沙箱预检。

沿用`tc128`、N=32768、Q13／P1及原w45配置。执行器只用评估密钥；本批无bootstrap、CUDA或层间解密再加密。

新绑定：`eea3613d91a9837387126945e404cb5e57d4324c81b4e23ea6a3f80e216644fc`。667个Agent／配置源码、93个历史案例文件、648个后端文件、6个reference／门限／检查器及12个历史文件核验保持。

各worker终态。记录worker约156.04秒，新增约28.83MiB；全局约22.79GiB、磁盘余量约45.95GiB。原1391个修改／未跟踪路径保留，Git diff check通过。安装、下载、SDK重建、后端／参数／reference修改、API及Git发布均为0。

r798把平铺fixture误作目录的检查错误，以及r802审计变量命名错误已保留。修正工具后核验完成，没有重跑密态模型或改写原结果。

## 下一步

当前编译／执行／验证身份已脱离生成服务和实验补跑工具。源身份继续保守绑定共享语义模块，worker的SDK／工作路径仍固定。

接下来优先推进通用PyTorch写法、算子组合和资源预测；只补新增或受影响范围。新增功能真实Agent须冻结新案例、绑定和预算后授权。bootstrap、密文分区、CUDA、x86原生／空白安装及ResNet20保留独立状态。

## 记录

- [机器报告](agent-capture-boundary-acceptance-r805.json)
- 预检／测试／审计：`/home/lhohy/poseidon-work/platforms/aarch64-linux/results/agent-capture-boundary-preflight-r798`
- 捕获／执行／程序包：`/home/lhohy/poseidon-work/platforms/aarch64-linux/results/agent-capture-boundary-backend-r801`
- [应用组件](application-agent-component-v1.md#可信-pytorch-捕获依赖)
