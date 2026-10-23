# 编译程序包资源分析

本批新增可选资源分析接口，宿主可在复用程序包前检查中间密文、明文及存储下界。原 Agent 输入、生成、修复、验收等级和执行接口保持；Poseidon、Runtime、Dacapo、参数及数值门限均未修改。

## 接口

- Python：`application_planning.compiled_resources.analyze_program()`、`check_analysis()`、`require_storage_precheck()`。
- CLI：`python3 -B scripts/analyze_program.py --bundle program --output compiled-resources.json`。
- 策略：`fixed-cpu-retained-coefficients-v1`，报告格式：`poseidon-compiled-resource-analysis-v1`。

程序包和依赖校验通过后，适配器读取绑定的RuntimePlan快照，检查原始及绑定产物、OperatorSpec、bundle与I/O。报告包括定义数量、输出别名、操作数量、旋转步数、重线性化需求以及level/scale概要。

当前Agent实际执行库是`third_party/ckks-runtime-agent`。该版本在一次执行中保留所有已定义值；参考目录`third_party/ckks-runtime`的释放逻辑不适用于此身份。按`degree × (level+1) × components × 8`计算逻辑系数数据，加入加载的槽向量数据，形成存储下界。输出别名不重复计数，各组样本依次执行。

## 验收结果

| 验收 | 结果 | 范围 |
|---|---:|---|
| 静态与组件测试 | 70通过，0失败，0跳过 | 新接口19项；已有资源、布局和执行上下文51项 |
| 已编译产物分析 | 30/30 | 复用r812规则候选，不重新编译或运行 |
| 真实Poseidon CPU重载 | 3/3 | 向量点积、双密文多维广播、公开批量矩阵与tuple输出 |
| 数值比较 | 12组、152值通过 | 最大绝对误差`8.8739546317e-7` |
| 真实程序包拒绝检查 | 13/13 | 预算不足、伪造峰值、陈旧产物、超原沙箱预算及候选文件篡改 |
| CLI | 2/2 | 默认分析成功；不足预算返回blocked及退出码1 |

数值门限保持`1e-5 + 1e-4*abs(reference)`。本批数值程序来自规则基线，复用已有编译产物，在当前分析身份下重新加密、真实执行、解密及比较；没有新增Agent生成，也没有新增模型或Benchmark案例。

30份产物的存储下界为23,330,816–1,671,430,144字节。双密文批量矩阵案例的输入下界为58.5MiB，完整定义下界约1.56GiB，说明输入大小无法代表运行时中间值规模。

## 边界

`memory_budget_bytes`与`--memory-mib`只约束存储下界预检，执行继续使用原生4GiB地址空间限制。下界超预算返回blocked；满足下界返回assessed，execution_readiness仍为unknown。评估密钥、分配容量、工作区、峰值及耗时尚未形成可信估计。此次开发侧/proc采样没有观察到runner，报告保留not_observed，不作为内存测量结果。

产物与源码改变会导致报告失效。适配器核对产物快照哈希，重新计算事实，拒绝自行改写报告后重算哈希的情况。它属于显式宿主接口，尚未自动接入默认Application流程。

## 保留的失败与身份

- r818首次测试驱动未向子进程传递PYTHONPATH，1项import错误；仅修正驱动环境，原记录保留。
- r820首次工具使用128KiB候选响应解析器读取RuntimePlan，发生大小拒绝；新适配器改用已有16MiB产物协议并加入边界测试，原记录保留。
- CLI首次SSH启动发生banner broken pipe，未进入客体；顺序重试后两项验收完成。
- 最终r824测试、r825产物审计、r826真实重载、r827 CLI建立各自绑定，不改写前序结果。
- r828收尾驱动曾把原始工具身份与组件包装身份直接比较；按原字段核验后端，源码和依赖身份保持，原驱动错误记录保留。
- r829收尾驱动的纯Nix路径未提供ps；改用客体已有/usr/bin/ps进行进程盘点，无新增依赖。
- r830收尾进程盘点未排除ps表头；修正驱动解析后复核，生产接口不受影响，原日志保留。
- r831收尾磁盘统计把顶层目录快照按文件路径比较，将历史占用误计为新增；按原快照范围和实际分配字节复核，原预算保持。
- r832完整性及预算核验通过，报告聚合引用了不存在的独立测试日志；r833使用已保留的合批日志完成聚合，原完整性记录不改写。
- r815报告及r816提案保留。r816新生成评测仍待单独批准。

分支为lhy-agent-dsl，HEAD为7588e48093631eaf8385506e3db058e7ca3cfe7e。代码及文档只在ARM Ubuntu `/home/lhohy/Code/Poseidon` 修改；x86配置保留，原生x86和CUDA不计本批通过。

原始报告、日志、数值数组及旧失败保存在ARM work root的r817、r825、r826目录。仓库机器报告只包含摘要、路径及哈希。[机器报告](agent-compiled-resources-acceptance-r828.json) · [使用接口](application-agent-component-v1.md#编译程序包的资源分析)。

## 下一阶段

继续通用算子、PyTorch写法与未见组合、资源预测及原函数应用精度。真实bootstrap、跨调用密文分区、CUDA、x86原生安装与ResNet20保持独立状态。本批不是整个通用化计划完成。
