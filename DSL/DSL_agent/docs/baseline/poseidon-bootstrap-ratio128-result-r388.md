# Poseidon CPU bootstrap ratio128试验

2026-10-01。r235单次有界试验获批，现已执行并消耗该授权。

**试验未通过冻结数值验收。** 安全参数校验、编译和真实bootstrap运行成功；零组完成两次刷新，有符号组在第二次刷新有15个值超限。按批准范围立即停止，随机与边界两组未运行。

## 实际执行

使用原Poseidon库与冻结探针，N=32768、Q17×48/P1×60，Q×P实际876位≤tc128表上限881。输入scale=2^40、工作scale=2^48，message/output ratio同时128。链路为原生refresh→平方/rescale→真实乘公开1/rescale→再次refresh。

乘1与rescale是实际密态运算。解密仅用于阶段比较，没有decrypt-reencrypt刷新，也未改当前Agent参数、Dacapo或Runtime。

| 输入组 | 阶段 | 比较值 | 失败值 | 最大绝对误差 | 状态 |
|---|---|---:|---:|---|
| zero | first_refresh | 16384 | 0 | 4.4739524e-07 | 通过 |
| zero | square_rescale | 16384 | 0 | 1.0658141e-14 | 通过 |
| zero | second_refresh | 16384 | 0 | 4.5201225e-07 | 通过 |
| signed | first_refresh | 16384 | 0 | 1.9322983e-05 | 通过 |
| signed | square_rescale | 16384 | 0 | 1.8216309e-05 | 通过 |
| signed | second_refresh | 16384 | 15 | 2.1589063e-05 | 失败 |

计划4组×3阶段=12项；实际执行6项，5通过、1失败、6未运行。不是5个模型通过。累计98,304次逐值比较，其中15次失败；重复阶段单独计数。

有符号第二次刷新的14个失败值reference=0.015625（门限1.15625e-5），1个reference=0.0625（门限1.625e-5）。最大总误差2.15891e-5并非最大超限比例所在项。最严重超限值reference=0.015625、actual=0.01560976655、error=1.52335e-5，超门限3.67095e-6。

数值门限保持 `abs(actual-reference)<=1e-5+1e-4*abs(reference)`。随机/边界未运行，不记skip或pass。执行约45.77秒，总建造与产物约13.95MiB，低于30分钟/512MiB；隔离进程RLIMIT_AS=6GiB。launcher的RSS记录不是沙箱峰值，不用它声称实际峰值内存。

## 误差判断

已确认：输入level16/scale40；第一次输出level2/scale约48；平方后level1/约48；乘1-rescale后level0/约40；第二次输出level2/约48，未出现容量或scale异常。失败层是重复刷新数值精度。

独立数学诊断只读取保留结果，不新增密态试验或替换reference。理想sine模型预测第一次最大偏差5.04945e-6、重复平方刷新约3.67604e-6；实际第一次相对surrogate残差max1.62011e-5，第二次相对上一平方结果新增差异max1.56529e-5。

基于证据的推断：已有平方误差与归一化/第二刷新新增误差共同累积，单独减小理想正弦近似偏差不足以解释少数槽位的误差尾部。未确认：carry、固定多项式、密钥噪声与归一化各自占比。冻结探针没有单独解密归一化中间值，不能给出更强归因。

下一步先只读分析内部trace与误差传播；若需要新增阶段解密诊断或参数试验，列范围和预算另申请，不自动重跑或扩大参数。

## 对主线的影响

核心Agent→Poseidon CPU迁移仍按已通过范围成立；bootstrap是独立精度/接线缺口，当前Runtime/Agent未开放。原12深多项式阻塞不因零组通过而消失。新容量和跨块算子工作继续推进。

同时补验README的1024元素mean CLI入口，1/1真实编译/密态/解密比较通过、四组输入、4输出值；规则源码，与Agent新生成分开。它重复现有容量模型，不增加模型数量。[CLI记录](poseidon-capacity-cli-acceptance-r387.json)

原始记录 `/home/lhohy/poseidon-work/platforms/aarch64-linux/results/poseidon-bootstrap-ratio128-r236`；源码/二进制/授权/日志/逐项值和数学诊断保留。批准提案仍保留原文pending_approval字段，它是提案快照，批准及消耗状态在独立授权和本报告中。

本批安装、下载、付费调用、commit、push为0。[机器可读报告](poseidon-bootstrap-ratio128-result-r388.json) · [迁移与容量状态](poseidon-capacity-acceptance-r386.md)
