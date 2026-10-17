# 线性层与 LM head 的数学侧瓶颈分析

本报告使用当前生产 DSL 和独立 FP64 槽 oracle；生产数学代码、HEFT、runtime、权重和接口均未改，未 SSH 或 commit。1418689 Prepare / 8273326 groups 是现有完整计划的上下文，不等于源级 MulCP 数。这些数字不能证明低 GPU 利用率的具体时间归因；需要与实际读取、Encode、API 提交及 GPU event 测量一起判断。

当前 BSGS 降低了 Rotate，仍保留每矩阵 `out+in-1` 条公开矩形对角线。下面是单次线性调用的原语计数。cyclic 列包含脏尾部保护所需的一个输入 mask；下表中的复制、Rotate、Add 不增加乘法深度。

| 矩阵 | [out,in] | 当前 MulCP / Rotate | cyclic MulCP / Rotate | 当前→cyclic CP 深度 |
| --- | --- | --- | --- | --- |
| q_proj | 2048,1024 | 3071 / 110 | 1025 / 81 | 1→2 |
| k_proj | 1024,1024 | 2047 / 94 | 1025 / 81 | 1→2 |
| v_proj | 1024,1024 | 2047 / 94 | 1025 / 81 | 1→2 |
| o_proj | 1024,2048 | 3071 / 110 | 2049 / 96 | 1→2 |
| gate_proj | 3072,1024 | 4095 / 126 | 1025 / 81 | 1→2 |
| up_proj | 3072,1024 | 4095 / 126 | 1025 / 81 | 1→2 |
| down_proj | 1024,3072 | 4095 / 126 | 4096 / 127 | 1→2 |

输入先用前缀 mask 去掉脏物理尾部，pad 到 `P=next_power_of_two(in)`，再通过 `log2(8192/P)` 次旋转相加复制到所有8192槽。周期向量满足 `Rot(s,x)[i]=x[(i+s) mod P]`；对 `s=0..P-1` 编码 `d_s[i]=W[i,(i+s) mod P]`，无效列置零。B64 BSGS 使用 `np.roll(d_s,g)` 后做 giant rotation，完整槽常量保证输出尾部零。MulCP 复杂度由 `out+in-1` 改为至多 P，Rotate 为 `63+(P/64-1)+log2(8192/P)`。没有截断或近似任何权重。

3072不能整除8192，直接按3072周期复制会产生环回错误；pad4096没有收益（一个结构性全空对角线略去，实际4095条，再加1个mask）。down 可把三个1024输入块的部分乘积放在同一个密文的三个1024输出区域：每条 `s=-1023..1023` 对角线在三个区域分别装入相应子矩阵系数，屏蔽跨输入块读。最后用两次旋转求和并mask到1024前缀。它不需要输入mask，脏尾部也不会被有效系数读到；计数为 **2048 MulCP / 96 Rotate、深度2**，其中2047是权重乘，1是输出mask。

## 完整 LM head 与接口

当前 LM head 为148个1024输出块加最后384，共149个密文、151936个logits。未修改生产 `linear` 的实测代表块计数为2047/94（1024块）及1407/84（384块）；总计 **304363 MulCP、13996 Rotate**。

只将中间输出块扩大到8192、继续旧矩形对角线，需18个完整8192块和最后4480：恢复原输出的149个extract mask计入后，计数是 **171522 MulCP / 3986 Rotate**。这只是计数，独立 oracle 验证的是下述周期方案。

周期输入可一次准备、供19个宽块共用。每块1024个权重对角线，总19456 MulCP；加1个共享输入mask及149个输出extract mask，为 **19606 MulCP / 1615 Rotate**。其中Rotate是19×78个BSGS、3个输入复制、130个输出切片。最后宽块拆成4个1024和1个384；其余18块各拆成8个1024。原149个logit输出和112个KV输出可保持原261接口；本 oracle 计算完整logits，未运行或替换KV网络。

## 全图源级计数与代价

七矩阵在28层、2个token中各调用56次，每层每token为22521 MulCP / 786 Rotate；加完整LM，共 **1565539 MulCP / 58012 Rotate**。按每次调用独立复制和mask的保守计数：

| 线性部分 | MulCP | Rotate | Add |
| --- | ---: | ---: | ---: |
| 当前生产 BSGS | 1565539 | 58012 | 1564998 |
| 七矩阵都用简单周期方案，加宽周期LM | 650726 | 36783 | 650784 |
| down 改用三个部分区域后fold，其余周期方案 | 536038 | 35047 | 536152 |

这不含norm、attention、SiLU、Boot、通信及runtime元数据。CSE/共享Encode、placement和最终scale/level选择会改变Prepare数，不应把536038当作已生成的新计划计数。周期复制也可在q/k/v或gate/up间共享；表中没有计入该优化。

原linear CP深度1；带显式输入mask的周期linear或packed down为2；周期宽LM再extract为3。在当前120-bit logical rescale配置下，额外乘法阶段通常意味着额外Rescale，可能消耗4个30-bit Q并增加Boot。当前rms_norm末尾已有前缀gain mask，数学上已产生零尾部，能否复用这一保证而省额外输入mask需要明确接口或融合证明；本 oracle 故意按任意脏尾部验证，没有假定它。不能把mask设低scale当成免费操作，也不能仅依据原语减少宣称FHE加速。

8192宽块和149个提取结果都需真实生命周期/容量检查；最终仍有149个logit密文，不能报告只剩19个。新数学IR需重新trace和编译，HEFT算法无需修改，但新图的最终placement与Boot必须实测。

## 验证证据与复现

`linear-cyclic-oracle.py` 完全独立实现槽操作和CP深度计数；dense `W @ x` 为参考。七个真实模型矩阵形状分别验证dense/short、clean/dirty物理尾部；另有7个小尺寸bias测试。七形状最大误差 **1.0658141036401503e-14**，输出尾部与脏尾部污染都为零。完整151936×1024合成LM矩阵分19块逐行生成，不同时占用整份约1.16GiB权重；每个恢复的149个输出都检查完整8192实数槽，最后384。完整LM最大误差 **1.199040866595169e-14**。本验证没有CKKS、GPU、预训练权重或完整网络前向。

```bash
OPENBLAS_NUM_THREADS=2 python test-results/waterline-20261013/linear-cyclic-oracle.py \
  --report test-results/waterline-20261013/linear-cyclic-oracle-new-report.json
```

命令要求新的report路径，保留原证据。当前结果为 `linear-cyclic-oracle-report.json` / `linear-cyclic-oracle.log`；`linear-cyclic-oracle-diagnostic-00.log` 记录初版检查器对空尾部做max的错误，已改为全槽输出时尾部定义为0，没有降低数值阈值。下一步应先对选中的数学改写做实际CKKS level/scale/Boot和数值验证，再测读取、Prepare、前台及端到端耗时；不能将本报告当作模型验收或加速结果。
