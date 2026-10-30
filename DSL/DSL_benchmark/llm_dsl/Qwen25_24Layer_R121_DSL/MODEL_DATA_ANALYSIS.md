# 24层 R121：实际编译的数据量、level与内存分析

本报告对应 `Qwen25_24Layer_R121_DSL`，上海服务器任务 `round_122_qwen24_r121_analysis`。24个真实宽度Block，hidden896 / intermediate4864，两个已知token，完整词表151936，29个输出。

**实际追踪、原始/优化计划验证及完整Mock均通过。** 编译分析流程耗时 1884.220 秒（31.40分钟），未执行完整CKKS或GPU。

## 1. 核心结果

| 指标 | 实际结果 | 统计口径 |
| --- | --- | --- |
| 优化后自举数量 | 866 | 计划中的native Boot；未执行真实自举 |
| 明文Value数 | 63789 | 优化后计划，包括再编码产生的新Value |
| 密文Value数 | 296335 | 优化后计划，包括输入和中间值 |
| 全保留明文系数 | 135.907227 GiB | 每个PT Value各保留一份 |
| 全保留密文系数 | 1,090.963867 GiB | 每个CT Value各保留一份 |
| 全保留合计 | 1,226.871094 GiB | 静态系数总和，不是实际RSS |
| 按Release释放时全图峰值 | 356.500 MiB | 最终计划同时存活系数 |
| 分析流程最大子进程RSS | 29.300831 GiB (compile-labelled) | /usr/bin/time -v实测 |
| Mock峰值RSS | 0.269775 GiB | 明文向量模拟器，非密文计算 |
| 完整CKKS进程峰值RSS | 未测 | 需要实际加密执行 |
| GPU峰值显存 | 未测 | 上海服务器没有GPU |

## 2. 明文、密文与内存为何不同

系数容量逐Value使用 `65536 × 8 × (level+1) × components` 字节统计，明文通常1分量，密文按实际2/3分量计算。每个Value各保留一份是逻辑容量模型；不把实现中的潜在别名共享当作已证实的节省。

图峰值按执行顺序加入输出、处理Release；客户端仍持有的外部输入不会随图内Release消失。它不含密钥、编码/自举内部工作区、原始常量bundle、运行时对象及分配器保留空间。因此不能将图峰值或全保留系数总和直接称为进程内存。

这次并未实际分配全保留数量的密文；全保留一列通过完整编译计划精确求和获得。完整CKKS含真实自举的峰值仍未知，不能拿父版两层27.08GiB直接当作24层保证。

## 3. 编译原计划与CPU内存优化后的对比

| 指标 | 原始编译计划 | 最终CPU计划 |
| --- | --- | --- |
| Boot次数 | 866 | 866 |
| PT Value数 | 59267 | 63789 |
| CT Value数 | 296335 | 296335 |
| 全保留系数 GiB | 1,216.996094 | 1,226.871094 |
| 同时存活峰值 MiB | 129,159.000 | 356.500 |
| 初始化Encode系数 GiB | 126.032227 | 0.000000 |
| 累计Encode系数 GiB | 126.032227 | 135.907227 |

CPU后处理依次做终端冗余刷新消除、release_ready调度、64MiB Encode根缓存。缓存限制只针对指定Encode根，不是进程内存上限；按需再编码可能增加累计PT Value数和Encode工作量，但减少同时存活内存。

## 4. level、自举和峰值所在阶段

输入level：`[5, 5]`；输出level按5个logits、24个KV顺序：`[1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]`。

全部 866 个自举均满足 L1→L5、scale43。明文在各消费者所需level编码，完整Encode分布见下表。

| Encode level / scale | 次数 |
| --- | --- |
| L1:s43 | 263 |
| L2:s43 | 31530 |
| L3:s43 | 549 |
| L4:s43 | 7656 |
| L5:s43 | 23791 |

全图同时存活峰值出现在 ordinal `630679`，操作 `rescale`；来源为 `qwen24-stage/token_0-1/layer_21/_decoder_block`。

峰值同一时刻PT 39.500 MiB、CT 317.000 MiB，合计 356.500 MiB。

逐层/逐阶段level、所有自举位置与峰值组成见 [COMPILED_LEVEL_DETAILS.md](COMPILED_LEVEL_DETAILS.md)。共享来源只计一次，不把一份融合Value重复分摊给多个子层。

## 5. 真实阶段耗时与RSS

| 阶段 | 耗时秒 | 峰值RSS GiB | 返回码 |
| --- | --- | --- | --- |
| verify-delivery | 0.125 | 0.016361 | 0 |
| compile-plain | 213.032 | 29.297256 | 0 |
| verify-eager-labelled | 4.198 | 0.223347 | 0 |
| full24-mock | 108.830 | 0.269775 | 0 |
| optimize-schedule-cache | 79.128 | 1.975677 | 0 |
| build-mock | 13.536 | 0.482193 | 0 |
| trace-labelled | 182.162 | 22.092297 | 0 |
| prepare-real-width-fixture | 32.658 | 2.433266 | 0 |
| trace-plain | 172.841 | 18.284657 | 0 |
| verify-optimized | 4.274 | 0.210808 | 0 |
| analyze-values-levels | 489.553 | 1.950611 | 0 |
| compile-labelled | 195.040 | 29.300831 | 0 |
| verify-eager-plain | 4.172 | 0.223270 | 0 |
| dump-locations | 12.885 | 29.273540 | 0 |

RSS由每条子进程命令的 `/usr/bin/time -v`采集；并非模型执行的CKKS峰值，也不包括远端守护worker所有对象。无标签/带标签编译各执行一次用于一致性证明，流程总时间包含二者。

## 6. 原始权重、常量文件和槽位

原始合成权重：291个数组、630167424个元素、float32 payload 2520669696字节（2.347557 GiB），embedding和LM head分别存放。若按共享权重仅计一次，则494032768个唯一元素；float16/bfloat16约0.920GiB。这不是CKKS编码体积。

本次原始常量bundle：59246个文件，15522781849字节（14.456717 GiB）。此项是序列化磁盘数据，不能再当成编码明文系数相加而不说明口径。

| 表示 | 有效槽 / 32768 | 利用率 |
| --- | --- | --- |
| 单token hidden | 896 | 2.734375% |
| 两token并列hidden | 1792 | 5.468750% |
| 单token MLP前缀 | 4864 | 14.843750% |
| 两token配对MLP两区 | 9728 | 29.687500% |
| 两token每层持久KV | 512 | 1.562500% |
| 5块logits合计 | 151936 / 163840 | 92.734375% |

这些是特定表示的有效槽占比，不是全模型平均slot利用率。临时重复和旋转可占更多物理槽；24个层KV仍独立存放。

## 7. 验证与可追溯性

独立dense参考的完整Mock最大误差 1.33912686107e-07，输出尾部 0；Mock自举输入最大绝对值 3.608604418。Mock直接模拟明文算子，不验证自举数值误差或CKKS噪声。

编译器SHA256：`12705f29938821d48ae3db3def7767af7a2af6507d40fb0c7e9bfdb98918de68`；183份编译源码哈希均检查。使用R121的同一实验normalize预算修正，不是泛称官方最新分支。

原始计划SHA256：`f4faa32fe8ee878113b98aff7ff117f58b2abced37da84c632d9dab49df2dd3d`；最终计划SHA256：`0f963b49633181d96392974d3a9726cd84d674d8daea806913ab197f55c33e7a`。

分别追踪无标签和仅修改诊断位置的版本，实际导出的RuntimePlan与全部常量bundle文件逐字节相同。最终后处理输入就是该已证明相同的计划；两份原计划及最终计划都通过Native PlanVerifier。

服务器任务：`/home/gy/qwen-ckks/experiments/qwen_round_122_1792657228975819700`。任务已退出。数据及校验：[compiled-analysis.json](evidence/compiled-analysis.json)、[compiled-verification.json](evidence/compiled-verification.json)。

## 8. 尚未由本报告证明的事项

尚未进行本版完整24层真实CKKS/GPU执行、跨独立decode图的密文level衔接、真实预训练权重校准及生成质量评测。这里的最大内存分别明确为静态系数峰值和编译/Mock实测RSS，没有虚构完整CKKS或显存数据。
