# v6 prepared AV集成（完整单block GPU已测）

同runner同窗control-B→v6：online中位8.236990537s→**6.303876115s**，
耗时降低23.468698%，v6/Cerium=**104.041527%**。两组各1warmup＋3完整测，
v6组合polynomial＋AV，不能分拆成单flag因果收益。实际23Boot/27Linear/6QK/6AV每run；
AV6IDs共享唯一1,037,303,808B MaskBundle，内部真实算术计数全部吻合下表。
Runtime wall中位6.490315884s、main565.974913868s（control main545.102443103s）；
较短online不等于较短进程耗时。最终98,304 complex slots有限、max误差37.94，**精度未通过**。
详见`test-results/bert-a800-20261115/v6-a800/20261115T144308Z/FINAL-REPORT.md`。

`--prepared-av`把原cyclic weighted-V改为标准V7 `bert.compact_av`调用。
仍全部128keys/12heads，保留每pair128个真实CC/relin/Rescale2，不用Host答案或私钥刷新。
实现与独立probe见[compact_av](compact_av/README.md)，adapter为`prepared_av_adapter.h`。

当前native能力有明确限制：N65536/T128/tile256/D64/baby8、selective q52-p11、初始L12，
实际scoreL10/valueL8→commonL6→outputL4。CLI与lowering拒绝不支持的区间，不改metadata凑数。
不同ID共享同一物理MaskBundle/runtime/keys；不能把每个pair报告的shared mask bytes相加。

## 后续输入packing＋A2 GPU诊断（与上面v6历史分开）

`--av-pack-inputs`显式选择distinct `bert.compact_av_packed_partial` v2，V理想gap0
公共证明、partial全槽函数和masked reciprocal三轮/全fanout清gap门禁完整接通。
pair中位baseline0.146629006s→partial0.120891714s；生产全部32768definedpartial槽及
16384live完整AV oracle通过（max约1.07e-8），混合key重传与每mode72ID共享核验通过。
实际每pair64CC/319CP/193Rescale/642drops/296rot/319adds/19hoist，不减128keys。

完整单block同新runner诊断control6.315600s/QKpack5.545888s/组合5.391346s，组合为
Cerium参考88.98%，但永久**shared-or-unresolved diagnostic，非clean因果或精度保持**。
组合最终max误差61.54、RMS约13，高于control5.57/~1.19，不能唯一归因算法（不同进程
keys/Boot物料）。后来独立同物料9forward已完成：AA全bitwise相同，但B参考误差更大、
C较小只是该固定物料的一次观察，仍无HE准确性/普适排序结论；见
[同物料结果](multi_plan_validation/GPU_RESULTS.md)。详见
[诊断数据](packing_validation/DIAGNOSTIC_RESULTS.md)和[A2语义](AV_PACKED_INPUTS_PLAN.md)。
此结果不混入旧6.303876s单层或76.417357s完整12层c34结果，不外推packed12耗时。

## 合法公开mask＋alignment合并

原非零shift是`mask CP→Rescale2→unity CP→Rescale2`。现在mask真正编码为
PTlogscale `S6+log2(q10*q9*q8*q7)-S10 ≈110.7127`，两来源L/R mask CP相加后
**真实Rescale4**。shift0仍是真unity CP+Rescale4。V selector积和仍是真Rescale2；
之后所有CC/relin/Rescale2照常执行。

相同polynomial-v5图下，每pair减少127个CP与127次逻辑Rescale调用，但**消耗prime数不减**。
原shift0本来已是Rescale4；整个old/new pair均770次physical prime drops。
公开0/1 mask不量化；FFT/Rescale噪声不同，不声称ciphertext bitwise。

## 完整单block CPU计划

| 普通/opaque调用 | v4 algebra/shared/QK | v5再加polynomial | v6再加AV |
|---|---:|---:|---:|
| compute/opaque | 12341 | 12299 | 1337 |
| MulCP | 3287 | 3269 | 209 |
| Rescale调用 | 2619 | 2595 | 291 |
| Rotate | 2026 | 2026 | 244 |
| MulCC/Relin | 856/856 | 856/856 | 88/88 |
| native Boot计划 | 23 | 23 | 23 |
| Linear/QK/AV callbacks | 27/6/0 | 27/6/0 | 27/6/6 |

这张表**不含opaque内部算术**：v6每层AV实际计划768CC/relin、2298CP、1542Rescale调用
（774Rescale2＋768Rescale4）、4620physical drops、1788direct rotations、2286adds。
加上QK内部768CC及普通88CC，仍总1624CC，未靠少算Attention降低工作量。

AV-only相对v5：真实CP/Rescale各减少762。整个v4→v6还包含poly pass，减少780CP/786Rescale。
**旋转净增6**：原DAG的denominator第一次reduce与AV g0/little1共享`scores.rotate(1)`；
opaque边界看不到该临时量，每pair重算一次。6×298内部rotation加普通244为2032，
不是原2026。此解释经独立只读源码/图/产物审计确认。

## 已验证与未验证

- v6冻结GPU源码112项、后续活跃源码最近132项Python CPU测试通过，含完整tiny双层独立oracle、QK/AV/两rewrite保留、
  多CT跨层SSA及资源准入新策略。native runner本地GCC15编译链接通过。
- `prepared-av-v6-bound/`是真实T128单层：输入、全部16参数SHA与冻结v3一致，
  28,339,200B dense binary逐bit相同；标准PlanVerifier通过，末端3CT均L12/Q13。
- `prepared-av-v6-full12-cpu/`已生成并逐SSA验证12层、324Linear＋72QK＋72AV、276计划Boot、
  一个AV物理mask tuple，标准PlanVerifier通过。绑定的是本地synthetic12数据，
  **不是远端实测v3的bitwise同权重数据**；正式12层比较必须绑定实际远端冻结资产。
- 默认CLI七主产物再次逐字节复现v3；新flags默认关闭。
- 上述v6整层另有自己的冻结source/bin/两计划和归档，不借旧v4或standalone probe替代。
  两进程随机keys/profile不同，不能把末端误差差异单独归因AV/poly。
- 原远端冻结12层权重的2→4→12实测已委派；本地synthetic12结构计划不是该GPU结果。
  Cerium12层仍无实测，不从单层×12构造百分比。
