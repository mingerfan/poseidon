# 单卡 API Trace / 共享 Hoist / 完整 Boot 验收

源码工作树 `poseidon-api-trace`，分支 `experiment/api-trace-hoist`；实现提交
`9acdbf1f`，Runtime `4829bff`，DaCapo `5e91402`。基础优化手写来源：
`SL640/FHE_GPU_Agent@55e51804`。代码、编译器和 runtime 都在独立工作树，不改原 checkout。

## 完整网络，188Server

Tesla V100-SXM2-32GB，CUDA 12.2、GCC11；网络使用物理 GPU0 单卡。Q49/P25，
主秘密 h32768，Boot 临时秘密 h32，degree22/DA3/K12。真实执行经过：

```
同一份手写代码 → 运行时 API trace → MLIR → HEFT/Release → RuntimePlan V8 → SequentialRuntime
```

- 27个加密输入已驻留 Device；7,448次真实 API dispatch，包括38个共享 Hoist、
  214个旋转后缀及18个完整 native Boot。无计划内 Encode/Transfer；最终密文 Q3。
- 原手写、trace 运行、runtime 分别使用**同一组密钥和输入密文**；trace 与手写以及
  runtime 与手写均逐一比较393,216个最终 RNS 残基及 metadata，全部相同。
- 38个 Hoist 均在最后使用后释放；实际 Hoist live 峰值6,291,456 bytes，最后为0。
- CUPTI 70,576个活动；计时区间 H2D/D2H 0次、0字节；计算后才下载/解密。
- 分类3，与明文参考相同；10个 logits 最大误差0.008690，小于0.1。这不是全槽精度或
  数据集准确率测试，也不构成安全性证明。
- 单次手写 wall 3614.882 ms；Runtime online 3677.479 ms（整个 runtime wall 3689.549 ms）。
  **runtime 挂CUPTI而手写没有，不能把差值解释为纯 runtime 开销或声称稳定速度比。**
- 计划内图值估计峰值582.5 MiB，包含27个外部输入与声明的 Buffer 字节；不包括
  密钥、Boot/旋转 workspace、明文缓存或30 GiB池预留，不是进程显存峰值。

`network.log` 为实际完整原始日志；`summary-reviewed.json` 由
`bench/resnet20_s2c_first/check_trace_run.py` 检查该日志和真实计划生成。
在审查前另一次完整GPU回放也通过，见 `network-first-pass.log`；两次随机新密钥，
不跨进程比较密文。最早 `trace-capture-preflight.log` 在确认 trace 精确后人为终止，
用于加入 Device 输入验证，不计为计划回放通过。

## 其他回归

- DaCapo 36/36；Runtime 11/11；Python 43项中35通过、8项缺少可选GPU证据而跳过。
- 真正 Q49 MLIR 的 V8 JSON/binary 导出互相一致，reader线程数1/4和 PlanVerifier通过：
  38 Buffer、7,448 ApiCall。**整网 GPU 实际回放只测 JSON，binary 只测读入/验证。**
- V100 GPU0 小规模与 GPU3 独立扩展回归：N4096/N65536、Q3/Q4/Q10/Q13/Q28、
  P2/P9/P10/P25，普通和逆预旋转密钥，拆分/原批量旋转逐 RNS 一致；复用与非法输入检查通过。
- 非阻塞流上传/下载、物理 Q 前缀裁剪、密钥缓存深拷贝与 plan 跨图绑定回归通过；
  N65536 P3–P8 小P kernel与旧路径逐 RNS 一致；QP-c0 有独立明文 oracle 数值检查。
- 编译器和 runtime还单独测试外部输入无 Buffer 的V8计划、共享 Buffer 的最后使用释放、
  Device 输入直接绑定和跨版本兼容。

## 验收与复现

```bash
# 从工作树根目录；最终已归档的真实trace、计划和日志
python3 bench/resnet20_s2c_first/check_trace_run.py \
  test-results/api-trace-v100-20261111/compiled-reviewed/scheduled.gpu_api_trace.runtime-plan.json \
  test-results/api-trace-v100-20261111/network.log
sha256sum -c test-results/api-trace-v100-20261111/tested-source-sha256.txt
```

从源码构建和跨机器编译复现见 `bench/resnet20_s2c_first/README.md` 与
`run_trace_v100.sh`。`provenance-reviewed.json` 记录源码、编译器和计划摘要；
服务端目录 `/home/xuming/poseidon-api-trace-20261111` 保留运行日志和独立 build。
`POOL_STATS`、密钥和 Boot 内部 scratch 不被计划边界值估计覆盖。

仅支持单rank、单卡、单流；Buffer 不跨设备传输；运行需要本次 trace 期间准备的
进程内资源，不是可跨进程序列化的明文/密钥 bundle。Q49/P25/h32 是不安全研究配置。
