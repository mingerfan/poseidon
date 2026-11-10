# 手写 ResNet20 → MLIR → RuntimePlan：188Server 单卡验收

已完成。工作树 `poseidon-manual-mlir`，分支 `experiment/manual-resnet20-mlir`，从
`framework@e45b10f7` 建立；实现提交 `30057fc2`。优化 Boot 来源为
`origin/integration/resnet20-manual-opt@a0ca0a89`，不是旧 `gpu-api` 的 degree-59 默认。
Runtime 子模块提交 `b324472`，DaCapo 提交 `39ac338`；均有同名本地实验分支，未推送。
原 `poseidon/` framework 工作树未修改。

## 结果

188Server：`/home/xuming/poseidon-manual-mlir-20261110`，Tesla V100-SXM2-32GB，
CUDA 12.2 / GCC 11。两次整网都只使用物理 GPU0；独立 high Boot 单测使用 GPU1，
没有任何多卡图执行。测试已结束，GPU 内存均已释放。

| 单次完整整网 | 原手写 wall | RuntimePlan wall | Runtime online | 最大 logit 误差 |
|---|---:|---:|---:|---:|
| Binary，RuntimePlan 挂 CUPTI | 4099.296 ms | 4150.300 ms | 4150.145 ms | 0.011084 |
| JSON，双方不挂 CUPTI | 4099.901 ms | 4096.629 ms | 4096.537 ms | 0.009947 |

这只是每种条件一次运行，不是稳定加速结论。clean 条件下与原手写耗时基本持平；
Binary 行只有 RuntimePlan 挂采集器，不能拿它与原手写时间直接归因比较。

两行均通过：
- 真正由通用 `SequentialRuntime<PreparedGpuApi>` dispatch 77 个 ApiCall，含
  18 次 degree22/DA3/K12/h32 native Boot、19 次 ReLU，最终 Q4。
- 每个进程使用同一组随机非零 h192 密钥、同一批 prepared 输入密文，与原
  `execute_graph(false)` 对照；最终 **524,288 个 RNS 残基及 metadata 完全相同**。
  不同进程重新生成密钥，不声称两行之间的密文相同。
- 预测均为 3，与独立明文参考相同；十个 logit 误差均小于 0.1。
  RNS 精确性是“相对原手写版”，不是“所有槽都等于明文参考”。
- 实际执行顺序逐项等于 compiler 导出的顺序，且不同于源顺序：例如 Conv 3
  被排在 clone 2 前。不是编译后继续绕过计划执行旧调度。
- CUPTI 记录 79,960 个活动，H2D/D2H **0 次、0 字节**；无在线 Encode/Prepare、
  中间解密重加密或换入换出。最终 RNS 下载和解密在计时后。
- 压缩 QP 明文、double hoist、C2S tile8、分层 P/dnum2、key-view cache、lazy
  base conversion、four-step、QP-c0、Conv 批量乘加和 ReLU 融合保持接入。
  实际开关见 `network-environment.txt` 和应用日志。

独立库级 native Boot 全槽单测：degree22 最大误差 `1.57338e-5`，degree59
`3.0556e-5`，均低于单测 `2e-3` guard，输出 Q31/2^40。未声称另跑了旧应用
`POSEIDON_S2C_LIBRARY_BOOT=0` 的整网。

## 编译和测试覆盖

- DaCapo CTest：35/35；Runtime CTest：10/10。
- 188Server CTest：5/5（metadata、prepared graph、support、DFT repack、Boot host）。
- Bench Python：39 项，31 项通过、8 项要求额外 GPU A/B 日志而跳过。
- 真实 Q50 spec 的 77-call JSON/binary reader、verifier、1/4 线程读取一致性通过；
  计划含 76 个 Release，无 reuse/Transfer/Encode。
- `--emit-mlir` 在 188Server 隐藏 GPU 后成功；其输出 SHA256 与本地编译输入相同。
  MLIR→RuntimePlan 编译在本地 LLVM/MLIR 18 构建的新 hecate-opt 上完成，GPU 执行在服务器。

## 验收和复现

```bash
# 本地检查原始日志与机器可读结果
python3 test-results/manual-mlir-20261110/summarize.py

# 服务器已存在独立源码、build、plan 和原始日志；重跑两个单卡案例
ssh 188Server 'bash /home/xuming/poseidon-manual-mlir-20261110/run-v100.sh'
```

从头构建、重新生成 MLIR/计划的命令见 `bench/resnet20_s2c_first/README.md`。
`run-v100.sh` 使用 GPU0 实验锁；重跑会覆盖服务器日志，不改其它工作树。
`summary.json`、`network-{bin,json}.log`、`boot-{low,high}.log` 是主要证据；
`provenance.json` 和 `tested-source-sha256.txt` 记录工具/源码身份。

## 范围与安全

当前是 **77 个 prepared API 任务**，不是每个 GPU kernel 的 MLIR 展开。
Conv/ReLU/Boot 内部 rescale、融合与 workspace 仍由原优化实现负责；compiler
只安排任务和边界值生命周期，不重新规划 scale/Boot。只支持单 rank、单卡、单流。

h32 dense→sparse key 在完整 Q50/P25 上生成，不安全；所有运行均为
`--unsafe-*` 的研究实验，不构成生产安全参数批准。
