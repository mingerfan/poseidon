# Step 3 在 188Server 上的四卡测试

测试日期：2026-10-08。四卡普通／Release 计划的 8 组数值检查全部通过，
两种布局各完成 1 次预热和 10 次连续运行。NCCL 使用两进程、每进程两张卡。

## 环境与构建

- 服务器：SSH 别名 `188Server`，主机名 `HPU-001`。
- GPU：4 × Tesla V100-SXM2-32GB，驱动 `535.288.01`。
- CUDA `12.2.140`，GCC `11.4.0`，CMake `3.31.6`，OpenMPI `4.1.6`，
  NCCL `2.20.5+cuda12.4`。
- 独立测试目录：`/home/xuming/poseidon-memory-step3-20261008/`，其中 `source/`
  保存本次源码快照，`build/` 是 Poseidon 构建，`build-runtime/` 是 Runtime 构建。
- Poseidon 使用 Release 构建，CUDA 架构 `70`，开启 Runtime CPU/GPU 测试、
  MPI 和 NCCL。Runtime 使用 Debug 构建，开启测试和 MPI。
- 使用仓库内依赖和服务器已有工具，离线构建。Runtime、GPU API、GPU 测试、
  两个 MPI 执行工具和计划生成器的远端源码 SHA-256 与本地一致。

服务器上已有其他 GPU 工作，测试期间各卡约占用 19 GiB，利用率经常接近
100%。本次结果用于检查执行和数值正确性，不据此比较性能。

## 测试计划

[生成器](../scripts/generate_runtime_release_smoke_plans.py) 直接生成手写小计划，
不依赖编译器。普通版是 RuntimePlan V1，Release 版是 V2，计算和传输相同。

- `1x4`：单进程使用 GPU 0—3。
- `2x2`：同一服务器上，rank 0 使用 GPU 0、1，rank 1 使用 GPU 2、3；
  `device_counts=2x2`，`rank_to_node=0x0`。
- Host 密文在初始化阶段上传。在线 Replicate 复制到其余三张卡；2×2 布局
  包含一份本地复制和一次向另一 rank 的两个目的地发送。
- 四张卡分别执行计算，共 10 次 Negate、9 次 AddCC，最终结果为 `-10*x`。
  收尾阶段再复制一次 Device 结果，数值检查工具负责下载和解密。
- 每份计划有 32 个值、10 条通信指令；Release 版加入 31 条 Release，保留
  最终输出。2×2 计划有 4 个跨 rank 传输目的地。
- CKKS 多项式次数 4096，Q 为 7 个 30 位模数，scale 为 `2^40`，输入 level 6。
  这是执行测试配置；OperatorSpec 的延迟字段为零，不是性能标定结果。

## 结果

| 测试 | 结果 |
| --- | --- |
| 独立 Runtime CTest，含 MPI 两／四进程和 Release 版本 | 9/9 通过 |
| Poseidon CPU API | 5/5 组通过 |
| Poseidon GPU API，含双卡普通／Release、两种模式各三轮 | 16/16 组通过 |
| 2×2 NCCL 初始化，合计四个 GPU 通信 rank | 通过 |
| 2×2 跨 rank 密文传输，逐字比较缓冲区 | 通过 |

下面是解密结果相对 `-10*x` 的最大实部误差。每项还检查虚部误差，原有阈值
均为 `1e-4`，保持不变。

| 布局 | 执行模式 | 普通计划 | Release 计划 |
| --- | --- | ---: | ---: |
| 1x4 | sequential | 8.23444e-8 | 1.99412e-7 |
| 1x4 | per_device_workers | 2.03078e-7 | 1.88197e-7 |
| 2x2 | sequential | 1.71780e-7 | 1.35046e-7 |
| 2x2 | per_device_workers | 1.46323e-7 | 1.30209e-7 |

8 组全部通过。普通版和 Release 版分别生成加密输入，最终解密数组的最大
差异为 `2.505e-7`。轨迹中，普通版没有 Release 事件，每份 Release 计划在
所有 rank 上合计恰好记录 31 次 Release。报告中的计划和 OperatorSpec 摘要
也与保存的最终文件一致。

1x4 和 2x2 的 Release 计划还分别在同一 Runtime/API 上完成 1 次预热和
10 次连续运行，使用 `per_device_workers`。这个通用执行工具不逐轮解密；
数值正确性由上面的 8 组检查和双卡 GPU API 重复用例验证。单卡 GPU Release
门控用例也在 V100 上通过，确认工作未完成时保留分配、Release 不等待，
工作完成后活跃分配在 Runtime 值表收尾前回到基线。

## 测试发现与修改

1. **修正 GPU 测试显存池的设备选择顺序。** `RmmPoolScope` 原先先构造
   pool，再在构造函数体里选择设备；第二张卡的 pool 因此建在上一张卡上，
   双卡普通计划第一次 AddCP 就报 `invalid resource handle`。现在先选择
   设备，再构造资源和 pool。修复后 16 组 GPU API 测试全部通过。
2. **为 MPI 数值检查工具增加执行模式选项。** 可用
   `--execution-mode sequential|per_device_workers` 检查同一计划，默认模式
   保持工作线程模式，JSON 报告记录所用模式。
3. **新增可复现的测试计划生成器。** 测试配置使用非 placeholder 的
   OperatorSpec，并满足 GPU 预检查要求的 `rescale.max_levels_per_op=4`。
   初始 30 位 scale 的一轮普通计划误差为 `1.66754e-4`，超过阈值；保留
   30 位 Q 模数，将 scale 提高到 40 位后重新运行全部矩阵。
4. **按既有多进程通信限制调整最终输出。** 最初的计划直接输出 Host 值，
   非参与 rank 的分布式 GPU 通信检查拒绝 Host 目的地；工作线程模式也要求
   多进程在线通信使用 Device 目的地。最终计划输出 Device 值，检查工具在
   输出所属 rank 下载解密。

这次多卡测试没有要求修改 Runtime 的 Release 计数逻辑或生产 GPU API。
失败日志也保留在结果目录中，最终结果以该目录顶层的 JSON 和日志为准。

## 复现与证据

本地保存的[测试脚本](../test-results/step3-188/run-remote-tests.sh) 在服务器上
设置现有 CUDA、MPI、NCCL 工具路径并依次执行各项检查，每项有 180 秒超时。
在已完成构建的测试目录内运行：

```bash
cd /home/xuming/poseidon-memory-step3-20261008
python3 source/scripts/generate_runtime_release_smoke_plans.py --output-dir plans
bash run-remote-tests.sh all
```

仅重跑数值矩阵和连续运行时，用 `plans-only` 代替 `all`。

- [汇总 JSON](../test-results/step3-188/summary.json)
- [Runtime CTest 日志](../test-results/step3-188/runtime-tests.log)
- [GPU API 日志](../test-results/step3-188/results/gpu-api.log)
- [2×2 NCCL 日志](../test-results/step3-188/results/nccl-2x2.log)
- [2×2 Release 工作线程数值报告](../test-results/step3-188/results/2x2-release-per_device_workers.json)
- [2×2 连续运行报告](../test-results/step3-188/results/2x2-release-repeat.json)

本次没有运行完整模型、native bootstrap 或编译器自动插入 Release，也没有
测量四卡显存峰值或逐轮分配回收量。2×2 是单服务器四卡测试，尚未验证跨服务器
NCCL。step 4 的原位执行仍未实现。
