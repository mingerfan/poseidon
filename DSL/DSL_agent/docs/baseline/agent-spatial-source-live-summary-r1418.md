# Poseidon Agent 空间源码测试结果

日期：2026-10-12。范围：PyTorch v27 的 34 个源码案例，DeepSeek deepseek-flash／high／stream，explicit-v13、P4／w45，Poseidon CPU。

## 结果

33 项通过、1 项失败，跳过和未运行均为 0。28 项首次通过，5 项经过一次自动修复后通过。

| 类型 | 案例数 | 通过 |
|---|---:|---:|
| 一维 nearest 尺寸调整 | 3 | 3 |
| 一维 nearest-exact 尺寸调整 | 3 | 3 |
| 一维 linear 尺寸调整 | 3 | 3 |
| 二维 nearest 尺寸调整 | 3 | 3 |
| 二维 nearest-exact 尺寸调整 | 3 | 3 |
| 二维 bilinear 尺寸调整 | 3 | 2 |
| UpsamplingNearest2d／UpsamplingBilinear2d | 2 | 2 |
| ConvTranspose1d | 3 | 3 |
| ConvTranspose2d | 3 | 3 |
| 空间算子组合 | 3 | 3 |
| 字典、关键字与公开默认参数 | 3 | 3 |
| float32 公开权重与命名别名 | 2 | 2 |

通过项均完成真实编译、Poseidon 密态执行、解密与独立 reference 比较，以及程序包重载、新输入、重复执行和缓存检查。

通过项共 264 组输入、3,720 个输出值；最大绝对误差 4.1622965e-07，MAE 9.8809178e-09。门限保持 abs(actual-reference) ≤ 1e-5 + 1e-4×abs(reference)。这些数值统计仅包含通过项。

全批产生 40 次生成、40 次 HTTP、38 次实际编译和 104 次密态调用。39 个完整响应有 usage；另 1 次失败请求的 usage／费用未知，账单未核验。

## 剩余失败

resize_2_bilinear_0 的首次候选已经编译和执行，但插值系数错误，最大绝对误差约 1。独立公开数学检查发现 6 个系数不符。随后修复请求触发本地 SSE 大小限制，终态为 Provider 层失败。

原响应、源码、报告及绑定保留。同一模型已在 r1416 独立重测通过，结果见“单项补测结果”；原失败源码和原批次状态保持不变。

原 r1413 在 API 前被进程检查误识别自身 timeout 启动器而拒绝。独立 v2 工具已修正判断，50 项控制测试通过；真实独立任务仍会阻止新批次启动。

## 后续

继续进行 51 个来源上下文的当前捕获和独立参考，按 26 个精确公开请求开展真实生成；其余 25 个来源回放对应的原 Agent DSL，各自保存来源和程序包。

这批结果覆盖上述配置与源码写法。Bootstrap、CUDA、大模型容量及完整通用化计划仍有独立待验工作。此次后端库、安全参数、模型、reference、误差门限与网络保持不变，未提交或推送。

## 记录

机器报告：docs/baseline/agent-spatial-source-live-acceptance-r1408.json

报告绑定：facdb1e39ad7627e53ad2927e13b1fe831d3af8e00e8e86fd4d5453d2a249807

源码 checkout：/home/lhohy/Code/Poseidon，分支 lhy-agent-dsl。

## 单项补测结果

r1416 的原 resize_2_bilinear_0 已在独立真实 Agent 重测中通过。一轮自动修复，2 次生成／2 次 HTTP、2 次编译、4 次密态调用；8 组输入、192 个输出值，最大绝对误差 2.3164798544639265e-8。程序包重载、新输入、重复执行与缓存检查通过，r1417 审计通过。

原 34 项批次的 33 通过／1 失败记录保持不变。结合独立补测，34 个源码案例均已有通过记录；这是同一模型的新 Agent 程序通过，并非原失败 DSL 获得了通过。

补测报告绑定：2ee1812bc43049e1a83a3bbf34157342c9a5316a131446a1583321d27896120a
