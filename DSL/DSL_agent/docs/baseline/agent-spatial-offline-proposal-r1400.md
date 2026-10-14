# 空间算子 Poseidon CPU 离线验收：r1400

状态：待批准。使用冻结规则候选；付费 API、HTTP、安装和下载为 0。

本批 45 项：18 个 resize、18 个 ConvTranspose、6 个组合、3 个 JSON 旧版本兼容。原模型、公开权重、独立 reference 和误差门限保持。45 项已经通过明文参考与候选静态检查。

| 限额 | 范围 |
|---|---|
| 编译／密态调用 | 最多 45 次编译／51 次 Poseidon CPU 调用 |
| 首次验收 | 每项 4 组冻结输入，逐项比较 |
| 包交付 | 45 包重载；3 包各执行 3 组新输入及 1 组重复输入，并检查无生成缓存复用 |
| 时间 | worker 40 分钟，启动器 45 分钟；首次任务 180 秒 |
| 并行与进程 | 密态串行，编译≤2／链接≤1；native 原 4 GiB／CPU150秒 |
| 空间 | 新增≤512 MiB，全局≤32 GiB，磁盘余量≥4 GiB |
| 请求及执行目录 | 原请求新增128 MiB、包执行新增64 MiB上限保持 |

43 项选择 P4，2 项选择 P8，使用已有 v9 公共布局、explicit-v13 和固定 Poseidon w45。packing 选择与加密参数分开，后端和安全参数保持。三个复用检查为 resize_1d_linear_0、resize_2d_bilinear_0、transpose_2d_1。

源码、模型、候选、请求及工具哈希在[机器方案](agent-spatial-offline-proposal-r1400.json)冻结。新 SDK：`388e891a44863d95e6a8ce7e83c4206799434e08e122c63ff58afa2b62452337`；方案绑定：`86b1c249c3844f350de4e2dc2304b08994d24d5744712eb2cd37585a74928202`。本方案取代未执行的 r1396/r1398，原预检失败、候选与绑定保留。

达到时间、空间或完整性边界时保留已完成和未运行状态，不追加调用、不删案例。单例编译或数值失败单独记录；环境与完整性失败停止相关批次。通过结果计作规则程序，真实 Agent 新生成单列。

## 案例

- `resize_1d_linear_0`
- `resize_1d_linear_1`
- `resize_1d_linear_2`
- `resize_1d_nearest_0`
- `resize_1d_nearest_1`
- `resize_1d_nearest_2`
- `resize_1d_nearest_exact_0`
- `resize_1d_nearest_exact_1`
- `resize_1d_nearest_exact_2`
- `resize_2d_bilinear_0`
- `resize_2d_bilinear_1`
- `resize_2d_bilinear_2`
- `resize_2d_nearest_0`
- `resize_2d_nearest_1`
- `resize_2d_nearest_2`
- `resize_2d_nearest_exact_0`
- `resize_2d_nearest_exact_1`
- `resize_2d_nearest_exact_2`
- `spatial_composition_0`
- `spatial_composition_1`
- `spatial_composition_2`
- `spatial_composition_3`
- `spatial_composition_4`
- `spatial_composition_5`
- `transpose_1d_0`
- `transpose_1d_1`
- `transpose_1d_2`
- `transpose_1d_3`
- `transpose_1d_4`
- `transpose_1d_5`
- `transpose_1d_6`
- `transpose_1d_7`
- `transpose_1d_8`
- `transpose_2d_0`
- `transpose_2d_1`
- `transpose_2d_2`
- `transpose_2d_3`
- `transpose_2d_4`
- `transpose_2d_5`
- `transpose_2d_6`
- `transpose_2d_7`
- `transpose_2d_8`
- `compat_v1_linear`
- `compat_v2_dot`
- `compat_v3_public_subtract`

## 执行入口

只读计划：

~~~bash
POSEIDON_PLATFORM=aarch64-linux POSEIDON_WORK_ROOT=/home/lhohy/poseidon-work \
  python3 -B scripts/baseline/benchmarks/tools/spatial_component_integration_v1/offline_spatial_r1400.py
~~~

取得本方案授权后，使用同一入口的 `--run-approved`。授权文件须绑定上述方案；当前尚未建立授权文件，也未启动编译或密态批次。

[接入结果](agent-spatial-component-checkpoint-r1399.md) · [用法](operator-decomposition-and-python.md#json-空间计算图)
