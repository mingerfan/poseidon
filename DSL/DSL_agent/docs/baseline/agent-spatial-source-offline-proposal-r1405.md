# PyTorch 空间入口 Poseidon CPU 离线验收 r1405

18 项已完成来源接入、独立参考与候选静态检查。使用冻结规则候选验证 v27 新入口，API 调用为 0。

| 范围 | 数量 |
|---|---:|
| 1D/2D resize 的模块和函数写法 | 6 |
| ConvTranspose1d/2d 的模块和函数写法 | 4 |
| resize→Linear、别名修改与多输出 | 2 |
| 字典／布尔 mask、嵌套多输入、float32 公开权重转换 | 3 |
| 旧 v26 关键字／嵌套调用、JSON v1 Linear | 3 |

额外检查 3 份程序包：字典输入、嵌套多输入、float32 公开权重。每包执行 3 组新输入、1 组重复输入及 3 组命名等价输入，再核对缓存复用。

## 预算与边界

- 最多 18 次真实编译、27 次 Poseidon CPU 密态调用。
- worker 40 分钟／启动器 45 分钟；首次任务 180 秒。
- 编译最多 2 jobs、链接 1 job、密态串行；native 地址空间 4 GiB／CPU 150 秒。
- 新增产物最多 512 MiB，全局最多 32 GiB，磁盘余量至少 4 GiB；触限保留现场。
- 四组初次验收输入，逐项门限保持 1e-5 + 1e-4*abs(reference)。
- API、安装、下载均 0；网络、后端、参数、模型及 reference 保持。

r1400 的 45 项已结算；本批属于新增入口与源码绑定的独立验收范围，当前待批准。

## 精确案例

- resize_1_nearest_0（P4/w45）
- resize_1_nearest_exact_1（P4/w45）
- resize_1_linear_0（P4/w45）
- resize_2_nearest_1（P4/w45）
- resize_2_nearest_exact_0（P4/w45）
- resize_2_bilinear_2（P4/w45）
- transpose_1_0（P4/w45）
- transpose_1_1（P4/w45）
- transpose_2_0（P4/w45）
- transpose_2_1（P4/w45）
- composition_0（P4/w45）
- composition_2（P4/w45）
- structured_dict（P4/w45）
- nested_keywords（P4/w45）
- float32_public_weight（P4/w45）
- compat_v26_keywords0（P4/w45）
- compat_v26_nested0（P4/w45）
- compat_v1_linear（P4/w45）

## 绑定与运行

SDK：`5bd7c37b67d6e94cd28c6ad46660c17600f90e29b4c213fa5e41cc09d57239d6`

方案：`c563ee93d44018613d6ee4affc315414efacbace38a77fb5e81a60e2e9f7e592`

[完整请求及文件哈希](agent-spatial-source-offline-proposal-r1405.json) · [来源接入](agent-spatial-source-component-checkpoint-r1404.md)

只读检查：
~~~bash
POSEIDON_PLATFORM=aarch64-linux POSEIDON_WORK_ROOT=/home/lhohy/poseidon-work \
  python3 -B scripts/baseline/benchmarks/tools/trusted_spatial_intake_v1/offline_source_r1405.py
~~~

取得本批授权后，使用同一入口的 --run-approved。旧结果和原绑定保留，新结果不计为真实 Agent 生成。
