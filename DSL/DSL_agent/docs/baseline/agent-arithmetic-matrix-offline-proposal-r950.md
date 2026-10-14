# 当前源码下的联合离线验收

状态：待批准。绑定：`49b5bee5cc88b9cc6ff9d7782895d6b73df3545a40043076679384645efaf4ac`。

复用原48项及已捕获的新入口案例。候选为固定规则源码；真实 Agent 新生成与 API 为0。旧失败、原模型、公开权重、reference及门限保留。

| 分片 | 内容 | 项数 |
|---|---|---:|
| 1 | 原8个编译失败、4个兼容例、36个JSON矩阵例 | 48 |
| 2 | 21个JSON算术与25个PyTorch入口例 | 46 |

第二片含同一数学计算的跨入口、写法及dtype变体；94为入口任务数量。原已通过的split24项不重跑。

验收包括真实Hecate tracing、Earth/CKKS IR、RuntimePlan／OperatorSpec／常量bundle、Poseidon CPU、四组输入解密比较；九份代表程序包重载、三组新输入各执行两次及缓存复用。失败与未运行单列。

预算：总90分钟、每项180秒；两片串行，密态1、编译≤2、链接≤1。新增产物≤4GiB、全局≤32GiB、磁盘余量≥4GiB。每个候选仍使用原沙箱与资源限制。

API、安装、下载、SDK重建、后端库及参数修改、删除和Git发布均0。数值条件保持 `abs(actual-reference) <= 1e-5 + 1e-4*abs(reference)`。

已核验681个运行源码、1367份案例／证据文件和4个新驱动。未获批准启动检查已确认在执行前拒绝；后端目录与批准收据尚未创建。

旧r914因共用源码更新而保留为未执行历史绑定。r870的30分钟窗口已在r878完成关闭；新的有限批次需单独批准。

批准并建立范围收据后的入口：

~~~bash
POSEIDON_PLATFORM=aarch64-linux POSEIDON_WORK_ROOT=/home/lhohy/poseidon-work \
python3 -B scripts/baseline/benchmarks/tools/frozen_arithmetic_backend/launch.py
~~~

## 案例清单

| 片 | 案例 | 入口 | 原状态 |
|---:|---|---|---|
| 1 | `attention_core_0` | trusted_pytorch_v7 | failed |
| 1 | `attention_core_1` | trusted_pytorch_v7 | failed |
| 1 | `attention_core_2` | trusted_pytorch_v7 | failed |
| 1 | `channel_broadcast_mix_0` | trusted_pytorch_v7 | failed |
| 1 | `channel_broadcast_mix_1` | trusted_pytorch_v7 | failed |
| 1 | `channel_broadcast_mix_2` | trusted_pytorch_v7 | failed |
| 1 | `norm_depth_branches_1` | trusted_pytorch_v7 | failed |
| 1 | `norm_depth_branches_2` | trusted_pytorch_v7 | failed |
| 1 | `indexed_batched_matmul_1` | trusted_pytorch_v7 | passed |
| 1 | `cnn_norm_head_1` | trusted_pytorch_v7 | passed |
| 1 | `public_weight_1` | trusted_pytorch_v7 | passed |
| 1 | `static_split_stack_1` | trusted_pytorch_v7 | passed |
| 1 | `matrix_matrix_0` | declared_json_v2 | not_run |
| 1 | `matrix_matrix_1` | declared_json_v2 | not_run |
| 1 | `matrix_matrix_2` | declared_json_v2 | not_run |
| 1 | `batched_cipher_0` | declared_json_v2 | not_run |
| 1 | `batched_cipher_1` | declared_json_v2 | not_run |
| 1 | `batched_cipher_2` | declared_json_v2 | not_run |
| 1 | `batch_broadcast_0` | declared_json_v2 | not_run |
| 1 | `batch_broadcast_1` | declared_json_v2 | not_run |
| 1 | `batch_broadcast_2` | declared_json_v2 | not_run |
| 1 | `higher_rank_cipher_0` | declared_json_v2 | not_run |
| 1 | `higher_rank_cipher_1` | declared_json_v2 | not_run |
| 1 | `higher_rank_cipher_2` | declared_json_v2 | not_run |
| 1 | `vector_left_0` | declared_json_v2 | not_run |
| 1 | `vector_left_1` | declared_json_v2 | not_run |
| 1 | `vector_left_2` | declared_json_v2 | not_run |
| 1 | `vector_right_0` | declared_json_v2 | not_run |
| 1 | `vector_right_1` | declared_json_v2 | not_run |
| 1 | `vector_right_2` | declared_json_v2 | not_run |
| 1 | `public_right_0` | declared_json_v2 | not_run |
| 1 | `public_right_1` | declared_json_v2 | not_run |
| 1 | `public_right_2` | declared_json_v2 | not_run |
| 1 | `public_left_0` | declared_json_v2 | not_run |
| 1 | `public_left_1` | declared_json_v2 | not_run |
| 1 | `public_left_2` | declared_json_v2 | not_run |
| 1 | `scalar_chain_0` | declared_json_v2 | not_run |
| 1 | `scalar_chain_1` | declared_json_v2 | not_run |
| 1 | `scalar_chain_2` | declared_json_v2 | not_run |
| 1 | `public_dot_0` | declared_json_v2 | not_run |
| 1 | `public_dot_1` | declared_json_v2 | not_run |
| 1 | `public_dot_2` | declared_json_v2 | not_run |
| 1 | `matrix_chain_0` | declared_json_v2 | not_run |
| 1 | `matrix_chain_1` | declared_json_v2 | not_run |
| 1 | `matrix_chain_2` | declared_json_v2 | not_run |
| 1 | `scalar_domain_0` | declared_json_v2 | not_run |
| 1 | `scalar_domain_1` | declared_json_v2 | not_run |
| 1 | `scalar_domain_2` | declared_json_v2 | not_run |
| 2 | `arithmetic_add_scalar_left` | declared_json_v3 | new_input_not_run |
| 2 | `arithmetic_add_scalar_right` | declared_json_v3 | new_input_not_run |
| 2 | `arithmetic_add_tensor_left` | declared_json_v3 | new_input_not_run |
| 2 | `arithmetic_add_tensor_right` | declared_json_v3 | new_input_not_run |
| 2 | `arithmetic_add_broadcast_left` | declared_json_v3 | new_input_not_run |
| 2 | `arithmetic_add_broadcast_right` | declared_json_v3 | new_input_not_run |
| 2 | `arithmetic_subtract_scalar_left` | declared_json_v3 | new_input_not_run |
| 2 | `arithmetic_subtract_scalar_right` | declared_json_v3 | new_input_not_run |
| 2 | `arithmetic_subtract_tensor_left` | declared_json_v3 | new_input_not_run |
| 2 | `arithmetic_subtract_tensor_right` | declared_json_v3 | new_input_not_run |
| 2 | `arithmetic_subtract_broadcast_left` | declared_json_v3 | new_input_not_run |
| 2 | `arithmetic_subtract_broadcast_right` | declared_json_v3 | new_input_not_run |
| 2 | `arithmetic_multiply_scalar_left` | declared_json_v3 | new_input_not_run |
| 2 | `arithmetic_multiply_scalar_right` | declared_json_v3 | new_input_not_run |
| 2 | `arithmetic_multiply_tensor_left` | declared_json_v3 | new_input_not_run |
| 2 | `arithmetic_multiply_tensor_right` | declared_json_v3 | new_input_not_run |
| 2 | `arithmetic_multiply_broadcast_left` | declared_json_v3 | new_input_not_run |
| 2 | `arithmetic_multiply_broadcast_right` | declared_json_v3 | new_input_not_run |
| 2 | `arithmetic_shared_multi_output` | declared_json_v3 | new_input_not_run |
| 2 | `arithmetic_matrix_chain` | declared_json_v3 | new_input_not_run |
| 2 | `arithmetic_broadcast_branches` | declared_json_v3 | new_input_not_run |
| 2 | `v8_add_scalar_left_operator` | trusted_pytorch_v8 | new_input_not_run |
| 2 | `v8_add_scalar_right_operator` | trusted_pytorch_v8 | new_input_not_run |
| 2 | `v8_add_tensor_left_operator` | trusted_pytorch_v8 | new_input_not_run |
| 2 | `v8_add_tensor_right_operator` | trusted_pytorch_v8 | new_input_not_run |
| 2 | `v8_add_broadcast_left_operator` | trusted_pytorch_v8 | new_input_not_run |
| 2 | `v8_add_broadcast_right_operator` | trusted_pytorch_v8 | new_input_not_run |
| 2 | `v8_subtract_scalar_left_operator` | trusted_pytorch_v8 | new_input_not_run |
| 2 | `v8_subtract_scalar_right_operator` | trusted_pytorch_v8 | new_input_not_run |
| 2 | `v8_subtract_tensor_left_operator` | trusted_pytorch_v8 | new_input_not_run |
| 2 | `v8_subtract_tensor_right_operator` | trusted_pytorch_v8 | new_input_not_run |
| 2 | `v8_subtract_broadcast_left_operator` | trusted_pytorch_v8 | new_input_not_run |
| 2 | `v8_subtract_broadcast_right_operator` | trusted_pytorch_v8 | new_input_not_run |
| 2 | `v8_multiply_scalar_left_operator` | trusted_pytorch_v8 | new_input_not_run |
| 2 | `v8_multiply_scalar_right_operator` | trusted_pytorch_v8 | new_input_not_run |
| 2 | `v8_multiply_tensor_left_operator` | trusted_pytorch_v8 | new_input_not_run |
| 2 | `v8_multiply_tensor_right_operator` | trusted_pytorch_v8 | new_input_not_run |
| 2 | `v8_multiply_broadcast_left_operator` | trusted_pytorch_v8 | new_input_not_run |
| 2 | `v8_multiply_broadcast_right_operator` | trusted_pytorch_v8 | new_input_not_run |
| 2 | `v8_add_tensor_right_method` | trusted_pytorch_v8 | new_input_not_run |
| 2 | `v8_add_tensor_left_function` | trusted_pytorch_v8 | new_input_not_run |
| 2 | `v8_subtract_tensor_right_method` | trusted_pytorch_v8 | new_input_not_run |
| 2 | `v8_subtract_tensor_left_function` | trusted_pytorch_v8 | new_input_not_run |
| 2 | `v8_multiply_tensor_right_method` | trusted_pytorch_v8 | new_input_not_run |
| 2 | `v8_multiply_tensor_left_function` | trusted_pytorch_v8 | new_input_not_run |
| 2 | `v8_add_tensor_left_operator_float32` | trusted_pytorch_v8 | new_input_not_run |

[完整绑定](agent-arithmetic-matrix-offline-proposal-r950.json) · [接入验收](agent-arithmetic-intake-acceptance-r949.md)。
