# PyTorch v7 真实 Agent 评测申请 r816

30个新入口上下文，十类矩阵用法各3例；自由生成，验证模型转换和自动修复。全部已通过规则候选的Poseidon CPU验收。

| 分类 | 精确案例 |
|---|---|
| 密文向量点积 | cipher_vv_0, cipher_vv_1, cipher_vv_2 |
| 密文矩阵乘向量 | cipher_mv_0, cipher_mv_1, cipher_mv_2 |
| 密文向量乘矩阵 | cipher_vm_0, cipher_vm_1, cipher_vm_2 |
| 多维密文批次广播 | cipher_batch_0, cipher_batch_1, cipher_batch_2 |
| 公开向量左乘 | public_vector_l_0, public_vector_l_1, public_vector_l_2 |
| 公开向量右乘 | public_vector_r_0, public_vector_r_1, public_vector_r_2 |
| 公开矩阵左乘 | public_matrix_l_0, public_matrix_l_1, public_matrix_l_2 |
| 公开矩阵右乘 | public_matrix_r_0, public_matrix_r_1, public_matrix_r_2 |
| 批量公开矩阵左乘 | public_batch_l_0, public_batch_l_1, public_batch_l_2 |
| 批量公开矩阵右乘 | public_batch_r_0, public_batch_r_1, public_batch_r_2 |

DeepSeek deepseek-flash/high/stream，explicit-v10、原Poseidon w45；每例最多3轮修复，每次生成最多3次传输重试；总计最多120次生成／480次HTTP。单次1200秒／384000 tokens；API与密态串行，编译≤2／链接≤1，全批12小时、每例4小时／4GiB、全局32GiB、余量4GiB。

只外发公开规范化图、固定权重或描述符、布局、DSL规则和受控反馈。首次请求按现有Provider白名单实际投影，最大47423字节，逐例哈希已冻结。规则答案、模型源码、私有测试数组／reference、凭据及私钥保持在执行端。

实际Agent源码、首次／修复后成功、失败层、预算耗尽、包重载和新输入复用分别记录。数值门限1e-5+1e-4*abs(reference)。没有明确货币上限；超时重试可能重复计费。状态为待批准，API调用0。

新增付费范围按当前计划单独批准；历史14项授权已结算。批准前仅保留准备记录。

[冻结请求／依赖与预算](agent-pytorch-matrix-live-proposal-r816.json) · [离线验收](agent-pytorch-matrix-acceptance-r815.md)。
