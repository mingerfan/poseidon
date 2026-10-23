# Attention Poseidon CPU 验收 r242

原 Attention decode 的数学定义、公开权重、六阶 exp 和三轮倒数近似均保留。新增人工等价表达，将共享公开投影和 RoPE 的当前 q/k 点积写成带公开系数的平方和；缓存点积提前放置原公开系数，KV 输出仍使用原计算。没有修改编译器、Runtime、Poseidon 库或安全参数。

| 验收 | 结果 |
|---|---|
| 两套独立明文 reference | 每个表达配置 16 组探针通过，最大差异 2.77556e-17 |
| 相关单元测试 | 9 passed / 0 failed / 0 skipped |
| w45：真实 tracing、编译、密态执行、解密比较 | 四组输入、40 值通过，最大误差 5.20717e-9 |
| w40：真实 tracing、编译、密态执行、解密比较 | 四组输入、40 值通过，最大误差 1.91664e-7 |
| 应用数值任务 | 自动选择 explicit-v10，通过；最大误差 6.65792e-9 |
| v4 程序包重载、缓存复用 | 通过；重载无生成/编译调用，缓存无 Provider 调用 |

所有密态比较采用 `1e-5 + 1e-4*abs(reference)`。三个输出分别为预测、更新后的 key 和 value 缓存。本案例是固定长度、单头、二维小模型，不代表任意 Attention 规模、多头或原 Softmax 的近似精度达标。

原规则基线及四项旧编译失败记录保持原绑定。新通过程序属于人工候选，真实 Agent 成功数没有增加，安装、下载和付费调用为 0。本次不依赖 bootstrap；这说明该原案例存在现参数下的可执行等价表达，不能把此前失败全部归因于后端不支持。

结果目录：`/home/lhohy/poseidon-work/platforms/aarch64-linux/results/poseidon-attention-score-r242`。

下一步是三个既定小模型的 Poseidon 真实 Agent 评测，须先冻结请求和调用预算。r235 独立 bootstrap 参数试验仍待单独批准；x86 原生、空白机器安装和 CUDA 暂缓。

[机器报告](poseidon-attention-acceptance-r242.json)。
