# PyTorch v22 真实 Agent 测试方案 r1364

## 目的与范围

验证 v22 公开张量算术入口的真实生成和预算内自动修复。复用现有应用组件、Provider 和修复循环。
已有规则候选的离线结果保留；本批对这五个模型生成新候选，单独记录生成来源、首次/修复后通过与失败层。

| 案例 | 计算 | 请求哈希 |
|---|---|---|
| bool_multiply | 公开布尔 mask × 张量 | 844c39dc591887d1e7b3729b46df906081d5172e08d5ba7ef710a99bfdfaa22f |
| mask_index_residual | 公开除法＋整数索引＋布尔掩码 | 05434d2e8f47cfd3f3fe91edd4f84c50ee4cdbc01b217f65a455276a95b844f2 |
| method_channel_mask | 公开除法＋按行掩码＋残差 | f4e624a999f06fc5ff91d4c99fae444bd910a30a654204a631846be90e4a6a21 |
| float32_batch_norm | float32 权重转换＋公开除法＋BatchNorm | 74482400297ab07d5103c7aaff4a49eefebcc7d5b55ec0f8d99f31ab436cc680 |
| divided_gelu_linear | 公开除法＋显式 GELU 近似＋Linear | 7adda19b6eaaa99bc9ff172dc0a87ef3330c72a0abc0498e4414180acd0d2864 |

输入均为固定 shape [2,3]；权重、模型、近似配置、reference 和门限冻结。
bool_multiply 的规则验收来自较早生产绑定，其余四例来自 r1363；原记录与来源全部保留。
GELU 按声明的近似计算作 CKKS 比较，原函数应用精度另计。

## 调用与资源上限

- DeepSeek deepseek-flash / high / stream，explicit-v13，P4 / Poseidon w45。
- 每例最多 3 轮修复、4 次生成；每次生成最多 3 次传输重试、4 次 HTTP。
- 全批最多 20 次生成、80 次 HTTP；单次 1200 秒、384000 tokens。
- API 与密态串行；编译最多 2 jobs，链接最多 1 job。
- 每例最多 4 小时、全批最多 12 小时；新产物合计最多 4 GiB，总产物最多 32 GiB，磁盘余量至少 4 GiB。
- 保留 native 4 GiB 内存与 CPU 150 秒限制。全局边界优先，达到边界时保留证据。
- 没有明确货币硬上限。超时与传输重试可能重复计费。

只外发公开模型/分解图、公开权重或描述符、布局、DSL 规则和受控诊断。
规则候选、私有测试输入、reference 输出、凭据、私钥均从生成上下文排除。
相应公开请求文件为 59–73 KB，已分别绑定。

[官方 Chat Completions](https://api-docs.deepseek.com/api/create-chat-completion/)当前列出 deepseek-flash、
high 与最大 393216 tokens；本批保留既有 384000 上限。

## 验收

真实源码捕获 → 新 Agent 候选 → tracing/编译 → RuntimePlan/OperatorSpec/bundle →
Poseidon CPU 密态执行 → 解密和四组输入逐项数值比较 → 包重载 → 三组新输入、一次重复执行与缓存复用。

门限保持 1e-5 + 1e-4*abs(reference)。人工候选不计入本批 Agent 成功。
失败、未运行、中断和未知调用单独记录；无自动续跑或预算扩限。

## 冻结与授权

- 分支 lhy-agent-dsl，HEAD 7588e48093631eaf8385506e3db058e7ca3cfe7e。
- 提案绑定 8752cc0b9e0fb913dedea3f717be0320e5f0339802cd046603a04a772a1a21e2。
- SDK 1066e8ae9e5ffed034a3c8341ac25aa4d1603f1070d2cec67235475a6ec8a7b6。
- 来源、权重、请求、四阶段代码、后端与控制脚本哈希见同名 JSON、runner-manifest.json。
- 34 项审批、账本、截止时间和进程归属检查通过；failed/errors/skipped0。静态/控制测试不计为密态验收。
- 当前 API/HTTP、编译及密态调用均为 0；授权记录尚未建立。
- 网络配置、依赖、后端库和参数保持。

只读入口：

~~~bash
POSEIDON_PLATFORM=aarch64-linux POSEIDON_WORK_ROOT=/home/lhohy/poseidon-work \
  python3 -B scripts/baseline/benchmarks/tools/public_tensor_live_v1/run.py --plan
~~~
