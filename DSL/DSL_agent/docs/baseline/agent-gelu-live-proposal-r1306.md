# GELU 新入口真实 Agent 测试提案

范围：tanh module、functional 后接 square、input keyword＋tuple 输出三例。
模型、公开权重、近似多项式、请求与当前源码身份已冻结。
[机器提案](agent-gelu-live-proposal-r1306.json)记录逐例 request_id 与完整绑定。

DeepSeek deepseek-flash／high／stream，explicit-v11，固定 Poseidon w45。
每例最多 3 轮修复，每次生成最多 3 次传输重试；合计最多 12 次生成、48 次 HTTP。
API与密态串行，编译≤2、链接≤1；单次 1200 秒、384000 tokens。
整批 12 小时、每例 4 小时；每例 4 GiB、全局 32 GiB，磁盘余量 4 GiB。

外发公开规范化模型、固定权重或描述符、布局、DSL规则和受控反馈。
验收真实生成、编译、密态、四组数值、包重载新输入及缓存。
CKKS 门限保持 1e-5 + 1e-4*abs(reference)；原函数应用精度单独记录。

无明确货币上限；超时与传输重试可能重复计费。
当前生成、HTTP、安装、下载均为0，状态 awaiting_human_approval。
新付费批次按用户要求单独批准；此前r1304已经关闭。
