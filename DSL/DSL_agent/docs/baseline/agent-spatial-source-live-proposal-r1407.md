# PyTorch 空间入口真实 Agent 测试

本批按用户“目前执行的计划内后续所有的离线和真实测试全部批准同意”的直接授权执行。每批保留独立的案例、源码、请求、调用额度和资源绑定；旧批次额度保持关闭。

## 范围

34 个 v27 来源上下文：18 个 resize、6 个 ConvTranspose、3 个组合、7 个结构化输入／公开状态／别名和模块写法。来源均通过当前应用入口的隔离捕获和独立参考验证。自由生成由真实 DeepSeek 完成，规则候选不进入生成提示。

每个 resize 维度／模式分区安排三个上下文；1D／2D 转置卷积各三个上下文，包含模块、函数和 output_size 写法。来源、权重和模型保持冻结。重叠案例按新生成身份单独计分。

## 配置和额度

- DeepSeek deepseek-flash／high／stream，explicit-v13，P4／w45。
- 每例最多 3 轮修复、4 次生成、16 次 HTTP 尝试；每次生成最多 3 次传输重试。
- 全批最多 136 次生成、544 次 HTTP、136 次候选编译、204 次密态调用。
- 单次 API 1200 秒、384000 tokens；每例 4 小时、全批 24 小时。
- API 和密态串行，编译最多 2 jobs、链接 1；native 4 GiB／CPU150 秒保持。
- 新增产物最多 4 GiB，全局最多 32 GiB，磁盘余量至少 4 GiB。触限保留证据并终止相关执行。
- 无明确货币上限，超时或传输重试可能重复计费。
- 数值门限：abs(actual-reference) <= 1e-5 + 1e-4*abs(reference)。

外发公开分解图、权重或描述符、布局、DSL 规则和受控反馈；测试数组、reference、规则答案、凭据和私钥不进入提示。每个成功程序包还验证新输入、结构化重复输入、重新加载及无生成无编译缓存复用。

官方[Chat Completions 接口](https://api-docs.deepseek.com/api/create-chat-completion/)于本次冻结前重新核对；deepseek-flash、high、stream 和请求 token 上限符合当前文档及本地固定配置。

## 绑定

SDK：`5bd7c37b67d6e94cd28c6ad46660c17600f90e29b4c213fa5e41cc09d57239d6`

方案：`d1f7942b50a359ba32ab57738adc06209b07f9cbdc948ae58e141b39df46867c`

运行工具清单：`99993078e375525daeced2b13169c1edd47ccb0e4f7fef7c9580ca7d181e3d9d`

来源准备耗时 272.11 秒；API、编译和密态调用均为 0。38 项授权、预算、监督和程序包调用控制检查通过。后端、参数、依赖、网络、x86 和当前生产源码保持。

[精确机器方案](agent-spatial-source-live-proposal-r1407.json) · [本计划总授权](current-plan-test-authorization-2026-10-11.json) · [本批授权](agent-spatial-source-live-authorization-r1407.json)

## 案例

| 案例 | 输入 shape | 输出 | 请求哈希 |
|---|---|---|---|
| resize_1_nearest_0 | [[1, 1, 3]] | output | 547c5beae9c1ee0d1eb1bb3695ec8416358e8d27bd85c0e3b11af911e5e88b92 |
| resize_1_nearest_1 | [[1, 1, 3]] | output | 547c5beae9c1ee0d1eb1bb3695ec8416358e8d27bd85c0e3b11af911e5e88b92 |
| resize_1_nearest_2 | [[1, 1, 3]] | output | e2a344e5618b826689cb4a22c4f20bc269ca9362b1d53fc9a844a49fc5694182 |
| resize_1_nearest_exact_0 | [[1, 1, 3]] | output | 30123f804d9ee20c67808bd16917f8d241cbb025051077655a2e499270442736 |
| resize_1_nearest_exact_1 | [[1, 1, 3]] | output | 30123f804d9ee20c67808bd16917f8d241cbb025051077655a2e499270442736 |
| resize_1_nearest_exact_2 | [[1, 1, 3]] | output | 9afed5cdcf753befedf520084ce63f0a0d6aea6378488df7e40b9a4ddf0fffa5 |
| resize_1_linear_0 | [[1, 1, 3]] | output | a12a1039613eb2e539bb92041a50bc85de68a19b44b990d18282f322e5ad086b |
| resize_1_linear_1 | [[1, 1, 3]] | output | 7bd49805e6997c99dfedf342d508968eefe449716f73cc7502fe8589f84fabf5 |
| resize_1_linear_2 | [[1, 1, 3]] | output | fac335a38bbfd228d9055d09f0bed49ee440b34d78abbb9952cbfc89636b19f9 |
| resize_2_nearest_0 | [[1, 2, 2, 3]] | output | 12a9ddace6a05404b8e2366677e025c5b7f4bc991c89f6a90092e0a8e5848cad |
| resize_2_nearest_1 | [[1, 2, 2, 3]] | output | 5de981eeed9b49331ca283057712a0f1d89e042a6d910847105bbf38a83614c1 |
| resize_2_nearest_2 | [[1, 2, 2, 3]] | output | c7b698b48b38abdcd526bc13dfd7a3284d3f0a37d2ed62d7fc18f9503c38916d |
| resize_2_nearest_exact_0 | [[1, 2, 2, 3]] | output | f626d4af971a8a37a5dc0587f7285276e1f71811d74a670405c5515a7f49c4c8 |
| resize_2_nearest_exact_1 | [[1, 2, 2, 3]] | output | 864afa140b688373fec3c3a84775d08143dddde11c5521da538f33295f664ea1 |
| resize_2_nearest_exact_2 | [[1, 2, 2, 3]] | output | 727c216ececba2de06ff2b66278fa738472e3661d1b4008b95e283bc56abce39 |
| resize_2_bilinear_0 | [[1, 2, 2, 3]] | output | 127b64f2fc0de15678e7797f4636c619ded0130e06babdbb67fba7d874b37f87 |
| resize_2_bilinear_1 | [[1, 2, 2, 3]] | output | 10719d91140eb632679c5f778f149418d3f408e861e2399d267ebe0acd12d286 |
| resize_2_bilinear_2 | [[1, 2, 2, 3]] | output | e05a516103c8480a86ea9f16b9daac5db4be6919422f65252446c17138505fe8 |
| transpose_1_0 | [[1, 2, 3]] | output | d964019d8ce5ca426d919f9bf4283fb4142e9074a1583f7fb788d46c10e5b6eb |
| transpose_1_1 | [[1, 2, 3]] | output | b249c30589f5ea29d79d5d449142b22ff73c07a6bd2068483bfe3afed097cb57 |
| transpose_1_2 | [[1, 2, 3]] | output | d964019d8ce5ca426d919f9bf4283fb4142e9074a1583f7fb788d46c10e5b6eb |
| transpose_2_0 | [[1, 4, 2, 3]] | output | e9b121d4a435b1d552cdca9ce7d12e3b69e66844c1e01939bc384dad712dccbf |
| transpose_2_1 | [[1, 4, 2, 3]] | output | 81c89b864e0a4de06547d03f6f8ddd27b3ac8f8d52dedfb100f7b7e4f4d070dc |
| transpose_2_2 | [[1, 4, 2, 3]] | output | e9b121d4a435b1d552cdca9ce7d12e3b69e66844c1e01939bc384dad712dccbf |
| composition_0 | [[1, 1, 3]] | output | cf8078f449f94d3385c9926f2f4145ef3e7e9964406fcc396efe605789367619 |
| composition_1 | [[1, 1, 3]] | output | 405d9dcb6db5ad903994304ae9c00f67d60327ba722f9b144b0c7a3e8b3fd136 |
| composition_2 | [[1, 1, 3]] | before, after | 12490b9240552f68290f586b28248686b9e6f74306c8b6f5f36ed6d448bf4191 |
| structured_dict | [[1, 1, 3]] | result, square | 3dfc3b0245c042974a07e5a56e76b0ed7e6c36fa24dbf5a8a450e5c207227ff6 |
| nested_keywords | [[1, 1, 3], [1, 1, 3]] | output | a8428d69d49a3df09382839dff3896d8fbc8b631efff79589f76ed65a58d1aa4 |
| public_defaults | [[1, 1, 3]] | output | a12a1039613eb2e539bb92041a50bc85de68a19b44b990d18282f322e5ad086b |
| float32_public_weight | [[1, 1, 3]] | output | dd56f18b9361ee8c742cd6980b4f82301c270483974a331434020cf2099cd3f6 |
| named_alias | [[1, 1, 3]] | before, after | 7cbe490f81219d3df627a1d9d278e812a1bb4359750b6108aec0884eccd5277d |
| nearest_2d_module | [[1, 1, 2, 3]] | output | a691f5fe4ae902061e02c5e7712974b0f23168781cc9c7b1bce248ffb7506ed2 |
| bilinear_2d_module | [[1, 1, 2, 3]] | output | f9168669955d429d452c56c82e4c99e03a6556ee2e57cb5a1f2059a00efa4be6 |

## 执行

~~~sh
POSEIDON_PLATFORM=aarch64-linux POSEIDON_WORK_ROOT=/home/lhohy/poseidon-work \
python3 -B scripts/baseline/benchmarks/tools/spatial_source_live_v1/run.py \
  --execute /home/lhohy/poseidon-work/platforms/aarch64-linux/results/agent-spatial-source-live-v1 \
  --approval-record /home/lhohy/Code/Poseidon/docs/baseline/agent-spatial-source-live-approval-r1407.json
~~~

失败、阻塞、未运行和未知调用不计通过；未知历史调用不补发。没有自动重启或自动扩限。安装、下载、后端库与参数修改、网络修改、commit、push 和 PR 均不在本次总测试授权内。
