# 节点布局语义说明 v2：独立六例实验

## 改动边界

`candidate_contract.PACKED_GUIDANCE_V2` 为包含 `permute`/`transpose` 的 schema5 模型增加
版本化说明：`schema=2`、`revision=node-layout-v1`。旧 `PACKED_GUIDANCE` 保持原文与验证行为。
其他 packed 模型仍使用旧说明。这里只改变 Agent 收到的语义说明，不扩展 AST、算子、ABI，
不改 reference、权重、测试输入、数值门限、CKKS 参数或编译配置。

明确的两类规则：

- Conv/Linear 的公开行常量对应该节点的输入布局；不能省略其上游 permutation，也不能靠
  交换明文/密文乘法操作数补回缺失的重排。
- `permutation_shift[d]` 是目标位置掩码。周期为 P 时，正旋转读取
  `rotate(x,d)[j]=x[(j+d) mod P]`；应在旋转后的目标位置应用该掩码。
  源位置 s 到目标位置 j 的位移是 `(s-j) mod P`，组合位移仅使用已配置的正二次幂步长。
  scalar-neuron 表示则重排整个密文引用。

此处说明规则而非提供 golden 程序。只向模型发送公开合成图、公开常量、layout、FX 分析与规则；
不发送测试输入数组、reference、密钥或规则转换器源码答案。

## 冻结的实验设计

清单：`scripts/baseline/cases/layout-guidance-6-manifest.json`。

| 分组 | 案例 | 验证内容 |
|---|---|---|
| 旧失败重测 | perm-negative-axes | 三维负轴 permutation |
| 旧失败重测 | perm-nhwc-conv | NHWC 到 NCHW 再卷积 |
| 新组合 | perm-v2-negative-rank4 | 四维负轴 permutation，48 个元素 |
| 新组合 | perm-v2-nhwc-grouped | 重排后分组卷积 |
| 新组合 | perm-v2-nhwc-linear | 重排后按末轴 Linear |
| 新组合 | perm-v2-two-permutations-bn | 两次重排后 BatchNorm |

前两例描述与历史实验完全相同，但新请求哈希不同，作为独立实验而非失败批次续跑。
历史 12 例报告仍为首次 10/12、最终 12/12、15 次生成；不回写历史成功率。
每例固定四组输入，合计 24 组、512 个输出值。
即便新批首次全过，两个旧案例的单次重测也不足以证明提示改进具有统计显著性。

## 确定性基线：已完成

报告：`/home/lhy/poseidon-work/results/layout-guidance-goldens-xn_25n73/report.json`。
SHA256：`7a941d16497ce4539076adf5eb55685160af6605950a8df8f065f0454029402e`。

- 规则生成 6/6 通过，0 API；不是 Agent 成绩。
- Dacapo 编译后在上游 SEAL HEVM CPU 真实加密执行，再解密比较。
- 24 组输入、512 个值；最大绝对误差 `8.752389368815727e-08`。
- 门限仍为 `1e-5 + 1e-4*abs(reference)`；N=32768，tc128，无 bootstrap。
- 使用显式 `seal-cpu-eva-w45-v1`，不修改默认 W40。
- 自动清理 3596758807 字节临时密钥；保留模型、请求、编译产物和解密证据。
- 包含历史证据与新基线的 190 项回归通过，0 跳过。

## 复现入口

```bash
cd '/mnt/d/Code Space/Poseidon'
timeout -k 5s 3650s python3 scripts/baseline/run_packed_model_goldens.py --layout-guidance
```

付费入口（仅在明确授权范围内使用；API 并发10，密态并发2）：

```bash
timeout -k 10s 11000s python3 scripts/baseline/run_agent_batch.py \
  --deepseek --case-manifest scripts/baseline/cases/layout-guidance-6-manifest.json \
  --provider deepseek --model deepseek-flash --reasoning-effort high \
  --jobs 10 --api-timeout 1200 --max-tokens 384000 --provider-retries 3 \
  --stream --loopback-proxy-port 6478 --compiler-configuration seal-cpu-eva-w45-v1
```

最多24次生成、96次HTTP尝试；每例最多3轮修复，每次生成最多3次网络重试。
网络探针和 Agent 结果必须分别记录。没有人民币费用上限，不凭 token 数捏造实际账单。
本实验不证明所有输入的形式化等价、完整 DSL 支持或 Poseidon GPU 已端到端通过。

## 真实 Agent：已完成

报告：`/home/lhy/poseidon-work/results/agent-batch-cij14tb3/report.json`。
SHA256：`1dc8abc94e6caee2067c06a2f0bd9de6120439b985cf04bbcd99601b0de46547`。

- 首次 parse/check/compile/execute/numerical correctness 全部 6/6；最终 6/6。
- 6 次 DeepSeek 官方 `deepseek-flash/high` 生成，0 修复，0 网络重试；未达到24次生成上限。
- 24 组真实密态输入、512 个比较值；最大绝对误差 `7.812636987347688e-08`。
- 按输出元素数加权 MAE `2.2378984542141434e-09`；各例非零 reference 相对误差、cosine
  和逐项误差保存在原始报告，不仅比较标签。
- 输入 tokens 44744，输出 tokens 204461，总计 249205；全部6次有 usage 回执。
- 运行约223.5秒。这里不是性能基准，不与上次墙钟时间比较。

| 案例 | 首次生成 | 最大绝对误差 |
|---|---|---:|
| perm-negative-axes | 通过 | 6.31346e-09 |
| perm-nhwc-conv | 通过 | 5.10376e-09 |
| perm-v2-negative-rank4 | 通过 | 1.56857e-08 |
| perm-v2-nhwc-grouped | 通过 | 3.49423e-09 |
| perm-v2-nhwc-linear | 通过 | 3.06376e-08 |
| perm-v2-two-permutations-bn | 通过 | 7.81264e-08 |

两个旧失败模型本次首稿通过，四个新组合也首稿通过。**事实**是这六个固定案例通过；
**有证据的推断**是显式布局说明有助于避免此前错误；**未证实**的是统计显著的整体成功率提升。
没有多随机种子、随机对照或大规模重复实验，不能将这次6/6推广为所有布局永久正确。

只读独立审计：`/home/lhy/poseidon-work/results/agent-lineage-audit-d7sw8s1d/report.json`。
审计通过6/6，0新增API、0新增密态执行、0遗留私钥目录；检查原始响应、候选源码、trace、
产物哈希、真实执行元数据、独立reference及解密结果。两个旧模型除 guidance/request_id 外
请求字段应保持一致，历史报告哈希保持不变。

```bash
timeout -k 5s 230s python3 scripts/baseline/audit_agent_lineage.py \
  /home/lhy/poseidon-work/results/agent-batch-cij14tb3/report.json
```

TLS报告：`/home/lhy/poseidon-work/results/deepseek-tls-probe-e45dj5h3`。
默认直连3/3连接超时；6478 CONNECT代理3/3完成证书校验并收到无凭据请求预期的401。
短探针不保证未来长连接永不失败；本次实际生成6次均未发生重试。没有禁用证书校验。

付费批次自动清理3596758807字节临时密钥；与规则基线合计7193517614字节，约6.70 GiB。
原随机密钥不能恢复，但可通过重跑生成新密钥；模型、原始响应、IR、HEVM/CST和数值证据保留。
不是清理无关目录，不删除旧失败证据。

最终192项回归全部通过，0跳过，包含全部此前已登记真实密态证据、新规则基线、
新Agent批次以及两个旧失败请求仅guidance/request_id变化的检查。
改动保持本地未提交；完整DSL目标继续进行，不以本批通过标记整体目标完成。
