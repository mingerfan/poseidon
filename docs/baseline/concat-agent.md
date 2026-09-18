# 静态 concat：模型语义、显式布局转换与真实测试

## 位置与语义

本次补齐受限模型图/FX的静态拼接，输入仍为公开固定权重和加密输入。
`model_graph`校验数据图并构建可信Torch模型；`concat_ops`计算shape与C-order索引；
`fx_to_hecate.Emitter.concat`把该语义降低成既有Hecate算术；Agent生成目标仍是Hecate Python。
不是新建FHE执行后端，也没有给候选开放文件、网络或任意Python执行权限。

- 支持torch.cat / torch.concat / torch.concatenate；dim可显式位置参数或关键字。
- 静态rank1..4，负轴按rank归一化，非拼接轴尺寸必须完全一致，不隐式broadcast。
- 1..8个非空分支，输出总元素数最多8；上限沿用当前scalar-neuron计算域。
- 当前用户输入ABI仍是每个输入4个逻辑元素；最终输出仍最多4个。
  8元素中间拼接可以再接Linear降到合法输出宽度。
- 已是scalar-neuron的值只按C-order重排句柄。
- packed period4输入先用one-hot公开掩码乘法和rotate-and-sum取出各元素，
  使每个结果密文的所有slot重复同一标量，再按目标轴拼接。
  不能只rotate到slot0就把它当作这种表示。
- 单元测试检查所有4个周期slot，而非仅最终selector读出的slot0。

模型reference使用独立的嵌套列表递归拼接；不用lowering的flat-index映射、
候选输出或解密结果来产生reference。另与原Torch模型交叉验证。

### 与上游HE_Concat的区别

固定Dacapo的`python/poly/poly/Func.py:HE_Concat`调用`close["CC"]`。
实际closure在`MPCB.py:Concat`，包含整密文数组拼接与部分slot拼接两条路径，
依赖ci/wi/hi/nt/po/ni/no等packing参数；CascadeConcat还限制特定channel shape。
本次不声称该closure或完整MPP packing已验证；是用现有原语实现明确的逻辑concat。
一般packing、空tensor、任意规模和上游helper直接调用仍是独立缺口。

## 冻结六例与人工判别

| 案例 | 覆盖 |
|---|---|
| concat-vector | 两个scalar-neuron分支2+2拼接 |
| concat-packed | 两个独立packed输入4+4，后接Linear |
| concat-axis0 | 二维轴0拼接，packed与scalar-neuron混合 |
| concat-axisneg | 二维负轴-1拼接，非平凡交错C-order |
| concat-three | 不等长三分支1+2+1拼接 |
| concat-spatial | 两个Conv1D结果沿空间轴拼接 |

每例四组固定输入；人工正确程序6/6、确定性转换器6/6均已通过真实
Hecate→Dacapo Earth/CKKS→HEVM/CST→upstream SEAL HEVM CPU→解密差分。
这不是Poseidon GPU证据。

人工错误顺序反例最大绝对误差2.000000002471632，错误轴反例4.593750002737876，
均在numerical_comparison拒绝，不把静态拒绝误算成密态反例。
人工正确程序最大绝对误差6.434130321616595e-8。
逐项门限固定为`abs(actual-reference)<=1e-5+1e-4*abs(reference)`。
CKKS参数保持N32768、[60]*14、tc128；没有bootstrap或decrypt-and-reencrypt。

人工证据：`/home/lhy/poseidon-work/results/concat-goldens-xb27kptt/report.json`。
规则转换器：`/home/lhy/poseidon-work/results/fx-batch-bdpd03b9/report.json`。
冻结清单：`scripts/baseline/cases/concat-agent-6-manifest.json`。

## 网络门禁及付费配置

付费前发现127.0.0.1:6478立即connection refused；Windows未查到该端口监听。
未改代理软件、.env或系统配置。现有probe_deepseek_tls.py对默认网络路径和
显式6478代理各进行三次无凭据GET /models：默认路径3/3返回401且TLS1.3
证书/主机名验证开启，显式代理3/3连接失败。
证据：`/home/lhy/poseidon-work/results/deepseek-tls-probe-hrmz8mme/report.json`。
短GET通过不保证长生成响应稳定，也不证明TUN/透明路由后没有中间网络组件。

本批仅使用命令参数`--loopback-proxy-port 0`，不持久化网络更改。
deepseek-flash/high，API并发上限10（本批最多6例）、native并发2，
max_tokens384000、API timeout1200s、传输重试最多3次、语义反馈修复最多3轮。
付费批次：`/home/lhy/poseidon-work/results/agent-batch-5l44gvj8/report.json`。
终态为6/6首次parse/static/compile/execution/numerical通过。实际6次API、
0次语义反馈修复、0次传输重试，prompt43068/completion66466/total109534 tokens，
usage缺失0；耗时202.13023138046265秒。未查询货币账单。

## 最终数值与独立审计

| 案例 | Agent最大绝对误差 |
|---|---:|
| concat-vector | 7.2570722831621914e-6 |
| concat-packed | 1.482733419227904e-7 |
| concat-axis0 | 2.1493208324230295e-9 |
| concat-axisneg | 6.48399396396826e-8 |
| concat-three | 8.406687790341039e-9 |
| concat-spatial | 1.8239180388235354e-8 |

24组密态输入、72个输出值，Agent加权MAE7.732637719182459e-7。
同模型的规则转换器最大绝对误差9.878178719446851e-8。
两者均通过固定门限，不据此声称数值质量完全相同。

concat-vector的Agent程序生成了`x*c0`及`x.rotate(1/2/3)*c0`四个返回值，
采用最终输出selector读取slot0，不是人工程序的每元素全slot归约。
最终输出契约不要求未读取slot是重复标量；中间scalar-neuron表示的全slot不变量
则仍由确定性lowering及相应单元测试验证。不能把这两种要求混同。

该候选的误差高于其他模型，故额外对同一原始response/request执行两次
新随机密钥离线重放，不重新生成、不付费、不改参数。两次都通过，最大误差
7.161368665964396e-6与6.713386591250536e-6。
这仅增加所测输入/密钥稳定性证据，不证明全部随机密钥或输入都通过，
误差根因尚未完全隔离，不能仅凭opcode或写法不同归因。
重放结果不计入“新Agent生成6例”的分母。

独立审计工具：`scripts/baseline/audit_concat_batch.py`，审计0次API/0次新FHE执行。

| 证据 | SHA256 |
|---|---|
| concat-goldens-xb27kptt/report.json | 060afdc3258b101bdd20e4ff95b2bebda7a440aee3e155f2073282d3c06ad9cb |
| fx-batch-bdpd03b9/report.json | fb444021d2d5f7e52544949f734c69f5b522079abd30ae96369281ac2548fa97 |
| agent-batch-5l44gvj8/report.json | bdb1b5cababfecee9af30c92e072681fb3f76828aa9f29eb6a92b7890efa6ef5 |
| concat-agent-audit-8slg7w79/report.json | 9ef61092f1389156239a5da2d7a9e317e3d3365faae4154deafc5bb76d662287 |
| concat-stability-g63hnuws/report.json | 2e529d4f7c013be09b99ca9e714e32f7303b4fefdbc242674764ae539dc5be70 |

表内证据均位于`/home/lhy/poseidon-work/results`。
冻结manifest SHA256：f44b92d4743037a9921e2b1622e543f5bf8656a62044571c3f387a64d571406b。
新累计模型域审计：`model-capability-audit-f155_n1s/report.json`，
141例=93个graph案例+48个历史catalog案例，当前16个有界graph算子均有
output-reachable的实际生成和数值证据，不等于完整DSL或全模型域已验证。
原135/132/126/114报告均保留；历史再审计不会追认concat覆盖。
该141例审计SHA256为72968bd8d46e31e3404abaacedc0a2bf1f302c401054a667fe28e909540fa9c0。
它不包含后文单独新增的1例轴0判别补批，不能改称142例统一审计。

## 已定位并修复的前置问题

- FX将构造列表存为immutable_list，原exact-list检查误拒绝concat。
  只把FX静态容器递归归一化为受检查的list/tuple，不扩大候选运行权限。
- 新历史源码归档由PowerShell读取时多了一个末尾空行，导致hash不一致。
  移除该多余空行后恢复原始SHA256；原135/132/126/114报告不修改。
- 一次多文件补丁第三文件上下文不匹配，前两个文件已更新；检查状态后补齐测试，
  没有重置用户文件。随后前置43项测试为41通过、2条件跳过。

## 回归入口

`test_concat`：几何轴映射、1..8分支、独立Torch/reference、所有scalar周期slot、
混合表示、别名入口及非法参数。
`test_concat_evidence`：真实人工正反例、冻结哈希、模型/输入/门限/安全参数和解密值。
`test_concat_paid`：规则对照、真实Agent、旧135例不追认concat覆盖。

实际证据测试必须显式设置：

```text
POSEIDON_CONCAT_GOLDENS=/home/lhy/poseidon-work/results/concat-goldens-xb27kptt/report.json
POSEIDON_CONCAT_RULE=/home/lhy/poseidon-work/results/fx-batch-bdpd03b9
POSEIDON_CONCAT_AGENT=/home/lhy/poseidon-work/results/agent-batch-5l44gvj8
POSEIDON_PRE_CONCAT_CAPABILITIES=/home/lhy/poseidon-work/results/model-capability-audit-nqn9hf4f/report.json
POSEIDON_CONCAT_STABILITY=/home/lhy/poseidon-work/results/concat-stability-g63hnuws/report.json
```

未设置则明确skip。通过既有hecate_python_env.enter_nix进入固定环境，
变量以env K=V形式传入Nix内部，不依赖外层变量隐式透传。

判别补批前47模块回归457项：435通过、22条件跳过、0失败。
模块为上一轮logical-reshape-agent.md记录的44项，加test_concat、
test_concat_evidence、test_concat_paid；旧真实证据环境变量保持，并加上本页5项。
provider retry日志来自mock故障注入，不是额外付费请求。

仅离线复核本次付费证据：

```bash
cd '/mnt/d/Code Space/Poseidon'
timeout -k 10s 500s python3 scripts/baseline/audit_concat_batch.py \
  /home/lhy/poseidon-work/results/agent-batch-5l44gvj8
```

人工、规则、付费、稳定性重放自动清理的可再生测试密钥分别为
5,431,840,080、297,283,048、4,073,880,060、1,357,960,020 bytes，
合计11,160,963,208 bytes（约10.39GiB），这是累计清理量，不是峰值占用。
已删除密钥不可恢复，可重生成；原请求、源码、模型/输入、产物、解密数组和诊断保留。
结束时WSL可用约189.15GiB，没有更改.env或系统网络设置。

完整DSL目标仍未完成；这六例不能穷举所有shape、分支数、轴和后续图组合。
保留本地working tree、原分支和sparse checkout；不commit/push。

## 轴0测试盲点与独立判别补批

收尾源码复核发现：原concat-axis0的Linear只读取拼接flat位置0/1/6/7，
对于两个2x2输入，这四个位置在轴0与轴1拼接中恰好相同。因此原模型结果
通过是有效的，但不能单独证明中间轴0拼接的完整顺序；不把它当作失败模型，
也不把“节点在输出路径上”升级为全部元素都有输出影响。

新增独立模型concat-axis0-sensitive使用覆盖全部8个位置的两行权重。
原6例manifest、权重、response和成功率不改写。手算输入x=[[1,2],[3,4]]、
y=[[5,6],[7,8]]，正确轴0结果[204,2.5]，错误轴1结果[196,4]。
回归同时保留旧盲点与新判别性，不能通过删除旧案例美化覆盖。

新人工正例最大误差2.226850082109877e-8；错误轴密态反例最大误差
4.593749998534052，明确在numerical_comparison失败。
人工证据：`/home/lhy/poseidon-work/results/concat-axis-goldens-ufswk3oo/report.json`。
新增人工密钥清理1,357,960,020 bytes，不计入上面的原批清理小计。

新冻结manifest：`scripts/baseline/cases/concat-axis0-sensitive-1-manifest.json`。
独立付费补批：`/home/lhy/poseidon-work/results/agent-batch-xjxqvy_e/report.json`。
审计必须显式传入新manifest，默认仍审计原6例：

```bash
timeout -k 10s 500s python3 scripts/baseline/audit_concat_batch.py \
  /home/lhy/poseidon-work/results/agent-batch-xjxqvy_e \
  --manifest scripts/baseline/cases/concat-axis0-sensitive-1-manifest.json
```

对应测试：test_concat_axis_evidence，test_concat_paid.ConcatAxisPaidTests。
环境变量分别为POSEIDON_CONCAT_AXIS_GOLDENS（人工report路径）与
POSEIDON_CONCAT_AXIS_AGENT（付费批次目录）。

补批终态首次1/1通过，1次API、0修复、0重试；prompt7624、completion11983、
total19607 tokens，耗时69.17259550094604秒。四组输入、8个输出值，
最大绝对误差9.264438741392015e-8，加权MAE3.474139838541734e-8。
补批独立审计：`concat-agent-audit-jq9f34i4/report.json`，
SHA256 cac7ec09cc00eb4e2486a08a74fc9f763322cf86e70f57845d203b4a7bc9a63e。
付费批report SHA256 b5d8b202e39f0591efcf00bdeba050da2dcfc20891d3fed14dadf49727d0e314。
新manifest SHA256 0782b415b7a232854ebd691fd1b64a038b955c1851bf97dc8c3dc49c5ce68c98。

补批的规则转换器也真实密态通过，最大绝对误差1.2491373047041066e-7。
规则证据`fx-batch-3ribhwhd/report.json`，SHA256
6647f94d7751ecabdf28afabe7399a0794bcc54cdfc7199cce174063332b2877。
POSEIDON_CONCAT_AXIS_RULE设置为该批次目录可启用同一严格规则证据检查。
新增模型同时加入Torch/reference/FX全slot回归，不改旧冻结六例。

两个独立付费批次合计7次API、7个首次成功候选、28组输入、80个输出值，
累计129141 tokens。清楚保留6+1批次边界，不重写原批统计。
补批付费与规则密钥又分别清理678,980,010与297,283,048 bytes。
加上补批人工程序，本阶段全部可再生密钥清理总量13,495,186,286 bytes，
约12.57GiB；源码/模型/请求/产物/解密值未删除，API Key不属于清理对象。

补批后的最终48模块回归461项：439通过、22条件跳过、0失败。
在原47模块基础上加入test_concat_axis_evidence，并显式设置本页三个补批环境变量。
检查覆盖新增人工/规则/Agent/稳定性结果、四轮旧构造付费证据及135/132/126/114历史报告。
所有修改留在本地：主逻辑concat_ops、model_graph、fx_to_hecate、model_semantic_coverage；
新增冻结案例/人工程序/审计和回归；历史源码按原hash归档。既有C++/GPU修改未触碰。
未安装依赖、未修改.env、未切换分支、未关闭sparse checkout、未commit/push。
建议将来获批后分为“逻辑concat与FX表示转换”“冻结测试/审计及历史兼容”“实验报告”提交。
