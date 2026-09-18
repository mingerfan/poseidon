# 联合 compiler 配置身份与显式 precision45

## 解决的问题

旧请求只有 `compiler_profile_sha256`，对应 compiler JSON 的字节哈希；waterline40
由执行器常量提供。同一 JSON 配合40或45会产生不同scale/level与rescale安排，
只记录JSON无法区分这两种配置。

新增可选字段 `compiler_configuration`：绑定命名版本、backend、EVA pipeline、
准确JSON哈希及waterline，并用域分离SHA256得到 `identity_sha256`。
该字段整体参与原请求 `request_id` 的计算；不改变候选响应允许字段。
配置由可信调用端选择，Agent只能返回schema、request_id和Hecate源码，
不能返回或改写waterline、profile、误差门限、安全参数。

注册配置：

- `seal-cpu-eva-w40-v1`：显式40，与旧隐式40有不同请求身份。
- `seal-cpu-eva-w45-v1`：显式45，基于已完成的[跨图精度实验](hevm-precision-graphs.md)。

两者固定同一JSON SHA256：
`ab53faeac14298e846a4ea8e9b2cbfaef4469a6b03ba25b80f37c79387fd246e`。

联合身份SHA256：

- 显式40：`3d52e20c11046ce47e6be45e1a46633079dcebe58d14f0d15774c4ac0a53f246`。
- 显式45：`b37bb8a8942013637366a241389a087cd0562c3e7c3d0b8326f701b73406bf46`。

任意其他JSON、waterline或pipeline不自动加入允许集合。
配置身份不是整个执行环境身份；构建、前端、运行时、安全参数及实际产物的证据
仍需由各运行报告保存和核对。

## 兼容性及执行检查

省略新选项时，仍生成不带新字段的旧请求，使用默认40；旧ID和历史报告不重写。
主动选择显式40也会生成新请求ID，因为其精度约束现在已经属于请求契约。

```bash
# 只准备新请求：不读取凭据、不调用API、不执行FHE
python3 scripts/baseline/run_candidate.py \
  --case scripts/baseline/cases/construction-sumslots4.json \
  --native-public-loops --compiler-configuration seal-cpu-eva-w45-v1 --prepare
```

检查覆盖以下位置：

1. 构造请求时对照固定注册表校验，拒绝额外字段、布尔/浮点伪装等。
2. API导出允许列表接受且校验这个纯公开配置对象，无路径、密钥或输入数组。
3. 静态候选检查重新校验新配置及请求哈希，拒绝跨配置response ID和候选参数覆盖。
4. 编译命令从可信配置取waterline，执行前后检查冻结请求及profile。
5. 编译产物gate检查实际输入scale/level与新配置一致。
6. 隔离worker在加载密钥/执行HEVM之前再次读取真实产物与挂载profile并检查配置；
   后续仍由原SEAL observer核对实际密文的输入、输出元数据。

旧的诊断脚本曾明确用旧payload仅固定I/O，对不同waterline做实验。
旧payload没有新精度字段，worker不会反向赋予它新的身份约束；这些实验继续明确
标记为诊断回放。正常旧candidate入口仍锁定40，不存在候选选择任意编译参数的接口。

## 批量与续跑

`run_agent_batch.py` 的 `--plan`、host→Nix、batch→candidate均传递新选项。
批次报告保存完整配置，并核对子运行返回的配置。
旧批次默认无配置扩展；40/45/旧隐式40之间的切换一律要求新批次。
`--allow-config-change` 是原有模型服务参数实验开关，不能绕过compiler配置的续跑限制。

本轮没有启动付费批次。新字段的服务导出校验和provider测试均为离线测试，
不能称为DeepSeek已在新精度配置下真实生成成功。

## 真实验证入口

`run_native_loop_precision_goldens.py` 使用原有8个正确循环程序和1个故意错误的
循环边界程序，从新显式45请求开始重新trace、编译和真实SEAL HEVM执行。
原native-loop legacy40失败批次保留，不能更新为通过。

```bash
cd '/mnt/d/Code Space/Poseidon'
timeout -k 10s 2900s python3 scripts/baseline/run_native_loop_precision_goldens.py
```

另有 `probe_compiler_configuration_guard.py`：在已安装Nix/venv里，对一个已完成的
显式45人工案例产物运行三个真实隔离worker拒绝探针。不挂载密钥、不提供运行输入、
不执行FHE、不调用API；必须在配置检查处报错，不能把随后缺少密钥导致的失败算作通过。

计划/防篡改/兼容性测试位于 `test_compiler_configuration.py`；完整真实链证据测试
通过 `POSEIDON_LOOP_CONFIG45_GOLDENS` 指定新批次。不提供证据时的skip不算通过。

完整DSL支持仍未完成：该改动修补精度契约和可复现性，不增加新的DSL语法，
不证明所有输入和随机密钥都满足门限，也不提供真实bootstrap或Poseidon GPU验证。
所有源码修改仅留在本地，不commit/push/PR。

## 2026-09-17 实际结果

Confirmed fact：新显式45请求从真实Hecate tracing到SEAL HEVM执行已通过。
8/8正确循环程序通过；故意写错边界的第9例完成编译、真实执行后被数值门限拒绝。
正例共32组输入、128个值，最大绝对误差1.6225815208104066e-7，
MAE1.31133571967444e-8。原失败sum4本次最大误差1.2139277993927777e-7。
错误边界程序相对正确reference的最大误差1.0000000789788397；对其自身错误数学
公式的CKKS误差仍在原门限内，说明负例被拒绝不是仅由数值噪声偶然导致。

报告：`/home/lhy/poseidon-work/results/native-loop-config45-goldens-tu_39c3f/report.json`

SHA256：`922c8fa5aea23d2ee940c2624c1211b753d46fb7fe3cd64dc6721931c58485be`

三个真实隔离worker均在配置门禁处拒绝，无密钥挂载、FHE执行或API调用：

| 负例 | 被拒绝的原因 |
|---|---|
| 合法40请求 + 实际45产物 | 产物输入精度不符合不可变请求 |
| 改写45配置对象的waterline字段 | 与注册的配置及身份不一致 |
| 同时改写请求及配置的JSON哈希 | 不符合固定profile哈希 |

报告：`/home/lhy/poseidon-work/results/compiler-configuration-guard-qv2ptpog/report.json`

SHA256：`260c8979ecc3824798f90d3dc65e60d0a67c81a083571aadb19547b5d4af467e`

两条40路径也分别完成真实trace/compile/encrypt/evaluate/decrypt：

- 默认旧请求：`candidate-replay-nht5_5yv`。
- 显式40新请求：`candidate-replay-wojavs6a`。
- 原始同案例：`candidate-replay-k61oyid5`。

以上目录均位于 `/home/lhy/poseidon-work/results/`；默认路径的request逐字段及ID
与原始同案例保持一致，显式40除新增配置字段与request_id外的全部字段保持一致。
权重和输入/reference数组完全相同，两个新运行均通过原数值门限。
这是默认路径的真实兼容性验证，不只是解析CLI选项。

本轮11个真实候选执行均为人工golden，API调用0次；正例10个通过、负例1个正确拒绝。
9例45循环与2例40兼容性运行生成的密钥合计7,468,780,110 bytes（约6.96GiB）已清理。
原随机私钥不可恢复，可重新生成新密钥；其他运行证据保留。

最终205项综合测试全部通过，无跳过：包含新配置的6项单测、3项真实证据审计、
原candidate/provider/batch定向测试，以及此前的163项语义、真实付费和精度回归。
集合有继承关系，不应把前后多次执行的测试数量累加为独立覆盖数。
整文件空白扫描仅报告candidate_contract.py原有第314行尾随空格；该行不属于本次修改，
未为格式检查改动无关内容。没有编译/执行基础设施失败。

Evidence-based inference：显式45可以作为下一步Agent新精度实验的受控配置，
且已有的循环精度问题在本次固定输入上消失。不能从这些有限结果推断全输入稳定。
Unconfirmed：真实付费Agent在该新配置下的生成表现、全部构造组合与完整上游语义。

建议把配置身份、candidate/provider/worker/batch接线及测试作为一个本地提交，
把循环验证与证据报告作为另一个提交；尚无提交授权，因此均保持未提交。
