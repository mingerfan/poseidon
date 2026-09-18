# 原生数组构造：逐项要求与真实frontend观察门禁

2026-09-17，本轮完成了原生数组的独立逐项构造契约，复用现有Hecate/SEAL执行路线。
10/10人工正例通过；错误转置索引的反例在真实密态执行后被数值比较拒绝。
本轮0次付费API调用。上一轮11项原生函数的真实Agent证据保持不变。

## 系统位置与数据流

新增 `native_array_exercises.py` 位于可信候选检查层，不是FHE backend：
它接收已通过类型/shape检查的AST、公开常量、固定构造要求，产生有限输出影响记录。
不会exec/eval候选Python，不读取密态测试输入/reference，也不替代数值比较。

`hecate-native-function-synthesis-v4` 在旧native-array v3之上增加独立构造要求。
旧v1/v2/v3提示词内容保持不变，旧11项函数构造不能挂到数组契约重标。
`--native-array-exercises` 选择固定10例manifest，将每例要求传入生成与修复闭环。
通用续跑若丢失这些冻结要求会被拒绝。

实际链路：模型描述 → 候选AST/c-p/shape检查 → 有界构造观察/结果扰动 →
隔离进程内真实Hecate数组操作 → Dacapo → HEVM/CST → SEAL HEVM CPU → 解密差分。
新观察只记录op、caller、源码span、输入shape、输出shape；不记录明密文数值。
操作成功后才记录，每个静态观察见证必须在实际frontend记录中找到同操作/shape/源码位置。
未调用的helper即使被hc.save单独trace，也不能满足从golden可达的探针要求。

## 冻结的10项构造案例

| ID | 构造要求 |
|---|---|
| na-reverse | helper返回矩阵，负步长切片的结果用于计算 |
| na-transpose | helper返回矩阵，使用.T或transpose结果 |
| na-rank-four | rank-4 reshape、transpose、rank-4返回和copy结果 |
| na-row-unpack | 矩阵返回，按第一轴解包为行数组 |
| na-mixed | 返回混合Plain/密文数组，两个类型都影响最终输出 |
| na-zero-item | 0-D数组返回，用item()提取Expr |
| na-zero-index | 0-D数组返回，用[()]提取Expr |
| na-zero-return | helper和golden直接返回0-D数组 |
| na-empty | 可达helper返回空对象数组，仅结构证据 |
| na-nested | helper调用返回数组的helper，并使用负步长切片 |

以上合计14个去重观察项：13项有限单元结果影响、1项空返回结构证据。
它们是同一仿射模型 `0.5*x+x+0.375` 的不同构造形式，不是10个模型家族。
数组单元是独立Expr，不能把转置/reshape误解释成密文slot rotation。

## 防无效填充，以及证据不能证明什么

固定公开探针与真实差分测试输入分离；对每个操作结果的单元分别施加固定扰动，
只有最终golden结果改变才记录影响。混合返回要求同一次返回的Plain和密文单元均有影响。
测试确认未调用helper、丢弃结果、结果在 `b-b` 中抵消、混合返回忽略Plain均被拒绝。
空数组没有可扰动的单元，明确仅作结构证据，不假装输出敏感。

这是有限结果参与证据，不证明某个操作在数学上不可消除（例如只读copy），
也不是全输入形式化等价或任意NumPy语义支持。shape正确、语法覆盖满足仍可能数值错误，
错误转置反例正是为了确认独立数值门禁仍然有效。
数组参数、写入、数组算术及与旧v22构造的组合仍未开放；bootstrap和高层helper等目标缺口仍在。

## 真实实验结果

- 人工正例10/10；40组输入执行、160个输出值。
- 最大绝对误差 `3.046570712372798e-8`，加权MAE `4.960859897353009e-9`。
- 反例额外执行4组输入、16个输出值，在numerical_comparison层拒绝，
  最大绝对误差 `0.1875000115343144`；不能把反例算作正确程序通过。
- 容差保持 `1e-5 + 1e-4*abs(reference)`。
- SEAL4.0.0，N32768，14×60-bit模数，tc128，waterline40；不是Poseidon GPU，不执行bootstrap。
- 本轮密钥按既有策略逐例清理，共7,468,780,110字节（约6.96GiB）；原密钥不可恢复。
  候选、编译产物、输入、reference、解密数组、日志及报告均保留。

真实批次：`/home/lhy/poseidon-work/results/native-array-exercise-goldens-qcummop0/report.json`。
SHA256：`e3afbd9f89179c7a15436aa77db2d7828aded29bebfbf8a4b12a1a89ad4ee274`。

独立审计：`/home/lhy/poseidon-work/results/native-array-exercise-audit-cf9u_u66/report.json`。
SHA256：`68a23c574fe9f9c6c6a0681c1b0f37a0681307f13eb13ba615ff387b3eedf4df`。
审计重新检查响应/trace payload、完整静态检查、真实存储操作、编译产物哈希、
独立明文公式、保存的解密数组和清理记录；本身0次新API/0次新FHE执行。
默认live审计拒绝人工数据；必须显式 `--manual`，不能升级成人工冒充Agent证据。

综合回归128项通过，0失败、0跳过，包含旧v22和上一轮11项真实Agent证据。
付费数组证据测试尚未执行，因为对应批次尚未获具体调用范围确认；不计入128项。

## 复现与下一门禁

```bash
cd '/mnt/d/Code Space/Poseidon'
# 只生成计划；不加载凭据、不发网络请求
timeout -k 3s 30s python3 scripts/baseline/run_agent_batch.py --plan \
  --native-array-exercises --provider deepseek --model deepseek-flash \
  --reasoning-effort high --jobs 10 --max-tokens 384000 \
  --api-timeout 1200 --provider-retries 3 --stream

# 审计已有人工密态证据，不重新执行FHE
timeout -k 5s 280s python3 scripts/baseline/audit_native_array_batch.py \
  /home/lhy/poseidon-work/results/native-array-exercise-goldens-qcummop0 --manual
```

已询问是否批准独立10例DeepSeek Flash/high付费批：API并发10/native2，
1200秒、384000输出tokens，每例最多3修复、每生成轮最多3传输重试，
合计最多40次逻辑生成/160次HTTP尝试；可能重复计费，无货币总额承诺。
自动goal继续消息不作为具体高费用/外发范围询问的答复。本轮没有发起新API请求。

源码、10例JSON和manifest、审计/测试/运行脚本均在本地仓库；构建和结果在WSL原生目录。
未安装依赖，未改变安全参数、reference或数值门限；未修改GPU/C++后端。
未commit/push/创建PR，保持用户已有修改。建议未来将本次构造契约、观察、门禁测试及文档
作为一组本地提交，付费结果作为后续独立证据提交；须先得到用户提交授权。
