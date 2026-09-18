# 原生函数逐项覆盖：人工基线历史记录

> 2026-09-16 更新：用户已明确批准，11项真实Agent批次已完成并独立审计通过，
> 首次10/11、最终11/11。详见 [最新付费结果](native-function-paid-results.md)。
> 下文保留此前人工基线和当时授权状态，不再代表当前付费门禁。

## 本轮结论

2026-09-16，已实现并冻结原生函数调用的11项构造要求，接通现有批量生成驱动，
并完成全部11个人工候选的真实隔离编译和 SEAL HEVM CPU 密态差分。
本轮没有付费API调用。不能把人工基线记作真实Agent生成覆盖。

新增 request `hecate-native-function-synthesis-v2` 用于逐项构造实验；v1文本与旧v22
请求规则不变。批量入口为 `--native-function-exercises`，自动选择独立 native core。
与旧构造语法混用、把原生exercise挂到v19、通用续跑丢失冻结要求都会被拒绝。
这不是全部 Hecate 语义支持，也不是 native core 与 v22 的语义并集。

## 覆盖为什么可信，以及不能说明什么

1. 候选先通过整个调用图的语法、c/p类型、返回数量、递归和资源检查。
2. 有界 AST 解释观察从 golden 可达的调用，不执行候选Python。
3. 对非空返回或要求的参数做固定公开探针上的扰动，检查最终输出是否改变。
   tuple要求每个返回单元都影响输出；重复调用至少两个返回分别有影响；
   public argument和双密文参数还须有参数影响证据。
4. 真实 tracing 在 native createCall 成功后写入 caller/callee/源码span。
   审计将有限探针观察到的关键调用位置与真实调用记录对齐。
5. 随后正常编译、真实加密执行、解密并比较固定独立reference；不使用探针代替reference。

探针是公开、固定的覆盖诊断输入，与密态差分测试输入分离，不是新FHE后端或模型oracle。
它能拒绝本次设计的死代码/忽略参数/部分元组使用/结果抵消反例，不是所有输入的形式化证明。
native helper body只trace一次，后续调用静态复制IR；native源码位置数量不是HEVM动态调用数。
空返回helper没有可影响输出的返回单元，因此明确只能取得可达调用与真实trace的结构证据。

## 冻结矩阵

| 案例 | 目标观察项 | 证据类别 |
|---|---|---|
| nf-scalar | 标量密文返回 | 有限返回影响 + 真实native调用 |
| nf-pair | tuple多个密文返回 | 每个单元影响 + 真实native调用 |
| nf-nested | helper中调用另一个helper | 内层返回影响 + 真实native调用 |
| nf-forward | 调用定义在后的helper | 返回影响 + 源码定义顺序 + 真实native调用 |
| nf-two-inputs | 两个密文参数 | 两个参数分别影响 + 真实native调用 |
| nf-public-argument | 公开参数通过p传入 | p参数影响 + 真实native调用 |
| nf-identity | 直接返回密文形参 | 返回影响 + identity结构 + 真实native调用 |
| nf-repeated | 同一helper重复调用 | 至少两次调用分别影响 + 真实native调用 |
| nf-zero-input | 无参数helper | 返回影响 + 真实native调用 |
| nf-empty | 空list/tuple返回 | 仅可达调用与真实trace结构 |
| nf-plain-return | Plain返回 | 返回影响 + 真实native调用 |

十例采用 `0.5*x+x+0.375`，双输入例采用 `left-right`。这是11种实现构造、两个数学图，
不是11个模型家族，也不能据此宣称模型泛化已完成。人工源码不进入provider请求。
权重、输入、reference、security profile与容差不由Agent修改。

## 真实执行和审计结果

- 人工基线11/11通过；44组输入，176个输出值。
- 最大绝对误差：`2.4863438335964716e-8`。
- 加权MAE：`4.416200485137924e-9`。
- 数值门限保持 `1e-5 + 1e-4*abs(reference)`。
- SEAL4.0.0、N32768、14×60-bit模数、tc128，bootstrap未执行；不是Poseidon GPU。
- 10项有限输出影响证据，1项明确的空返回结构证据，均与实际native调用位置对齐。
- 真实Agent覆盖仍为0/11；新API调用为0。

真实批次：`/home/lhy/poseidon-work/results/native-exercise-goldens-hq1cqoct`。
report SHA256：`6200f1fb2a88efec096a6d1abd280ea8145b29521a46955bbc959f604c0b1920`。

独立离线审计：`/home/lhy/poseidon-work/results/native-exercise-audit-cwgretlk`。
report SHA256：`75b1f42a14802e8601247bf5df1fb39b65c4359447f704bc1b054a46a01e868b`。
审计校验冻结输入/产物哈希、实际trace payload与响应的一致性、调用记录、独立明文公式、
保存的解密输出，以及密钥清理；审计本身不重新执行FHE或调用模型。

本批已删除 `7468780110` 字节（约6.96 GiB）可再生私钥/评估密钥；原密钥不可恢复。
模型、候选、IR、HEVM/CST、请求、输入、reference、解密输出和日志保留。

## 测试及失败诊断

- 含新逐项审计、真实11例证据、前一轮正反例和旧native/v22证据的综合回归：
  106通过，0失败，0跳过。
- 批处理、manifest、旧构造版本、provider及重试回归：103项，100通过、3条件跳过。
  三项跳过因未配置旧48例规则转换器/v20付费批证据路径；mock重试输出不是付费请求。
  两组测试有重叠，不能相加为独立测试总数。
- 最初误用系统Python运行依赖NumPy的旧v22测试，产生ModuleNotFoundError；
  切回既有Nix/venv后通过，没有安装新依赖。
- 初版纯明文probe测试错误要求浮点逐bit相等，仿射加法重排产生约1e-16差异；
  单测改为1e-15绝对精度检查。FHE差分门限、reference和模型均未放宽或更换。

## 入口（以下全部不收费）

```bash
cd '/mnt/d/Code Space/Poseidon'

# 只生成批次计划，不读取API key或发出网络请求
timeout -k 3s 30s python3 scripts/baseline/run_agent_batch.py --plan \
  --native-function-exercises --provider deepseek --model deepseek-flash \
  --reasoning-effort high --jobs 10 --max-tokens 384000 \
  --api-timeout 1200 --provider-retries 3

# 人工候选真实编译/密态测试；每例完成后清理临时密钥
timeout -k 5s 3650s python3 scripts/baseline/run_native_function_exercise_goldens.py

# 回放已有人工证据，不重新执行FHE
timeout -k 5s 280s python3 scripts/baseline/audit_native_function_batch.py \
  /home/lhy/poseidon-work/results/native-exercise-goldens-hq1cqoct --manual
```

付费审计默认不带 `--manual`，会拒绝将这批人工结果当作真实Agent批次。
真正付费运行仍等待此前提出的11例外发/费用范围确认；自动goal继续消息不视为对
此前执行环境审批拒绝的绕过许可。没有发起新的付费请求，没有读取或修改.env。

## 本轮文件与后续

新增 `native_function_exercises.py`、冻结JSON catalog/manifest和11例描述/人工源码，
`audit_native_function_batch.py`、`run_native_function_exercise_goldens.py`及对应测试。
修改请求路由、批处理flag、真实trace观察、sandbox只读挂载和语义清单。
没有修改C++ compiler、依赖版本、安全配置、GPU backend或bootstrap实现。

下一步首先是获批后的11例真实Agent生成/反馈修复及独立审计。之后仍需推进native函数
与已有构造语法的组合、尚未开放的返回/容器形式、上游高层helper和真实bootstrap等
剩余语义；不能以这11项通过替代完整目标的验收。
所有修改保留在当前working tree，不commit/push/PR。
