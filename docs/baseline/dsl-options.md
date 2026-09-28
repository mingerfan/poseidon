# DSL 构造与 helper 选项

普通模型生成不需要设置这些选项。它们用于约束生成程序的写法，或要求调用固定上游 helper。选项不会改变模型的数学定义。

## 构造配置

| 选项 | 用途 |
|---|---|
| `--unified-profile native` | 默认；类型化 native 函数、Expr 与对象数组 |
| `--unified-profile public-v1` | 公开函数、容器、闭包、循环等受限构造 |
| `--unified-exercise ID` | 要求采用一个已登记构造 |
| `--unified-guidance explicit-v9` | 显式启用类型、槽位与多项式表达指导 |
| `--unified-chunk-period P` | 启用分块输入输出；P 为 4/8/16/32/64/128/256 |

native 与 public-v1 是不同契约；旧 schema 的构造开关也不能直接拼到统一图上。选项组合不合法会在请求准备阶段拒绝。

Expr 对象数组的每个单元是完整表达式。对象数组 transpose/reshape 不等于密文 rotation。公开 if/for 在构造阶段运行，不支持根据密文值作分支。

## 上游 helper

`--unified-helpers` 接受下列版本化配置。配置只允许受信适配器绑定的函数和参数，不允许任意导入上游库。

| 配置 | 主要用途 |
|---|---|
| upstream-poly-silu-v1 | HE_SiLU 固定多项式 |
| upstream-poly-bn-silu-v2 | 增加固定统计量 BN |
| upstream-poly-concat-bn-silu-v3 | 增加 Concat |
| upstream-poly-spatial-v4 | 有界 Conv / Avg / Pool |
| upstream-poly-spatial-mapped-v5 | 通过公开窗口映射接入空间 helper |
| upstream-poly-fused-spatial-v6 | ConvBN / DwConv 的固定子图 |
| upstream-poly-downsample-v7 | 两个空间轴 step=2 的下采样 |
| upstream-poly-virtual-prefix-v8 | MPBN / Linear / ReshapeLinear 的固定槽位适配 |
| upstream-poly-chunked-virtual-v9 | 分块 MPBN / Linear / ReshapeLinear / SiLU |
| upstream-poly-fixed-polynomials-v10 | 固定 Poly_Default、Tree、Leaf 调用 |

这些配置保留各自的 shape、packing、常量、工作量与组合检查。新版本号不表示所有旧用法都能无条件组合。函数签名与绑定由 [upstream_candidate_helpers.py](../../scripts/baseline/upstream_candidate_helpers.py) 和 [适配器目录](../../scripts/baseline/upstream_adapters/) 定义。

运行真实 poly helper 还需要 [锁定 einops 依赖](../../src/poseidon/tools/dacapo/poly-python-wheels.lock.json)。基础环境安装器不自动增加这个依赖。

`--unified-helper-exercise CALLEE` 要求指定调用，允许重复该参数。检查器会检查调用、返回依赖及相应构造贡献；仅在源码中写出名字不算完成。

当前唯一显式开放的额外能力组合是 `--capability-composition native-bn-directed-v1`：绑定 BN 与受支持 native 函数调用、对象数组复制或标量增强赋值的定向构造；不分块，不扩展为任意 helper/public 混用。具体字段见[worker 接口快照](agent-component-v1.md)。

## 查看实际选项

```bash
python3 -B scripts/agent.py --backend local candidate -- --help
```

使用 `--prepare` 检查模型与这些选项的组合，不调用 API。生成候选必须遵守请求中的完整规则，不能改变权重、布局、参数或误差门限。

HE_ReLU、HE_Max 和 HE_MaxPad 的上游路径依赖 bootstrap；当前 SEAL HEVM 后端没有真实 bootstrap。
