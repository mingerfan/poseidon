# 编译配置

当前 SEAL CPU 配置：

| 名称 | waterline |
|---|---:|
| `seal-cpu-eva-w40-v1` | 40 |
| `seal-cpu-eva-w45-v1` | 45 |

两者使用固定安全参数与 EVA 编译流程。应用组件默认显式 w45；旧 CLI 未指定配置时保留原隐式 w40 行为。

```bash
python3 -B scripts/agent.py --backend local candidate -- \
  --case scripts/baseline/cases/unified-two-input-two-output.json \
  --prepare --compiler-configuration seal-cpu-eva-w45-v1
```

## 配置身份

配置名、后端、pipeline、profile 哈希和 waterline 一起参与请求身份。候选只返回 DSL，不能自行更改这些参数。

省略配置、显式 w40、显式 w45 是不同请求身份。切换配置后必须重新验收，不能继续使用旧通过记录。

版本和哈希以 [compiler_configuration.py](../../scripts/baseline/compiler_configuration.py) 为准。编译产物的 level/scale 检查见 [SEAL 产物检查](seal-artifact-gate-v2.md)。

## 失败处理

提高 waterline 不等于增加可执行深度。高次近似、多次密文乘法和不合适的表达顺序可能耗尽 level 或超过 scale 容量。

生成器可以在请求预算内改写等价 DSL；它不会改编译器、安全参数或数值门限。当前后端没有真实 bootstrap。

原函数近似误差与 CKKS 执行误差分别处理，见[误差定义](approximation-error-decomposition.md)。
