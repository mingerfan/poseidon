# PyTorch MultiheadAttention 示例

三份清单使用同一条 v16 导入接口：self Attention 后接 Linear 与残差、投影后的 cross Attention、布尔 mask 与 padding 的 Attention。

清单包含静态输入、公开 NPY 权重和近似配置。exp 一阶 Taylor 与 reciprocal 一次 Goldschmidt 用于小规模执行回归；原函数误差单独记录。当前配置用于验证编译和密态链路，模型应用精度另行评测。

来源文件及权重与已有固定案例逐字节一致。[输入说明](../../../../docs/baseline/operator-decomposition-and-python.md#pytorch-v16-multiheadattention)。
