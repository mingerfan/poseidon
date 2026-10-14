# PyTorch 新入口真实 Agent 测试提案

状态：**待批准，新增调用0**。15个请求及源码已经冻结；本提案覆盖未执行的外部状态输入与新增函数式输入。

| 案例 | 数量 |
|---|---:|
| 外部NPY状态：MLP、Conv→Pool、残差、BatchNorm、矩阵融合、形状→Linear | 6 |
| F.layer_norm：直接归一化、公开affine与残差、归约与stack | 3 |
| F.group_norm：整体分组、公开affine、批量分组与残差 | 3 |
| 总体方差：跨块轴归约、多轴与均值融合、全归约与stack | 3 |

精确案例列表、请求哈希和公开上下文哈希见[冻结JSON](agent-pytorch-functional-live-proposal-r675.json)。绑定：

`c089b9c4ce2d49e8c7ed7b8ff23218c91c10d5245779f889f26f6072b109d982`

使用DeepSeek deepseek-flash/high/stream、explicit-v10、Poseidon w45；每例最多3轮修复，每次生成最多3次传输重试。全批最多 **60次生成／240次HTTP**。单次timeout 1200秒、max tokens 384000。

API与密态串行，编译≤2／链接≤1；全批12小时、每项4小时／4GiB、总产物32GiB、磁盘余量4GiB。达到边界保留证据。无明确货币上限，超时重试可能重复计费。

外发内容为公开图、权重描述符／公开常量、布局、合法DSL规则与受控反馈。规则候选、私有测试数组、reference、凭据和宿主环境信息留在可信端。

验收采用当前链路重新生成、编译、Poseidon CPU执行、逐项比较及程序包重载。门限 `1e-5 + 1e-4*abs(reference)`；首次成功、修复后成功、失败层和预算耗尽分别记录。归一化验证已声明的近似计算，原函数应用精度另计。

r620原14项已执行完毕。r635外部状态六项未执行，原绑定与文件保留；源码改动后由本提案替代执行范围。六项模型及请求哈希与旧提案相同，来源已重新隔离捕获。新增九项已完成规则候选的真实后端验收，尚无新真实Agent结果。

获本范围批准后执行：

~~~bash
POSEIDON_PLATFORM=aarch64-linux POSEIDON_WORK_ROOT="$HOME/poseidon-work" \
python3 -B scripts/baseline/benchmarks/tools/run_application_matrix.py \
  --execute docs/baseline/agent-pytorch-functional-live-proposal-r675.json \
  --approved-binding c089b9c4ce2d49e8c7ed7b8ff23218c91c10d5245779f889f26f6072b109d982 \
  --output "$HOME/poseidon-work/platforms/aarch64-linux/results/agent-pytorch-functional-live-r676"
~~~

批准范围包含这15项新绑定，其他付费评测另行冻结。需要单独确认，源于本轮用户计划要求每批新功能真实Agent评测单独授权。
