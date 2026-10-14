# PyTorch 常用调用与融合矩阵入口验收

已将验收过的联合版本接入应用入口。开发分支保持 lhy-agent-dsl；只修改 Agent 层。

## 输入扩展

- `poseidon-trusted-pytorch-v13`：在有序张量操作上组合静态 alpha、eval Dropout 和近似 Attention。
- `poseidon-trusted-pytorch-v14`：增加 addmm、baddbmm、addbmm；支持公开 alpha/beta、合法偏置广播和已登记的公开／加密操作数组合。
- 模型继续通过静态签名、公开权重包和显式近似配置导入，沿用任务、验证和程序包接口。

## 验收结果

| 范围 | 模型 | 编译 | 密态阶段 | 数值比较 | 结果 |
|---|---:|---:|---:|---|---|
| 本批 r1301 | 11 | 11 | 22 | 77 组、1778 值 | 33/33 阶段通过 |
| 复用 Boolean 原证据 | 1 | 原 1 次 | 原 2 阶段 | 原 7 组、84 值 | 原 3/3 阶段通过 |
| 组合兼容索引 | 12 | 合计 12 | 合计 24 | 84 组、1862 值 | 36/36 阶段通过 |

每例包括资格验收、程序包重载后的三组新输入执行和无再次生成的缓存复用。
本批最大绝对误差 `5.701685747983021e-7`，所有输出满足
`abs(actual-reference) <= 1e-5 + 1e-4*abs(reference)`。

本批是规则基线；Boolean 为已记录的人工选择表达。API/HTTP 均为 0，
不增加真实 Agent 生成通过数。原 76 项真实 Agent 结果仍为 56 通过、20 失败。

164 项切换／证据控制、切换后 27 项能力／程序包兼容测试全部通过；
失败、错误和跳过均为 0。前者使用控制夹具，真实编译与密态结果由上表另计。

## 源码与历史记录

应用源码由 704 变为 706 个绑定文件：8 个文件更新、2 个文件新增。
安装日志、原文件及权限恢复材料保存在：

`/home/lhohy/Code/Poseidon/scripts/baseline/benchmarks/tools/staged_common_fused_integration_v1/cutover-records/20261008T072609-37b52747`

648 个后端文件、6 个 reference／门限文件及全部冻结文件哈希通过核验。
Poseidon、CKKS Runtime、Dacapo、参数和 x86 配置保持。
本轮没有 x86 密态回归，也没有 bootstrap 或 CUDA 执行。

r1296 原失败报告保持：3 阶段通过、11 失败、22 未运行。
新批独立保存；组合索引只复用其已通过 Boolean，不改写原结论。

## 资源与下一步

本批含启动约 28.51 分钟，
产物约 73.4 MiB；全局结果约
26.41 GiB，可用磁盘约 43.53 GiB，
均在原批准范围内。未安装、下载、提交或推送。

下一步核验待批真实生成任务是否匹配当前源码身份，再继续通用输入组合、容量与整模型工作。
Bootstrap、密文交接、应用精度和 CUDA 实机分别保持未完成状态。

[机器可读报告](agent-common-fused-cutover-acceptance-r1303.json) ·
[输入用法](operator-decomposition-and-python.md#pytorch-v13-有序操作与常用调用) ·
[源码切换记录](../../scripts/baseline/benchmarks/tools/staged_common_fused_integration_v1/cutover-records/20261008T072609-37b52747/record.json)

下一条只读检查命令：

~~~bash
POSEIDON_PLATFORM=aarch64-linux POSEIDON_WORK_ROOT=/home/lhohy/poseidon-work \
python3 -B scripts/baseline/benchmarks/tools/staged_common_fused_integration_v1/cutover.py --evidence-mode composite
~~~
