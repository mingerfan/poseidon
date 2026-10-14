# 联合来源接入离线验收范围

状态：待独立批准。生产入口继续使用现有版本。

此批在同一份 706 文件 Agent 草案中，验收 common-call 与融合矩阵接入，
复用现有 SDK、参考实现和隔离执行器。模型、公开权重、完整生成请求与 reference
均保留原内容。候选使用冻结的通用 balanced scalar lowering；结果归类为
人工选择的规则表达，不记作真实 Agent 新生成。

## 案例

| 案例 | 静态输入形状 | 分解图节点 | 命名输出 |
|---|---|---:|---:|
| common_alpha | x:[2, 3] | 15 | 3 |
| common_dropout | x:[2, 3] | 9 | 3 |
| common_boolean | x:[2, 2], y:[2, 2], z:[2, 2] | 54 | 3 |
| addmm_cipher | bias:[2], left:[2, 3], right:[3, 2] | 22 | 3 |
| addmm_public_right | bias:[2], left:[2, 3], right:[3, 2] | 23 | 3 |
| addmm_public_bias | bias:[2], left:[2, 3], right:[3, 2] | 21 | 3 |
| baddbmm_cipher | bias:[2], left:[2, 2, 3], right:[2, 3, 2] | 22 | 3 |
| baddbmm_public_right | bias:[2], left:[2, 2, 3], right:[2, 3, 2] | 55 | 3 |
| baddbmm_public_bias | bias:[2], left:[2, 2, 3], right:[2, 3, 2] | 21 | 3 |
| addbmm_cipher | bias:[2], left:[2, 2, 3], right:[2, 3, 2] | 25 | 3 |
| addbmm_public_right | bias:[2], left:[2, 2, 3], right:[2, 3, 2] | 58 | 3 |
| addbmm_public_bias | bias:[2], left:[2, 2, 3], right:[2, 3, 2] | 24 | 3 |

common_alpha、common_dropout、common_boolean 分别覆盖 alpha 参数、
eval Dropout 和布尔掩码 Attention。九个融合矩阵案例覆盖 addmm、baddbmm、
addbmm，各自包含密文参与、公开右矩阵和公开偏置上下文。
所有案例保留 current／previous 别名状态输出，按现有绑定逐项解码。

## 执行与预算

- 每例三个阶段：原四组输入的编译与密态数值验收、程序包重载后执行三组新输入、
  缓存复用。共 12 例／36 阶段。
- 最多 12 次编译尝试、12 次原输入密态验收和 12 次新输入密态执行；
  缓存阶段不编译或生成。
- 整批 3,600 秒，每阶段 180 秒；密态串行，编译最多 2 jobs，链接 1。
- 新产物累计最多 4 GiB，全部结果最多 32 GiB，磁盘余量至少 4 GiB。
- API、HTTP、安装、下载、后端与参数改动均为 0。
- Nix、bubblewrap、网络隔离、AST、资源限制和哈希检查沿用现有实现；
  数值门限保持 `1e-5 + 1e-4*abs(reference)`。
- 一个固定结果目录和独占阶段记录；每阶段最多启动一次。
  时间、空间、完整性、环境异常或缺失阶段收据停止相关批次并保留现场。
  正常编译或数值失败有明确收据时可继续其他独立案例。

布尔掩码原 scale 735>720 失败和 r1286 的规则表达成功记录继续保留。
它们对应旧绑定，不能代替本批联合源码的实际编译、密态和程序包复用结果。

## 身份与位置

提案 JSON：`docs/baseline/agent-common-fused-offline-proposal-r1296.json`  
提案绑定：`4a781002ea16c71bdbff063e5fde449f95e75c1f6bcb2dc265ca329802f8c187`  
联合草案绑定：`1f2f80e9da4db0731f77954e79428e36de3407c305b0fbb3b9bd68735e5ff433`

源码：`/home/lhohy/Code/Poseidon`，分支 `lhy-agent-dsl`。  
结果：`/home/lhohy/poseidon-work/platforms/aarch64-linux/results/agent-common-fused-runtime-r1296`。  
后端：Poseidon CPU w45、现有 tc128 参数。SDK 与依赖身份已按实际文件核验。
生产 704、后端 648 和 reference／门限 6 文件保持。

批准记录为单独文件；r1263 和 r1286 的执行额度已经结算。
此范围不继承付费批次，也不包含源码切换、bootstrap 或 CUDA 验收。

## 下一命令

批准记录写入后，在 Ubuntu checkout 运行：

```sh
POSEIDON_PLATFORM=aarch64-linux \
POSEIDON_WORK_ROOT=/home/lhohy/poseidon-work \
python3 -B scripts/baseline/benchmarks/tools/staged_common_fused_integration_v1/runtime_acceptance.py \
  --execute /home/lhohy/poseidon-work/platforms/aarch64-linux/results/agent-common-fused-runtime-r1296
```

批准前可用 `--plan` 读取冻结范围。
