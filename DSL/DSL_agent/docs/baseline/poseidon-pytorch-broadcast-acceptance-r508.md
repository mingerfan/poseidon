# PyTorch 矩阵输入与分块广播验收

2026-10-02，ARM Ubuntu，分支 lhy-agent-dsl。263 项静态测试通过；24 个案例完成真实编译、Poseidon CPU 密态执行、解密和程序包重载。没有新增 Agent 生成或付费调用。

## 本批范围

| 范围 | 案例 | 来源 | 结果 |
|---|---:|---|---|
| PyTorch dot、mm、bmm | 各 3，共 9 | 隔离捕获后的规则基线 | 9/9 |
| 两密文张量广播 add、subtract、multiply | 各 3，共 9 | 规则基线 | 9/9 |
| 原归约融合与 Linear 融合 | 2 | 隔离 PyTorch 捕获后的规则基线 | 2/2 |
| 奇数宽度二维池化、尾块一维池化 | 2 | 规则基线 | 2/2 |
| 原真实 Agent MLP、RMSNorm 源码 | 2 | 原响应与候选保持不变 | 2/2 |

22 项为规则基线，2 项为历史真实 Agent 源码重放。不是24项当前 Agent 新生成。补充案例独立于冻结1200集，不据此增加全项目去重模型总数。

首次执行：96组输入、2972个输出值，最大绝对误差8.16099409127e-07。24个程序包重载同样96组、2972值，最大误差5.58704196257e-07。重载生成及编译次数均为0。门限保持1e-5 + 1e-4*abs(reference)。

## 输入与布局

可信本地 PyTorch 支持 torch.mm、torch.bmm、torch.dot 和相应 Tensor 方法，并补齐 matmul 方法及静态关键字参数。矩阵计算沿用注册表的dot/matmul规则；注册表仍为34项。dot是匹配长度的一维向量；mm是二维矩阵；bmm是批量数相同的三维矩阵；matmul保留既有rank-2/3广播范围。当前注册规则要求两个密文操作数；公开权重矩阵可采用Linear。标量输出在源码中以reshape(1)表示。

分块规则基线新增通用非标量广播。按静态C-order坐标映射，提取并复用源元素，再放到输出位置；使用已有mask0和旋转密钥，不增加公开常量、不改变请求数学语义。密态上下文包含二维、四维和period-4跨块；加减乘各三个上下文。源码、AST、工作量和128KiB请求限制保持，因此更大广播仍可能触及构造预算。

9个矩阵来源包各通过16组原模块／高层图／分解图独立参考检查；9个广播案例各通过16组数学、PyTorch与规则源码检查。两套参考不调用DSL转换器或待测helper计算期望答案。

## 修复与历史绑定

r491的旧策略版本断言已修正。r495暴露dot关键字名错误，依据锁定Torch2.0.1的实际签名修正为tensor；原日志保留。r497的6个mm/bmm案例通过捕获与参考，但规则基线缺少非标量广播；补齐后原6项在r503全部通过，模型、来源文件与请求参数保持不变。

最终601执行源码文件绑定与263项静态记录匹配，24包完整性审计通过。Dacapo、Runtime核心、Poseidon库、runner、参数和门限保持。原三个中型模型的当前规则源码与r487逐字节一致；两个成功模型重新验收，空间残差保留原失败，没有以简单模型替代。

空间残差r487在Rescale第587条处bad_alloc，随后触及150秒上限。Runtime保留所有中间密文；仅密文系数累计存储估计4,857,266,176字节，超过4GiB限制。这是计划结构估算，不是实测RSS，且不含密钥、临时量和分配器。最后使用释放补丁r490仍待批准，未应用。

## 原计划状态

- 中型1024元素配置与跨块运算已有代表性执行；更大组合和真实Agent生成验收继续补充。
- MLP、RMSNorm真实Agent通过；Attention规则基线通过，Agent编译失败及新反馈重测仍未关闭。
- 分解注册、独立参考、来源及递归／预算检查已有实现；每项Poseidon和Agent上下文尚未全部完成。
- PyTorch入口继续扩展；本批矩阵和广播尚无当前真实Agent生成结果。
- Bootstrap重复刷新精度、12个深多项式案例和ResNet20整网未完成。x86原生、空白安装与CUDA暂缓。

## 使用

~~~sh
POSEIDON_PLATFORM=aarch64-linux POSEIDON_WORK_ROOT=/home/lhohy/poseidon-work python3 -B scripts/prepare_model.py \
  --trusted-pytorch-manifest scripts/baseline/cases/trusted-pytorch-matrix-v1/fx_bmm_2.json \
  --trust-local-code --resource-profile medium-v1
~~~

默认显示导入计划。写入使用--write与新的--output目录；生成的model.json沿用现有Agent入口。

当前已编译程序包可直接重放：

~~~sh
POSEIDON_PLATFORM=aarch64-linux POSEIDON_WORK_ROOT=/home/lhohy/poseidon-work python3 -B scripts/dsl_bundle.py replay \
  --bundle /home/lhohy/poseidon-work/platforms/aarch64-linux/results/poseidon-pytorch-matrix-r503/fx_bmm_2/program --execute
~~~

完整路径、逐例请求／报告哈希和程序包位置见[机器报告](poseidon-pytorch-broadcast-acceptance-r508.json)。当前源码绑定只支持有限输入上的数值证据，不构成形式化等价证明。
