# PyTorch 来源选择与多文件 CLI

`--source-format latest` 现在生成 v34 清单，涵盖已登记的公开张量运算、混合拼接和多文件模型。
默认版本继续使用 v18；旧 v28 及其他显式清单保持原内容。
v29–v34 的包判断统一复用源码包策略，`--source-root` 与 `--helper` 选择显式文件。

| 验收 | 结果 |
|---|---|
| CLI 与旧清单兼容 | 17/17，failed / skipped 0 |
| 真实 Poseidon CPU | 3/3，3 次编译、9 次密态调用 |
| 数值比较 | 24 组输入、120 个输出值，最大绝对误差 2.05790805e-07 |
| 程序包 | 三包重载、新输入、重复执行、零生成／零编译缓存通过 |
| API | 0 |

三种来源为根目录模型与嵌套 helper、包内模型入口与嵌套 helper、关键字输入与 Linear 残差。
真实验收使用规则候选；当前真实 Agent v34 的 18/18 结果另见
[拼接生成验收](agent-public-join-acceptance-r1480.md)。

审计重新校验来源、RuntimePlan、OperatorSpec、常量 bundle 和依赖，重新计算原四组输入与新输入的数学参考。
模型及权重与原上下文保持，门限仍为 `1e-5 + 1e-4*abs(reference)`。
原先 latest=v28 的记录保留；本次清单保存实际 v34，而非 latest 别名。

后端库、参数、网络和依赖保持；安装、下载及 Git 操作为 0。
任意 Python、动态 shape、最大容量、应用精度、bootstrap、CUDA 与完整 ResNet20 仍分别待验证。

SDK：0a58989024185fead91aa91e5b8079e1bacc3eeee591fa58dbff60e5a2f79968。报告绑定：0bc97c00d67778a6661782ec01d7612a143be63b012ebec254574df2856cca0f。
[机器可读报告](agent-source-cli-acceptance-r1482.json)。
