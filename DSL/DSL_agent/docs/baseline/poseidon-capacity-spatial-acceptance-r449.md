# 跨密文卷积与平均池化验收

2026-10-02，ARM Ubuntu、lhy-agent-dsl。新增模型使用 medium-v1；旧模型、冻结1200集与历史报告保留。

| 检查 | 结果 |
|---|---:|
| 静态、双参考、布局及请求兼容测试 | 143 passed / 0 failed / 0 skipped |
| 真实编译、Poseidon CPU 密态及数值比较 | 20/20 |
| 新模型 / 兼容模型 | 15 / 5 |
| 四组输入逐项比较 | 80组，17,736值 |
| 最大绝对误差 | 1.4257347624151845e-6 |
| 程序包重载 | 7/7，28组，5,480值 |
| 重载最大绝对误差 | 6.204409971899416e-7 |
| 真实Agent生成 / 付费调用 | 0 / 0 |

新增范围：Conv1d/2d各三个上下文，含分组、深度卷积、batch、stride、padding与dilation；AvgPool1d/2d各三个上下文，含边缘窗口与count_include_pad两种分母。另有Conv→square→Pool、卷积残差双输出、Conv→Pool→Flatten→Linear三个组合。每个新模型另有16组双参考和DSL算术解释检查。

可信布局使用分块旋转与baby/giant分组，处理卷积的通道、核坐标及尾块。当前公开张量契约为capacity-public-tensors-v5，规则路径为capacity-chunks-v6。零权重与bias-only路径有静态检查；未将其算作本批密态验收。

v5请求文件可无损压缩公开向量的重复数值；解码后仍重建常量并验证原逻辑请求哈希。旧v1–v4请求格式及身份保持。128KiB请求/沙箱payload、4096 AST、1024展开工作、256公开绑定及原沙箱限制未放宽。r443–r446中的请求大小、AST和不支持公开循环尝试按原绑定保留。

本批复用已有SDK，串行执行约210.99秒，产物约592.6MiB。Dacapo、Runtime核心、Poseidon库、安全参数、模型reference与误差门限未改；安装、下载、API、提交及推送为0。全部候选为规则基线，没有增加Agent成功数。

当前输入输出各1024的执行路径已含逐项运算、Linear、Conv、AvgPool、concat、stack、sum/mean与代表组合。逻辑准入不保证所有参数组合可运行；公开绑定、源码工作、编译深度与物理资源检查仍适用。ResNet20配置的4096元素输入及65536中间元素尚未完成物理验收。

原始证据：
- /home/lhohy/poseidon-work/platforms/aarch64-linux/results/poseidon-capacity-spatial-static-r447
- /home/lhohy/poseidon-work/platforms/aarch64-linux/results/poseidon-capacity-spatial-numerical-r448

完整源码、后端、模型、请求与程序包绑定见[机器报告](poseidon-capacity-spatial-acceptance-r449.json)。

下一步：原三整模型真实Agent验收、容量生成验收、其他切片/形状组合与ResNet20物理布局。12个深多项式和重复bootstrap精度缺口分别保留。

~~~sh
POSEIDON_PLATFORM=aarch64-linux POSEIDON_WORK_ROOT=/home/lhohy/poseidon-work python3 -B scripts/dsl_bundle.py replay --bundle /home/lhohy/poseidon-work/platforms/aarch64-linux/results/poseidon-capacity-spatial-numerical-r448/capacity_spatial_composition_1/program --execute
~~~
