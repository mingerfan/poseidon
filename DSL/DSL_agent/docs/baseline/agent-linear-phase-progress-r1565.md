# Linear 坐标与组件配置恢复

完整 PyTorch 批次的独立收尾审计通过：4 项全部有终态，2 passed、2 failed；7 次生成／7 次 HTTP 全部结算，unknown=0。实际 4 次编译、8 次密态调用。万参数 Linear 与 rank32 两项均为原始 Agent 源码，来源、编译、数值、包重载、新输入、重复执行和缓存已核对。通过范围 16 组／1,888 值，最大绝对误差 1.73005922144e-7。

两项失败保留：rank_axis_8 触及流接收上限，尚未编译；wide_tail_linear 有两次静态拒绝和两次数值失败。审计首版误用了导入目录读取接口，修正为读取程序包中的完整来源载荷后再次核验；原首版报告保留。

## 公共信息诊断

大输入 Linear 最后候选错误地移动了已经在第二行位置的广播掩码；2,046 个逻辑位置的 y1 系数不同。其投影的两个非 bias 计算项，都不覆盖所需的第二行输出槽 3、4、5，因此这一行始终只剩 bias。诊断使用公开源码、模型与向量，没有读取私有测试输入或 reference 数组。

## 修改与检查

explicit-v18 按实际 tile 绑定登记每个 expand/compress 阶段的源坐标掩码、行宽、stride 与 shift。固定和移动掩码在阶段输入坐标中相乘，移动项随后旋转；后续阶段使用前一阶段的结果。未给模型提供规则 baseline 源码。

同时修复 request_options() 丢失显式 capacity packing 版本的问题；此前恢复 v10 请求会选择默认 v8。旧请求重建与接口检查继续通过。

112 检查 passed，failed/skipped0；包括三个不同几何上下文的紧凑／扩展行坐标差分。旧首次测试失败记录保留。默认来源、x86 配置、后端库、参数与门限保持。

## 当前验收与下一步

原 wide_tail_linear 的规则基线主执行已完成：1 passed，0 failed。同模型、权重、布局和数值门限，最多 1 次编译／3 次密态，900 秒；API=0。检查首轮四组输入、程序包、新输入、重复和缓存。原主owner PID939105/start25951461 已退出，终态已核对；正在进行独立保存证据审计。

~~~sh
python3 -B -m json.tool /home/lhohy/poseidon-work/platforms/aarch64-linux/results/agent-linear-phase-offline-r1564/report.json
~~~

本项完成后冻结原数值失败任务的新 Agent 请求，沿用计划内总测试授权；旧通过项不全量重跑。stream_limit 另作为有界输出与相关上下文选择问题处理，保持网络与接收上限。

后续继续其他容量／算子组合、原函数应用精度、Agent bootstrap、CUDA 实机和完整 ResNet20。当前改动未触及 Poseidon、Runtime、Dacapo，没有安装下载、提交或推送。


## 离线结算与真实生成

规则基线独立审计通过：1 次真实编译、3 次密态调用，8 组／112 个输出值，最大绝对误差 2.76549276965e-7。重载、新输入、重复与缓存全部通过，API=0。

原 wide_tail_linear 的 explicit-v18 新 Agent 测试已启动，沿用计划内总授权。仅此 1 项，最多 4 次生成／16 次 HTTP、4 次编译／6 次密态，4 小时；新增≤4 GiB／总48 GiB。公开权重、模型、布局、参数和门限保持。外发内容为公开模型、权重或描述符、布局、规则和受控反馈；规则源码、测试数组和 reference 不进入请求。源入口为 graph，此轮不声称再次验证完整 PyTorch 捕获。

主owner PID940859/start26069746，后置只读等待owner PID941450/start26088790。源码冻结到两个 owner 终态且 HTTP 结算。[绑定](agent-linear-phase-acceptance-r1567.json)。
