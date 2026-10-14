# MHA v12 运行入口

该入口用于 r1333 的两个已冻结任务：self Attention 后接 Linear 与残差，以及投影后的 cross Attention。生产 Agent 和后端代码保持其 r1334 绑定。

```sh
python3 -B scripts/baseline/benchmarks/tools/mha_tensor_axis_live_v1/run.py --plan
```

61 项运行器控制测试通过。两个 PyTorch 来源已完成无收费捕获，生成上下文与冻结请求逐字段一致，分别为125428和131069字节。此次捕获只验证请求准备；没有编译、密态执行或模型API调用。

运行器沿用 Application 的生成、修复、验证、预算和程序包接口。执行前校验人类授权、冻结源码、请求、原始证据、运行器及预算；输出进入独立目录。每例最多4次生成、16次HTTP，原P4/w45与native限制保持。

最终工具修正只涉及传输侧录校验：详细诊断使用独立provider-transport.json，账单侧录保持旧格式。原预检和运行器manifest保留；该工具调整没有改变生产代码、来源捕获或冻结请求。

当前等待r1333授权。r1330已关闭；历史两个失败保留。readiness.json保存本地控制与捕获证据。
