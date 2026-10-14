# 通道布局与 BatchNorm 验收

2026-10-02，ARM Ubuntu，lhy-agent-dsl。新增 capacity-public-tensors-v8；默认 v7 和旧请求保持原样。

## 改动

- 固定统计量 BatchNorm 按密文块绑定 gain/offset，保留通道顺序和尾部补零。统计量被其他节点使用时仍保留其公开绑定。
- 对齐的二次幂后缀归约先做窗口加法，再选取每行起点。支持 sum/mean 和 keepdims，其他轴与宽度继续走原路径。
- v8 使用有界、无损的行字典压缩；保留浮点符号零和数值类型，拒绝非法长度、索引、膨胀和未使用字典条目。
- CLI 增加 --unified-packing，组件选项为 capacity_packing_version。省略时保留原默认行为。

## 验收

静态163 passed、0 failed/errors/skipped。六个新增模型各16组双参考和独立生成算术检查通过，包含三个BatchNorm及三个后缀归约上下文、跨行非对称探针、绑定篡改和旧请求/源码重建。

六个规则候选及两份历史真实Agent原源码（MLP/RMSNorm）8/8通过真实tracing、编译、RuntimePlan/spec/bundle、Poseidon CPU密态、解密及逐项比较。32组输入、3868个输出，最大绝对误差2.77961e-8。八包重载同样32组/3868值通过，最大误差5.74698e-8，生成和编译0。

公开CLI完成v8请求准备，不另算编译或密态通过。603执行文件一致，底层编译器、库、runner及参数与r535相同。新Agent生成/API、安装、下载、commit/push均0。

## 原尺寸 ResNet20 首层

原模型及权重未改。P4096结构测算：输入1块、中间4块、卷积144路由。原16516条绑定减至168条，公开数据无损压缩后55784字节。

完整 [1,3,32,32] → [1,16,32,32] stem的16组输入、256个通道均值与两套独立reference一致，最大差异6.66134e-16。压缩向量逐项、逐类型恢复一致。

**P4096尚未接通执行ABI。** 该stem只有公开布局数学及成本证据，没有编译、密态或Agent生成。现有执行周期仍为4–256。

下一步接通大周期的版本化输入输出、rotation授权、常量绑定和解码，再验收同一原尺寸stem。bootstrap、Runtime内存、Attention及新增输入的Agent测试和整网校准仍待完成。

## 使用

~~~sh
python3 -B scripts/agent.py --backend local candidate -- \
  --case scripts/baseline/cases/capacity-channel-v1/capacity_bn_0.json \
  --unified-chunk-period 256 --unified-packing capacity-public-tensors-v8 \
  --unified-guidance explicit-v10 --self-test
~~~

组件选项：

~~~python
{"chunk_period": 256, "capacity_packing_version": "capacity-public-tensors-v8"}
~~~

开发批次只读计划：

~~~sh
python3 -B scripts/baseline/benchmarks/tools/qualify_poseidon_channels.py \
  --output "$POSEIDON_WORK_ROOT/platforms/$POSEIDON_PLATFORM/results/channels-next"
~~~

原始记录：poseidon-channel-static-r543、poseidon-channel-numerical-r544、poseidon-full-stem-layout-r545、poseidon-channel-stage-audit-r546。r538源码备份、r539早期测试失败保留。旧profile断言已改为验证显式graph-v2，并拒绝去掉profile的512元素模型。Nix内重复启动外层CLI的开发调用失败保留；正常Ubuntu外层CLI复核通过。

门限保持 abs(actual-reference) <= 1e-5 + 1e-4*abs(reference)。x86原生、空白安装和CUDA暂缓。

[机器报告](poseidon-channel-layout-acceptance-r547.json)
