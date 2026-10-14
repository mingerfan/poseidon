# PyTorch v23 Poseidon CPU 离线验收 r1369

18 个来源上下文和候选已冻结；本阶段只准备请求，编译／密态／API 为 0。

范围：reshape、permute、slice、bool_cast、identity 各三个不同 shape／参数上下文，加 float32 原始状态、命名字典输出和未使用公开状态三例。四组固定输入逐项比较；permute_1、bool_cast_1、named 三份包另检三组新输入和一组重复输入。

候选来源为确定性规则基线，真实 Agent 成功另计。每例采用独立应用任务目录，避免跨例缓存代替新的编译验收。

| 案例 | 冻结 request_id |
|---|---|
| reshape_0 | 9cc698679022136975ea433b2fffc7cff6ff343b3e4de8560babd16bf194385a |
| reshape_1 | d83974f41e7d848483c300d636eb38f58d0f59077acfbc5acb56accd7f104f33 |
| reshape_2 | 30baf614dcd1a8e2c37cb63a652e7df8f33a6b9b64b96c350dcacf546585904f |
| permute_0 | ad6bae0e9224108eb611f7db2d5e2140e8e215d8d00ba85c2720a5afbdb44fa9 |
| permute_1 | e465db200bcf99e26b65da89f83b7c1c4b43a186966c6921450874056d455be8 |
| permute_2 | c8b7d1f986f462671c214eae4728b630ba2c746f6557e5d6a2d1d3a8d6579ee7 |
| slice_0 | bd9caec3d7047760650c13857759c41c9d907988f5f00d4045482fa0ecb598c5 |
| slice_1 | dbbd92d8fa56d038f5af9fff2ca19d5bcf6b2bcead4de97fa4f00db48e62d79f |
| slice_2 | 1bce88ae9b54465ce0c0589a0bc2de97cc9188bcd36b188591c75b1720a188c6 |
| bool_cast_0 | 1de53aa804f0c312e41775809ad13ea95c8dede914c71eb396c212b1d3f15f35 |
| bool_cast_1 | 5b500ffbaac96efb3beef1d72edd9da6f8f9edf7e63f0dbad915695f13c7681e |
| bool_cast_2 | 6992b1189a0965d1f097426c067c32a8dc1069ba511da64df0a728e4e01d30a0 |
| identity_0 | ad6bae0e9224108eb611f7db2d5e2140e8e215d8d00ba85c2720a5afbdb44fa9 |
| identity_1 | 9cc698679022136975ea433b2fffc7cff6ff343b3e4de8560babd16bf194385a |
| identity_2 | 3c6ee0fdae53d10a5eeb33be5c8982e847bcf545b3abe37bb1e4c07dae88614f |
| float32 | 9cc698679022136975ea433b2fffc7cff6ff343b3e4de8560babd16bf194385a |
| named | 61e277c6e9d3062e24c98c14cadb54702edd201785b4f8c3b780c50257e6372d |
| unused | 9cc698679022136975ea433b2fffc7cff6ff343b3e4de8560babd16bf194385a |

预算：全批 900 秒，每例 180 秒；新增产物 512 MiB，总 results ≤32 GiB、磁盘余量 ≥4 GiB。编译≤2、链接≤1、密态串行；native 内存4 GiB／CPU150秒保持。预计最多18次初始编译／密态验收、6次已编译程序包执行，共84组执行输入；API和HTTP为0。

后端保持 Poseidon CPU、w45、P4，门限仍为 1e-5+1e-4*abs(reference)。环境、网络、依赖、后端和参数保持。失败和超限保留现场，不扩大范围或续用旧批准。

方案绑定：faa04c4a585dfca471c8eca23484a149002c6d24d7d2b6bfef21c416afb854c1  
SDK：b41fe9599da34b14b013ce1aad5237969897132f0f0130643c795fc780841d5a

首次准备已完成全部18份数据，在核对配置字段 name/waterline 时修正工具字段检查；后续仅核对并冻结既有请求，未重新执行案例。所有原始记录保留。

准确文件哈希、任务、配置与工具身份见同名 JSON。执行入口要求本方案的新批准记录。
