# LLM DSL benchmarks

| 目录 | 用途 |
|---|---|
| [Qwen25_24Layer_R121_DSL](Qwen25_24Layer_R121_DSL/README.md) | 对应已验证R121两层优化的24层工程；两组小尺寸24层、四条路径算术通过，真实尺寸编译/CKKS/GPU待验证；含数据分析与level统计边界 |
| [Qwen25_2Layer](Qwen25_2Layer/README.md) | 两个 Decoder Block 的模型版本归档；当前已验证推荐版为 [Qwen25_2Layer_R121_DSL](Qwen25_2Layer/Qwen25_2Layer_R121_DSL/README.md) |
| [Qwen25_2Token_Block_DSL](Qwen25_2Token_Block_DSL/README.md) | 单个 Block、两个 token 的历史测试包 |
| [Qwen25_24Layer_R071_DSL](Qwen25_24Layer_R071_DSL/README.md) | 对应 071 两层优化方案的 24 层工程；小尺寸算术已检查，真实尺寸编译/CKKS/GPU 待验证 |
| [Qwen25_24Layer_DSL](Qwen25_24Layer_DSL/README.md) | 24 层模型 DSL；验证状态以该包说明为准 |
| [Qwen_Layer_DSL_Source](Qwen_Layer_DSL_Source/README.md) | Attention、RMSNorm 等独立子层源码 |

后续两层模型更新统一放在 `Qwen25_2Layer/<版本名>/`，每版附介绍 README、输入输出与运行说明、精度/性能证据及文件哈希。不要覆盖历史版本，也不要将 CPU 验证等同于 GPU 已通过。
