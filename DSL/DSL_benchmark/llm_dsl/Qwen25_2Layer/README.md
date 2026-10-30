# Qwen2.5 两层模型版本

这里集中存放 **两个独立 Decoder Block** 的 Qwen2.5 0.5B DSL。以后更新两层模型，请在此目录下新增独立版本文件夹，保留已验证历史版本，并更新本表。

| 版本 | 内容与状态 | CPU 完整测试 |
|---|---|---|
| [Qwen25_2Layer_R121_DSL](Qwen25_2Layer_R121_DSL/README.md) | 当前推荐；完整 CPU CKKS 通过，新打包与自举段预算、32线程、释放调度和64MiB明文缓存 | 74 次自举，45.680 分钟，峰值 RSS 27.079472 GiB，误差 < 1e-3 |
| [Qwen25_2Layer_R071_DSL](Qwen25_2Layer_R071_DSL/README.md) | 历史已验证版本；060 DSL + 070 后端、32 线程、Release + DFS + 64 MiB 明文缓存 | 154 次真实自举，约 2.254 小时，峰值 RSS 28.988 GiB，误差 < 1e-3 |
| [Qwen25_2Layer_Decode_Autoboot_DSL](Qwen25_2Layer_Decode_Autoboot_DSL/README.md) | 原两层自动自举交付包；从上级目录整体迁入，文件内容不变 | 604 次真实自举，约 25.21 小时，全保留模式峰值 RSS 382.082 GiB，误差 < 1e-3 |

这些版本均为两层、两 token、合成权重的 CPU 验证；保留策略、打包布局与后端不同，内存和耗时差不能仅归因于线程数。它们不代表完整 24 层、预训练模型或 GPU 已通过。

071 的输入是两个 embedding 密文，输出是 5 个 logits 密文和 2 个打包 KV 密文；旧版本输出布局不同，不能混用解包程序。详细介绍、使用方法和验证证据见对应 README。

上级的 `Qwen25_2Token_Block_DSL` 是 **单个 Block、两个 token**，不属于本目录；`Qwen25_24Layer_DSL` 为另一验证范围。不要将 token 数量当作网络层数。

旧路径 `DSL/DSL_benchmark/llm_dsl/Qwen25_2Layer_Decode_Autoboot_DSL` 已迁移到本目录下同名文件夹。进入包目录后的相对路径命令仍有效。
