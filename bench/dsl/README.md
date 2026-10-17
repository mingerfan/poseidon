# DSL

本分支的 DSL 文件统一放在此目录。

```text
bench/dsl/
├── DSL_benchmark/
│   └── llm_dsl/
│       ├── Qwen_Layer_DSL_Source/
│       ├── Qwen25_2Token_Block_DSL/
│       ├── Qwen25_24Layer_DSL/
│       └── Qwen25_2Layer_Decode_Autoboot_DSL/
├── DSL_agent/
└── MLP.py
```

- [DSL_benchmark/llm_dsl](DSL_benchmark/llm_dsl)：Qwen DSL 交付包，随原根目录 `DSL/` 整体迁入，包内文件保持不变。使用方法和验证范围见各包的 README。
- [DSL_agent](DSL_agent)：预留 DSL 生成 agent 的文件目录。
- [MLP.py](MLP.py)：从 DaCapo 编译器示例复制的 MNIST MLP tracing 入口，需要 Hecate Python 环境和 `bench/data/mlp.model` 权重。

进入各交付包目录后，原有相对路径命令仍可使用。引用仓库根目录旧路径的命令，请将 `DSL/DSL_benchmark/llm_dsl/` 或更早的 `llm_dsl/` 改为 `bench/dsl/DSL_benchmark/llm_dsl/`。
