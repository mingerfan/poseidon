# DSL

本分支的 DSL 文件统一放在此目录。

```text
DSL/
├── DSL_benchmark/
│   └── llm_dsl/
│       ├── Qwen_Layer_DSL_Source/
│       ├── Qwen25_2Token_Block_DSL/
│       └── Qwen25_24Layer_DSL/
└── DSL_agent/
```

- [DSL_benchmark/llm_dsl](DSL_benchmark/llm_dsl)：由本分支原根目录 `llm_dsl/` 整体迁移的三个交付包，包内文件保持不变。使用方法和验证范围见各包的 README。
- [DSL_agent](DSL_agent)：预留 DSL 生成 agent 的文件目录。

进入各交付包目录后，原有相对路径命令仍可使用。引用仓库根目录旧路径的命令，请将 `llm_dsl/` 改为 `DSL/DSL_benchmark/llm_dsl/`。
