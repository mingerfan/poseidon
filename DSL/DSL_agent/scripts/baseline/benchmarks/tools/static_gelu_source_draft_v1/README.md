# GELU source variant draft

Agent 侧默认 GELU 与 tanh GELU 的输入适配草稿。

- `source_call()`：检查静态调用参数、来源版本和显式近似，生成多项式图与绑定。
- `from_fx_call()`：在可信捕获进程中识别固定 functional 或模块调用。
- `gelu_source_reference.py`：独立原函数数学参考。
- `audit.py --output <fresh-results-directory>`：使用锁定 Nix Python 环境完成签名、FX 与明文差分。

沿用 Python 3.10.14、NumPy 1.25.2、Torch 2.0.1 及现有 polynomial 分解。来源 none/tanh 与声明的近似多项式分别记录；输入按已有资源配置检查。

当前完成明文参考验收，生产输入和真实后端尚待接入。原模型精度单独认证。

详见 `docs/baseline/agent-gelu-source-draft-acceptance-r1060.md`。
