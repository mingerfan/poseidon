# 公开张量除法原型

本目录属于 Agent 开发工具。活动 PyTorch 来源仍为 v21，显式原型策略为 `poseidon-public-tensor-division-draft-v1`。

- `public_tensor_division.py`：公开除数校验、派生倒数、FX 投影和来源绑定。
- `test_division.py`：数学、广播、共享视图、权重和篡改测试。
- `controls-v3.json`：当前 52 项控制测试收据；前两版和错误快照保留。
- `backend-cases.json` 与 `backend.py`：三个固定等价图的 Poseidon CPU 验收。
- `final-audit.json`：CPU 验收前的来源和数学检查。
- `backend-final-audit.json`：三个程序包的产物、密态结果和复用审计。

运行器使用固定结果目录，已完成的验收记录保持原样。后续复跑使用新的输出目录、源码绑定和执行预算。

[范围、结果和下一步](../../../../../docs/baseline/agent-public-tensor-division-draft-r1350.md)
