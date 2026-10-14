# 命名输出程序包离线补验 r1372

仅补验原 named 案例，返回结构为 `{"primary": y, "saved": [x, None]}`。模型、权重、候选 DSL、reference、布局与 w45／P4 保持原字节；规范化请求与原 r1369 相同。

重新提交任务使用修复后的执行身份，完成 1 次真实编译与四组密态资格验收，然后对同一包执行 3 组新输入及 1 组重复输入。共最多 **1 编译、3 密态调用、8 组／96 个输出值**，同时检查字典／列表中的 None 恢复和逐项误差。候选为原规则程序，API／HTTP 为 0。

全批≤240秒、任务≤180秒；新增产物≤128 MiB、全局≤32 GiB、磁盘余量≥4 GiB。编译≤2／链接≤1／密态串行，native 4 GiB／CPU150秒保持。没有安装、下载、网络、后端或参数修改。失败或超限保留现场。

- SDK：`e2b3e63c27e8ed49f6ddca34b2c5a5d8be366eefa9cdb1ab67d9b953b8a64992`
- request_id：`61e277c6e9d3062e24c98c14cadb54702edd201785b4f8c3b780c50257e6372d`
- 方案绑定：`0e0d796df8da1070f56c39209036bf1239490b1c8acbd0425e18d0042fc0e63a`
- 原批次审计：`43f485c321dce9572e9d4abbd5b61b5122d9fb40af282f5574232b18d174145f`

工具只准备了新任务，编译、密态、API 均未启动。执行要求此新范围的明确批准；旧批次及旧程序包不改写。

只读检查：

~~~sh
POSEIDON_PLATFORM=aarch64-linux POSEIDON_WORK_ROOT=/home/lhohy/poseidon-work python3 -B scripts/baseline/benchmarks/tools/public_expression_output_tree_v1/run.py --plan
~~~

详细文件、工具与阶段身份见同名 JSON。
