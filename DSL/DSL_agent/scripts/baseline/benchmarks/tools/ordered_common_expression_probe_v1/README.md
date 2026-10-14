# 规则表达检查

probe.py 复用现有 scalar lowering，检查固定请求的新表达。它对 candidate 使用算术 AST 解释，并分别与 benchmark_math、benchmark_torch 比较。候选源码为 scalar-candidate.py。

报告中的 request_sha256 指执行传输文件字节的 SHA-256；原始任务的 canonical 请求哈希另列在 r1286 提案中。两种表示通过现有 request_file_bytes 校验，数学字段相同。
报告覆盖 16 组／192 值的明文槽位计算；frontend、编译、密态及 API 计数均为 0。
原失败候选和实际后端报告保存在 r1283，不改写历史记录。
