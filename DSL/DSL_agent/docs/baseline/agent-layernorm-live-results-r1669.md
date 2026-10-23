# LayerNorm 真实 Agent 测试

本批使用原始模型、权重、Newton 近似配置和 P4/w45，explicit-v20。
原始响应、失败候选和旧绑定保留；成功源码与 Provider 响应一致。
数值参考为已冻结近似图，原 LayerNorm 应用精度另计。

| 案例 | 真实 Agent 终态 | 生成 / HTTP | 验收 |
|---|---|---:|---|
| layer_module_base | succeeded | 2 / 2 | 一次修复后数值通过，包重载、新输入、重复执行通过 |
| layer_module_affine | service_error | 3 / 3 | 编译 scale810→750，均超过720；第三次生成触及响应流资源上限 |
| layer_function_rank3 | service_error | 2 / 2 | 首轮编译 scale810>720；第二次生成触及响应流资源上限 |

最终1/3通过，failed=2，skipped/not_run=0。7次生成、7次HTTP。
5次真实编译尝试，1次成功；3次真实密态调用。
成功首次比较4组8值，max absolute error 5.4634448466e-8。
包复用4组8值，max absolute error 8.9032543907e-7。
门限保持 abs(actual-reference) <= 1e-5+1e-4*abs(reference)。

两次服务错误均在 HTTP200 响应体阶段记录 stream_limit，107281 / 107242 个SSE事件；
失败均为非重试项，retry_scheduled=false。现有检查共用 stream_limit 代码，旧记录缺少总字节/单行分类，
因此没有足够证据判断具体触发哪一种上限，也没有证据将其归为网络故障。

规则基线仍有2项编译失败；两项人工等价表达另行通过。
人工表达只证明同一参数下存在可执行表达，不计为真实 Agent 成功。
后续改进集中在生成上下文、公开系数处理指导和受控服务诊断，保持编译器、Runtime 和 Poseidon 不变。

SDK 316b4fcb795526fb96402050de45d6edadd294f76a3a7a25de3bf98d59a16084
审计绑定 ed052ba55e1476e50aa3da95005aa90e78f70364cb5ca6efa2c0887190c5eafa

执行目录：ARM work root/results/agent-layernorm-calibration-live-r1668。
审计：scripts/baseline/benchmarks/tools/layernorm_calibration_live_v1/audit-r1669.json。
本批已终态；原进程退出后才修改共享代码。网络、参数、reference、门限保持。
