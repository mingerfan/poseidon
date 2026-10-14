# MHA 程序包输入域检查

三个保留模型（self Attention→Linear→残差、投影后的 cross Attention、布尔 mask／padding）完成 8 项输入域测试：8 passed，0 failed/errors/skipped。检查使用原数学图和既有独立参考；没有重新捕获、生成、编译或密态执行。

测试覆盖合法零／有符号输入、首次越界节点定位、第二组输入拒绝、诊断脱敏、16／17 组边界、数据快照、value 输入与分数域的区分，以及 masked／unmasked key 的区分。布尔模型程序包读取的原图与保留捕获图一致，输出结构仍按 v16 规则恢复。区间内可执行输入不因此取得数值或原函数精度认证。

应用源码和待批准两项重测绑定保持。691 个冻结源码文件、654 个保护文件核验一致。新增内容仅为开发审计脚本和报告。API、编译、密态执行、安装、下载、参数及后端修改、提交／推送均为 0。

原三项付费结果仍见 [阶段报告](agent-mha-live-results-r1315.md)。两项重测仍待原有审批问题回复，范围按 [修正方案](agent-mha-feedback-live-proposal-r1317.md)，原审核与修正的等价关系见 [入口报告](agent-mha-retest-readiness-r1318.md)。Bootstrap、CUDA、大图和 ResNet20 状态保持。

复查命令（使用现有固定 Poseidon Nix/Python 环境）：

```sh
python -B scripts/baseline/benchmarks/tools/mha_input_domain_audit_v1/audit.py
```

审计工具采用一次性输出，已有 report.json 时保留原结果；后续复验应建立新输出目录或使用标准测试收集方式，不覆盖当前证据。运行范围只含输入域控制，不能计为编译、密态或 Agent 新生成通过。

机器报告：`docs/baseline/agent-mha-input-domain-audit-r1319.json`。审计绑定：`975ac526ba0a594250d55044f652029ac8ea00f1a70c0bad07d9d65e26a82f2e`。
