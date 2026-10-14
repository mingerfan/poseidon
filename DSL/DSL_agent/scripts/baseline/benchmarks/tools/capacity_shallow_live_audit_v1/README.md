# r1540 终态证据审计

仅处理 capacity_shallow_live_v1 的冻结 9 项，以及其中 3 份程序包复用记录。
生成预算为 36 次生成 / 144 次 HTTP；编译与密态调用上限分别为 36 / 42。
产物总上限 48 GiB。真实 Agent 来源依据保留的传输账本、原始响应和候选字节；
这些是本地来源证据，不是供应商签名或账单证明。

审计使用独立 NumPy 数学解释器与 CPU float64 Torch，重新读取冻结输入、reference
和实际 decrypted.npy；不会将规则候选或最终候选作为 expected 计算来源。
失败、未运行、启动后未记账、未结算调用和未知后端启动均分别记录。
空 compile.log 和预先保存的 compile_command 不足以证明编译器实际启动。

只能在现有 owner 退出且 terminal.json 已落盘后执行一次：

```sh
cd /home/lhohy/Code/Poseidon
POSEIDON_PLATFORM=aarch64-linux POSEIDON_WORK_ROOT=/home/lhohy/poseidon-work \
python3 -B scripts/baseline/benchmarks/tools/capacity_shallow_live_audit_v1/audit.py --execute
```

入口使用既有 Poseidon Nix 环境，最长 300 秒，保留原进程资源限制。
不生成、不编译、不执行密态计算、不读取凭据。唯一输出是本目录 report.json；
已存在时拒绝覆盖。audit_status 表示证据审计是否完整，不代表 9 个案例全部通过。
退出码 2 表示证据不完整，保留已核对结果及明确问题，不可算通过。
工具目前仅准备、内存编译检查，尚未启动终态审计。
