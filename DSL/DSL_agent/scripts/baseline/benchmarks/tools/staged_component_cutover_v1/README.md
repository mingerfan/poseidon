# Agent 输入与反馈联合切换

合并 V12 顺序张量／别名输入与 explicit-v11 公开绑定、结构化修复反馈。生产改动限于 plan.json 列出的 25 个 Agent 文件，后端库保持当前版本。

run_joint.py 在已有 portable Nix store 上建立独立网络隔离，检查来源捕获、双参考、旧格式、应用接纳和反馈。r1196 为 251 项通过；替身后端检查与真实来源捕获分别记录。

guarded_apply_joint.py 核对原授权 76 项、账本、进程、Nix 锁、源码及恢复副本。只读命令：

~~~sh
python3 -B scripts/baseline/benchmarks/tools/staged_component_cutover_v1/guarded_apply_joint.py
~~~

到达 ready 后，--apply 安装指定 Agent 文件，保存原文件与 Git 状态；后端验收仍为 pending。

post_cutover_acceptance.py --plan 列出 Linear、LayerNorm、公开 split 权重三例。--execute <ARM results下的新目录> 仅在切换完成后进入一个已有 Nix worker，串行完成真实来源捕获、规则候选数值验收、包新输入执行和缓存复用。规则候选与真实 Agent 分别记录。

本目录为开发工具。应用请求继续调用现有 Application，沿用任务预算与后端接口。

当前27文件（含Poseidon沙箱／包身份修正）已接入。三例真实来源、规则编译／密态、包新输入／缓存全部通过，独立审计r1230，21组／119值。原76项关闭后才切换；原失败和r1225保留。新入口真实Agent评测另行冻结批准。
