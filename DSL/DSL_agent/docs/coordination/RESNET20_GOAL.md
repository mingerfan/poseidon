# Goal B：整模型、后端认证与 ResNet20

建议目标文本：在 lhy-resnet20 的独立 Ubuntu 工作目录中，复用通用 Agent 组件和最新固定后端，通过真实 bootstrap 与 MLP、RMSNorm、Attention 组合验收，最终由真实 Agent 生成 CIFAR-10 ResNet20 的完整程序包，完成准确率、密态差分和复用验收；CUDA 实机验收独立交付。

## 目标与所有权

拥有 workloads 下的模型、公开权重描述、数据规范、近似配置、校准/验证脚本和报告，以及 validation/backend_certification 的公共后端能力认证。
使用 A 的接入、分解、布局、生成、预算和包接口；发现缺口时提交最小复现与能力需求，不在本线复制或改写通用 lowering。
ResNet20 是整网目标之一，不代替通用性。

## 里程碑

| 编号 | 交付 | 完成条件 |
|---|---|---|
| B0 | 最新后端联合审计 | 核验 poseidon、Runtime、Dacapo 固定提交与语义补丁；真实 bootstrap/密钥/参数/ABI/输入输出接口分类；CPU/CUDA 状态独立；对 A 提交能力认证格式 |
| B1 | 非 bootstrap 路线 | 复用历史 Agent Linear 和平方 MLP 的原样源码；当前链路真实编译、密态、四组数值、包重载；代表性 packing/rotation/多输入输出/native 回归 |
| B2 | Bootstrap 与 CUDA 认证 | 真正消耗 level → 刷新 → 继续计算 → 多次刷新；检查原消息误差、level/scale 和评估密钥；CUDA 在另行获准 NVIDIA 环境执行，CPU/预检不算 CUDA |
| B3 | 三个整模型 | 小型平方 MLP、RMSNorm、Attention 自由 Agent 生成与数值/原函数精度分项；Attention 含投影、RoPE、密文点积、近似 Softmax、因果 mask、显式 KV 和多头代表例 |
| B4 | ResNet20 分层验收 | 残差块、下采样块、多个块和全网；中间例仅用于定位；冻结权重、实现、预处理、数据和近似配置，整网另由真实 Agent 生成 |
| B5 | 全网准确率与交付 | 完整测试集原模型/近似模型 Top-1 差≤1百分点；40张每类4张固定样本+4组有效域探针整网密态通过；导出、重载及重复执行 |

ResNet20 使用 CIFAR-10：batch=1，输入 [1,3,32,32]，16/32/64 通道，9 个残差块，10 logits；固定训练权重，标准 ReLU 原模型保持作为比较对象。
校准样本从训练集固定取每类100张共1000张；测试集不拟合近似。数据/权重尚缺时先核对已有文件，需要下载或训练另行批准。

## 验收

密态输出比较声明近似 reference，门限固定为 1e-5 + 1e-4*abs(reference)；另报告原函数误差和 Top-1，不修改权重或 reference 取得通过。
完整网络在同一密文、参数和密钥上下文内执行，层间解密再加密或 SEAL 回退不计通过。
历史人工源、规则源、历史 Agent 原样重放、新 Agent 生成分别记录；后端迁移可用历史 Agent 源证明，整模型生成要求仍保留。
bootstrap 采用公共真实实现；decrypt_reencrypt、直接 GPU 与 Runtime 一致或放宽误差后的旧结果均不能代替验收。
若当前接口不支持安全密文交接，分区只报告可行性，不串接多个重新生成密钥的 runner。
库、Runtime、编译器、参数或依赖需要改动时，先保留原例和诊断，给出最小方案及具体批准范围。无硬件时继续 CPU 和可独立完成项，CUDA 里程碑保持待验收。

## 执行与完成

遵守 PARALLEL_DEVELOPMENT.md 的全局上限与文件所有权。A 负责能力实现，B 在其版本上验收；共同请求哈希和身份吻合时引用既有证据，避免重复生成。
只有 B0–B5 的目标均有真实记录时完成对应 Goal；硬件、精度或 bootstrap 未通过保持未完成，并按 Goal 的阻塞规则处理。
不重建冻结1200模型，不用本线的特定网络例子作为输入白名单，也不自动扩大资源。

## 开始指令

先读 AGENTS.md、TASK_STATE.md、PARALLEL_DEVELOPMENT.md 和迁移报告；核验独立 checkout、分支、任务/预算/源码冻结。
确认共同基线后创建本目标对应 Goal，从 B0/B1 开始；B2 的认证结果交给 A 接线后，再推进 B3–B5。
