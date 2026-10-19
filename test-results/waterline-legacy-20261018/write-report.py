#!/usr/bin/env python3
"""Render the checked measurements as a concise Chinese experiment record."""
import json
from pathlib import Path

out = Path(__file__).resolve().parent
data = json.loads((out / "comparison.json").read_text())
by_key = {(r["group"], r["name"]): r for r in data["comparisons"]}
groups = {
    "local-mlp": "4060 MLP，64 MiB/窗口 128",
    "local-qproj": "4060 合成 q_proj，64 MiB/窗口 128",
    "local-pressure": "4060 大负载，3.125 GiB 权重，256 MiB/窗口 512",
    "server-mlp1": "V100 单卡 MLP，64 MiB/窗口 128",
    "server-mlp2": "V100 双卡 MLP，64 MiB/窗口 128",
    "server-pressure": "V100 大负载，12.5 GiB 权重，1024 MiB/窗口 512",
}
names = ["legacy-eager", "unfixed-eager", "unfixed-waterline", "current-eager", "current-waterline"]


def value(group, name, metric="online_ms"):
    return by_key[group, name][metric]["median"] / 1000


text = [
    "Waterline 引入前的执行器与修复前后实现对照，2026-10-18",
    "",
    "比较方法：",
    "历史执行器取自 ckks-runtime 的 f7e9115eaec22e44c8e12ff9f53528e421691688，即 04819dd 引入 Waterline 调度器之前的最后一个版本。runtime/runtime.hpp 原样提取，没有修改其中的执行逻辑。",
    "修复前执行器取自 a7245b81e647e55974eb295da9f86d4066f390a3 的 HEAD 文件；当前执行器使用本轮已验证的工作区版本。三个头文件和 SHA-256 均保存于 headers/、provenance.json。",
    "每台机器的三个版本使用相同的当前 GPU 后端、验证器、计划加载器、runner、编译优化参数和链接库，只替换 Runtime 头文件。编译依赖文件确认实际选中了对应头文件。这是历史执行器对照，不是把整个仓库和所有依赖恢复到旧版。保留当前后端能排除旧版数值实现、流排序和依赖环境差异。",
    "相同负载的计算指令、参数文件、fixture 和全部权重文件指纹一致；每个样本独立加密相同逻辑输入。三种全加载计划完全相同。水位版本只调整准备和内存管理，计算指令不变。",
    "",
    "在线执行时间，单位秒，越小越好。水位列均使用 GPU Prepare。",
    "负载 | 历史全加载 | 修复前全加载 | 修复前水位 | 当前全加载 | 当前水位",
]
for group, label in groups.items():
    if (group, "legacy-eager") in by_key:
        text.append(label + " | " + " | ".join(f"{value(group, name):.6f}" for name in names))
text += ["", "当前全加载相对历史全加载的耗时变化："]
for group, label in groups.items():
    if (group, "legacy-eager") in by_key:
        ratio = value(group, "current-eager") / value(group, "legacy-eager")
        waterline = value(group, "current-waterline") / value(group, "legacy-eager")
        text.append(f"{label}：当前全加载增加 {(ratio-1)*100:.2f}%；当前水位耗时为历史全加载的 {waterline:.2f} 倍。")
text += [
    "",
    "完整首轮 forward，包含初始化与权重准备，单位秒：",
    "负载 | 历史全加载 | 修复前全加载 | 修复前水位 | 当前全加载 | 当前水位",
]
for group, label in groups.items():
    if (group, "legacy-eager") in by_key:
        text.append(label + " | " + " | ".join(f"{value(group, name, 'forward_ms'):.6f}" for name in names))
text += [
    "",
    "计时范围与解释：",
    "全加载 online 排除提前编码和上传；水位 online 包含运行中的读取、编码和上传。两者都等待最终结果并排空 GPU 工作，没有只测异步提交后立即返回。完整 forward 包含初始化和权重准备；密钥生成、输入加密均单独计时，不含在上述两种指标中。",
    "全加载使用 CPU 提前编码，水位列使用 GPU 编码，因此首轮 forward 的差距同时包含编码后端差异，不能全部归因于内存调度。",
    "当前全加载的明显加速，主要体现为恢复 Waterline 执行器改造后丢失的速度。当前仍经过逐项派发和线程通知；历史版本由工作线程直接遍历自己的队列。部分服务器负载仍有明显额外耗时，不声称已经在所有负载上完全恢复原始性能。逐项派发和线程间往返是剩余开销的可疑来源，尚未逐项拆分证明各因素的贡献。",
    "",
    f"q_proj 默认 Auto/窗口 8 的在线时间为 {value('local-qproj', 'unfixed-auto-w8'):.6f} → {value('local-qproj', 'current-auto-w8'):.6f} 秒；历史全加载为 {value('local-qproj', 'legacy-eager'):.6f} 秒。GPU/窗口 128 的水位结果不能代替默认 Auto/8 的结果。",
    "q_proj 是合成算子负载，并非完整 Qwen。完整 Qwen 本轮没有重测，不能据此宣称其所有速度问题已经解决。",
    "",
    "验证与样本：",
    "RTX 4060 MLP 最终每项 7 次，其余每项 3 次；每个样本使用新进程。每轮轮换配置顺序；修复前水位的本地补测是另一个独立批次。初始本地 MLP 每项 3 次的数据完整保留为 pilot，因短耗时样本波动较大，最终 MLP 中位数统一取新的 7 次对照，不挑选混合样本。",
    f"共 {data['verified_samples_total']} 个未加 profiling 的样本校验通过，其中 {data['timing_samples_used']} 个用于最终表格，另外 15 个是上述 pilot。最早的本地 smoke 单独保存，不计入这些数量。",
    "MLP 对照独立参考输出，q_proj 和大负载逐一检查全部 8192 个复数槽。另检查相同计算次数、输入指纹、池上限、水位账本边界及结束时归还额度；q_proj/大负载的 pinned staging 在运行和校验结束后均归零。",
    "独立 Nsight 本地 RTX 4060 MLP 采样不进入时延中位数：三个执行器均为 16018 个 kernel，名称、grid/block 和次数完全相同。历史/修复前/当前的 kernel 活动并集分别为 104.90/102.67/104.71 ms，首末 kernel 跨度为 132.26/236.70/134.13 ms，没有 kernel 活动的间隙为 27.36/134.03/29.43 ms。这些间隙可能包含 memcpy 活动，不代表整个 GPU 完全空闲；此采样覆盖进程的全部 GPU kernel，不能直接当作未加 profiling 的 online 时间。",
    "",
    "环境与复现：",
    "本地 RTX 4060 Laptop 8 GiB，固定物理池 6 GiB；V100-SXM2 32 GiB，MLP 固定池 6 GiB/卡，大负载固定 24 GiB。服务器只使用物理 GPU 2、3。服务器访问开始时短暂不可达，用户恢复连接后完成服务器测试；连接失败记录单独保留，不计入运行失败或计时。",
    "没有锁定 GPU/CPU 频率，没有清理 OS 文件缓存。误差范围与全部原始样本见 comparison.json 和 measurements.txt，图中误差线为最小值到最大值。",
    "生产代码未在本轮修改，也没有提交或推送。实验二进制位于 build-waterline-legacy-20261018；服务器位于 /home/xuming/poseidon-waterline-legacy-20261018。",
    "build-variants.py 构建三个执行器；local-jobs.json、local-unfixed-waterline-jobs.json、local-mlp-repeat-jobs.json、server-jobs.json 记录确切配置；run-comparison.py 轮换运行。输出目录有意禁止覆盖，复现时需换新目录。summarize.py 检查原始样本，merge-results.py 合并结果，plot.py 生成图表。",
]
(out / "diagnosis.zh-CN.txt").write_text("\n".join(text) + "\n")
