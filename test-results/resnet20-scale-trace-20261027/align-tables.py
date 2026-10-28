"""Reorder the recorded traces for comparison; never change a recorded operation."""

from collections import defaultdict
from pathlib import Path
import csv
import heapq
import json
import math
import re


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "scale-trace.json"
data = json.loads(SOURCE.read_text())
sides = ("manual", "compiler")
graphs = {side: data[side] for side in sides}
ops = {side: {op["id"]: op for op in graphs[side]["operations"]} for side in sides}
position = {side: {op["id"]: i for i, op in enumerate(graphs[side]["operations"])} for side in sides}
muls = {side: {op["id"]: op for op in graphs[side]["multiplications"]} for side in sides}
moduli = {q["q_count"]: q for q in data["moduli"]}


def is_core(op):
    return op["kind"] == "mul_cc" or (op["kind"] == "mul_cp" and op["scalar"] != 1)


def core_key(op):
    label = op["label"]
    if op["kind"] == "mul_cc":
        if label.startswith("T"):
            name, expression = label.split("：", 1)
            label = name + "：" + " × ".join(sorted(expression.split(" × ")))
        elif label.startswith("H"):
            label = label.replace("商项 × ", "").replace(" × 商项", "") + " × 商项"
    return op["stage"], op["kind"], label


core = {side: {core_key(op): op["id"] for op in ops[side].values() if is_core(op)} for side in sides}
assert len(core["manual"]) == len(core["compiler"]) == 57
assert core["manual"].keys() == core["compiler"].keys()

ancestor_cache = {}


def core_parents(side, oid):
    key = side, oid
    if key in ancestor_cache:
        return ancestor_cache[key]
    op = ops[side][oid]
    if is_core(op):
        result = frozenset([core_key(op)])
    else:
        result = frozenset().union(*(core_parents(side, p) for p in op["inputs"]))
    ancestor_cache[key] = result
    return result


def input_core_parents(side, oid):
    return frozenset().union(*(core_parents(side, p) for p in ops[side][oid]["inputs"]))

# One common anchor for each multiplication and each completed mathematical
# subproblem. Completed subproblems can end in different primitive operations.
pairs = []
for key, mid in core["manual"].items():
    cid = core["compiler"][key]
    assert ops["manual"][mid]["kind"] == ops["compiler"][cid]["kind"]
    if ops["manual"][mid]["kind"] == "mul_cp":
        assert math.isclose(ops["manual"][mid]["scalar"], ops["compiler"][cid]["scalar"], rel_tol=0, abs_tol=1e-12)
    pairs.append((mid, cid, "multiplication", key[0], key[2]))

# Pair the five unit multiplications that align a basis correction branch.
# Distinguish the correction and product branches by arithmetic ancestors.
for mid, op in ops["manual"].items():
    if op["kind"] != "mul_cp" or op["scalar"] != 1:
        continue
    basis = re.match(r"(T\d+)：", op["label"])[1]
    candidates = [cid for cid, other in ops["compiler"].items()
                  if other["kind"] == "mul_cp" and other["scalar"] == 1
                  and other["stage"] == op["stage"]
                  and other["label"].endswith(basis + "：基底校正")
                  and input_core_parents("manual", mid) == input_core_parents("compiler", cid)]
    assert len(candidates) == 1, (mid, candidates)
    pairs.append((mid, candidates[0], "scale_alignment", op["stage"], basis + "：校正支路 ×1 对齐"))


def group_key(op):
    match = re.match(r"([TH]\d+)：", op["label"])
    return (op["stage"], match[1]) if match else None


checkpoints = {}
for side in sides:
    groups = {}
    for op in ops[side].values():
        key = group_key(op)
        if key and op["kind"] not in ("encode", "transfer"):
            groups[key] = op["id"]
    checkpoints[side] = groups
assert checkpoints["manual"].keys() == checkpoints["compiler"].keys()
assert len(checkpoints["manual"]) == 40
for key, mid in checkpoints["manual"].items():
    cid = checkpoints["compiler"][key]
    assert not is_core(ops["manual"][mid]) and not is_core(ops["compiler"][cid])
    pairs.append((mid, cid, "checkpoint", key[0], key[1] + " 完成"))
pairs.append((graphs["manual"]["output"], graphs["compiler"]["output"], "checkpoint", "尾部", "ReLU 输出"))
input_ids = {side: next(o["id"] for o in ops[side].values() if o["kind"] == "input") for side in sides}
pairs.append((input_ids["manual"], input_ids["compiler"], "input", "输入", "ReLU 输入"))

# Merge the corresponding anchors, then sort the union of BOTH dependency
# graphs. A successful sort guarantees that neither displayed column violates
# its original dependencies, including operations hidden in the compact table.
node_for = {}
nodes = {}
for mid, cid, kind, stage, label in pairs:
    nid = "pair:" + mid
    assert ("manual", mid) not in node_for and ("compiler", cid) not in node_for
    node_for["manual", mid] = node_for["compiler", cid] = nid
    nodes[nid] = dict(manual=mid, compiler=cid, alignment=kind, stage=stage, label=label)
for side in sides:
    for oid, op in ops[side].items():
        if (side, oid) not in node_for:
            nid = side + ":" + oid
            node_for[side, oid] = nid
            nodes[nid] = {side: oid, "alignment": "auxiliary", "stage": op["stage"], "label": op["label"]}
successors = defaultdict(set)
indegree = dict.fromkeys(nodes, 0)
for side in sides:
    for oid, op in ops[side].items():
        child = node_for[side, oid]
        for parent_id in op["inputs"]:
            parent = node_for[side, parent_id]
            assert parent != child
            if child not in successors[parent]:
                successors[parent].add(child)
                indegree[child] += 1

priorities = {}


def priority(nid, active=None):
    if nid in priorities:
        return priorities[nid]
    active = set() if active is None else active
    assert nid not in active, "Pairing corresponding steps introduced a dependency cycle"
    active = active | {nid}
    node = nodes[nid]
    if "manual" in node:
        value = position["manual"][node["manual"]]
    else:
        value = min((priority(child, active) for child in successors[nid]), default=10000)
    priorities[nid] = value
    return value


ready = []
for nid, degree in indegree.items():
    if degree == 0:
        heapq.heappush(ready, (priority(nid), nid))
order = []
while ready:
    _, nid = heapq.heappop(ready)
    order.append(nid)
    for child in sorted(successors[nid]):
        indegree[child] -= 1
        if indegree[child] == 0:
            heapq.heappush(ready, (priority(child), child))
assert len(order) == len(nodes), "Cannot align both graphs topologically"
rank = {nid: i for i, nid in enumerate(order)}
for side in sides:
    for oid, op in ops[side].items():
        assert all(rank[node_for[side, p]] < rank[node_for[side, oid]] for p in op["inputs"])

# Check the arithmetic dependencies of all matched multiplications. Ignore
# scale-management operations, but keep every common multiplication ancestor.
for key in core["manual"]:
    parents = {}
    for side in sides:
        parents[side] = input_core_parents(side, core[side][key])
    assert parents["manual"] == parents["compiler"], (key, parents)


def f(value):
    return f"{value:.6f}"


def qtext(q):
    m = moduli[q]
    return f'Q{q}；q[{m["index"]}]={m["value"]}（{m["bits"]} bit）'


def effect_text(row):
    effects = row["rescale_effects"]
    if not effects:
        return "无后续 rescale"
    parts = []
    for effect in effects:
        bits = "/".join(str(p["bits"]) for p in effect["primes"])
        parts.append(f'RS {bits} bit [{",".join(effect["ids"])}] → Q{effect["q_after"]}／s={f(effect["scale_after"])}')
    return "<br>".join(parts)


def mul_cell(side, oid):
    if oid is None:
        return "—"
    row = muls[side][oid]
    return (f'[{oid}] {row["kind"]}；CC/全部深度={row["cc_depth"]}/{row["mul_depth"]}'
            f'<br>{qtext(row["q_before"])}'
            f'<br>s：{f(row["left_scale"])} + {f(row["right_scale"])} → {f(row["product_scale"])}'
            f'<br>{effect_text(row)}')


def op_cell(side, oid):
    if oid is None:
        return "—"
    op = ops[side][oid]
    args = ", ".join(f'[{p}] Q{ops[side][p]["q"]}/s={f(ops[side][p]["scale"])}' for p in op["inputs"])
    result = f'[{oid}] {op["kind"]}：{op["label"]}'
    result += f'<br>输入：{args or "无"}<br>输出：{qtext(op["q"])}；s={f(op["scale"])}'
    if op["removed"]:
        result += '<br>除去：' + '；'.join(f'q[{p["index"]}]={p["value"]}（{p["bits"]} bit）' for p in op["removed"])
    if op["kind"] == "mul_cp":
        result += f'<br>明文值：{op["scalar"]:.12g}'
    return result


mul_rows = []
full_rows = []
checkpoint_rows = []
for nid in order:
    node = nodes[nid]
    record = {**node, "topological_row": rank[nid] + 1}
    full_rows.append(record)
    if node["alignment"] in ("checkpoint", "input"):
        checkpoint_rows.append(record)
    if any(oid in muls[side] for side in sides if (oid := node.get(side)) is not None):
        for side in sides:
            assert side not in node or node[side] in muls[side]
        mul_rows.append(record)
assert len(mul_rows) == 99
assert sum(r["alignment"] == "multiplication" for r in mul_rows) == 57
assert sum(r["alignment"] == "scale_alignment" for r in mul_rows) == 5
assert sum("manual" in r for r in mul_rows) == 62
assert sum("compiler" in r for r in mul_rows) == 99

summary = """# 手动版与编译器版 ReLU：按同一拓扑顺序逐步对照

本表只调整已有跟踪记录的展示顺序，未修改编译器、计算图、scale、模数或 rescale。范围仍是输入 Q31、log₂(scale)=40 的同一个 15/15/27 ReLU，不含 Boot、Stem、卷积；编译器记录使用原诊断下界。

原来两张“Q 消耗路径”各自选取了不同分支，因此不能按行号比较。这里改为同一张左右对照表：先按手动版的基底和子多项式顺序排列，再合并两边依赖并做拓扑排序。每列都检查了全部原始输入依赖，没有把子节点排到父节点之前。独立分支允许换序，原始操作 ID 保留，可回查原记录。

57 个数学乘法全部一一对应：27 个 CC 和 30 个系数 CP；系数值及去掉 scale 管理后的乘法依赖均已核对。另有 5 个基底校正支路的 ×1 对齐，按同一用途和上游依赖放在同一行。编译器其余 37 个 ×1 管理乘法单独占行，手动侧写“—”。手动版共有 62 个逻辑乘法，编译器版有 99 个实际乘法；本表共 99 行乘法，覆盖两边所有分支。

这里的“相同数学乘法”允许左右操作数交换。基底倍增也可能在乘法前或后，rescale 可能在校正前或后，因而不表示原语图完全相同。这些实际差异保留在[逐原语对照表](aligned-steps.md)中。

所有 s 都是 log₂(scale)。Qn 表示剩余 n 个素数，Runtime level=n−1；当前 rescale 素数是 q[n−1]，表内列出其整数值和位宽，scale 实际减去 log₂(q)。RS 表示 rescale。乘法表中的“后续 RS”可能经过加减法才发生，多个乘法可共享同一 RS ID；它是注释，不代表紧接该行执行，准确先后请看逐原语表。

先看同一结果检查点，再看全部乘法。完整原始数值保留在 [scale-trace.json](scale-trace.json)，左右同排的全部字段另见 [aligned-multiplications.csv](aligned-multiplications.csv)。

## 同一结果检查点

每行是一个基底或一个子多项式完成后的结果，已包含该实现为此节点安排的校正、scale 对齐和 rescale。Hn 是原分解树的节点编号，H(2n+1) 为商子树，H(2n) 为余子树；H1 是阶段根节点。

|顺序／同名结果|手动版完成操作／Q／s|编译器版完成操作／Q／s|
|---|---|---|
"""
for i, row in enumerate(checkpoint_rows, 1):
    cells = []
    for side in sides:
        op = ops[side][row[side]]
        cells.append(f'[{op["id"]}] {op["kind"]}；Q{op["q"]}／{f(op["scale"])}')
    summary += f'|{i}. {row["stage"]} {row["label"]}|{cells[0]}|{cells[1]}|\n'
summary += "\n## 全部乘法：同一计算同一行\n\n"
current_stage = None
core_number = 0
for i, row in enumerate(mul_rows, 1):
    if row["stage"] != current_stage:
        current_stage = row["stage"]
        summary += f'\n### {current_stage}\n\n|对照行／数学步骤|手动版|编译器版|\n|---|---|---|\n'
    if row["alignment"] == "multiplication":
        core_number += 1
        label = f'数学步骤 {core_number}：{row["label"]}'
    elif row["alignment"] == "scale_alignment":
        label = row["label"] + "（同一用途）"
    else:
        label = ("手动版独有管理：" if "manual" in row else "编译器版独有管理：") + row["label"]
    summary += f'|{i}. {label}|{mul_cell("manual", row.get("manual"))}|{mul_cell("compiler", row.get("compiler"))}|\n'
summary += "\n## 模数链\n\n|剩余 Q／下标|当前模数|整数位宽|实际 log₂(q)|\n|---|---:|---:|---:|\n"
for m in reversed(data["moduli"]):
    summary += f'|Q{m["q_count"]}／q[{m["index"]}]|{m["value"]}|{m["bits"]}|{m["log2"]:.12f}|\n'
(ROOT / "comparison.md").write_text(summary)

detail = """# 手动版与编译器版：逐原语拓扑对照

这是[同排乘法表](comparison.md)的完整顺序依据；s=log₂(scale)。左右相同的数学乘法、同一校正支路的 ×1 对齐、以及同一数学子问题的完成检查点被对齐；检查点两侧可能是不同原语，不能当成同一种指令。其余原语单独占行，空侧为“—”。原始操作、依赖、数值均未改动。

编码和数据传输不消耗乘法深度，在此省略以方便阅读，但参与了全图拓扑检查，完整记录见 aligned-order.json。ModSwitch、倍增、取负、加减、重线性化、×1 和每一次 rescale 全部列出。原标签来自跟踪位置；共享的倍增／取负可能带有某个使用者的 Tn 标签，实际依赖以 ID 为准。按任意一列自上而下读，都是该列的合法拓扑顺序。
"""
current_stage = None
display_row = 0
for row in full_rows:
    if all(ops[side][row[side]]["kind"] in ("encode", "transfer") for side in sides if side in row):
        continue
    display_row += 1
    if row["stage"] != current_stage:
        current_stage = row["stage"]
        detail += f'\n## {current_stage}\n\n|行／对齐依据|手动版|编译器版|\n|---|---|---|\n'
    anchor = row["label"] if row["alignment"] != "auxiliary" else "单侧原语"
    detail += f'|{display_row}. {anchor}|{op_cell("manual", row.get("manual"))}|{op_cell("compiler", row.get("compiler"))}|\n'
(ROOT / "aligned-steps.md").write_text(detail)

validation = dict(
    manual_operations=len(ops["manual"]),
    compiler_operations=len(ops["compiler"]),
    matched_mathematical_multiplications=57,
    matched_correction_scale_alignments=5,
    matched_result_checkpoints=41,
    multiplication_rows=99,
    manual_multiplications=62,
    compiler_multiplications=99,
    all_original_dependencies_checked=True,
    matched_multiplication_dependencies_equal=True,
    coefficient_absolute_tolerance=1e-12,
    source_operations_and_numeric_values_unchanged=True,
)
aligned = dict(source="scale-trace.json", validation=validation, order=full_rows)
(ROOT / "aligned-order.json").write_text(json.dumps(aligned, ensure_ascii=False, indent=2) + "\n")
fields = ["row", "stage", "label", "alignment"]
for side in sides:
    fields += [side + "_" + key for key in next(iter(muls[side].values()))]
with (ROOT / "aligned-multiplications.csv").open("w", newline="") as out:
    writer = csv.DictWriter(out, fieldnames=fields)
    writer.writeheader()
    for i, row in enumerate(mul_rows, 1):
        record = dict(row=i, stage=row["stage"], label=row["label"], alignment=row["alignment"])
        for side in sides:
            if side not in row:
                continue
            for key, value in muls[side][row[side]].items():
                record[side + "_" + key] = json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
        writer.writerow(record)
print(json.dumps(validation, ensure_ascii=False, indent=2))
