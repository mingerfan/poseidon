"""E10c/E10d (API 0, no Nix): bootstrap gain and wall-clock breakdowns from existing results.

E10c lists, per model, the compiled bootstrap count after each step that changes it: the rule program with
depth-based placement, the same program under the program-level planner, the rule softmax search (Transformers),
the best uniform recipe (CNNs, E10b), the tuner alone and the agent. E10d splits each session's wall clock into main
LLM calls, the tuner (with the LLM calls it makes) and the rest, and adds model intake and the compile step, so the
end-to-end time from the plaintext model to the compiled program can be reported. Reads results only.

Outer:  python3 -B -m pt_agent.experiments.e10c_breakdown --root <E10 dir>
"""
import argparse
import glob
import json
from pathlib import Path

from ..paths import RESULTS_DIR

R = RESULTS_DIR
E9 = R/'E9-same-pipeline-20261110'
# A1 rule programs under depth-based placement (R121 build); source: ch1-2-sources.md, Fig. softmax data.
A1_DEPTH = dict(bert=dict(R0=12081, R1=9095), qwen=dict(R0=38849, R1=22721))
INTAKE_S = dict(bert=R/'A1-bert-prepare-20261027T123333-7967/REPORT.json',
                qwen=R/'A1-qwen-prepare-v2-20261028T135355-6529/REPORT.json')
TRANSFORMER_SESSIONS = (   # label, model, session folder, compiled bootstraps of its frozen program, compile REPORT
    ('E5d', 'bert', 'E5d-agent-sessions-20261106/bert/session', 7175, 'E9-same-pipeline-20261110/c/E5d/acceptance/pars'),
    ('E6-s1', 'bert', 'E6-llm-select-20261107/e6b/bert/s1/session', 6808, 'E9-same-pipeline-20261110/c/E6-s1/acceptance/pars'),
    ('E6r2-p1', 'bert', 'E6r2-resume-20261107/e6r2-b/bert/p1/session', 7141, 'E9-same-pipeline-20261110/c/E6r2-p1/acceptance/pars'),
    ('E6r2-p2', 'bert', 'E6r2-resume-20261107/e6r2-b/bert/p2/session', 6977, 'E9-same-pipeline-20261110/c/E6r2-p2/acceptance/pars'),
    ('E5d', 'qwen', 'E5d-agent-sessions-20261106/qwen/session', 21857, None),
    ('E6-s1', 'qwen', 'E6-llm-select-20261107/e6b/qwen/s1/session', 22272, None),
    ('E8b-r1', 'qwen', 'E8-align-fallback-20261108/e8b/qwen/r1/session', 20800, 'E8-align-fallback-20261108/e8b/qwen/r1/acceptance/pars'),
    ('E8b-r2b', 'qwen', 'E8-align-fallback-20261108/e8b/qwen/r2b/session', 21536, 'E8-align-fallback-20261108/e8b/qwen/r2b/acceptance/pars'))
BERT_REPLAYS = dict(E5d=7175, **{'E6-s1': 7046, 'E6r2-p1': 7141, 'E6r2-p2': 6977})   # E9c, final rules
CNN = ('mlp', 'cnn_plain', 'resnet20_silu', 'resnet20')


def read(path):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else {}


def compiled(item):
    return ((item or {}).get('boots') or {}).get('compiled')


def trace_split(path):
    """Main LLM seconds, tuner seconds and the LLM seconds inside the tuner, from an agent trace."""
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()] if Path(path).exists() else []
    tuned = [row.get('tuned') or {} for row in rows]
    return dict(llm_s=sum(row.get('seconds') or 0 for row in rows), tuner_s=sum(t.get('seconds') or 0 for t in tuned),
                tuner_llm_s=sum(t.get('llm_s') or 0 for t in tuned), llm_calls=len(rows),
                estimates=sum(t.get('estimates') or 0 for t in tuned))


def steps_seconds(session):
    return sum(read(p).get('seconds') or 0 for p in glob.glob(str(Path(session)/'steps/*/REPORT.json')))


# ---------------------------------------------------------------- E10c

def gains(root):
    e9a, e9b, e9d = read(E9/'E9a.json').get('cases', {}), read(E9/'E9b.json').get('models', {}), read(E9/'E9d.json').get('models', {})
    e10b = read(Path(root)/'E10b.json').get('models', {})
    rows = []
    for model in ('bert', 'qwen'):
        r0, r1 = e9a.get(model+'-R0', {}), e9a.get(model+'-R1', {})
        agents = sorted(BERT_REPLAYS.values()) if model == 'bert' else sorted(
            boots for label, name, *_ in TRANSFORMER_SESSIONS if name == model for boots in [_[1]])
        rows.append(dict(model=model, rule_depth=A1_DEPTH[model]['R0'], rule_plan=compiled(r0.get('rule')),
                         softmax_search=compiled(r1.get('rule')), uniform=None,
                         tuner_only=min(filter(None, (compiled(r0.get('acceptance')), compiled(r1.get('acceptance'))))),
                         agent_min=agents[0], agent_max=agents[-1]))
    for model in CNN:
        b, d, u = e9b.get(model, {}), e9d.get(model, {}), e10b.get(model, {})
        # The best uniform recipe is the frozen uniform program with the fewest compiled bootstraps; the rule
        # program itself counts, since a lower degree can need more bootstraps (resnet20: deg 63 210, deg 127 209).
        options = [(compiled(b.get('rule')), u.get('rule_degree'))] if u else []
        if u.get('compile'):
            options.append((compiled(u.get('compile')), u.get('best_degree')))
        options = [option for option in options if option[0] is not None]
        uniform, uniform_degree = min(options) if options else (None, None)
        rows.append(dict(model=model, rule_depth=dict(mlp=18, cnn_plain=54, resnet20_silu=230, resnet20=344)[model],
                         rule_plan=compiled(b.get('rule')), softmax_search=None, uniform=uniform,
                         uniform_degree=uniform_degree, rule_degree=u.get('rule_degree'),
                         tuner_only=compiled(d.get('compile')), agent_min=compiled(b.get('compile')),
                         agent_max=compiled(b.get('compile'))))
    return rows


# ---------------------------------------------------------------- E10d

def times():
    rows = []
    intake = {model: read(path).get('seconds') for model, path in INTAKE_S.items()}
    for label, model, session, boots, pars in TRANSFORMER_SESSIONS:
        split = trace_split(R/session/'agent_trace.jsonl')
        total = steps_seconds(R/session)
        compile_s = read(R/pars/'REPORT.json').get('seconds') if pars else None
        rows.append(dict(kind='agent', model=model, label=label, boots=boots, intake_s=intake[model], session_s=total,
                         compile_s=compile_s, **split))
    for name, case in read(E9/'E9a.json').get('cases', {}).items():
        model = case.get('model') or name.split('-')[0]
        session = case.get('session') or {}
        rows.append(dict(kind='tuner-only', model=model, label=name, boots=compiled(case.get('acceptance')),
                         intake_s=intake.get(model), session_s=session.get('seconds'),
                         compile_s=read(E9/'a'/name/'acceptance/pars/REPORT.json').get('seconds'),
                         **trace_split(E9/'a'/name/'session/agent_trace.jsonl')))
    for kind, summary, folder in (('agent', 'E9b.json', 'b'), ('tuner-only', 'E9d.json', 'd')):
        for model, entry in read(E9/summary).get('models', {}).items():
            if model not in CNN:
                continue
            rows.append(dict(kind=kind, model=model, label=summary[:3], boots=compiled(entry.get('compile')),
                             intake_s=0.0, session_s=(entry.get('session') or {}).get('seconds'),
                             compile_s=(entry.get('compile') or {}).get('seconds'),
                             **trace_split(E9/folder/model/'session/agent_trace.jsonl')))
    for row in rows:
        parts = (row['intake_s'], row['session_s'], row['compile_s'])
        row['end_to_end_s'] = sum(parts) if all(part is not None for part in parts) else None
        row['other_s'] = (row['session_s'] - row['llm_s'] - row['tuner_s']) if row['session_s'] else None
    return rows


# ---------------------------------------------------------------- output

def _pct(new, base):
    return f'{100*(new-base)/base:+.1f}%' if base and new is not None else '—'


def _m(seconds):
    return '—' if seconds is None else f'{seconds/60:.1f}'


def write_reports(root, data):
    show = lambda value: '—' if value is None else f'{value:,}' if isinstance(value, int) else str(value)
    lines = ['# E10c：自举收益分解（编译自举数）', '',
             '每一列在上一列基础上多做一步；括号内为相对上一列的变化。Transformer 的 Agent 为四个会话的范围（BERT 用 E9c 最终规则下的重放）。', '',
             '| 模型 | 规则程序 + 深度放置 | + 规划器 | + 规则 softmax 搜索 | 最佳统一配方（阶数） | 只跑精调器 | Agent |',
             '|---|---|---|---|---|---|---|']
    for row in data['gains']:
        prev = row['rule_plan']
        middle = row['softmax_search'] if row['softmax_search'] is not None else row['uniform']
        uniform = '—' if row['uniform'] is None else f'{show(row["uniform"])}（{row.get("uniform_degree")}，{_pct(row["uniform"], prev)}）'
        softmax = '—' if row['softmax_search'] is None else f'{show(row["softmax_search"])}（{_pct(row["softmax_search"], prev)}）'
        agent = show(row['agent_min']) if row['agent_min'] == row['agent_max'] else f'{show(row["agent_min"])}–{show(row["agent_max"])}'
        lines.append(f'| {row["model"]} | {show(row["rule_depth"])} | {show(row["rule_plan"])}（{_pct(row["rule_plan"], row["rule_depth"])}） | '
                     f'{softmax} | {uniform} | {show(row["tuner_only"])}（{_pct(row["tuner_only"], middle or prev)}） | '
                     f'{agent}（{_pct(row["agent_min"], row["tuner_only"])} 至 {_pct(row["agent_max"], row["tuner_only"])}） |')
    (Path(root)/'E10c.md').write_text('\n'.join(lines)+'\n')
    lines = ['# E10d：端到端耗时分解（分钟）', '',
             '端到端 = 模型导入与校准 + 会话 + 编译步骤（下降、规划、导出、DaCapo 编译）。CNN 的导入在会话内。'
             '精调器一栏含精调器内部的 LLM 调用（llm-select）。', '',
             '| 类型 | 模型 | 会话 | 自举 | 导入 | 会话 | 其中主 LLM | 其中精调器（内含 LLM） | 其中其他 | 编译 | 端到端 |',
             '|---|---|---|---|---|---|---|---|---|---|---|']
    for row in data['times']:
        lines.append(f'| {row["kind"]} | {row["model"]} | {row["label"]} | {show(row["boots"])} | {_m(row["intake_s"])} | '
                     f'{_m(row["session_s"])} | {_m(row["llm_s"])} | {_m(row["tuner_s"])}（{_m(row["tuner_llm_s"])}） | '
                     f'{_m(row["other_s"])} | {_m(row["compile_s"])} | {_m(row["end_to_end_s"])} |')
    (Path(root)/'E10d.md').write_text('\n'.join(lines)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    data = dict(gains=gains(args.root), times=times())
    (args.root/'E10cd.json').write_text(json.dumps(data, indent=2, ensure_ascii=False)+'\n')
    write_reports(args.root, data)
    print('E10CD_DONE='+str(args.root/'E10cd.json'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
