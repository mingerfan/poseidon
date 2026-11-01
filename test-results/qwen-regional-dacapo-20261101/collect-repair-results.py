"""Summarize bounded remote experiments after copying their small records."""
import json
from pathlib import Path
import re

root = Path(__file__).resolve().parent
revision = root / 'repair-targets'
summary = json.loads((root / 'repair-summary.json').read_text())
rows = []
for strategy in ('greedy', 'optimized'):
    case = revision / f'regional-{strategy}'
    result = json.loads((case / 'result.json').read_text())
    log = (case / 'compile.log').read_text()
    starts, intervals = {}, {}
    for name, event, seconds in re.findall(r'^\[compile\] (.*?) (begin|end|failed) total_s=([\d.eE+\-]+)', log, re.M):
        if name == 'input-parser':
            continue
        if event == 'begin':
            starts.setdefault(name, []).append(float(seconds))
        elif starts.get(name):
            intervals.setdefault(name, []).append(float(seconds) - starts[name].pop())
    first = re.search(r'^\[compile\] .*? begin total_s=([\d.eE+\-]+)', log[log.index('\n')+1:], re.M)
    result.update(scope='Full-model precompile: parse, Boot/scale planning and logical CKKS lowering; bytecode discarded. No RuntimePlan or weight bundle export.',
                  phase_seconds=intervals,
                  input_parse_seconds_approx=float(first[1]) if first else None,
                  sampled_peak_rss_gib=result['peak_rss_kib_sampled']/1024**2,
                  valid_complete_runtime_plan=False)
    boot = re.search(r'Number of Bootstrapping: (\d+)', log)
    result['compiled_boot_count'] = int(boot[1]) if boot else None
    result['unfinished_passes'] = {name: value[-1] for name, value in starts.items() if value}
    attempts = re.findall(r'dacapo-attempt trials=(\d+) reachable_cuts=(\d+) last_reachable=(\d+)', log)
    result['completed_search_attempts'] = [dict(trials=int(a), reachable_cuts=int(b), last_reachable=int(c)) for a,b,c in attempts]
    sets = re.findall(r'Regional DaCapo candidate set (\d+): (\d+) cuts', log)
    result['last_candidate_set'] = dict(size=int(sets[-1][0]), cuts=int(sets[-1][1])) if sets else None
    rows.append(result)
summary['qwen_results'] = rows
summary['resolution_status'] = ('Both Qwen precompiles succeeded; full export remains unvalidated.' if all(r['returncode'] == 0 for r in rows)
                                else 'Greedy Qwen precompile repaired; optimized Qwen planning remains unresolved within the 900-second bound.')
summary['limitations'][-1] = 'The latest comparison is precompile-only. The prior full greedy export timed out at 900 seconds; no valid complete Qwen RuntimePlan was produced.'
summary['validation']['qwen_status'] = summary['resolution_status']
(root / 'repair-summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps([dict(strategy=r['strategy'], seconds=r['seconds'], returncode=r['returncode'], boots=r['compiled_boot_count'], last_candidate_set=r['last_candidate_set']) for r in rows], indent=2))
