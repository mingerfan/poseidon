"""Apply frozen round-071 host DFS schedule, then a 64 MiB Encode-root cache."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT/'source/model'), str(ROOT/'source/tools')]
from reorder_runtime_plan import transform_dfs
from bounded_cpu_encodes import transform_bounded

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--report', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists() or a.report.exists() or a.output.resolve() == a.report.resolve():
        p.error('Use distinct new output and report paths')
    original = json.loads(a.input.read_text('utf-8'))
    scheduled, schedule_report = transform_dfs(original)
    final, cache_report = transform_bounded(scheduled, 64*2**20)
    assert len(final['external_inputs']) == 2 and len(final['final_outputs']) == 29
    raw = json.dumps(final, indent=2).encode('utf-8')
    with a.output.open('xb') as f:
        f.write(raw)
    with a.report.open('x', encoding='utf-8') as f:
        json.dump(dict(schedule=schedule_report, cache=cache_report,
            plan_sha256=hashlib.sha256(raw).hexdigest(), ckks_executed=False,
            note='Keep the original constant bundle; run native PlanVerifier before execution'), f, indent=2)

if __name__ == '__main__':
    main()
