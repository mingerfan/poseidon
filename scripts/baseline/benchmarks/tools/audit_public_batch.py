"""Replay all successful public manual records in a frozen batch; no FHE rerun."""
import argparse
from collections import Counter
import importlib.util
import json
from pathlib import Path
from benchmark_runner import ROOT,DEFAULT
from benchmark_graph import require
from audit_semantic_benchmark import audit

# Reuse the independently verified reference/artifact/trace audit without changing
# the historical checker file or its published source identity.
CHECKER=ROOT/'scripts/baseline/cases/public-control-packed-goldens-v1/audit.py'
spec=importlib.util.spec_from_file_location('public_replay_checks',CHECKER)
checks=importlib.util.module_from_spec(spec);spec.loader.exec_module(checks)


def run(batch,plaintext):
    state=audit(DEFAULT,plaintext,[batch])
    records=[dict(task_id=r['id'],**checks.verify_candidate(Path(r['evidence'])))
             for r in state['batches'][0]['summary']['reports']]
    values=sum(r['comparison']['compared_values'] for r in records)
    require(records and values>0,'No passed public records to audit')
    return dict(schema=1,records=records,passed=len(records),
                batch=str(batch),batch_plan_sha256=checks.sha(batch/'plan.json'),
                counts=state['directed_counts'],plaintext=state['counts']['plaintext'],
                model_set_sha256=state['model_set_sha256'],
                compared_values=values,
                max_absolute_error=max(r['comparison']['max_absolute_error'] for r in records),
                weighted_mae=sum(r['comparison']['compared_values']*r['comparison']['mae'] for r in records)/values,
                periods=dict(Counter(r['period'] for r in records)),
                inputs=dict(Counter(r['inputs'] for r in records)),
                outputs=dict(Counter(r['outputs'] for r in records)),
                numeric=sum(not r['coverage']['structural_only'] for r in records),
                structural=sum(r['coverage']['structural_only'] for r in records),
                checker_sources={str(p.relative_to(ROOT)):checks.sha(p) for p in (CHECKER,Path(__file__))},
                agent_calls=0,all_dsl_semantics_covered=False,formal_equivalence_proven=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--batch',type=Path,required=True)
    p.add_argument('--plaintext',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();require(not a.output.exists(),'Preserve existing audit')
    result=run(a.batch,a.plaintext)
    a.output.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2))
