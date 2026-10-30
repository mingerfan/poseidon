"""U4 §7.8 finite cost-DAG acceptance, reusing saved frontend/RuntimePlan evidence."""

import argparse
from collections import Counter, defaultdict
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

from ..paths import REPO,RESULTS_DIR
from .u4_point_diagnosis import write


def inventory(args):
    from ..tests.test_p2_export import cases
    from .s1_managed import inventory as s1_inventory
    rows,lookup = [],{}
    s1 = json.loads((RESULTS_DIR/'s1-final-local-checks-20261026T071428925546/SUMMARY.json').read_text())
    s1_cases = {row['program']:row for row in s1['cases']}
    u3 = json.loads((RESULTS_DIR/'u3a-complete-20261024T161354727454/REPORT.json').read_text())

    def add(row,group,native=None):
        key = row.get('source') or 'unit:'+row['unit']
        if key not in lookup:
            item = dict(row,id='case-'+str(len(rows)).zfill(2),groups=[],native=[])
            rows.append(item);lookup[key]=item
        item = lookup[key]
        item['groups'].append(dict(group=group,label=row['label']))
        if native is not None and native not in item['native']:
            item['native'].append(native)

    p2 = cases()
    for name,model,path in p2[:4]:
        evidence = path.parent/'REPORT.json'
        saved = json.loads(evidence.read_text())
        add(dict(label=name,model=model,source=str(path)),'frozen',
            dict(boots=saved['boots_plan'],source=str(evidence)))
    for item in u3['models']:
        directory = Path(item['acceptance']).parent
        add(dict(label=item['model'],info=str(directory),source=str(directory/'program.ptdsl')),'u3',
            dict(boots=item['boots_compiled'],source=item['runtime_plan']) if item['boots_compiled'] is not None else None)
    r6b = json.loads((RESULTS_DIR/'r6b-refresh-20261021T025529/REPORT.json').read_text())
    agent = next(row for row in r6b['cases'] if row['mode']=='agent' and row['deg']==127)
    for name,model,path in p2:
        saved = s1_cases.get(name)
        native = (dict(boots=saved['boots']['compiler'],source=saved['report'])
                  if saved and saved.get('boots') else
                  dict(boots=agent['plan']['boots'],source=str(RESULTS_DIR/'r6b-refresh-20261021T025529/REPORT.json'))
                  if name=='rule-agent' else None)
        add(dict(label=name,model=model,source=str(path)),'p2',native)
    for row in s1_inventory():
        saved = s1_cases[row['label']]
        native = dict(boots=saved['boots']['compiler'],source=saved['report']) if saved.get('boots') else None
        add(row,'s1',native)
    sizes = Counter(group['group'] for row in rows for group in row['groups'])
    if dict(sizes)!=dict(frozen=4,u3=9,p2=7,s1=20):
        raise ValueError('Retained cohort sizes differ: '+str(dict(sizes)))
    order = sorted(rows,key=lambda row:(0 if any(g['group']=='frozen' for g in row['groups']) else
                                       1 if 'unit' in row else 2 if any(g['group']=='p2' for g in row['groups']) else
                                       3 if not row.get('info') else 4,int(row['id'].split('-')[1])))
    result = dict(round='U4',section='7.8.2.3',complete=True,cases=rows,
                  cohorts=dict(sizes),memberships=sum(sizes.values()),unique=len(rows),
                  execution_order=[row['id'] for row in order],llm_calls=0,report_images=0)
    write(args.out/'MANIFEST.json',result)
    return result


def unit_cost(name,normalized):
    from ..backends import DagBackend
    from ..backends.cost_dag import CostDag
    from ..cost import load_latency
    from ..placement import depth_dp
    from .s1_managed import replay
    from ..tests.test_s1_manage import record_case
    recorded,_ = record_case(name)
    backend = CostDag(recorded.slots) if normalized else DagBackend(recorded.slots)
    backend.op_context = 12
    replay(recorded,backend)
    if normalized:
        backend.finalize()
    placement = depth_dp(backend,last_boot_level=4,boot_in_level=1,boot_out_level=5,
                         latency_table=load_latency('qwen48'))
    return dict(boots_pred=placement.boots,latency_pred_us=placement.latency_us,
                boot_nodes=placement.boot_nodes,statement_costs=None,hotspots=None)


def regression_case(args):
    import gc
    from ..cost import estimate
    from ..dsl.parser import parse
    manifest = json.loads(args.manifest.read_text())
    row = next(row for row in manifest['cases'] if row['id']==args.case)
    result = dict(round='U4',section='7.8.2.3',case=row,complete=False,llm_calls=0,report_images=0)
    if 'unit' in row:
        before,after = unit_cost(row['unit'],False),unit_cost(row['unit'],True)
    else:
        if row.get('info'):
            from ..models.info import ModelInfo
            graph = ModelInfo.load(Path(row['info'])).graph
        else:
            from ..models.zoo import get
            graph = get(row['model'],RESULTS_DIR/'models-v2').import_graph()
        program = parse(Path(row['source']).read_text())
        before = estimate(program,graph.weights,frontend_normalize=False)
        gc.collect()
        after = estimate(program,graph.weights,frontend_normalize=True)
    keys = sorted(set(before)|set(after))
    result.update(before=before,after=after,changed=[key for key in keys if before.get(key)!=after.get(key)],
                  requested_fields={key:dict(before=before.get(key),after=after.get(key),equal=before.get(key)==after.get(key))
                    for key in ('boots_pred','latency_pred_us','boot_nodes','statement_costs')})
    checks = []
    for native in row['native']:
        checks.append(dict(**native,before_equal=before['boots_pred']==native['boots'],
                           after_equal=after['boots_pred']==native['boots']))
    result['native_checks'] = checks
    result['new_native_mismatch'] = any(check['before_equal'] and not check['after_equal'] for check in checks)
    result['complete'] = not result['new_native_mismatch']
    if not result['complete']:
        result['stop_reason'] = '§7.8.2.3: previously matching native bootstrap count changed to a mismatch'
    return result


def s1_readonly(args):
    from .. import toolchain
    from ..manage import RecordingBackend,emit_raw
    examples = []
    for case in ('plain','scalar','cipher','plain-left'):
        b = RecordingBackend(8)
        x,y = b.input(0),b.input(1)
        value = x-b.const([2.]) if case=='plain' else x-2. if case=='scalar' else x-y if case=='cipher' else 2.-y
        b.output([value])
        path = emit_raw(b,args.out/(case+'.mlir'),toolchain.management_profile(),name=case.replace('-','_'))
        examples.append(dict(case=case,kinds=[n.kind for n in b.nodes],
                             plaintext_values=[n.data.tolist() for n in b.nodes if n.kind=='constant'],
                             contains_earth_sub='"earth.sub"' in path.read_text()))
    native = (toolchain.host_paths()['frontend'].parents[3]/'tools/frontend.cpp')
    # This check records the unchanged frontend builder and current S1 emitter.
    # It does not run a native pass or change either implementation.
    return dict(round='U4',section='7.8.2.4',complete=not any(row['contains_earth_sub'] for row in examples),
                examples=examples,source='manage/greedy.py:RecordingBackend._apply',native_source=str(native),
                conclusion='S1 represents subtraction by add plus negated plaintext or negate(cipher), matching the native builder',
                s1_modified=False,known_remaining='S1 last case is pending resource/server acceptance; no subtraction representation discrepancy found',
                llm_calls=0,report_images=0,compile_calls=0)


def bert(args):
    from unittest.mock import patch
    from .. import cost
    from ..dsl.parser import parse
    from ..models.info import ModelInfo
    result = dict(round='U4',section='7.8.2.2',model='bert',complete=False,
                  program=str(args.program),diagnosis=str(args.diagnosis),llm_calls=0,report_images=0,ckks_calls=0)
    program = parse(args.program.read_text())
    info = ModelInfo.load(args.prepared/'model')
    original = cost.lower
    recorded = {}

    def capture(*values,**kwargs):
        recorded['backend'] = values[2]
        return original(*values,**kwargs)

    with patch.object(cost,'lower',capture):
        result['cost'] = cost.estimate(program,info.graph.weights,frontend_normalize=True)
    backend = recorded['backend']
    result['normalization'] = backend.normalization
    by_owner = defaultdict(Counter)
    for kind,owner in zip(backend.kind,backend.prog_op):
        if kind!='input':
            by_owner[owner][kind] += 1
    before = json.loads(args.diagnosis.read_text())
    reference = before['graphs']['frontend']
    result['expected_counts'] = {kind:reference['counts'].get(kind,0)
                                 for kind in ('add','sub','neg','mul_cc','mul_cp','rot')}
    result['actual_counts'] = {kind:result['cost']['counts'].get(kind,0) for kind in result['expected_counts']}
    result['counts_equal'] = result['expected_counts']==result['actual_counts']
    differences = []
    rows = []
    for op,front in zip(program.ops,reference['by_line']):
        counts = dict(by_owner[op.id])
        classes = set(counts)|set(front['counts'])
        equal = all(counts.get(kind,0)==front['counts'].get(kind,0) for kind in classes)
        row = dict(id=op.id,line=op.lineno,op=op.kind,counts=counts,frontend=front['counts'],equal=equal)
        rows.append(row)
        if not equal:
            differences.append(row)
    result['by_line'] = rows
    result['first_line_difference'] = differences[0] if differences else None
    result['boots_equal'] = result['cost']['boots_pred']==15399
    result['boundaries_equal'] = result['cost']['barriers']==110
    result['depth_equal'] = result['cost']['max_depth']==323
    result['complete'] = all(result[k] for k in ('counts_equal','boots_equal','boundaries_equal','depth_equal')) and not differences
    if not result['complete']:
        result['stop_reason'] = '§7.8.2.2: normalized BERT cost graph differs from saved frontend/RuntimePlan'
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inside',action='store_true')
    parser.add_argument('--part',choices=('bert','inventory','regression-case','s1-readonly'),required=True)
    parser.add_argument('--case')
    for name in ('out','prepared','program','diagnosis','manifest'):
        parser.add_argument('--'+name,type=Path)
    args = parser.parse_args()
    if args.inside:
        resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3))
        resource.setrlimit(resource.RLIMIT_FSIZE,(8*1024**3,8*1024**3))
        resource.setrlimit(resource.RLIMIT_CORE,(0,0))
        result = dict(round='U4',section='7.8',complete=False,llm_calls=0)
        try:
            result = {'bert':bert,'inventory':inventory,'regression-case':regression_case,'s1-readonly':s1_readonly}[args.part](args)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            result.update(error=f'{type(exc).__name__}: {exc}')
        write(args.out/'REPORT.json',result)
        if not result['complete']:
            write(args.out/'STOP.json',result)
        return 0 if result['complete'] else 1
    stamp = datetime.now(timezone.utc)
    run_id = 'u4-cost78-'+args.part+'-'+stamp.strftime('%Y%m%dT%H%M%S%f')
    args.out = args.out or RESULTS_DIR/run_id
    args.out.mkdir(parents=True,exist_ok=False)
    command = [sys.executable,'-B','-m','pt_agent.tools.nixrun','--seconds','3600','--','-m',
               'pt_agent.experiments.u4_cost_normalization','--inside','--part',args.part]
    for key in ('out','prepared','program','diagnosis','manifest','case'):
        value = getattr(args,key)
        if value is not None:
            command += ['--'+key,str(value)]
    write(args.out/'argv.json',command)
    print('RUN='+str(args.out),flush=True)
    started = time.monotonic()
    with (args.out/'run.log').open('w') as log:
        status = subprocess.run(command,cwd=REPO/'DSL/DSL_agent/scripts',stdout=log,stderr=subprocess.STDOUT)
    path = args.out/'REPORT.json'
    result = json.loads(path.read_text()) if path.exists() else dict(complete=False,error='see run.log')
    rev = subprocess.check_output(['git','rev-parse','--short','HEAD'],cwd=REPO,text=True).strip()
    result.update(run_id=run_id,git_rev=rev,exit=status.returncode,seconds=time.monotonic()-started)
    write(path,result)
    from ..registry import append
    os.environ.setdefault('PT_RESULTS',str(RESULTS_DIR))
    append(dict(run_id=run_id,round='U4',model=args.case or ('bert' if args.part=='bert' else 'cohorts'),mode='cost78-'+args.part,git_rev=rev,
                metrics=dict(complete=result['complete'],llm_calls=0),artifacts_dir=str(args.out),ts=stamp.isoformat()))
    print(json.dumps(dict(report=str(path),complete=result['complete'],exit=status.returncode),indent=2),flush=True)
    return status.returncode


if __name__=='__main__':
    raise SystemExit(main())
