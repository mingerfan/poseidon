"""Frozen directed helper tasks. Default plan is read-only and never calls APIs."""
import argparse,json,os,shlex,sys
from pathlib import Path
from benchmark_graph import require,digest
from benchmark_runner import ROOT,DEFAULT,load,strict_file


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['helper-directed'])
    parser.add_argument('--suite',type=Path,default=DEFAULT)
    parser.add_argument('--task-id',action='append')
    parser.add_argument('--limit',type=int,default=48)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--inside',action='store_true',help=argparse.SUPPRESS)
    args=parser.parse_args()
    require(1<=args.limit<=48,'Helper shard limit is 48')
    load(args.suite)
    tasks=strict_file(args.suite/'coverage.json',4*1024**2)['helper_directed_tasks']
    from upstream_helper_directed_cases import tasks as definitions
    require(tasks==definitions(),'Frozen helper definitions changed')
    names=args.task_id or [t['id'] for t in tasks]
    require(len(names)==len(set(names)) and set(names)<={t['id'] for t in tasks},'Unknown or repeated helper task')
    selected=[t for t in tasks if t['id'] in names][:args.limit]
    if not args.execute:
        print(json.dumps(dict(mode='plan',tasks=selected,agent_calls=0,execution='manual_candidate_not_agent',
             native_concurrency=1,max_wall_seconds=1800,max_result_mib=512,
             resumable=True,old_evidence_preserved=True),indent=2));return 0
    from hecate_python_env import WORK,VENV,enter_nix
    require(args.output is not None and args.output.resolve().is_relative_to(WORK/'results') and (not args.output.exists() or args.resume),
            'New output directory in selected platform results required')
    if not args.inside:
        cmd=[str(VENV/'bin/python'),'-B',str(ROOT/'scripts/benchmark.py'),*sys.argv[1:],'--inside']
        return enter_nix('LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" '+shlex.join(cmd),seconds=1800)
    require(os.environ.get('IN_NIX_SHELL') and Path(sys.prefix)==VENV,'Locked Nix Python required')
    sys.path.insert(0,str(ROOT/'scripts/baseline/benchmarks/tools'))
    from run_upstream_helper_directed_v3 import main as run
    options=['--output',str(args.output),'--suite',str(args.suite),'--limit',str(args.limit)]
    if args.resume:options+=['--resume']
    for t in selected:options+=['--task-id',t['id']]
    return run(options)
