"""U4b finite primitive and encrypted-selection checks; no rule feedback."""

import argparse
from dataclasses import replace
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


def unit(args):
    import unittest
    suite=unittest.defaultTestLoader.loadTestsFromName('pt_agent.tests.test_u4_decoder')
    result=unittest.TextTestRunner(verbosity=2,failfast=True).run(suite)
    return dict(complete=result.wasSuccessful(),tests=result.testsRun,failures=len(result.failures),
                errors=len(result.errors),skipped=len(result.skipped))


def inventory(args):
    from ..models.info import ModelInfo
    from ..intake.coverage import report,check
    rows=[]
    for name,path in (('bert',args.bert_prepared),('qwen',args.prepared)):
        info=ModelInfo.load(path/'model')
        rows.append(dict(model=name,prepared=str(path),coverage=report(info.graph),check=check(info.graph)))
    return dict(complete=all(r['check']['ok'] and r['coverage']['lowering_ratio']==1 for r in rows),models=rows)


def selection(args):
    import numpy as np
    from ..models.info import ModelInfo
    from ..data.samples import StoredSplits
    from ..dsl.parser import parse
    from ..dsl.printer import print_program
    from ..dsl.typecheck import check
    from ..transformer_validation import execute
    from ..analyzer import reference_forward
    info=ModelInfo.load(args.prepared/'model');program=parse(args.program.read_text())
    arrays,_=StoredSplits(args.prepared/'model/samples',info).get('calibration',2)
    head=program.producer(program.outputs[0]);select=program.producer(head.inputs[0])
    if head.kind!='linear' or select.kind!='select':
        raise ValueError('The selection check requires select followed by the final linear')
    variant=replace(program,ops=tuple(replace(op,inputs=(select.inputs[0],)) if op.id==head.id else op
                                    for op in program.ops if op.id!=select.id))
    checked=check(variant,{k:v.shape for k,v in info.graph.weights.items()})
    if not checked.ok:raise RuntimeError('Diagnostic all-position variant does not typecheck')
    (args.out/'diagnostic-all-positions.ptdsl').write_text(print_program(variant))
    lengths=arrays['attention_mask'].sum(-1).astype(int)
    client=np.zeros_like(arrays['sel']);client[np.arange(len(lengths)),lengths-1]=1
    np.testing.assert_array_equal(client,arrays['sel'])
    selected=execute(program,info.graph.weights,arrays,engine='mock',exact=True)
    full=execute(variant,info.graph.weights,arrays,engine='mock',exact=True)
    expected=full[np.arange(len(lengths)),lengths-1][:,None,:]
    np.save(args.out/'selected-exact.npy',selected)
    np.save(args.out/'client-selected-exact.npy',expected)
    np.testing.assert_array_equal(selected,expected)
    target=reference_forward(info.graph,arrays)[info.graph.outputs[0]]
    delta=np.abs(selected-target)
    if np.any(delta>1e-12+1e-12*np.abs(target)):
        raise RuntimeError('Exact selection route differs from the independent reference')
    return dict(complete=True,program=str(args.program),split='calibration',samples=len(lengths),
                original_program_unchanged=True,diagnostic_variant_only=True,
                lengths=lengths.tolist(),client_selection_valid=True,
                all_position_shape=list(full.shape),selected_shape=list(selected.shape),
                selected_vs_client_array_equal=True,selected_vs_client_max_abs=float(np.max(np.abs(selected-expected))),
                independent_reference_max_abs=float(delta.max()))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--part',choices=('unit','selection','inventory'),required=True)
    parser.add_argument('--inside',action='store_true')
    for key in ('out','prepared','program','bert-prepared'):parser.add_argument('--'+key,type=Path)
    args=parser.parse_args()
    if args.inside:
        resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,6*1024**3))
        resource.setrlimit(resource.RLIMIT_FSIZE,(8*1024**3,8*1024**3));resource.setrlimit(resource.RLIMIT_CORE,(0,0))
        try:report={'unit':unit,'selection':selection,'inventory':inventory}[args.part](args)
        except Exception as exc:
            import traceback
            traceback.print_exc();report=dict(complete=False,error=f'{type(exc).__name__}: {exc}')
            write(args.out/'STOP.json',report)
        write(args.out/'REPORT.json',report)
        return 0 if report['complete'] else 1
    stamp=datetime.now(timezone.utc);run_id='u4-decoder-'+args.part+'-'+stamp.strftime('%Y%m%dT%H%M%S%f')
    args.out=args.out or RESULTS_DIR/run_id;args.out.mkdir(parents=True,exist_ok=False)
    command=[sys.executable,'-B','-m','pt_agent.tools.nixrun','--seconds','3600','--',
             '-m',__spec__.name,'--inside','--part',args.part,'--out',str(args.out)]
    for key in ('prepared','program','bert_prepared'):
        if getattr(args,key) is not None:command+=['--'+key.replace('_','-'),str(getattr(args,key))]
    print('RUN='+str(args.out),flush=True);started=time.monotonic()
    with (args.out/'run.log').open('w') as log:status=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=3630)
    path=args.out/'REPORT.json';report=json.loads(path.read_text()) if path.exists() else dict(complete=False,error='see run.log')
    rev=subprocess.check_output(['git','rev-parse','--short','HEAD'],cwd=REPO,text=True).strip()
    report.update(round='U4',milestone='U4b',run_id=run_id,part=args.part,git_rev=rev,exit=status.returncode,
                  seconds=time.monotonic()-started,llm_calls=0,report_images=0,ckks_calls=0)
    write(path,report)
    from ..registry import append
    os.environ.setdefault('PT_RESULTS',str(RESULTS_DIR))
    append(dict(run_id=run_id,round='U4',model='qwen',mode='decoder-'+args.part,git_rev=rev,
        metrics=dict(complete=report['complete'],tests=report.get('tests'),llm_calls=0),artifacts_dir=str(args.out),ts=stamp.isoformat()))
    print(json.dumps(dict(report=str(path),complete=report['complete'],exit=status.returncode)),flush=True)
    return status.returncode


if __name__=='__main__':raise SystemExit(main())
