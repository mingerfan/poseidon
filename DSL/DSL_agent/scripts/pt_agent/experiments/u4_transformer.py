"""Finite U4 acceptance; system Python owns results and the append-only registry."""

import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import re
import resource
import subprocess
import sys
import time

from ..paths import REPO,RESULTS_DIR,PLATFORM_DIR


def write(path,record):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(record,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def corpus(path):
    sources=[REPO/'third_party/dacapo/README.md',REPO/'third_party/ckks-runtime/README.md',
             REPO/'third_party/spdlog/README.md',REPO/'scripts/README.md']
    sentences=[];seen=set();used=[]
    for source in sources:
        if not source.is_file():continue
        count=0
        for line in re.split(r'[。！？.?\n]+',source.read_text()):
            words=line.strip().split()
            for start in range(0,len(words),24):
                text=' '.join(words[start:start+24])
                # A Chinese word without spaces is capped at an equivalent
                # 48 characters; long text is split, never synthesized.
                for c in range(0,len(text),48 if re.search(r'[\u4e00-\u9fff]',text) else max(1,len(text))):
                    sample=text[c:c+48] if re.search(r'[\u4e00-\u9fff]',text) else text
                    if sample and sample not in seen:
                        seen.add(sample);sentences.append(sample);count+=1
        used.append(dict(file=str(source.relative_to(REPO)),sentences=count))
        if len(sentences)>=223:break
    if len(sentences)<223:raise ValueError('Existing Markdown corpus has fewer than 223 distinct sentences')
    sentences=sentences[:223]
    path.write_text('\n'.join(json.dumps(dict(text=v),ensure_ascii=False) for v in sentences)+'\n')
    return dict(samples=len(sentences),sources=used,max_words=max(len(v.split()) for v in sentences))


def prepare(args):
    old=RESULTS_DIR/'u1-cli-20261024T104532576122'/('bert_tiny' if args.model=='bert' else 'qwen_alternative')/'model_info.json'
    local=json.loads(old.read_text())['source']['local_path']
    text=args.out/'sentences.jsonl'
    data=corpus(text)
    command=[sys.executable,'-B','-m','pt_agent.cli','--model',local,'--inputs',str(text),
             '--out',str(args.out/'model'),'--stages','probe,import,calibrate,plan,check']
    with (args.out/'cli.log').open('w') as log:
        status=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=3720)
    path=args.out/'model/REPORT.json'
    report=json.loads(path.read_text()) if path.exists() else {}
    return dict(complete=status.returncode==0,model=args.model,corpus=data,cli_argv=command,
                cli_exit=status.returncode,cli_report=report,model_dir=str(args.out/'model'))


def unit(args):
    import unittest
    files=sorted(Path('pt_agent/tests').glob('test_u4_*.py'))
    files.append(Path('pt_agent/tests/test_r1_parser.py'))
    suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName('pt_agent.tests.'+p.stem) for p in files)
    result=unittest.TextTestRunner(verbosity=2,failfast=True).run(suite)
    return dict(complete=result.wasSuccessful(),tests=result.testsRun,failures=len(result.failures),
                errors=len(result.errors),skipped=len(result.skipped))


def evaluate(args):
    import numpy as np
    from ..models.info import ModelInfo
    from ..data.samples import StoredSplits
    from ..dsl.parser import parse
    from ..transformer_validation import execute
    from ..analyzer import reference_forward
    from ..metrics import compare
    info=ModelInfo.load(args.prepared/'model')
    source=args.program or args.prepared/'model/program.ptdsl'
    program=parse(source.read_text())
    arrays,_=StoredSplits(args.prepared/'model/samples',info).get('heldout')
    count=len(next(iter(arrays.values())))
    result=dict(complete=False,model=args.model,sequence=info.sequence,heldout=count,
                prepared=str(args.prepared),program=str(source),rows=[],llm_calls=0,report_images=0)
    exact=reference_forward(info.graph,arrays)[info.graph.outputs[0]]
    np.save(args.out/'exact.npy',exact)
    # Keep per-case evidence immediately; no generated program is edited.
    outputs=[]
    for start in range(0,count,20):
        batch={k:v[start:start+20] for k,v in arrays.items()}
        row=dict(start=start,count=len(next(iter(batch.values()))),complete=False)
        result['rows'].append(row);write(args.out/'REPORT.json',result)
        began=time.monotonic()
        mock=execute(program,info.graph.weights,batch,engine='mock')
        np.save(args.out/f'mock-{start}.npy',mock)
        tensor=execute(program,info.graph.weights,batch,engine='tensor')
        np.save(args.out/f'tensor-{start}.npy',tensor)
        difference=np.abs(tensor-mock)
        row.update(max_diff=float(difference.max()),seconds=time.monotonic()-began,
                   finite=bool(np.isfinite(tensor).all() and np.isfinite(mock).all()),
                   classes_equal=bool(np.array_equal(tensor.argmax(-1),mock.argmax(-1))))
        if not row['finite'] or np.any(difference>1e-9*(1+np.abs(mock))) or not row['classes_equal']:
            raise RuntimeError('U4 held-out Mock/tensor numerical comparison differs')
        row['complete']=True;outputs.append(tensor);write(args.out/'REPORT.json',result)
    actual=np.concatenate(outputs)
    measured=dict(values=exact,task=info.task)
    if info.task=='causal_lm':measured.update(lengths=arrays['attention_mask'].sum(1).astype(int),selected_last=True)
    result.update(complete=True,metrics=compare(measured,actual,None),
                  max_diff=max(r['max_diff'] for r in result['rows']))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--part',choices=('prepare','unit','evaluate'),required=True)
    parser.add_argument('--model',choices=('bert','qwen'),default='bert')
    parser.add_argument('--inside',action='store_true')
    parser.add_argument('--out',type=Path)
    parser.add_argument('--prepared',type=Path)
    parser.add_argument('--program',type=Path)
    args=parser.parse_args()
    if args.inside:
        resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,6*1024**3))
        resource.setrlimit(resource.RLIMIT_FSIZE,(8*1024**3,8*1024**3))
        resource.setrlimit(resource.RLIMIT_CORE,(0,0))
        report=dict(complete=False)
        try:report={'unit':unit,'evaluate':evaluate}[args.part](args)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            if (args.out/'REPORT.json').exists():report.update(json.loads((args.out/'REPORT.json').read_text()))
            report.update(complete=False,error=f'{type(exc).__name__}: {exc}')
        write(args.out/'REPORT.json',report)
        return 0 if report['complete'] else 1
    stamp=datetime.now(timezone.utc)
    run_id='u4-'+args.model+'-'+args.part+'-'+stamp.strftime('%Y%m%dT%H%M%S%f')
    args.out=args.out or RESULTS_DIR/run_id
    args.out.mkdir(parents=True,exist_ok=False)
    print('RUN='+str(args.out),flush=True)
    started=time.monotonic()
    if args.part=='prepare':report=prepare(args)
    else:
        seconds=3600 if args.part=='evaluate' else 1200
        command=[sys.executable,'-B','-m','pt_agent.tools.nixrun','--seconds',str(seconds),'--',
                 '-m','pt_agent.experiments.u4_transformer','--inside','--part',args.part,'--model',args.model,'--out',str(args.out)]
        if args.prepared is not None:command.extend(['--prepared',str(args.prepared)])
        if args.program is not None:command.extend(['--program',str(args.program)])
        with (args.out/'run.log').open('w') as log:
            status=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=seconds+30)
        path=args.out/'REPORT.json'
        report=json.loads(path.read_text()) if path.exists() else dict(complete=False,error='see run.log')
        report['exit']=status.returncode
    rev=subprocess.check_output(['git','rev-parse','--short','HEAD'],cwd=REPO,text=True).strip()
    report.update(round='U4',run_id=run_id,part=args.part,git_rev=rev,seconds=time.monotonic()-started,
                  llm_calls=0,report_images=0)
    write(args.out/'REPORT.json',report)
    from ..registry import append
    os.environ.setdefault('PT_RESULTS',str(RESULTS_DIR))
    append(dict(run_id=run_id,round='U4',model=args.model,mode=args.part,git_rev=rev,
                metrics=dict(complete=report['complete'],tests=report.get('tests'),llm_calls=0),
                artifacts_dir=str(args.out),ts=stamp.isoformat()))
    print(json.dumps(dict(report=str(args.out/'REPORT.json'),complete=report['complete']),indent=2),flush=True)
    return 0 if report['complete'] else 1


if __name__=='__main__':raise SystemExit(main())
