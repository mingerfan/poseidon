"""Bounded U1 CLI/split acceptance with the already approved local assets."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import resource
import subprocess
import sys

from ..paths import PLATFORM_DIR, RESULTS_DIR


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def prepare(args):
    import numpy as np
    if args.worker == 'prepare':
        from ..data.cifar import read_rows, normalize
        pixels, labels = read_rows('train', range(223))
        images = normalize(pixels)
        np.savez(args.out/'cifar-labelled.npz', input=images, labels=labels)
        np.savez(args.out/'cifar-unlabelled.npz', input=images)
        np.savez(args.out/'mlp.npz', input=images.reshape(len(images), -1), labels=labels)
        from ..data.samples import read, split_indices
        a = read(args.out/'cifar-labelled.npz').split()
        b = read(args.out/'cifar-labelled.npz').split()
        assert a == b
        assert [len(a[key]) for key in ('calibration','heldout','confirm','report')] == [23,20,20,160]
        try:
            split_indices(222)
        except ValueError as exc:
            message = str(exc)
        else:
            raise AssertionError('222-sample input was accepted')
        write(args.out/'SPLIT.json', dict(reproducible=True, source='CIFAR train[0:223]',
              counts=dict(zip(('calibration','heldout','confirm','report'),(23,20,20,160))),
              boundary_222=message, report_samples_read=0))
    elif args.worker == 'prepare-onnx':
        from PIL import Image
        files = sorted((args.inputs/'images').rglob('*.png'))
        mnist, squeeze = [], []
        for file in files:
            with Image.open(file) as image:
                mnist.append(np.asarray(image.convert('L').resize((28,28),Image.Resampling.BILINEAR),dtype=np.float32)[None]/np.float32(255))
                squeeze.append(np.asarray(image.convert('RGB').resize((224,224),Image.Resampling.BILINEAR),dtype=np.float32).transpose(2,0,1)/np.float32(255))
        np.savez(args.out/'mnist.npz', Input3=np.stack(mnist))
        np.savez(args.out/'squeezenet.npz', data=np.stack(squeeze))
    return 0


def limits():
    # The inherited soft limit is 4 GiB; the existing trusted compiler child
    # alone can raise it to its separately approved 6 GiB setting.
    resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,6*1024**3))
    resource.setrlimit(resource.RLIMIT_FSIZE,(8*1024**3,8*1024**3))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path)
    parser.add_argument('--inputs',type=Path,required=True)
    parser.add_argument('--worker',choices=('prepare','prepare-onnx'))
    parser.add_argument('--only')
    parser.add_argument('--prepared-root',type=Path)
    parser.add_argument('--intake-python',default=str(PLATFORM_DIR/'venvs/intake-2.0.1-cpu/bin/python'))
    args=parser.parse_args()
    if args.worker:
        if args.worker=='prepare-onnx' and Path(sys.executable).absolute()!=Path(args.intake_python).absolute():
            os.execv(args.intake_python,[args.intake_python,'-B','-m','pt_agent.experiments.u1_cli_acceptance',*sys.argv[1:]])
        return prepare(args)
    args.out=args.out or RESULTS_DIR/('u1-cli-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f'))
    args.out.mkdir(parents=True,exist_ok=False)
    result=dict(complete=False,clis=[],llm_calls=0,new_downloads=0)
    base=[sys.executable,'-B','-m','pt_agent.tools.nixrun','--seconds','3600','--']

    def run(command,log):
        with (args.out/log).open('w') as stream:
            return subprocess.run(command,cwd=Path(__file__).resolve().parents[2],stdout=stream,
                stderr=subprocess.STDOUT,preexec_fn=limits,timeout=3720).returncode

    def freeze(label):
        if run([*base,'-m','pip','freeze'],label+'.log')!=0:
            raise AssertionError('Main environment freeze check failed')
        return sorted(line for line in (args.out/(label+'.log')).read_text().splitlines()
                      if '==' in line or ' @ ' in line)

    before=freeze('main-before')
    try:
        for stage in (() if args.prepared_root is not None else ('prepare','prepare-onnx')):
            code=run([*base,'-m','pt_agent.experiments.u1_cli_acceptance','--worker',stage,
                '--out',str(args.out),'--inputs',str(args.inputs)],stage+'.log')
            if code:
                raise AssertionError(stage+' failed')
        assets=PLATFORM_DIR/'cache/pt-agent-models'
        prepared=args.prepared_root or args.out
        cases=[('mlp','pt_agent.models.builders:mlp',prepared/'mlp.npz',
                'probe,import,info,calibrate,plan,check,mock,count,export,compile')]
        cases.extend((name,'pt_agent.models.builders:'+name,prepared/'cifar-labelled.npz','probe,import,info')
                     for name in ('resnet20','resnet20_silu','cnn_plain'))
        for group in ('resnet18','bert_tiny','qwen_alternative'):
            configs=list((assets/group).glob('*/config.json'))
            if len(configs)!=1:
                raise AssertionError('Expected one approved local config for '+group)
            cases.append((group,str(configs[0].parent),args.inputs/('images' if group=='resnet18' else 'prompts.txt'),'probe,import,info'))
        for group,filename in (('mnist','mnist-12.onnx'),('squeezenet','squeezenet1.1-7.onnx')):
            paths=list((assets/group).glob('*/'+filename))
            if len(paths)!=1:
                raise AssertionError('Expected one approved local asset for '+group)
            cases.append((group,str(paths[0]),prepared/(group+'.npz'),'probe,import,info'))
        for name,model,inputs,stages in cases:
            if args.only and name not in args.only.split(','):
                continue
            directory=args.out/name
            command=[sys.executable,'-B','-m','pt_agent.cli','--model',model,'--inputs',str(inputs),
                     '--out',str(directory),'--stages',stages]
            code=run(command,name+'.log')
            report=json.loads((directory/'REPORT.json').read_text())
            row=dict(model=name,exit=code,report=report,artifacts_dir=str(directory))
            result['clis'].append(row)
            write(args.out/'REPORT.json',result)
            if code or not report['complete']:
                raise AssertionError(name+' CLI failed')
        result['complete']=True
    except Exception as exc:
        result['error']=f'{type(exc).__name__}: {exc}'
    after=freeze('main-after')
    result.update(main_freeze_equal=before==after,main_packages=len(before),report_samples_read=0)
    if before!=after:
        result.update(complete=False,error='Main pip freeze changed')
    write(args.out/'REPORT.json',result)
    from ..registry import append,git_rev,new_run_id
    os.environ.setdefault('PT_RESULTS',str(RESULTS_DIR))
    append(dict(run_id=new_run_id('U1','cli','acceptance'),round='U1',model='cli',mode='acceptance',
        git_rev=git_rev(),metrics=result,artifacts_dir=str(args.out),ts=datetime.now(timezone.utc).isoformat()))
    print('REPORT='+str(args.out/'REPORT.json'),flush=True)
    return 0 if result['complete'] else 1


if __name__=='__main__':
    raise SystemExit(main())
