"""Reproduce metadata, GPU prefixes, native/DSL full, or same-ciphertext control."""
import argparse,datetime,json,os,re,subprocess
from pathlib import Path
from prepare_assets import ROOT,prepare
def execute(args):
    assets=prepare(args.work.resolve(),True)
    artifacts=ROOT/'artifacts/resnet18.hevm'
    output=args.work.resolve()/('run-'+datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    output.mkdir(exist_ok=False)
    env=os.environ.copy()
    for key in ('POSEIDON_RESCALE_META_BITS','POSEIDON_RESCALE_MODEL_BITS','POSEIDON_DSL_SCALE_SHIFT','POSEIDON_GPU_DNUM'):env.pop(key,None)
    env.update(POSEIDON_GPU_LINEAR_TRANSFORM_MODE='no_hoist',POSEIDON_NTT_ALGO='fourstep',
        POSEIDON_TRIDENT_RESNET18_ROOT=str(args.work.resolve()/'data/resnet18'))
    gpu_uuid=None
    if args.mode!='metadata':
        assert args.gpu is not None,'choose an idle GPU with --gpu'
        gpu_uuid,free=[x.strip() for x in subprocess.check_output(['nvidia-smi','-i',str(args.gpu),
            '--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],text=True).strip().split(',')]
        occupied=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True).splitlines()
        assert gpu_uuid not in [x.strip() for x in occupied] and int(free)>=28672,'GPU occupied or below28GiB free'
        env['CUDA_VISIBLE_DEVICES']=str(args.gpu)
    binary=args.bin.resolve()
    base=[str(binary/'resnet18_dsl'),'18',str(artifacts),str(assets/'resnet18.native_constants.jsonl'),
        str(assets/'resnet18.inputs.jsonl'),str(assets/'resnet18.payload.zlib')]
    def run(command,name):
        log=output/(name+'.log')
        with log.open('x') as stream:
            stream.write(json.dumps(dict(command=command,gpu_uuid=gpu_uuid,N=65536,Q=50,P=25,pool_gib=26,no_hoist=True))+'\n');stream.flush()
            result=subprocess.run(command,env=env,stdout=stream,stderr=subprocess.STDOUT)
            stream.write('\nPROCESS_EXIT='+str(result.returncode)+'\n')
        assert result.returncode==0,log
        return log.read_text()
    metadata=run(base+[str(output/'unused.json'),'0','--metadata-only'],'metadata')
    assert 'IMAGENET_DSL_PREFLIGHT PASS' in metadata and 'boots=60' in metadata
    if args.mode=='short':
        first=int(re.search(r'first_boot_action=(\d+)',metadata)[1])
        for limit in (700,first+5000):
            text=run(base+[str(output/f'short{limit}.json'),str(limit)],f'short{limit}')
            match=re.search(r'IMAGENET_DSL_TRUNCATED PASS actions=(\d+) boots=(\d+) exact_vector=(\d+) exact_scalar=(\d+) exact_product=(\d+)',text)
            assert match and int(match[1])==limit
            if limit>700:assert all(int(match[i])>=1 for i in range(2,6))
    elif args.mode in ('full','native'):
        report=output/(args.mode+'.json')
        command=base+[str(report),'0'] if args.mode=='full' else [str(binary/'resnet18_native'),str(report)]
        run(command,args.mode);value=json.loads(report.read_text())
        assert value['status']=='COMPLETE' and value['prediction']==value['label']==65
        if args.mode=='full':assert value['bootstrap_calls']==60
    elif args.mode=='same-cipher':
        run([str(binary/'resnet18_same_cipher'),'--probe-only',str(output/'probe')],'probe')
        assert json.loads((output/'probe.paired.json').read_text())['status']=='PASS'
        run([str(binary/'resnet18_same_cipher'),str(artifacts),str(assets/'resnet18.native_constants.jsonl'),
             str(assets/'resnet18.inputs.jsonl'),str(assets/'resnet18.payload.zlib'),str(output/'control')],'control')
        value=json.loads((output/'control.paired.json').read_text())
        assert value['status']=='PASS' and value['native_rns_outputs']==value['dsl_rns_outputs']==1000
        assert value['rns_mismatch_words']==value['metadata_mismatch_classes']==value['max_logit_error']==0
        assert value['native_prediction']==value['dsl_prediction']==65
    print(args.mode.upper()+'_PASS',output)
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('metadata','short','full','native','same-cipher'))
    p.add_argument('--bin',type=Path,default=ROOT/'build/gpu/bin');p.add_argument('--work',type=Path,default=ROOT/'work')
    p.add_argument('--gpu',type=int);execute(p.parse_args())
