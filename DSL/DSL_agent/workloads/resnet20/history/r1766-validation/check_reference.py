"""Offline original-network checks using the already approved small assets."""
import ast,hashlib,json,os,shutil,sys,time,traceback
from pathlib import Path
A=Path('/home/lhohy/Code/Poseidon-generalization/DSL/DSL_agent/scripts/baseline')
B=Path('/home/lhohy/Code/Poseidon-resnet20/DSL/DSL_agent/workloads/resnet20')
WORK=Path('/home/lhohy/poseidon-work/platforms/aarch64-linux')
VENV=WORK/'venvs/hecate-2.0.1-cpu'
RESULT=WORK/'results/agent-resnet20-reference-r1766'
DOWNLOAD=WORK/'results/agent-resnet20-assets-recovery-r1759/downloads'
sys.path[:0]=[str(B),str(A)]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def create(p,v):
    with Path(p).open('x')as f:json.dump(v,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
def main():
    if Path(sys.prefix)!=VENV:
        env=dict(os.environ,LD_LIBRARY_PATH=os.environ['HECATE_PYTHON_LIBRARY_PATH'],PYTHONNOUSERSITE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
        os.execve(str(VENV/'bin/python'),[str(VENV/'bin/python'),'-B',str(Path(__file__)),*sys.argv[1:]],env)
    os.umask(0o077)
    from process_resource_limits import limits
    limits()
    from shared_execution import Coordinator,KINDS,digest,process
    frozen=json.loads(Path(sys.argv[1]).read_text())
    def unchanged():
        for p,h in frozen['source'].items():assert sha(p)==h,p
    unchanged();assert 'DEEPSEEK_API_KEY'not in os.environ
    RESULT.mkdir(mode=0o700)
    coordinator=Coordinator('/home/lhohy/poseidon-work/coordination/v1',[WORK/'results'])
    plan=dict(branch='lhy-resnet20',source_binding=digest(frozen),reserve_bytes=64*1024**2,wall_seconds=300,stages={kind:0 for kind in KINDS})
    batch,usage=coordinator.register('resnet20-reference-r1766',plan,RESULT);status='failed'
    try:
        create(RESULT/'plan.json',dict(plan=plan,plan_binding=batch.binding,input=frozen,initial_usage=usage,api_calls=0,encrypted_calls=0,downloads=0,execute_downloaded_code=False,checkpoint_loader='Original torch.load(weights_only=True), no unrestricted fallback',probes='Four fixed valid uint8 RGB inputs; no CIFAR test accuracy claim'))
        create(RESULT/'owner-ready.json',dict(owner=process(),nonce=frozen['nonce']))
        deadline=time.monotonic()+60
        while not(RESULT/'owner-ack.json').exists():
            assert time.monotonic()<deadline
            time.sleep(.05)
        assert json.loads((RESULT/'owner-ack.json').read_text())==dict(nonce=frozen['nonce'],owner_verified=True)
        from public_assets import ChunkReader
        public=RESULT/'public';public.mkdir(mode=0o700);assets={}
        for index in range(4):
            record=json.loads((DOWNLOAD/('completed-%02d.json'%index)).read_text())
            name=record['asset']['id'];manifest=record['stream']
            raw=ChunkReader(DOWNLOAD/name,manifest).read(manifest['bytes'])
            assert len(raw)==manifest['bytes']and hashlib.sha256(raw).hexdigest()==manifest['sha256']
            target=public/(name+'.data')
            with target.open('xb')as f:f.write(raw)
            assets[name]=dict(path=str(target),sha256=sha(target),bytes=len(raw),url=record['asset']['url'])
        assert assets['resnet20-weights']['sha256']=='12fca82f0bebc4135bf1f32f6e3710e61d5108578464b84fd6d7f5c1b04036c8'
        # Parse source descriptions as data. Nothing from downloaded Python is executed.
        upstream=ast.parse(Path(assets['resnet-source']['path']).read_text())
        entry=next(node for node in upstream.body if isinstance(node,ast.FunctionDef)and node.name=='resnet20')
        assert ast.unparse(entry.body[0].value)=='ResNet(BasicBlock, [3, 3, 3])'
        trainer=ast.parse(Path(assets['trainer-source']['path']).read_text())
        normalizers=[n for n in ast.walk(trainer)if isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)and n.func.attr=='Normalize']
        assert len(normalizers)==1
        normalization={k.arg:ast.literal_eval(k.value)for k in normalizers[0].keywords}
        import numpy as np,torch
        from model_reference import load_original_weights,independent_stages,valid_domain_probes,MEAN,STD
        assert normalization==dict(mean=list(MEAN),std=list(STD))
        model,loading=load_original_weights(assets['resnet20-weights']['path'])
        assert loading['parameter_elements']==269722
        state={name:value.detach().numpy().copy()for name,value in model.state_dict().items()}
        pixels,normalized=valid_domain_probes()
        assert pixels.dtype==np.uint8 and pixels.shape==(4,3,32,32)
        mean=torch.tensor(MEAN,dtype=torch.float64).reshape(1,3,1,1)
        std=torch.tensor(STD,dtype=torch.float64).reshape(1,3,1,1)
        supplied=(torch.from_numpy(pixels).to(torch.float64)/255.0-mean)/std
        np.testing.assert_allclose(supplied.numpy(),normalized,atol=1e-12,rtol=1e-12)
        with torch.no_grad():
            native32=model((torch.from_numpy(pixels).to(torch.float32)/255.0-mean.float())/std.float()).numpy()
            stages=model.double().forward_stages(supplied)
        references=independent_stages(normalized,state)
        assert list(stages)==list(references)
        from numerical_results import compare
        rows=[]
        for name,reference in references.items():
            actual=stages[name].numpy();comparison=compare(actual,reference,1e-12,1e-12)
            assert comparison['passed'],name
            rows.append(dict(stage=name,shape=list(actual.shape),comparison=comparison))
        np.save(RESULT/'uint8-probes.npy',pixels,allow_pickle=False)
        np.save(RESULT/'normalized-f64.npy',normalized,allow_pickle=False)
        np.save(RESULT/'torch-f64-logits.npy',stages['logits'].numpy(),allow_pickle=False)
        np.save(RESULT/'numpy-f64-logits.npy',references['logits'],allow_pickle=False)
        np.save(RESULT/'torch-f32-logits.npy',native32,allow_pickle=False)
        report=dict(format='poseidon-original-resnet20-reference-v1',status='succeeded',plan_binding=batch.binding,
            architecture=dict(input=[1,3,32,32],channels=[16,32,64],blocks=[3,3,3],logits=10,shortcut='Option A stride-2 spatial sampling and symmetric zero channel padding',relu='Original exact ReLU',batch_norm_eps=1e-5),
            approved_assets=assets,loading=loading,preprocessing=normalization,
            source_correspondence='Pinned source AST architecture and preprocessing audit; downloaded code was never imported or executed',
            groups=4,stage_comparisons=len(rows),compared_values=sum(row['comparison']['compared_values']for row in rows),
            max_independent_difference=max(row['comparison']['max_absolute_error']for row in rows),
            independent_math_tolerance=dict(atol=1e-12,rtol=1e-12),stage_results=rows,
            f32_f64_max_difference=float(np.max(np.abs(native32-references['logits']))),
            f32_f64_probe_top1_agreement=int(np.count_nonzero(np.argmax(native32,axis=1)==np.argmax(references['logits'],axis=1))),
            original_weight_bytes_changed=False,learned_weights_changed=False,downloaded_code_executed=False,
            unsafe_pickle_used=False,api_calls=0,encrypted_calls=0,new_downloads=0,
            cifar_test_accuracy_measured=False,approximation_calibrated=False,b4_complete=False,b5_complete=False,goal_complete=False)
        create(RESULT/'report.json',report)
        unchanged();batch.checkpoint();status='succeeded'
        print(json.dumps({key:value for key,value in report.items()if key not in ('stage_results','approved_assets','loading')}),flush=True)
    except BaseException:
        (RESULT/'exception.log').write_text(traceback.format_exc())
        if not(RESULT/'report.json').exists():create(RESULT/'report.json',dict(status='failed',plan_binding=batch.binding,api_calls=0,encrypted_calls=0,new_downloads=0,unsafe_pickle_fallback=False,original_assets_preserved=True))
        raise
    finally:batch.finish(status)
    return 0
if __name__=='__main__':raise SystemExit(main())
