"""Restore frozen assets and regenerate the large CKKS payload from the pinned model."""
import argparse,gzip,hashlib,json,os,shutil,subprocess,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''):h.update(block)
    return h.hexdigest()
def prepare(work,payload=False):
    assets=work/'assets';assets.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((ROOT/'fixtures/assets.json').read_text())
    for item in manifest['gzip_assets']:
        dst=assets/item['unpacked']
        if not dst.exists():
            tmp=dst.with_name(dst.name+'.partial')
            with gzip.open(ROOT/item['compressed'],'rb') as src,tmp.open('xb') as out:shutil.copyfileobj(src,out)
            assert tmp.stat().st_size==item['bytes'] and sha(tmp)==item['sha256'],item['unpacked']
            tmp.rename(dst)
        assert dst.stat().st_size==item['bytes'] and sha(dst)==item['sha256'],dst
    for src,name in ((ROOT/'source/lowering_report.json','resnet18.lowering_report.json'),
                     (ROOT/'fixtures/resnet18.inputs.jsonl','resnet18.inputs.jsonl')):
        dst=assets/name
        if dst.exists():assert sha(dst)==sha(src),dst
        else:shutil.copy2(src,dst)
    if payload:
        expected=manifest['generated_payload'];dst=assets/expected['name']
        if not dst.exists():
            data=work/'data';data.mkdir(exist_ok=True)
            with tarfile.open(ROOT/'fixtures/model_data.tar.gz','r:gz') as tar:
                for item in tar:
                    path=(data/item.name).resolve()
                    assert path.is_relative_to(data.resolve()) and item.isfile(),item.name
                    path.parent.mkdir(parents=True,exist_ok=True)
                    with tar.extractfile(item) as source:
                        raw=source.read()
                    if path.exists():assert path.read_bytes()==raw,path
                    else:path.write_bytes(raw)
            from build_trace import build
            binary=build(work/'cpu-trace')
            trace=work/'generated-native.jsonl'
            env=os.environ.copy();env['POSEIDON_TRIDENT_RESNET18_ROOT']=str(data/'resnet18')
            for key in ('POSEIDON_RESCALE_META_BITS','POSEIDON_RESCALE_MODEL_BITS','POSEIDON_DSL_SCALE_SHIFT','POSEIDON_GPU_DNUM'):env.pop(key,None)
            with (work/'generate-native.log').open('x') as log:
                subprocess.run([str(binary),str(trace)],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            assert sha(trace)==sha(assets/'resnet18.native_trace.jsonl'),'regenerated graph differs'
            generated=Path(str(trace)+'.zlib')
            assert generated.stat().st_size==expected['bytes'] and sha(generated)==expected['sha256'],'regenerated payload differs'
            generated.rename(dst)
        assert dst.stat().st_size==expected['bytes'] and sha(dst)==expected['sha256'],dst
    return assets
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--work',type=Path,default=ROOT/'work')
    p.add_argument('--payload',action='store_true');a=p.parse_args();print(prepare(a.work.resolve(),a.payload))
