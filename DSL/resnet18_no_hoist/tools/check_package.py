"""Verify package hashes and independently recompute ResNet18 acceptance evidence."""
import hashlib,json,math,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def read(name):return json.loads((ROOT/name).read_text())
def error(a,b):
    assert len(a)==len(b)==1000 and all(math.isfinite(v) for v in a+b)
    delta=[x-y for x,y in zip(a,b)]
    return max(map(abs,delta)),math.sqrt(math.fsum(v*v for v in delta)/1000)
def check():
    manifest=read('package_manifest.json')
    assert manifest['files']['artifacts/resnet18.hevm']=='e9f6a906d490766134e810d4f16224f05a9f0c7c7f7ebdf67037595d5c84b4df'
    for name,expected in manifest['files'].items():
        path=ROOT/name;assert path.is_file() and path.stat().st_size<100_000_000,name
        assert hashlib.sha256(path.read_bytes()).hexdigest()==expected,name
    base='evidence/resnet18-dsl-gpu2-v1-'
    d,n=read(base+'full-report.json'),read(base+'native-logits.json')
    assert d['status']==n['status']=='COMPLETE' and d['bootstrap_calls']==60
    assert all(x['prediction']==x['label']==65 for x in (d,n))
    assert all(max(range(1000),key=x['logits'].__getitem__)==65 for x in (d,n))
    assert 'DSL_PROCESS_EXIT code=0' in (ROOT/(base+'full.log')).read_text()
    assert 'NATIVE_LOGITS_EXIT=0' in (ROOT/(base+'native-logits.log')).read_text()
    maximum,rms=error(d['logits'],n['logits'])
    assert math.isclose(maximum,0.005010604173461886,rel_tol=1e-13)
    assert math.isclose(rms,0.0017419935604740106,rel_tol=1e-13)
    stem='evidence/resnet18-same-cipher-gpu1-v1-full'
    paired=read(stem+'.paired.json');a,b=read(stem+'.native.json'),read(stem+'.dsl.json')
    assert paired['status']=='PASS' and paired['same_context_and_keys'] and paired['same_input_ciphertexts']
    assert paired['bit_exact_final_ciphertexts'] and paired['input_ciphertexts']==147
    assert paired['native_rns_outputs']==paired['dsl_rns_outputs']==1000
    assert all(paired[k]==0 for k in ('rns_mismatch_words','rns_mismatch_classes','metadata_mismatch_classes'))
    assert error(a['logits'],b['logits'])==(0,0)
    assert a['prediction']==b['prediction']==a['label']==b['label']==65
    assert all(max(range(1000),key=x['logits'].__getitem__)==65 for x in (a,b))
    assert b['bootstrap_calls']==60
    log=(ROOT/(stem+'.log')).read_text()
    assert 'SHARED_CIPHER_PROCESS_EXIT=0' in log
    assert re.findall(r'DSL_NATIVE_BOOTSTRAP call=(\d+)',log)==[str(i) for i in range(1,61)]
    capture=read(stem+'.capture-audit.json')
    assert capture['status']=='PASS' and capture['rns_words']==524288000 and capture['records']==1000
    assert capture['sha256']=='0743069a4361a116b7911a793a6288137a2f018a11b2dd5b43b3dc0c83ec60f0'
    packaged=read('evidence/packaged-validation-20261016.json')
    assert packaged['status']=='PASS' and packaged['backend_source_files_matched']==237
    assert packaged['real_compiler_hevm_and_two_sidecars_byte_equal'] and packaged['metadata_preflight']
    extended=packaged['gpu_prefixes']['95990']
    assert extended['boots']==8 and all(extended[k]>=1 for k in ('exact_vector','exact_scalar','exact_product'))
    result=dict(status='PASS',package_files=len(manifest['files']),classification=65,
        label=65,independent_max_error=maximum,independent_rms_error=rms,
        same_cipher_max_error=0,same_cipher_rms_error=0,bit_exact_final_ciphertexts=1000)
    print(json.dumps(result,indent=2));return result
if __name__=='__main__':check()
