"""Verify source integrity, 24-layer interfaces and arithmetic evidence scope."""
import ast
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text('utf-8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    manifest=read(ROOT/'manifest.json')
    for name,record in manifest['files'].items():
        p=(ROOT/name).resolve()
        assert p.is_relative_to(ROOT) and p.is_file(),name
        assert sha(p)==record['sha256'] and p.stat().st_size==record['bytes'],name
        if p.suffix=='.py':ast.parse(p.read_text('utf-8-sig'),filename=name)
    binding=read(ROOT/'source-binding.json')
    for name,entry in binding['files'].items():
        assert sha(ROOT/name)==entry['delivered_sha256'],name
    for name in binding['unchanged_arithmetic_files']:
        entry=binding['files'][name]
        assert entry['unchanged'] and sha(ROOT/name)==entry['source_sha256'],name
    for name,inputs in [('decode_pair',2),('prefill_2token',2),('prefill_1token',1),('decode_second_token',25)]:
        sig=read(ROOT/('interfaces/'+name+'.json'))
        assert sig['config']['num_layers']==24 and len(sig['inputs'])==inputs and len(sig['outputs'])==29
        assert sig['native_compiled']==(name=='decode_pair')
        assert not sig['full_model_ckks_validated'] and not sig['gpu_validated']
        assert [v['name'] for v in sig['outputs'][5:]]==[f'layer_{i}.kv_cache' for i in range(24)]
    profiles=read(ROOT/'frozen/model/package/config/approximations.provisional.json')
    assert len(profiles['blocks'])==24
    schema=read(ROOT/'frozen/model/package/config/weight_schema.json');assert len(schema)==291
    for i in range(24):assert len([k for k in schema if k.startswith(f'model_list.{i}.')])==12
    for name in ('basic','tiled'):
        result=read(ROOT/('evidence/small-'+name+'.json'))
        assert result['passed'] and not result['ckks'] and not result['gpu'] and result['cache_immutable']
        assert result['config']['num_layers']==24 and len(result['checks'])==4 and result['pair_calls']==46
        assert all(r['executed_blocks']==24 and r['comparison']['passed'] for r in result['checks'])
        if name=='tiled':assert any(t['tile']==256 and t['width']==385 for t in result['tile_calls'])
    proof=read(ROOT/'evidence/validation-binding.json')
    for name,digest in proof['source_files'].items():assert sha(ROOT/name)==digest,name
    proof=read(ROOT/'evidence/compiled-verification.json')
    assert proof['passed'] and proof['plan_verified'] and proof['full_mock_passed'] and not proof['ckks_executed']
    assert sha(ROOT/'evidence/compiled-analysis.json')==proof['result_sha256']
    print(json.dumps(dict(passed=True,files=len(manifest['files']),blocks=24,weight_arrays=291,
        unchanged_arithmetic=len(binding['unchanged_arithmetic_files']),small_cases=2,paths=8,
        native_compiled=True,full_model_ckks_validated=False,gpu_validated=False)))

if __name__=='__main__':main()
