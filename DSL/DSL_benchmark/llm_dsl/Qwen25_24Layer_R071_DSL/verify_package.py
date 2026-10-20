"""Verify delivery and that all inherited arithmetic kernels remain byte-identical."""
import ast
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def main():
    m=json.loads((ROOT/'manifest.json').read_text('utf-8'))
    for name,h in m['files'].items():
        p=(ROOT/name).resolve();assert p.is_relative_to(ROOT.resolve())
        assert hashlib.sha256(p.read_bytes()).hexdigest()==h,name
        if p.suffix=='.py':ast.parse(p.read_text('utf-8-sig'),filename=name)
    b=json.loads((ROOT/'source-binding.json').read_text('utf-8'))
    for name in b['unchanged_arithmetic_files']:
        assert b['files'][name]['unchanged']
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==b['files'][name]['source_sha256']
    pair=json.loads((ROOT/'interfaces/decode_pair.json').read_text('utf-8'))
    assert pair['config']['num_layers']==24 and len(pair['inputs'])==2 and len(pair['outputs'])==29
    assert all(not pair[k] for k in ('native_compiled','full_model_ckks_validated','gpu_validated'))
    print(json.dumps(dict(passed=True,files=len(m['files']),unchanged_kernels=len(b['unchanged_arithmetic_files']),
        blocks=24,decode_pair_inputs=2,decode_pair_outputs=29,native_compiled=False,ckks_validated=False,gpu_validated=False)))
if __name__=='__main__':main()
