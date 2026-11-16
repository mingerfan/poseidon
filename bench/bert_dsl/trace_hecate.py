#!/usr/bin/env python3
"""Native hc.func/hc.save frontend for the same BERT arithmetic as generate.py.

Requires the bundled Hecate frontend environment (including NumPy and Torch).
Use a fresh process/output directory per trace. Automatic --dacapo will replan
Earth Boot hints; preserve manual scale/Boot through the physical CKKS path.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--layout',choices=('prefix','packed'),default='packed')
    p.add_argument('--tokens',type=int,default=128);p.add_argument('--layers',type=int,default=1)
    p.add_argument('--hidden',type=int,default=768);p.add_argument('--intermediate',type=int,default=3072)
    p.add_argument('--heads',type=int,default=12);p.add_argument('--slots',type=int,default=32768)
    p.add_argument('--row-stride',type=int,default=4096);p.add_argument('--seed',type=int,default=20261114)
    p.add_argument('--boot-policy',choices=('none','stage'),default='none')
    p.add_argument('--output-dir',type=Path);p.add_argument('--describe',action='store_true')
    a=p.parse_args()
    from model import BertConfig,create_weights,build_model
    fields=dict(tokens=a.tokens,layers=a.layers,hidden=a.hidden,intermediate=a.intermediate,heads=a.heads,slots=a.slots)
    if a.layout=='packed':
        from packed_model import PackedConfig,build_model
        config=PackedConfig(**fields,row_stride=a.row_stride);count=config.groups
    else:
        config=BertConfig(**fields,head_dim=a.hidden//a.heads);count=a.tokens
    signature=dict(model='BERT encoder',layout=a.layout,config=asdict(config),input_ciphertexts=count,
                   output_ciphertexts=count,seed=a.seed,boot_policy=a.boot_policy,
                   arithmetic='post-LN, bidirectional exp4/Newton3 attention, GELU4, rsqrt3')
    if a.describe:
        print(json.dumps(signature,indent=2));return
    if a.output_dir is None:p.error('--output-dir is required unless --describe')
    if a.output_dir.exists():p.error('--output-dir must be new')
    import hecate as hc
    class Public:
        def __init__(self,values):self.values=values;self.raw=hc.Plain(values)
    class Cipher:
        def __init__(self,value):self.raw=value
        @staticmethod
        def unwrap(value):return value.raw if isinstance(value,(Cipher,Public)) else value
        def __add__(self,other):return Cipher(self.raw+self.unwrap(other))
        def __radd__(self,other):return Cipher(self.unwrap(other)+self.raw)
        def __mul__(self,other):return Cipher(self.raw*self.unwrap(other))
        def __rmul__(self,other):return Cipher(self.unwrap(other)*self.raw)
        def __sub__(self,other):
            # The existing frontend lowers x-Plain to illegal earth.negate(Plain).
            # Negate the PUBLIC coefficients on the client, not a plaintext IR op.
            return Cipher(self.raw+hc.Plain(-other.values)) if isinstance(other,Public) else Cipher(self.raw-self.unwrap(other))
        def __rsub__(self,other):return Cipher(self.unwrap(other)-self.raw)
        def __neg__(self):return Cipher(-self.raw)
        def rotate(self,step):return Cipher(self.raw.rotate(step))
    class NativeHecate:
        def plain(self,values):return Public(values)
        def bootstrap(self,value,label='stage'):return Cipher(hc.bootstrap(value.raw))
    weights=create_weights(config,a.seed);backend=NativeHecate()
    @hc.func(','.join(['c']*count))
    def bert_encoder(*inputs):
        outputs=build_model(backend,[Cipher(x) for x in inputs],weights,config,boot_policy=a.boot_policy)
        return [x.raw for x in outputs]
    a.output_dir.mkdir(parents=True)
    exported=hc.save(str(a.output_dir),str(a.output_dir),bytecode=True)
    signature['frontend_artifact']=str(exported)
    (a.output_dir/'signature.json').write_text(json.dumps(signature,indent=2)+'\n')
    print(json.dumps(signature,indent=2))


if __name__=='__main__':main()
