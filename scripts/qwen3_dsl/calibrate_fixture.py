#!/usr/bin/env python3
"""Finish nonlinear calibration without rewriting verified converted weights."""
import argparse, json, math, struct
from pathlib import Path
import numpy as np
from prepare_checkpoint import REVISION, WEIGHT_SHA256, calibrate
from qwen3 import reference, weights


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fixture',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--tokens',type=int,nargs=2,default=[9707,11])
    a=p.parse_args();out=a.fixture;checkpoint=a.checkpoint
    if weights.file_hash(checkpoint/'model.safetensors')!=WEIGHT_SHA256:
        raise ValueError('Pinned checkpoint SHA-256 mismatch')
    c=json.loads((out/'model.json').read_text())
    arrays=weights.load(out/'weights',c)
    embedding=np.load(out/'embeddings.npy',allow_pickle=False)
    if not np.array_equal(embedding,arrays['emb_weight.weight'][a.tokens]):
        raise ValueError('Fixture embeddings do not match token weights')
    exact=dict(np.load(out/'reference-exact.npz',allow_pickle=False))
    profiles=calibrate(json.loads((out/'activation-bounds.json').read_text()))
    approximate,observed=reference.forward(arrays,embedding,[0,1],c,profiles=profiles)
    (out/'approximations.json').write_text(json.dumps(profiles,indent=2)+'\n')
    np.savez(out/'reference-approximate.npz',**approximate)
    (out/'approximate-activation-bounds.json').write_text(json.dumps(observed,indent=2)+'\n')
    error=approximate['logits']-exact['logits']
    tokenizer=json.loads((checkpoint/'tokenizer.json').read_text())
    inverse={v:k for k,v in tokenizer['model']['vocab'].items()}
    with (checkpoint/'model.safetensors').open('rb') as f:
        length=struct.unpack('<Q',f.read(8))[0];header=json.loads(f.read(length))
    provenance=dict(model='Qwen/Qwen3-0.6B',revision=REVISION,weight_sha256=WEIGHT_SHA256,
        config=json.loads((checkpoint/'config.json').read_text()),tensor_count=len(arrays),
        stored_parameters=sum(math.prod(r['shape']) for k,r in header.items() if k!='__metadata__'),
        tied_weights_identical=bool(np.array_equal(arrays['emb_weight.weight'],arrays['lm_head.weight'])),
        tokens=a.tokens,token_strings=[inverse[t] for t in a.tokens],positions=[0,1],
        reference='independent dense NumPy FP64 equations; original rsqrt/SiLU/causal softmax and a separate polynomial reference',
        profile_scope='calibrated on this fixed two-token prefill; not a general input accuracy certificate',
        ckks_tolerance={'absolute':.001,'relative':.0001},
        exact_top1=int(np.argmax(exact['logits'])),approximate_top1=int(np.argmax(approximate['logits'])),
        approximate_vs_exact={'logit_max_abs':float(np.max(np.abs(error))),
          'logit_rms':float(np.sqrt(np.mean(error**2))),
          'cosine':float(np.dot(approximate['logits'],exact['logits'])/(np.linalg.norm(approximate['logits'])*np.linalg.norm(exact['logits'])))})
    (out/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(json.dumps(provenance,indent=2),flush=True)


if __name__=='__main__':main()
