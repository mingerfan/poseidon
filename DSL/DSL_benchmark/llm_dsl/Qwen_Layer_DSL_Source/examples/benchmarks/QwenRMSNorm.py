"""Trace RMSNorm for one token using an explicit inverse-square-root profile."""
import argparse
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights',type=Path,required=True,help='NPZ containing weight: [hidden_size]')
    parser.add_argument('--profiles',type=Path,default=Path(__file__).with_name('qwen_layer_profiles.json'))
    parser.add_argument('--slots',type=int,required=True)
    parser.add_argument('--eps',type=float,default=1e-6)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    if args.slots<1 or args.slots&(args.slots-1):
        parser.error('--slots must be a positive power of two')
    import numpy as np
    import hecate as hc
    from hecate import qwen_layers as ql

    with np.load(args.weights,allow_pickle=False) as archive:
        gain=archive['weight']
    if gain.ndim!=1:
        parser.error('weight must be a one-dimensional gain vector')
    profile=ql.approximation_from_dict(json.loads(args.profiles.read_text(encoding='utf-8'))['rsqrt'])

    @hc.func('c')
    def QwenRMSNorm(x):
        return ql.rms_norm(ql.PackedVector(x,len(gain),args.slots),gain,rsqrt=profile,eps=args.eps).expr

    args.output_dir.mkdir(parents=True,exist_ok=True)
    print(hc.save(str(args.output_dir),str(args.output_dir)))


if __name__=='__main__':
    main()
