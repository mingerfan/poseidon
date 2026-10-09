"""Trace complete causal Attention, with explicit public weights and KV inputs."""
import argparse
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights',type=Path,required=True,help='NPZ: q_weight, k_weight, v_weight, o_weight; optional q_bias/k_bias/v_bias')
    parser.add_argument('--profiles',type=Path,default=Path(__file__).with_name('qwen_layer_profiles.json'))
    parser.add_argument('--slots',type=int,required=True)
    parser.add_argument('--positions',type=int,nargs='+',required=True)
    parser.add_argument('--cached-positions',type=int,nargs='*',default=[])
    parser.add_argument('--mode',choices=('prefill','decode'),default='prefill')
    parser.add_argument('--query-heads',type=int,default=14)
    parser.add_argument('--kv-heads',type=int,default=2)
    parser.add_argument('--head-dim',type=int,default=64)
    parser.add_argument('--theta',type=float,default=1_000_000.)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    if args.slots<1 or args.slots&(args.slots-1):
        parser.error('--slots must be a positive power of two')
    if args.mode=='prefill' and args.cached_positions:
        parser.error('prefill starts with an empty cache')
    if args.mode=='decode' and (len(args.positions)!=1 or not args.cached_positions):
        parser.error('decode requires one new position and a nonempty cache')
    import numpy as np
    import hecate as hc
    from hecate import qwen_layers as ql

    with np.load(args.weights,allow_pickle=False) as archive:
        weights={name:archive[name] for name in archive.files}
    if not {'q_weight','k_weight','v_weight','o_weight'}<=weights.keys():
        parser.error('weights require q_weight, k_weight, v_weight, o_weight')
    profile=ql.approximation_from_dict(json.loads(args.profiles.read_text(encoding='utf-8'))['softmax'])
    hidden_size=weights['q_weight'].shape[1]
    count=len(args.positions)
    cached=len(args.cached_positions)
    sizes=[hidden_size]*count+[args.kv_heads*args.head_dim]*(2*cached)

    @hc.func(','.join('c' for _ in sizes))
    def QwenAttention(*expressions):
        vectors=[ql.PackedVector(expr,size,args.slots) for expr,size in zip(expressions,sizes)]
        outputs,keys,values=ql.attention(
            vectors[:count],weights['q_weight'],weights['k_weight'],weights['v_weight'],weights['o_weight'],
            positions=args.positions,cached_positions=args.cached_positions,
            cached_keys=vectors[count:count+cached],cached_values=vectors[count+cached:],
            q_bias=weights.get('q_bias'),k_bias=weights.get('k_bias'),v_bias=weights.get('v_bias'),
            query_heads=args.query_heads,kv_heads=args.kv_heads,head_dim=args.head_dim,
            theta=args.theta,normalization=profile)
        return [vector.expr for vector in outputs+keys+values]

    args.output_dir.mkdir(parents=True,exist_ok=True)
    print(hc.save(str(args.output_dir),str(args.output_dir)))


if __name__=='__main__':
    main()
