"""Trace one fixed two-token block using the recipient's native Hecate frontend."""
import argparse
import json
from pathlib import Path

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('prefill','decode-pair'),required=True)
    parser.add_argument('--weights',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    if args.output_dir.exists():
        parser.error('output directory must be new')
    import hecate as hc
    from qwen2token.block import load_model,register,signature
    model=load_model(args.weights)
    register(hc,model,args.mode)
    args.output_dir.mkdir(parents=True)
    print(hc.save(str(args.output_dir),str(args.output_dir)))
    with (args.output_dir/'signature.json').open('x',encoding='utf-8') as stream:
        json.dump(signature(args.mode),stream,indent=2)

if __name__=='__main__':
    main()
