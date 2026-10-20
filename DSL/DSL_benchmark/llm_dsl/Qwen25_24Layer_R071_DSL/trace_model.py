"""24 blocks derived from 071; fixed approximation strategy, at most two visible tokens."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parent
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=('decode-pair','prefill','prefill-one','decode'),default='decode-pair')
    p.add_argument('--weights',type=Path)
    p.add_argument('--output-dir',type=Path)
    p.add_argument('--describe',action='store_true')
    a=p.parse_args()
    modes={'decode-pair':('decode-pair',[0,1],[],'decode_pair'), 'prefill':('prefill',[0,1],[],'prefill_2token'),
           'prefill-one':('prefill',[0],[],'prefill_1token'), 'decode':('decode',[1],[0],'decode_second_token')}
    mode,pos,cache,name=modes[a.mode]
    if a.describe:
        print((ROOT/('interfaces/'+name+'.json')).read_text('utf-8'));return
    if a.weights is None or a.output_dir is None:p.error('--weights and --output-dir required')
    cmd=[sys.executable,'-B',str(ROOT/'source/model/trace_combined_model.py'),'--mode',mode,
         '--positions',*map(str,pos),'--rms-degree-strategy','degree10p2_pair13_silu_canonical_fold2',
         '--weights',str(a.weights.resolve()),'--output-dir',str(a.output_dir.resolve())]
    if cache:cmd+=['--cached-positions',*map(str,cache)]
    subprocess.run(cmd,check=True)
if __name__=='__main__':main()
