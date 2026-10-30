"""Trace the 24-layer derivative configuration; dry-run does not import Hecate."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parent
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=('decode-pair','prefill','prefill-one','decode'),default='decode-pair')
    p.add_argument('--describe',action='store_true')
    p.add_argument('--weights',type=Path)
    p.add_argument('--output-dir',type=Path)
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args()
    c=json.loads((ROOT/'delivery-config.json').read_text())
    modes={'decode-pair':('decode-pair',[0,1],[],'decode_pair'),'prefill':('prefill',[0,1],[],'prefill_2token'),'prefill-one':('prefill',[0],[],'prefill_1token'),'decode':('decode',[1],[0],'decode_second_token')}
    mode,pos,cache,name=modes[a.mode]
    if a.describe:
        print((ROOT/('interfaces/'+name+'.json')).read_text('utf-8'));return
    if a.weights is None or a.output_dir is None:p.error('--weights and --output-dir required')
    cmd=[sys.executable,'-B',str(ROOT/'frozen/model/trace_combined_model.py'),
         '--mode',mode,'--positions',*map(str,pos),
         '--weights',str(a.weights.resolve()),'--output-dir',str(a.output_dir.resolve())]
    for key,value in c['trace_options'].items():
        if isinstance(value,bool):
            if value:cmd.append('--'+key.replace('_','-'))
        else:cmd.extend(['--'+key.replace('_','-'),str(value)])
    if cache:cmd+=['--cached-positions',*map(str,cache)]
    print(json.dumps(cmd,indent=2),flush=True)
    if not a.dry_run:subprocess.run(cmd,check=True)
if __name__=='__main__':main()
