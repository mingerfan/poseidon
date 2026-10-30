"""Trace the exact accepted configuration; dry-run does not import Hecate."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parent
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--weights',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args()
    c=json.loads((ROOT/'delivery-config.json').read_text())
    cmd=[sys.executable,'-B',str(ROOT/'frozen/model/trace_combined_model.py'),
         '--mode','decode-pair','--positions','0','1',
         '--weights',str(a.weights.resolve()),'--output-dir',str(a.output_dir.resolve())]
    for key,value in c['trace_options'].items():
        if isinstance(value,bool):
            if value:cmd.append('--'+key.replace('_','-'))
        else:cmd.extend(['--'+key.replace('_','-'),str(value)])
    print(json.dumps(cmd,indent=2),flush=True)
    if not a.dry_run:subprocess.run(cmd,check=True)
if __name__=='__main__':main()
