"""Apply the accepted host-only postpasses; verify and Mock before CKKS."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'frozen/model'),str(ROOT/'frozen/tools')]
from elide_terminal_refresh import transform as terminal
from release_ready_schedule import transform as schedule
from bounded_cpu_encodes import transform_bounded
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--report',type=Path,required=True)
    a=p.parse_args()
    assert len({x.resolve() for x in (a.input,a.output,a.report)})==3
    assert not a.output.exists() and not a.report.exists()
    plan=json.loads(a.input.read_text())
    plan,t=terminal(plan)
    plan,s=schedule(plan)
    plan,c=transform_bounded(plan,64*2**20)
    payload=json.dumps(plan,separators=(',',':')).encode()
    with a.output.open('xb') as f:f.write(payload)
    with a.report.open('x',encoding='utf-8') as f:
        json.dump(dict(terminal=t,schedule=s,cache=c,output_sha256=hashlib.sha256(payload).hexdigest(),
            independently_verified=False,ckks_executed=False),f,indent=2)
if __name__=='__main__':main()
