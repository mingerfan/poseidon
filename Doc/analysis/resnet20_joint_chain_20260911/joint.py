"""One-chain bootstrap/ReLU/ConvBN/S2C metadata audit; no encryption or GPU.

Numerical outputs are generated artifacts, never production parameters.
"""
import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'resnet20_scale_chain_20260911'))
from search import Planner, ntt_primes

# Exact public primes from the reproduced historical mixed-Q34 schedule.
HISTORICAL = [4255252481,4255645697,4258922497,4260364289,4261675009,
 4264427521,4265476097,4266393601,4267442177,4269015041,4271374337,4271505409,
 265420801,268042241,2135031809,2135162881,4272291841,4272685057,1071513601,
 2135818241,4273340417,2142502913,4274126849,4276092929,2144468993,2146041857,
 2146959361,4277403649,4279369729,2147352577,4280025089,4280156161,4281204737,1073479681]

def make_chain(full_q=50, app_bits=30, prefer_relu=False):
    # Q9 lower prefix: two base primes, four S2C primes, three ConvBN primes.
    # The top 19 primes retain the historical C2S/EvalMod schedule exactly.
    low = HISTORICAL[:9]
    high = HISTORICAL[15:]
    count = full_q-len(low)-len(high)
    if prefer_relu:
        # Preserve the exact 22-prime ReLU witness, plus an explicit 32-bit
        # output-normalization prime when full_q=51. Replace collisions in
        # the bootstrap suffix; the source planner must recheck that suffix.
        application=list(ntt_primes(app_bits)[:min(count,22)])
        used=set(low+application)
        new_high=[]
        for value in high:
            if value in used:
                value=next(v for v in ntt_primes(value.bit_length()) if v not in used and v not in high)
            new_high.append(value);used.add(value)
        extra=[]
        for _ in range(max(0,count-22)):
            v=next(v for v in ntt_primes(32) if v not in used)
            extra.append(v);used.add(v)
        q=low+list(reversed(application))+extra+new_high
    else:
        excluded = set(low+high)
        application = [q for q in ntt_primes(app_bits) if q not in excluded][:count]
        q = low + list(reversed(application)) + high
    p = [v for v in ntt_primes(32) if v not in set(q)][:math.ceil(full_q/2)]
    assert len(q)==full_q and len(set(q+p))==len(q+p)
    return q,p

def linear(q, count, scale, plaintexts):
    trace = []
    for plain in plaintexts:
        pre=scale+plain
        out=count
        post=pre
        while out>1 and post-math.log2(q[out-1])>=44-1e-12:
            post-=math.log2(q[out-1]);out-=1
        if count==out: raise ValueError('linear stage cannot rescale')
        trace.append(dict(q_in=count,s_in=scale,plain=plain,pre=pre,
                          drop=count-out,q_out=out,s_out=post))
        count,scale=out,post
    return trace

def prepare(q, qin=6, sin=40):
    s2c=linear(q,qin,sin,[45]*3)
    n,s=s2c[-1]['q_out'],s2c[-1]['s_out']
    guard=[]
    while s>54:
        before=n
        s-=math.log2(q[n-1]);n-=1
        guard.append(dict(q_in=before,q_out=n,s_out=s))
    if n<2: raise ValueError('S2C output falls below Q2 in preparation')
    target=math.log2(q[0])+math.log2(q[1])-5
    multiplier=max(1,math.floor(2**(target-s)+0.5))
    prepared=s+math.log2(multiplier)
    # Exactly the source's calibrated C2S physical/logical scale relation.
    c2s_in=prepared+45+5-round(math.log2(q[0])+math.log2(q[1]))
    return dict(s2c=s2c,guard=guard,q_out=2,prepared_scale=prepared,
                target_scale=target,multiplier=multiplier,c2s_in=c2s_in)

def bootstrap(bridge,q,p,entry,sweep=False,fold=False):
    payload=f'{len(q)} {len(p)} {entry:.17g}\n'+' '.join(map(str,q+p))+'\n'
    done=subprocess.run([bridge]+(['sweep'] if sweep else ['fold'] if fold else []),input=payload,text=True,capture_output=True,check=True)
    return json.loads(done.stdout)

def find_relu(q, qin, sin, seconds=35):
    planner=Planner([math.log2(x) for x in reversed(q[:qin])])
    best=None
    started=time.monotonic()
    # Start with the previously studied schedules, then vary both interfaces.
    scales=[(40.1,44),(40.2,44),(42.5,44),(45,45),(50,45)]
    scales += [(40+i/2,40+j/2) for i in range(41) for j in range(41)]
    checked=0
    for a,b in scales:
        c=planner.relu_fixed(input_s=sin,stage_scales=(a,b),trace=True)
        checked+=1
        if c is not None and (best is None or c['total']<best['total']):
            best=c
            print('RELU_BEST',json.dumps(dict(input_q=qin,input_scale=sin,
              scales=[a,b],total=c['total'],output_q=qin-c['total'])),flush=True)
        if best and qin-best['total']>=9:
            break
        if time.monotonic()-started>seconds:
            break
    return best,checked

def conv(q, qin, sin=40):
    # Actual fast replicated ConvBN: q_last weights, q_next support mask,
    # two rescales, then q_next selector and one rescale. Rotations/adds
    # preserve scale. All groups share these parameters before their sums.
    a,b,c=map(math.log2,(q[qin-1],q[qin-2],q[qin-3]))
    return [dict(name='weight_mac',q_in=qin,q_out=qin,s_in=sin,plain=a,s_out=sin+a),
      dict(name='support_mac',q_in=qin,q_out=qin,s_in=sin+a,plain=b,s_out=sin+a+b),
      dict(name='rescale_weight',q_in=qin,q_out=qin-1,s_in=sin+a+b,s_out=sin+b),
      dict(name='rescale_support',q_in=qin-1,q_out=qin-2,s_in=sin+b,s_out=sin),
      dict(name='selector_mac',q_in=qin-2,q_out=qin-2,s_in=sin,plain=c,s_out=sin+c),
      dict(name='rescale_selector',q_in=qin-2,q_out=qin-3,s_in=sin+c,s_out=sin)]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--bridge',required=True)
    ap.add_argument('--output',required=True)
    ap.add_argument('--full-q',type=int,default=50)
    ap.add_argument('--app-bits',type=int,default=30)
    ap.add_argument('--seconds',type=float,default=35)
    ap.add_argument('--sweep',action='store_true')
    ap.add_argument('--normalize',action='store_true')
    ap.add_argument('--prefer-relu',action='store_true')
    ap.add_argument('--fold-output',action='store_true')
    args=ap.parse_args()
    q,p=make_chain(args.full_q,args.app_bits,args.prefer_relu)
    entry=prepare(q)
    assert not (args.fold_output and args.normalize)
    cases=bootstrap(args.bridge,q,p,entry['c2s_in'],args.sweep,args.fold_output)
    if args.sweep:
        Path(args.output).write_text(json.dumps(dict(q_bottom_first=q,p=p,
          preparation=entry,cases=cases),indent=2)+'\n')
        for case in cases:
            print('CASE',case['last_plain'],case['eval_q_out'],case['eval_s_out'],flush=True)
        return
    boot=cases[0]
    print('BOOT',json.dumps({k:boot[k] for k in ('eval_q_in','eval_q_out','eval_s_in','eval_s_out')}),flush=True)
    relu_q,relu_s=boot['eval_q_out'],boot['eval_s_out']
    if args.fold_output:relu_s=boot['fold']['application_scale']
    normalization=None
    if args.normalize:
        # Integer multiplication is exact on residues; preserve the actual
        # resulting scale instead of overwriting it with nominal 2^40.
        k=math.ceil(2**(40+math.log2(q[relu_q-1])-relu_s))
        out_s=relu_s+math.log2(k)-math.log2(q[relu_q-1])
        normalization=dict(q_in=relu_q,q_out=relu_q-1,s_in=relu_s,s_out=out_s,integer=k)
        relu_q,relu_s=relu_q-1,out_s
    relu,checked=find_relu(q,relu_q,relu_s,args.seconds)
    report=dict(q_bottom_first=q,p=p,preparation=entry,bootstrap=boot,
                relu=relu,normalization=normalization,search_candidates=checked,precision_scope='PUBLIC_SCALE_METADATA_ONLY')
    if relu:
        after_relu=relu_q-relu['total']
        report['conv']=conv(q,after_relu)
        after_conv=after_relu-3
        report['after_conv_q']=after_conv
        try:
            report['next_preparation']=prepare(q,after_conv)
            report['loop_counts_match']=after_conv==6 and report['next_preparation']==entry
        except ValueError as error:
            report['failure']=str(error)
    Path(args.output).write_text(json.dumps(report,indent=2)+'\n')
    print('REPORT',args.output, 'after_conv_q=',report.get('after_conv_q'),
          'loop_counts_match=',report.get('loop_counts_match',False),flush=True)

if __name__=='__main__':main()
