"""CPU semantic replay of real 27-input compiler bytecode, ideal bootstraps only."""
import argparse
import collections
import json
from pathlib import Path
import struct
import numpy as np
from benchmark_utils import operation
from prepare_assets import digest


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('hevm','data_prefix','native_log','report'):
        parser.add_argument(name)
    args=parser.parse_args()
    metadata=json.loads(Path(args.data_prefix+'.data-report.json').read_text())
    assert digest(Path(args.data_prefix+'.cst'))==metadata['cst_sha256']
    assert digest(Path(args.data_prefix+'.inputs.f64'))==metadata['input_sha256']
    data=Path(args.hevm).read_bytes()
    _,header,nargs,nresults=struct.unpack_from('<IIQQ',data)
    body,nops,_,_,_=struct.unpack_from('<5Q',data,header)
    assert nargs==27 and nresults==1
    output,=struct.unpack_from('<Q',data,header+40+(2*nargs+2*nresults)*8)
    raw=[struct.unpack_from('<4H',data,header+body+8*i) for i in range(nops)]
    cst=np.memmap(args.data_prefix+'.cst',dtype=np.uint8,mode='r')
    count,=struct.unpack_from('<q',cst)
    offsets=[];cursor=8
    for _ in range(count):
        length,=struct.unpack_from('<q',cst,cursor);cursor+=8
        offsets.append((cursor,length));cursor+=8*length
    assert cursor==len(cst)
    initial=np.memmap(args.data_prefix+'.inputs.f64',dtype='<f8',mode='r').reshape(27,32768)
    ct={i:initial[i].astype(np.longdouble) for i in range(27)};pt={}
    uses=collections.Counter()
    for opcode,_,lhs,rhs in raw:
        if opcode not in (65535,0):
            uses[lhs]+=1
            if opcode in (6,8):uses[rhs]+=1
    uses[output]+=1;boots=0
    for opcode,dst,lhs,rhs in raw:
        if opcode==65535:continue
        if opcode==0:
            offset,length=offsets[lhs]
            pt[dst]=np.frombuffer(cst,dtype='<f8',count=length,offset=offset)
            continue
        right=ct[rhs] if opcode in (6,8) else pt[rhs] if opcode in (7,9) else rhs
        out=operation(opcode,ct[lhs],right)
        assert np.all(np.isfinite(out))
        ct[dst]=out
        boots+=opcode==10
        consumed=[lhs]+([rhs] if opcode in (6,8) else [])
        for source in consumed:
            uses[source]-=1
            if uses[source]==0:del ct[source]
    line=next(row for row in Path(args.native_log).read_text().splitlines() if row.startswith('NETWORK_REFERENCE_LOGITS '))
    reference=np.array([float(v) for v in line.split()[1:]],dtype=np.longdouble)
    logits=ct[output][:10];error=float(np.max(np.abs(logits-reference)))
    assert boots==18 and error<1e-8,error
    result=dict(status='PASS',scope='real compiler HEVM CPU semantics with ideal bootstraps, not GPU acceptance',
                raw_ops=nops,inputs=27,bootstrap_count=boots,max_logit_error=error,
                prediction=int(np.argmax(logits)),reference_prediction=int(np.argmax(reference)),
                logits=[float(v) for v in logits])
    Path(args.report).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
