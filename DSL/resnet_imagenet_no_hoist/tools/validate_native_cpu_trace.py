"""Audit native-driven CPU plan and lossless payload; not numerical/GPU acceptance."""
import argparse
from array import array
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import struct
import zlib

def f64(bits):return struct.unpack('<d',struct.pack('<Q',bits))[0]
def word(value):return struct.unpack('<Q',struct.pack('<d',value))[0]
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def validate(trace,model,fixture,report):
    tokens=fixture.read_text().split();assert tokens[0]=='RELU_PRECISION_V1'
    nq,np=map(int,tokens[1:3]);assert (nq,np)==(50,25)
    primes=list(map(int,tokens[3:3+nq]))
    qvalues=array('Q',[0]);scales=array('Q',[0]);components=array('B',[0])
    counts=Counter();stages=Counter();bootstrap_stages=Counter();writes=0;end=None
    payload=trace.with_name(trace.name+'.zlib');payload_end=0;payload_vectors=0
    with trace.open() as lines,payload.open('rb') as data:
        for number,line in enumerate(lines,1):
            r=json.loads(line);op=r['op'];counts[op]+=1
            if op=='outputs':
                assert end is None;end=r;continue
            assert end is None and r['id']==len(qvalues),(number,'nonmonotonic DAG ID')
            q=r['q'];s=r['scale_bits'];value=f64(s)
            assert 0<q<=50 and value>0 and math.isfinite(value),(number,'invalid metadata')
            stages[r['stage']]+=1
            arguments=r['inputs'];args=[]
            for a in arguments:
                i=a['id'];assert 0<i<len(qvalues),(number,'undefined dependency')
                assert a['q']==qvalues[i],(number,'unrecorded Q mutation')
                if a['scale_bits']!=scales[i]:
                    # Original ReLU makes explicit roundoff metadata assignments.
                    # Preserve these per-use snapshots for literal lowering.
                    ratio=f64(a['scale_bits'])/f64(scales[i])
                    assert math.isfinite(ratio) and abs(math.log2(ratio))<=1e-6,(number,'large metadata mutation')
                    writes+=1
                args.append((a['q'],a['scale_bits'],components[i]))
            if op in ('input','encode_vector'):
                assert not args and r['elements']<=32768
                assert r['payload_offset']==payload_end,(number,'payload gaps/overlap')
                data.seek(payload_end);header=data.read(16);assert len(header)==16
                elements,compressed_bytes=struct.unpack('<QQ',header)
                assert elements==r['elements'] and compressed_bytes==r['payload_compressed_bytes']
                assert compressed_bytes<=elements*8+65536
                compressed=data.read(compressed_bytes);assert len(compressed)==compressed_bytes
                decoder=zlib.decompressobj();raw=decoder.decompress(compressed,elements*8+1)
                assert len(raw)==elements*8 and decoder.eof and not decoder.unused_data
                payload_end=data.tell();payload_vectors+=1
            if op=='input':
                assert q==32 and s==word(2.0**40) and r['stage']=='stem.conv1_bn1' and r['elements']==32768
                component=2
            elif op=='encrypted_constant':
                assert not args and q==32 and s==word(2.0**40) and r['value_bits']==word(1.0)
                component=2
            elif op in ('encode_vector','encode_scalar'):
                if op=='encode_scalar':assert math.isfinite(f64(r['value_bits']))
                component=1
            elif op in ('multiply_integer','rotate','negate','relinearize'):
                assert len(args)==1 and (q,s)==args[0][:2]
                if op=='relinearize':assert args[0][2]==3
                else:assert args[0][2]==2
                component=2
            elif op=='set_scale':
                assert len(args)==1 and q==args[0][0];component=args[0][2]
            elif op=='drop':
                assert len(args)==1 and q<args[0][0] and s==args[0][1];component=args[0][2]
            elif op in ('add','sub','add_plain'):
                assert len(args)==2 and args[0][0]==args[1][0]==q and s==args[0][1]
                assert args[0][2]==2 and args[1][2]==(1 if op=='add_plain' else 2)
                assert abs(math.log2(f64(args[0][1])/f64(args[1][1])))<=1e-6
                component=2
            elif op in ('multiply_plain','multiply_raw'):
                assert len(args)==2 and args[0][0]==args[1][0]==q
                assert s==word(f64(args[0][1])*f64(args[1][1]))
                assert args[0][2]==2 and args[1][2]==(1 if op=='multiply_plain' else 2)
                component=2 if op=='multiply_plain' else 3
            elif op=='rescale_many':
                assert len(args)==1;before,bits,component=args[0];n=r['count']
                assert before-q==n and 0<n<before
                result=f64(bits)
                if n==2:result/=float(primes[before-1]*primes[before-2])
                else:
                    for j in range(n):result/=float(primes[before-1-j])
                assert s==word(result),(number,'physical rescale recurrence differs')
            elif op=='bootstrap_native_api':
                assert len(args)==1 and args[0][0]==6 and abs(math.log2(f64(args[0][1]))-40)<1e-8
                assert q==31 and s==word(2.0**40) and r['stage'].endswith('.bootstrap')
                bootstrap_stages[r['stage']]+=1;component=2
            else:raise AssertionError((number,'unsupported op',op))
            qvalues.append(q);scales.append(s);components.append(component)
    assert end and end['gpu_executed'] is False and len(end['values'])==1000
    assert len(set(end['values']))==1000
    for i in end['values']:assert 0<i<len(qvalues) and qvalues[i]==4 and components[i]==2
    assert counts['input']==end['inputs']==147
    assert counts['encrypted_constant']==end['encrypted_constants']==1
    assert end['bootstrap_calls']==counts['bootstrap_native_api']
    block_counts=(2,2,2,2) if model==18 else (3,4,6,3)
    expected={f'layer{stage}.{block}.act{act}.bootstrap'
              for stage,n in enumerate(block_counts,1) for block in range(n)
              for act in range(1,3 if model==18 else 4)}
    assert set(bootstrap_stages)==expected,(set(bootstrap_stages)^expected)
    assert len(bootstrap_stages)==(16 if model==18 else 48)
    if model==18:assert counts['bootstrap_native_api']==60
    assert payload_end==payload.stat().st_size,'trailing/unbound payload bytes'
    result={'status':'PASS','model':f'resnet{model}','scope':'CPU native-driven metadata/DAG/payload audit only',
            'gpu_executed':False,'numerical_semantics_validated':False,'dsl_compiled':False,
            'ops':len(qvalues)-1,'input_layouts':147,'encrypted_constant_one':1,'outputs':1000,
            'output_q':4,'bootstrap_positions':len(bootstrap_stages),
            'bootstrap_calls':counts['bootstrap_native_api'],'bootstrap_stage_counts':dict(bootstrap_stages),
            'implicit_roundoff_metadata_assignments':writes,'op_counts':dict(counts),
            'payload_vectors':payload_vectors,'payload_bytes':payload.stat().st_size,
            'trace_sha256':sha(trace),'payload_sha256':sha(payload),'stage_op_counts':dict(stages)}
    report.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('bootstrap_stage_counts','op_counts','stage_op_counts')}))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model',type=int,choices=[18,50]);parser.add_argument('trace',type=Path)
    parser.add_argument('fixture',type=Path);parser.add_argument('report',type=Path)
    args=parser.parse_args();validate(args.trace,args.model,args.fixture,args.report)
