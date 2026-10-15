"""Replay native-driven slots with ideal bootstrap/relinearize/rescale semantics.

Preserves integer-scaled stem weights and per-use metadata interpretation.
This is not a noise simulator, compiler gate, or encrypted precision result.
"""
import argparse
from array import array
import json
from pathlib import Path
import struct
import zlib
import numpy as np
from validate_native_cpu_trace import f64,sha

def replay(trace,oracle,report):
    uses=array('I',[0]);outputs=None;metadata=array('Q',[0])
    with trace.open() as stream:
        for line in stream:
            r=json.loads(line)
            if r['op']=='outputs':outputs=r['values'];continue
            assert r['id']==len(uses);uses.append(0);metadata.append(r['scale_bits'])
            for a in r['inputs']:uses[a['id']]+=1
    assert outputs and len(outputs)==1000
    output_index={v:i for i,v in enumerate(outputs)}
    logits=np.empty(1000,dtype=np.float64);values={};peak=0;boots=0
    payload=trace.with_name(trace.name+'.zlib')
    with trace.open() as stream,payload.open('rb') as constants:
        for line in stream:
            r=json.loads(line);op=r['op']
            if op=='outputs':continue
            args=[]
            for a in r['inputs']:
                value=values[a['id']]
                if a['scale_bits']!=metadata[a['id']]:
                    value=value*(f64(metadata[a['id']])/f64(a['scale_bits']))
                args.append(value)
            if op in ('input','encode_vector'):
                constants.seek(r['payload_offset']);elements,size=struct.unpack('<QQ',constants.read(16))
                assert elements==r['elements'] and size==r['payload_compressed_bytes']
                raw=zlib.decompress(constants.read(size));assert len(raw)==elements*8
                value=np.frombuffer(raw,dtype='<f8')
                if elements!=32768:
                    # Poseidon scalar/vector encoding pads short vectors with zeros.
                    value=np.pad(value,(0,32768-elements))
            elif op in ('encrypted_constant','encode_scalar'):
                value=np.full(32768,f64(r['value_bits']))
            elif op=='rotate':value=np.roll(args[0],-r['step'])
            elif op=='negate':value=-args[0]
            elif op=='sub':
                ratio=f64(r['inputs'][1]['scale_bits'])/f64(r['scale_bits'])
                value=args[0]-args[1]*ratio
            elif op in ('add','add_plain'):
                ratio=f64(r['inputs'][1]['scale_bits'])/f64(r['scale_bits'])
                value=args[0]+args[1]*ratio
            elif op in ('multiply_plain','multiply_raw'):value=args[0]*args[1]
            elif op=='multiply_integer':value=args[0]*r['integer']
            elif op=='set_scale':value=args[0]*(f64(r['inputs'][0]['scale_bits'])/f64(r['scale_bits']))
            elif op in ('drop','rescale_many','relinearize'):value=args[0]
            elif op=='bootstrap_native_api':value=args[0];boots+=1
            else:raise ValueError(op)
            assert np.all(np.isfinite(value)),(r['id'],op,r['stage'])
            if r['id'] in output_index:logits[output_index[r['id']]]=value[0]
            if uses[r['id']]:values[r['id']]=value
            for a in r['inputs']:
                source=a['id'];uses[source]-=1
                if not uses[source]:del values[source]
            peak=max(peak,len(values))
    assert not values and not any(uses)
    reference=np.fromfile(oracle,dtype='<f8');assert len(reference)==1000
    difference=logits-reference
    result={'status':'COMPLETE','scope':'native-driven CPU slots, ideal bootstrap/no HE noise; not compiler or GPU acceptance',
            'prediction':int(np.argmax(logits)),'reference_prediction':int(np.argmax(reference)),
            'classification_matches_reference':bool(np.argmax(logits)==np.argmax(reference)),
            'max_logit_error':float(np.max(np.abs(difference))),'rms_logit_error':float(np.sqrt(np.mean(difference*difference))),
            'bootstrap_calls':boots,'peak_live_values':peak,'logits':[float(v) for v in logits],
            'trace_sha256':sha(trace),'payload_sha256':sha(payload),'oracle_sha256':sha(oracle),
            'gpu_executed':False,'dsl_compiled':False}
    report.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='logits'}))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trace',type=Path);parser.add_argument('oracle',type=Path);parser.add_argument('report',type=Path)
    args=parser.parse_args();replay(args.trace,args.oracle,args.report)
