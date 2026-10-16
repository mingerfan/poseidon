"""Check the retained native Q4 RNS capture; DSL word equality was checked live."""
import argparse,hashlib,json,struct,math
from pathlib import Path
import numpy as np

def audit(path,output):
    size=path.stat().st_size;words=2*4*65536;record_bytes=88+words*8
    assert size==1000*record_bytes,(size,record_bytes)
    digest=hashlib.sha256();parms=None;scales=set()
    moduli=np.array([4255252481,4255645697,4258922497,4260364289],dtype=np.uint64)
    with path.open('rb') as stream:
        for index in range(1000):
            raw_header=stream.read(88);digest.update(raw_header);header=struct.unpack('<11Q',raw_header)
            assert header[:4]==(65536,4,0,2) and header[5:7]==(1,1),(index,header)
            # Header logscale=40 is rounded; retain the natural physical f64.
            scale=struct.unpack('<d',struct.pack('<Q',header[4]))[0]
            assert math.isfinite(scale) and abs(math.log2(scale)-40)<1e-8,(index,header[4],scale)
            scales.add(header[4])
            if parms is None:parms=header[7:]
            assert parms==header[7:]
            raw=stream.read(words*8);assert len(raw)==words*8;digest.update(raw)
            data=np.frombuffer(raw,dtype='<u8').reshape(2,4,65536)
            assert np.all(data<moduli[None,:,None]),('RNS residue outside native Q modulus',index)
        assert not stream.read(1)
    result=dict(status='PASS',scope='native capture integrity/ranges; live runtime separately compared DSL against every retained word',
        records=1000,N=65536,Q=4,components=2,scale_bits=sorted(f'{v:016x}' for v in scales),
        physical_log2_scales=[math.log2(struct.unpack('<d',struct.pack('<Q',v))[0]) for v in sorted(scales)],
        rns_words=words*1000,bytes=size,sha256=digest.hexdigest(),parms_id=list(parms))
    output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('capture',type=Path);p.add_argument('report',type=Path);a=p.parse_args();audit(a.capture,a.report)
