"""Audit literal registers/order and replay real V2 HEVM slots (ideal bootstrap)."""
import argparse, hashlib, json, math, struct, zlib
from pathlib import Path
import numpy as np

OPCODES={'encode':0,'rotatec':1,'negatec':2,'rescalec':3,'modswitchc':4,
         'addcc':6,'addcp':7,'mulcp':9,'bootstrapc':10,'setscalec':13,
         'multiplyscalarc':14,'rawmulcc':15,'relinearizec':16,'rescalemanyc':17}
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()
def f64(bits):return struct.unpack('<d',struct.pack('<Q',int(bits)))[0]
def read(hevm):
    with hevm.open('rb') as stream:
        magic,header,argc,results=struct.unpack('<IIQQ',stream.read(24))
        assert (magic,header,argc,results)==(0x48455632,24,148,1000)
        body,nops,nc,np_,level=struct.unpack('<QQQQQ',stream.read(40))
        assert body==40+8*(148*2+1000*3) and level==49
        metadata=np.frombuffer(stream.read(body-40),dtype='<u8')
    assert np.all(metadata[:148]==40) and np.all(metadata[148:296]==31)
    assert np.all(metadata[296:1296]==40) and np.all(metadata[1296:2296]==3)
    assert hevm.stat().st_size==24+body+16*nops
    ops=np.memmap(hevm,dtype='<u4',mode='r',offset=24+body,shape=(nops,4))
    return ops,int(nc),int(np_),metadata[-1000:]
def audit(hevm,prefix,report,native_trace=None):
    ops,nc,np_,returns=read(hevm);lower=json.loads(prefix.with_suffix('.lowering_report.json').read_text())
    rows=0;output_ids=None;output_regs={}
    if native_trace:
        with native_trace.open('rb') as stream:
            stream.seek(max(0,native_trace.stat().st_size-65536))
            last=json.loads(stream.read().splitlines()[-1])
        assert last['op']=='outputs';output_ids=last['values'];assert len(output_ids)==1000
        assert sha(native_trace)==lower['source_sha256']
        wanted=set(output_ids)
    with prefix.with_suffix('.native_trace_map.jsonl').open() as stream:
        for line in stream:
            row=json.loads(line);index=row['hevm_instruction_index'];code=OPCODES[row['op']]
            expected=[code,row['raw_dst'],0,0];args=row['operands']
            if code==0:expected[2:]=[row['constant_index'],row['parameter']]
            else:
                expected[2]=args[0]['register']
                if code in (6,7,9,15):expected[3]=args[1]['register']
                elif code in (1,4,10,14,17):expected[3]=row['parameter']&0xffffffff
            assert index==2*rows+1 and int(ops[index-1,0])==0xffffffff,('instruction order',rows)
            assert [int(x) for x in ops[index]]==expected,('opcode/register/args',rows,expected,ops[index])
            if output_ids is not None and row['source_id'] in wanted:output_regs[row['source_id']]=row['raw_dst']
            rows+=1
    assert rows*2==len(ops)==lower['compiler_instructions']
    assert nc==lower['cipher_registers'] and np_==lower['plain_registers']
    if output_ids is not None:assert list(map(int,returns))==[output_regs[i] for i in output_ids], 'return register/order'
    result=dict(status='PASS',scope='actual compiler opcode/register/argument/order audit; not GPU',
        hevm_sha256=sha(hevm),trace_map_sha256=sha(prefix.with_suffix('.native_trace_map.jsonl')),
        source_sha256=lower['source_sha256'],operations=len(ops),mapped_operations=rows,args=148,outputs=1000,
        output_order_bound_to_native=output_ids is not None,gpu_executed=False)
    assert result['trace_map_sha256']==lower['trace_map_sha256']
    report.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)

def replay(hevm,prefix,inputs,payload_path,oracle,report,q):
    ops,nc,np_,returns=read(hevm);codes=np.asarray(ops[:,0]);cu=np.zeros(nc,dtype=np.uint32);pu=np.zeros(np_,dtype=np.uint32)
    for code in OPCODES.values():
        if code==0:continue
        selected=ops[codes==code]
        np.add.at(cu,selected[:,2],1)
        if code in (6,15):np.add.at(cu,selected[:,3],1)
        if code in (7,9):np.add.at(pu,selected[:,3],1)
    output_at={}
    for index,r in enumerate(returns):output_at.setdefault(int(r),[]).append(index)
    scales=np.zeros(nc);ps=np.zeros(np_);levels=np.zeros(nc,dtype=np.uint8)
    literals={}
    for suffix in ('.native_scales','.native_primitives'):
        with Path(str(hevm)+suffix).open() as stream:
            next(stream)
            for line in stream:
                row=line.split();literals[int(row[0])]=f64(int(row[-1],16))
    constants=[json.loads(s) for s in prefix.with_suffix('.native_constants.jsonl').open()]
    ins=[json.loads(s) for s in inputs.open()][1:];c={};p={};logits=np.empty(1000);boots=0;peak=0;actions=0
    with payload_path.open('rb') as stream:
        def vector(row):
            stream.seek(row['payload_offset']);n,size=struct.unpack('<QQ',stream.read(16))
            assert (n,size)==(row['elements'],row['payload_compressed_bytes'])
            data=zlib.decompress(stream.read(size));assert len(data)==n*8
            return np.pad(np.frombuffer(data,dtype='<f8'),(0,32768-n))
        for i,row in enumerate(ins):
            assert row['register']==i and row['q']==32 and int(row['scale_bits'],16)==0x4270000000000000
            if cu[i]:c[i]=vector(row) if row['kind']=='layout' else np.full(32768,1.)
            scales[i]=2.**40;levels[i]=32
        for raw in range(len(ops)):
            code,dst,lhs,rhs=map(int,ops[raw])
            if code==0xffffffff:continue
            args=[]
            if code==0:
                k=constants[lhs];assert k['scale_bits']==struct.unpack('<Q',struct.pack('<d',literals[raw]))[0]
                value=vector(k) if k['op']=='encode_vector' else np.full(32768,f64(k['value_bits']))
                ps[dst]=literals[raw]
                if pu[dst]:p[dst]=value
            else:
                a=c[lhs];s=scales[lhs];level=int(levels[lhs]);args=[(False,lhs)]
                if code==1:value=np.roll(a,-((rhs^0x80000000)-0x80000000))
                elif code==2:value=-a
                elif code in (3,17):
                    n=1 if code==3 else rhs
                    if n==2:s/=float(int(q[level-1])*int(q[level-2]))
                    else:
                        for j in range(n):s/=float(q[level-1-j])
                    level-=n;value=a
                elif code==4:level-=rhs;value=a
                elif code in (6,7,9,15):
                    plain=code in (7,9);b=(p if plain else c)[rhs];bs=(ps if plain else scales)[rhs];args.append((plain,rhs))
                    if code in (6,7):value=a+b*(bs/s)
                    else:value=a*b;s*=bs
                elif code==10:
                    assert level==6 and abs(math.log2(s)-40)<1e-8
                    value=a;level=31;s=2.**40;boots+=1
                elif code==13:value=a*(s/literals[raw]);s=literals[raw]
                elif code==14:value=a*rhs
                elif code==16:value=a
                else:raise ValueError(('unsupported opcode',code,raw))
                assert np.all(np.isfinite(value)),('nonfinite',raw,code)
                scales[dst]=s;levels[dst]=level
                if dst in output_at:
                    assert level==4 and abs(math.log2(s)-40)<1e-8
                    for index in output_at[dst]:logits[index]=value[0]
                if cu[dst]:c[dst]=value
            for plain,r in args:
                uses=pu if plain else cu;values=p if plain else c;assert uses[r]>0
                uses[r]-=1
                if not uses[r]:del values[r]
            actions+=1;peak=max(peak,len(c)+len(p))
            if actions%100000==0:print('COMPILED_CPU_PROGRESS',actions,'boots',boots,flush=True)
    assert not c and not p and not np.any(cu) and not np.any(pu)
    reference=np.fromfile(oracle,dtype='<f8');assert len(reference)==1000
    error=logits-reference;maximum=float(np.max(np.abs(error)))
    result=dict(status='PASS' if maximum<=1e-6 else 'FAIL',scope='real compiler HEVM CPU slots; ideal bootstrap/no HE noise; not GPU acceptance',
        prediction=int(np.argmax(logits)),reference_prediction=int(np.argmax(reference)),label=65,
        max_logit_error=maximum,rms_logit_error=float(np.sqrt(np.mean(error**2))),logits=logits.tolist(),
        bootstrap_calls=boots,peak_live_values=peak,hevm_sha256=sha(hevm),oracle_sha256=sha(oracle),gpu_executed=False)
    report.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='logits'}),flush=True)
    assert result['status']=='PASS'

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('hevm',type=Path);p.add_argument('prefix',type=Path);p.add_argument('report',type=Path)
    p.add_argument('--replay',action='store_true');p.add_argument('--inputs',type=Path);p.add_argument('--payload',type=Path);p.add_argument('--oracle',type=Path);p.add_argument('--config-source',type=Path);p.add_argument('--native-trace',type=Path)
    a=p.parse_args()
    if not a.replay:audit(a.hevm,a.prefix,a.report,a.native_trace)
    else:
        import re
        source=a.config_source.read_text();moduli=re.search(r's2c_first_q_moduli\(\).*?return\s*\{(.*?)\}',source,re.S)[1]
        q=[int(v) for v in re.findall(r'(\d+)ULL',moduli)];assert len(q)==50
        replay(a.hevm,a.prefix,a.inputs,a.payload,a.oracle,a.report,q)
