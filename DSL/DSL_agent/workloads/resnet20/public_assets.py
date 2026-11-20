"""Fixed public ResNet20/CIFAR-10 acquisition and lossless bounded storage.

This B-owned data adapter adds no model lowering or backend capability. Import
is offline. Downloads require a separately recorded exact-spec approval.
"""
import hashlib,io,json,time,tarfile
from pathlib import Path

CHUNK=8*1024**2
ROW=3073
REVISION='d5489e8995e81e91ce6b1d69dcc98ad579b0b153'
RAW='https://raw.githubusercontent.com/akamaster/pytorch_resnet_cifar10/'+REVISION+'/'
CIFAR_MD5='c32a1d4ab5d03f1284b67883e8d87530'

def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def check(ok,message):
    if not ok:raise ValueError(message)
def dump(path,value):
    with Path(path).open('x')as f:json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
def specification():
    assets=[dict(id='cifar10-binary',url='https://cave.cs.toronto.edu/kriz/cifar-10-binary.tar.gz',max_bytes=192*1024**2,md5=CIFAR_MD5),dict(id='resnet20-weights',url=RAW+'pretrained_models/resnet20-12fca82f.th',max_bytes=2*1024**2),dict(id='resnet-source',url=RAW+'resnet.py',max_bytes=64*1024),dict(id='trainer-source',url=RAW+'trainer.py',max_bytes=64*1024),dict(id='license',url=RAW+'LICENSE',max_bytes=16*1024)]
    value=dict(format='poseidon-public-resnet20-assets-v1',source_commit=REVISION,assets=assets,max_http_gets=5,retries=0,redirects=0,max_seconds_per_asset=120,wall_seconds=660,max_chunk_bytes=CHUNK,max_new_artifact_bytes=512*1024**2,min_free_disk_bytes=4*1024**3,asset_requests_include_project_data=False,credential_reads=0,executed_downloaded_code=False)
    value['binding']=digest(value);return value

def recovery_specification():
    """Separate proposal: small files first; one new 600-second archive attempt."""
    value=specification();value.pop('binding')
    archive=dict(value['assets'][0],max_seconds=600)
    value.update(format='poseidon-public-resnet20-assets-recovery-v2',
        assets=value['assets'][1:]+[archive],wall_seconds=1080,
        preceding_failed_batch='resnet20-public-assets-r1754',
        automatically_authorized=False)
    value['binding']=digest(value);return value

def store_stream(stream,directory,max_bytes,*,expected_md5=None,checkpoint):
    """Preserve returned bytes in bounded parts, including an interrupted tail."""
    check(type(max_bytes)is int and 0<max_bytes<=192*1024**2,'Stream byte limit')
    check(callable(checkpoint),'Checkpoint required')
    directory=Path(directory);directory.mkdir(mode=0o700)
    sha=hashlib.sha256();md5=hashlib.md5();total=0;parts=[];buffer=bytearray()
    def flush(*,check_after=True):
        nonlocal buffer
        name='part-%05d.bin'%len(parts)
        with(directory/name).open('xb')as f:f.write(buffer)
        parts.append(dict(name=name,bytes=len(buffer),sha256=hashlib.sha256(buffer).hexdigest()))
        buffer=bytearray()
        if check_after:checkpoint()
    try:
        while True:
            count=min(65536,CHUNK-len(buffer),max_bytes-total+1)
            raw=stream.read(count)
            check(type(raw)is bytes and len(raw)<=count,'Invalid bounded stream')
            if not raw:break
            total+=len(raw);sha.update(raw);md5.update(raw);buffer.extend(raw)
            check(total<=max_bytes,'Download byte budget')
            if len(buffer)==CHUNK:flush()
        if buffer:flush()
        check(total>0,'Empty public asset')
        result=dict(format='poseidon-byte-chunks-v1',bytes=total,sha256=sha.hexdigest(),md5=md5.hexdigest(),parts=parts)
        check(expected_md5 is None or result['md5']==expected_md5,'Public asset checksum mismatch')
        dump(directory/'stream.json',result)
        return result
    except BaseException as error:
        # A checkpoint/deadline failure must not discard the already returned
        # sub-chunk. This record cannot be consumed as a verified complete asset.
        if buffer:flush(check_after=False)
        dump(directory/'incomplete-stream.json',dict(format='poseidon-incomplete-byte-stream-v1',complete=False,bytes_returned_by_stream=total,sha256=sha.hexdigest(),md5=md5.hexdigest(),parts=parts,error_type=type(error).__name__,expected_md5=expected_md5,retry_authorized=False))
        raise

class ChunkReader(io.RawIOBase):
    def __init__(self,directory,manifest):
        super().__init__();self.directory=Path(directory);self.manifest=manifest;self.index=0;self.offset=0;self.data=b''
        check(manifest.get('format')=='poseidon-byte-chunks-v1','Chunk manifest')
        sha=hashlib.sha256();md5=hashlib.md5();total=0
        for i,part in enumerate(manifest['parts']):
            check(part['name']=='part-%05d.bin'%i and type(part['bytes'])is int and 0<part['bytes']<=CHUNK,'Chunk binding')
            p=self.directory/part['name'];check(p.is_file()and not p.is_symlink(),'Indirect chunk')
            check(p.stat().st_size==part['bytes'],'Chunk size')
            data=p.read_bytes();check(hashlib.sha256(data).hexdigest()==part['sha256'],'Chunk digest')
            sha.update(data);md5.update(data);total+=len(data)
        check(total==manifest['bytes']and sha.hexdigest()==manifest['sha256']and md5.hexdigest()==manifest['md5'],'Stream digest')
    def readable(self):return True
    def read(self,size=-1):
        check(type(size)is int and 0<=size<=CHUNK,'Bounded chunk read required')
        output=bytearray()
        while len(output)<size:
            if self.offset==len(self.data):
                if self.index==len(self.manifest['parts']):break
                p=self.manifest['parts'][self.index];self.data=(self.directory/p['name']).read_bytes();self.index+=1;self.offset=0
            count=min(size-len(output),len(self.data)-self.offset)
            output.extend(self.data[self.offset:self.offset+count]);self.offset+=count
        return bytes(output)
    def readinto(self,target):
        data=self.read(len(target));target[:len(data)]=data;return len(data)

def prepare_binary_records(directory,manifest,output,*,checkpoint):
    """Stream fixed official rows without extracting tar paths or unpickling."""
    check(manifest.get('md5')==CIFAR_MD5,'Only approved official CIFAR archive')
    reader=ChunkReader(directory,manifest);output=Path(output);output.mkdir(mode=0o700)
    datasets={'data_batch_%d.bin'%i:('train',(i-1)*10000)for i in range(1,6)}
    datasets['test_batch.bin']=('test',0)
    metadata={'batches.meta.txt','readme.html'};seen=set();records=[];counts=dict(train=[0]*10,test=[0]*10)
    with tarfile.open(fileobj=reader,mode='r|gz')as archive:
        members=0
        for member in archive:
            members+=1;check(members<=10,'Archive member budget')
            if member.isdir():
                check(member.name.rstrip('/')=='cifar-10-batches-bin','Unexpected archive directory');continue
            check(member.isfile(),'Archive links and special members forbidden')
            path=Path(member.name)
            check(len(path.parts)==2 and path.parts[0]=='cifar-10-batches-bin'and path.parts[1]in set(datasets)|metadata,'Unexpected archive member')
            name=path.parts[1];check(name not in seen,'Duplicate archive member');seen.add(name)
            stream=archive.extractfile(member);check(stream is not None,'Missing archive member')
            if name in metadata:
                check(0<=member.size<=65536,'Metadata byte budget')
                raw=stream.read(65537);check(len(raw)==member.size,'Metadata size')
                with(output/name).open('xb')as f:f.write(raw)
                continue
            check(member.size==10000*ROW,'Official CIFAR member size')
            split,base=datasets[name];sha=hashlib.sha256()
            for start in range(0,10000,1000):
                raw=stream.read(1000*ROW);check(len(raw)==1000*ROW,'Incomplete CIFAR rows')
                labels=raw[::ROW];check(all(label<10 for label in labels),'Invalid CIFAR class')
                for label in labels:counts[split][label]+=1
                sha.update(raw);target='%s-%05d.records'%(split,base+start)
                with(output/target).open('xb')as f:f.write(raw)
                records.append(dict(file=target,split=split,start=base+start,count=1000,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()));checkpoint()
            check(stream.read(1)==b'','Extra CIFAR rows')
    check(set(datasets)<=seen and counts['train']==[5000]*10 and counts['test']==[1000]*10,'Incomplete official CIFAR dataset')
    result=dict(format='poseidon-cifar10-binary-records-v1',source_sha256=manifest['sha256'],source_md5=manifest['md5'],record_bytes=ROW,pixel_layout='uint8 planar RGB 3x32x32, C order',row_label_byte=0,train_count=50000,test_count=10000,class_counts=counts,shards=records,lossless_original_rows=True,preprocessing_applied=False)
    dump(output/'dataset.json',result);return result

def tls_context():
    import ssl
    context=ssl.create_default_context()
    bundle=Path('/etc/ssl/certs/ca-certificates.crt')
    if not context.get_ca_certs() and bundle.is_file():context.load_verify_locations(cafile=str(bundle))
    check(bool(context.get_ca_certs())and context.check_hostname and context.verify_mode==ssl.CERT_REQUIRED,'Installed TLS trust unavailable')
    return context

def download_assets(spec,output,approval,*,checkpoint):
    """One fixed GET per asset, no credentials, redirect or retry. Not automatic."""
    check(spec in (specification(),recovery_specification()),'Changed fixed asset specification')
    check(type(approval)is dict and approval.get('format')=='poseidon-explicit-public-asset-approval-v1'and approval.get('status')=='approved'and approval.get('proposal_binding')==spec['binding']and approval.get('allowed_urls')==[a['url']for a in spec['assets']],'Public asset download requires explicit matching approval')
    check(callable(checkpoint),'Checkpoint required')
    import http.client,ssl
    from urllib.parse import urlsplit
    output=Path(output);output.mkdir(mode=0o700);completed=[];started=time.monotonic()
    for i,asset in enumerate(spec['assets']):
        checkpoint();check(time.monotonic()-started<spec['wall_seconds'],'Asset batch wall budget')
        url=urlsplit(asset['url']);check(url.scheme=='https'and not url.query and not url.fragment,'Fixed HTTPS asset URL')
        state=dict(format='poseidon-public-asset-request-v1',asset_id=asset['id'],method='GET',url=asset['url'],request_index=i,credential_used=False,project_data_sent=False,max_bytes=asset['max_bytes'],retries=0,redirects=0)
        dump(output/('request-%02d.json'%i),state)
        seconds=asset.get('max_seconds',spec['max_seconds_per_asset'])
        connection=http.client.HTTPSConnection(url.hostname,timeout=seconds,context=tls_context())
        tick=time.monotonic()
        try:
            connection.request('GET',url.path,headers={'Accept':'application/octet-stream','User-Agent':'Poseidon-ResNet20-Validation/1'})
            response=connection.getresponse()
            dump(output/('response-%02d.json'%i),dict(format='poseidon-public-asset-http-response-v1',request_index=i,status=response.status,elapsed_seconds=time.monotonic()-tick))
            check(response.status==200,'Public asset HTTP status')
            length=response.getheader('Content-Length')
            if length is not None:check(length.isdigit()and 0<int(length)<=asset['max_bytes'],'Public asset Content-Length')
            class Bounded:
                def read(self,n):
                    remaining=seconds-(time.monotonic()-tick)
                    check(remaining>0,'Public asset wall deadline')
                    if connection.sock is not None:connection.sock.settimeout(min(remaining,30))
                    return response.read(n)
            value=store_stream(Bounded(),output/asset['id'],asset['max_bytes'],expected_md5=asset.get('md5'),checkpoint=checkpoint)
            if length is not None:check(value['bytes']==int(length),'Public asset response size')
            completed.append(dict(asset=asset,stream=value))
            dump(output/('completed-%02d.json'%i),completed[-1]);checkpoint()
        finally:connection.close()
    result=dict(format='poseidon-public-resnet20-download-v1',spec_binding=spec['binding'],assets=completed,http_gets=len(completed),retries=0,redirects=0,credentials_used=False,executed_downloaded_code=False)
    dump(output/'downloads.json',result);return result
