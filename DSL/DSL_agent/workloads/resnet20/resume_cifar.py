"""Single separately approved GET; combine a verified prefix and response bytes."""
import hashlib,json,re,time
from pathlib import Path
from public_assets import CHUNK,CIFAR_MD5,ChunkReader,check,digest,dump,store_stream,tls_context

PREFIX=Path('/home/lhohy/poseidon-work/platforms/aarch64-linux/results/agent-resnet20-assets-recovery-r1759/downloads/cifar10-binary')
PREFIX_BYTES=38666240
PREFIX_SHA='c80313d971d5fd1bc62daa52770f37eae9e4f9e3f8d3d3bf5c7ce4e5b4cd0fbd'
URL='https://cave.cs.toronto.edu/kriz/cifar-10-binary.tar.gz'
MAX_BYTES=192*1024**2

def specification():
    path=PREFIX/'incomplete-stream.json';raw=path.read_bytes();value=json.loads(raw)
    check(value['format']=='poseidon-incomplete-byte-stream-v1'and value['complete']is False,'Retained incomplete archive prefix')
    check(value['bytes_returned_by_stream']==PREFIX_BYTES and value['sha256']==PREFIX_SHA,'Fixed retained prefix')
    spec=dict(format='poseidon-cifar-single-range-resume-v1',method='GET',url=URL,
        headers={'Accept':'application/octet-stream','Accept-Encoding':'identity',
            'User-Agent':'Poseidon-ResNet20-Validation/1','Range':'bytes=%d-'%PREFIX_BYTES},
        prefix_directory=str(PREFIX),prefix_manifest_sha256=hashlib.sha256(raw).hexdigest(),
        prefix_bytes=PREFIX_BYTES,prefix_sha256=PREFIX_SHA,prefix_parts=value['parts'],
        max_http_gets=1,retries=0,redirects=0,max_seconds=3600,socket_seconds=30,
        max_archive_bytes=MAX_BYTES,official_archive_md5=CIFAR_MD5,
        max_new_artifact_bytes=512*1024**2,min_free_disk_bytes=4*1024**3,
        accepted_statuses=[206,200],http_206='Append response to verified prefix',
        http_200='Use complete same response without prefix; no second request',
        credentials_used=False,project_data_sent=False,downloaded_code_executed=False)
    spec['binding']=digest(spec);return spec

def prefix_reader(directory,value):
    check(value.get('format')=='poseidon-incomplete-byte-stream-v1'and value.get('complete')is False,'Incomplete prefix manifest')
    count=value.get('bytes_returned_by_stream')
    check(type(count)is int and 0<count<=MAX_BYTES,'Prefix byte count')
    # This verifies a complete byte prefix, not a complete CIFAR archive.
    checked=dict(format='poseidon-byte-chunks-v1',bytes=count,sha256=value['sha256'],md5=value['md5'],parts=value['parts'])
    return ChunkReader(directory,checked)

def response_mode(status,content_range,length,encoding,start):
    check(encoding in (None,'identity'),'Unexpected content encoding')
    if length is not None:check(type(length)is str and length.isdigit()and 0<int(length)<=MAX_BYTES,'Response byte length')
    if status==206:
        check(type(content_range)is str and len(content_range)<=100,'Missing Content-Range')
        match=re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)',content_range)
        check(match is not None,'Invalid Content-Range')
        first,last,total=map(int,match.groups())
        check(first==start and last==total-1 and 0<start<total<=MAX_BYTES,'Changed Content-Range')
        check(length is None or int(length)==total-start,'Range response size')
        return 'prefix-plus-range',total
    check(status==200 and content_range is None,'HTTP resume status')
    return 'complete-response',None if length is None else int(length)

class JoinedReader:
    def __init__(self,prefix,response):self.prefix=prefix;self.response=response
    def read(self,size):
        if self.prefix is not None:
            data=self.prefix.read(size)
            if data:return data
            self.prefix=None
        return self.response.read(size)

def fetch(spec,approval,output,*,checkpoint):
    check(spec==specification(),'Changed resume specification')
    check(type(approval)is dict and approval.get('format')=='poseidon-explicit-cifar-resume-approval-v1'and approval.get('status')=='approved'and approval.get('proposal_binding')==spec['binding']and approval.get('allowed_url')==URL and approval.get('allowed_headers')==spec['headers']and approval.get('max_http_gets')==1 and approval.get('retries')==0 and approval.get('redirects')==0,'Explicit matching resume approval required')
    import http.client
    from urllib.parse import urlsplit
    checkpoint();output=Path(output);output.mkdir(mode=0o700)
    raw=(PREFIX/'incomplete-stream.json').read_bytes()
    check(hashlib.sha256(raw).hexdigest()==spec['prefix_manifest_sha256'],'Changed prefix manifest')
    prefix=prefix_reader(PREFIX,json.loads(raw))
    dump(output/'request.json',dict(method='GET',url=URL,headers=spec['headers'],request_index=0,credentials_used=False,project_data_sent=False))
    tick=time.monotonic();url=urlsplit(URL)
    connection=http.client.HTTPSConnection(url.hostname,timeout=30,context=tls_context())
    try:
        connection.request('GET',url.path,headers=spec['headers'])
        response=connection.getresponse()
        content_range=response.getheader('Content-Range');length=response.getheader('Content-Length');encoding=response.getheader('Content-Encoding')
        mode,total=response_mode(response.status,content_range,length,encoding,spec['prefix_bytes'])
        dump(output/'response.json',dict(status=response.status,content_range=content_range,content_length=length,content_encoding=encoding,mode=mode,total_bytes=total,elapsed_seconds=time.monotonic()-tick))
        class Bounded:
            received=0
            def read(self,size):
                remaining=spec['max_seconds']-(time.monotonic()-tick)
                check(remaining>0,'Resume HTTP wall deadline')
                sock=connection.sock
                if sock is None:sock=getattr(getattr(response.fp,'raw',None),'_sock',None)
                if sock is not None:sock.settimeout(min(remaining,30))
                data=response.read1(size);self.received+=len(data)
                check(self.received<=MAX_BYTES,'HTTP byte budget')
                return data
        bounded=Bounded()
        joined=JoinedReader(prefix if mode=='prefix-plus-range'else None,bounded)
        archive=store_stream(joined,output/'archive',MAX_BYTES,expected_md5=CIFAR_MD5,checkpoint=checkpoint)
        check(total is None or archive['bytes']==total,'Complete archive size')
        result=dict(format='poseidon-cifar-range-download-v1',spec_binding=spec['binding'],
            mode=mode,http_gets=1,retries=0,redirects=0,network_bytes=bounded.received,
            reused_prefix_bytes=spec['prefix_bytes']if mode=='prefix-plus-range'else 0,
            archive=archive,official_md5_verified=True,downloaded_code_executed=False)
        dump(output/'download.json',result);return result
    finally:connection.close()
