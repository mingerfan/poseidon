"""Run the unchanged CPU evidence auditor in one bounded, offline worker.

This moves the existing verification work out of the orchestration process.
It never generates, compiles, executes a Runtime plan or changes old evidence.
"""
import hashlib
import json
import os
from pathlib import Path
import secrets
import signal
import stat
import subprocess
import sys
import tempfile
import time

FORMAT='poseidon-bounded-original-candidate-audit-v1'
MAX_BYTES=4*1024**2
WALL_SECONDS=180
BASE=Path('/home/lhohy/Code/Poseidon/DSL/DSL_agent/scripts/baseline')

def _require(value,reason):
    if not value:raise ValueError(reason)

def _sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def _canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()

def _digest(value):return hashlib.sha256(_canonical(value)).hexdigest()

def _read(path):
    from strict_json_files import strict_file
    return strict_file(Path(path),MAX_BYTES)

def _save(path,value):
    raw=_canonical(value);_require(len(raw)<=MAX_BYTES,'Bounded audit result bytes')
    with Path(path).open('xb')as stream:
        os.chmod(path,0o600);stream.write(raw)

def _closure():
    from application_identity import source_closure
    return source_closure(BASE,['audit_unified_candidate.py'])

def _directory(path):
    path=Path(path)
    _require(path.is_absolute()and path==path.resolve()and path.is_dir()and not path.is_symlink(),
             'Bounded audit directory')
    return path

def _checked_result(value,request,folder):
    _require(type(value)is dict and set(value)=={'format','nonce','request_binding','audit','worker'},
             'Bounded audit response fields')
    _require(value['format']==FORMAT and value['nonce']==request['nonce']and
             value['request_binding']==_digest(request),'Bounded audit response binding')
    audit=value['audit'];worker=value['worker']
    _require(type(audit)is dict and set(audit)=={'evidence','files','request_id','model_sha256',
        'exercise','upstream_helper_trace','comparison','coverage','helper_coverage','period','inputs','outputs'}and
             audit.get('evidence')==str(folder)and
             type(audit.get('files'))is dict and bool(audit['files']), 'Bounded audit evidence binding')
    _require(type(worker)is dict and set(worker)=={'pid','limits','net_namespace','CPU_seconds','torch_threads'}and
             type(worker['pid'])is int and worker['pid']>0 and type(worker['torch_threads'])is int and worker['torch_threads']==2 and
             type(worker['limits'])is dict and
             set(worker['limits'])=={'AS','CPU','FSIZE','NOFILE','CORE'}and
             all(type(pair)is list and len(pair)==2 and all(type(v)is int for v in pair)
                 for pair in worker['limits'].values())and
             worker['limits']=={'AS':[4*1024**3]*2,'CPU':[150,155],'FSIZE':[16*1024**2]*2,'NOFILE':[128]*2,'CORE':[0,0]}and
             type(worker['net_namespace'])is str and worker['net_namespace']!=request['parent_net_namespace']and
             type(worker['CPU_seconds'])in(int,float)and 0<=worker['CPU_seconds']<=155,
             'Bounded audit worker boundary')
    from audit_unified_candidate import verify_files
    verify_files(folder,audit['files'])
    _require(set(audit['files'])=={str(p.relative_to(folder))for p in folder.rglob('*')if p.is_file()},
             'Bounded audit complete file coverage')
    report=_read_report(folder/'report.json')
    passed=[row for row in report['attempts']if row.get('status')=='passed']
    _require(report['status']=='passed'and len(passed)==1 and
             report['request_id']==audit['request_id']and
             audit['comparison']==(passed[0]['comparison']if request['validation_level']=='numerical'else None),
             'Bounded audit original result binding')
    return audit

def _read_report(path):
    from strict_json_files import strict_file
    return strict_file(Path(path),8*1024**2)

def verify_candidate(folder,*,live_approval=None,validation_level='numerical',workspace=None,artifact_observer=None):
    """Same auditor result, with the original limits applied to its own worker."""
    _require(validation_level in('compiled','numerical'),'Audit validation level')
    if live_approval is not None:
        # Existing historical explicit-live provenance stays on its original path.
        from audit_unified_candidate import verify_candidate as original
        return original(folder,live_approval=live_approval,validation_level=validation_level)
    from hecate_python_env import WORK,VENV
    from process_resource_limits import limits
    _require(Path(sys.prefix)==VENV and os.environ.get('IN_NIX_SHELL')=='pure',
             'Bounded audit requires pinned original environment')
    folder=_directory(folder)
    _require(folder.is_relative_to((WORK/'results').resolve()),'Bounded audit platform evidence')
    workspace=_directory(WORK/'results'if workspace is None else workspace)
    directory=Path(tempfile.mkdtemp(prefix='candidate-replay-audit-',dir=workspace))
    if artifact_observer is not None:artifact_observer(directory)
    script=Path(__file__).resolve();closure=_closure()
    request=dict(format=FORMAT,nonce=secrets.token_hex(16),evidence=str(folder),
        validation_level=validation_level,script=str(script),script_sha256=_sha(script),
        source_closure=closure,parent_net_namespace=os.readlink('/proc/self/ns/net'))
    _save(directory/'request.json',request)
    command=['/usr/bin/bwrap','--unshare-user','--unshare-net','--die-with-parent','--new-session',
        '--ro-bind','/','/','--bind',str(directory),str(directory),'--tmpfs','/tmp',
        '--proc','/proc','--dev','/dev','--',sys.executable,'-B',str(script),'--worker',str(directory)]
    started=time.monotonic();child=None
    with(directory/'worker.log').open('xb')as log:
        child=subprocess.Popen(command,cwd=BASE,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
            close_fds=True,start_new_session=True,preexec_fn=limits)
        try:code=child.wait(timeout=WALL_SECONDS)
        except BaseException as error:
            if child.poll()is None:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=5)
            _save(directory/'host-exit.json',dict(returncode=child.returncode,
                elapsed_seconds=time.monotonic()-started,request_binding=_digest(request),
                error_type=type(error).__name__,new_API=0,new_compile=0,new_encrypted=0,retries=0))
            raise
    _save(directory/'host-exit.json',dict(returncode=code,elapsed_seconds=time.monotonic()-started,
        request_binding=_digest(request),new_API=0,new_compile=0,new_encrypted=0,retries=0))
    _require(code==0,'Original isolated auditor failed')
    _require(_sha(script)==request['script_sha256']and _closure()==closure and
             _read(directory/'request.json')==request,'Bounded audit authority changed')
    result=_checked_result(_read(directory/'result.json'),request,folder)
    _save(directory/'host-receipt.json',dict(format=FORMAT,status='verified',request_binding=_digest(request),
        result_sha256=_sha(directory/'result.json'),original_audit_unchanged=True,
        API=0,compile=0,encrypted=0,retries=0))
    return result

def _worker(directory):
    from process_resource_limits import limits
    limits();os.umask(0o077)
    import resource
    started=time.process_time();directory=_directory(directory)
    _require(stat.S_IMODE(directory.stat().st_mode)==0o700,'Private bounded audit workspace')
    request=_read(directory/'request.json')
    _require(type(request)is dict and set(request)=={'format','nonce','evidence','validation_level','script',
        'script_sha256','source_closure','parent_net_namespace'},'Bounded audit request fields')
    _require(request['format']==FORMAT and type(request['nonce'])is str and len(request['nonce'])==32 and
        all(x in'0123456789abcdef'for x in request['nonce'])and request['validation_level']in('compiled','numerical')and
        type(request['parent_net_namespace'])is str and request['parent_net_namespace'].startswith('net:[')and
        request['parent_net_namespace'].endswith(']')and request['parent_net_namespace'][5:-1].isdigit()and
        request['script']==str(Path(__file__).resolve())and _sha(__file__)==request['script_sha256']and
        _closure()==request['source_closure'],'Bounded audit worker authority')
    net=os.readlink('/proc/self/ns/net')
    _require(net!=request['parent_net_namespace'],'Bounded audit offline namespace')
    from hecate_python_env import WORK,VENV
    _require(Path(sys.prefix)==VENV and os.environ.get('IN_NIX_SHELL')=='pure','Pinned audit worker')
    folder=_directory(request['evidence'])
    _require(folder.is_relative_to((WORK/'results').resolve())and
        os.statvfs(folder).f_flag&os.ST_RDONLY and os.statvfs(BASE).f_flag&os.ST_RDONLY,
        'Read-only audit evidence/source')
    import torch
    torch.set_num_threads(2)  # Match QualificationSession.qualify's original reference owner.
    from audit_unified_candidate import verify_candidate as original
    audit=original(folder,validation_level=request['validation_level'])
    _require(_sha(__file__)==request['script_sha256']and _closure()==request['source_closure'],
             'Audit source changed')
    bounds={k:list(resource.getrlimit(v))for k,v in [('AS',resource.RLIMIT_AS),('CPU',resource.RLIMIT_CPU),
        ('FSIZE',resource.RLIMIT_FSIZE),('NOFILE',resource.RLIMIT_NOFILE),('CORE',resource.RLIMIT_CORE)]}
    _save(directory/'result.json',dict(format=FORMAT,nonce=request['nonce'],request_binding=_digest(request),
        audit=audit,worker=dict(pid=os.getpid(),limits=bounds,net_namespace=net,CPU_seconds=time.process_time()-started,torch_threads=torch.get_num_threads())))

if __name__=='__main__':
    sys.path.insert(0,str(BASE))
    _require(len(sys.argv)==3 and sys.argv[1]=='--worker','Bounded audit worker invocation')
    _worker(sys.argv[2])
