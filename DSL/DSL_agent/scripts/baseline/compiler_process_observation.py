"""Passive observations of the original Hecate frontend/compiler sandbox.

No process is launched, stopped, modified or certified here. A verified
snapshot is only one input to future source/build/terminal provenance, never
proof of a successful compilation, CKKS execution or first delivery.
"""
import os
from pathlib import Path
import re
import stat
import time

from delivery_component import bound, canonical, checked_bound, read_bytes, require, sha
from normal_cpu_overlay_host_evidence import _command_parts, _mounts, LIMITS

CONTRACT='poseidon-original-compiler-process-observation-contract-v1'
RECORD='poseidon-original-compiler-process-observation-v1'
NAMESPACES=('pid','user','mnt','net','ipc','uts')
MAX_PROC=1024**2
FORBIDDEN={'.env','.ssh','.aws','client-private','secret.key','secret.bin'}


def _path(value):
    path=Path(value)
    require(path.is_absolute()and '..'not in path.parts and not any(p in FORBIDDEN for p in path.parts),
        'Only explicitly selected public paths')
    return path


def _identity(path):
    st=os.stat(path)
    require(stat.S_ISREG(st.st_mode)or stat.S_ISDIR(st.st_mode),'Original regular file or directory')
    return dict(device=st.st_dev,inode=st.st_ino,kind=stat.S_IFMT(st.st_mode))


def _file_identity(path):
    import hashlib
    path=_path(path);fd=os.open(path,os.O_RDONLY|os.O_NONBLOCK|os.O_CLOEXEC)
    try:
        start=os.fstat(fd);require(stat.S_ISREG(start.st_mode)and start.st_size<=512*1024**2,'Bounded public executable')
        digest=hashlib.sha256();size=0
        while True:
            piece=os.read(fd,1024**2)
            if not piece:break
            digest.update(piece);size+=len(piece)
        end=os.fstat(fd)
        signature=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
        require(signature(start)==signature(end)and size==start.st_size,'Executable changed while observed')
        return dict(device=start.st_dev,inode=start.st_ino,kind=stat.S_IFREG,bytes=size,sha256=digest.hexdigest())
    finally:os.close(fd)


def _owner(value):
    require(type(value)is dict and set(value)=={'pid','start_ticks','boot_id'}
        and type(value['pid'])is int and value['pid']>1 and type(value['start_ticks'])is int and value['start_ticks']>0
        and re.fullmatch('[0-9a-f-]{36}',value['boot_id'])is not None,'Exact original shared owner')
    return value


def _namespace(value):
    require(type(value)is dict and set(value)==set(NAMESPACES)
        and all(type(value[n])is str and re.fullmatch(n+r':\[[0-9]+\]',value[n])for n in NAMESPACES),
        'Every raw process namespace')
    return value


def source_manifest():
    import delivery_component
    import normal_cpu_overlay_host_evidence
    return {str(Path(m.__file__).absolute()):sha(read_bytes(m.__file__))
        for m in(delivery_component,normal_cpu_overlay_host_evidence)}|{str(Path(__file__).absolute()):sha(read_bytes(__file__))}


def _factory(command,test_only):
    binds,environment,special,argv=_command_parts(command)
    require(command.count('--chdir')==1 and command[command.index('--chdir')+1]=='/out',
        'Original frontend/compiler working directory')
    destinations={b['destination']:b for b in binds}
    require('/payload.json'in destinations and destinations['/payload.json']['mode']=='ro'
        and '/out'in destinations and destinations['/out']['mode']=='rw'
        and '/'not in destinations and '/home'not in destinations and '/keys'not in destinations,
        'Original public payload and output; no broad host or key mount')
    require(argv and all(type(x)is str and '\x00'not in x for x in argv),'Exact public tool arguments')
    if argv[0]=='/hecate-opt':
        require('/hecate-opt'in destinations and destinations['/hecate-opt']['mode']=='ro','Original read-only compiler')
        role='compiler';executable=destinations['/hecate-opt']['source']
    else:
        require(len(argv)==2 and argv[1]=='/app/candidate_trace.py'and Path(argv[0]).is_absolute(),
            'Only original AST/Hecate frontend or compiler roles')
        role='frontend';executable=argv[0]
    if not test_only:
        from poseidon_candidate_sandbox import command as original
        expected=original(Path(destinations['/payload.json']['source']),Path(destinations['/out']['source']),argv,
            helpers='/poly-deps'in destinations)
        require(command==expected,'Original sandbox factory command must remain byte-for-byte equal')
    return binds,environment,special,argv,role,executable


def contract(command,owner,owner_namespaces,nonce,*,test_only=False):
    require(type(test_only)is bool and type(nonce)is str and re.fullmatch('[0-9a-f]{32}',nonce),
        'Frozen fresh observation nonce and explicit fixture role')
    _owner(owner);_namespace(owner_namespaces)
    binds,environment,_,argv,role,executable=_factory(command,test_only)
    require(not any(part in FORBIDDEN for b in binds for part in _path(b['source']).parts),'No private bind source')
    return bound(dict(format=CONTRACT,command=command,role=role,argv=argv,cwd='/out',
        owner=owner,owner_namespaces=owner_namespaces,nonce=nonce,test_only=test_only,
        bindings=[dict(b,identity=_identity(b['source']))for b in binds],
        environment=dict(environment),executable=dict(path=str(_path(executable)),**_file_identity(executable)),
        observer_sources=source_manifest(),execution_or_approval_authority=False))


def _process(pid):
    require(type(pid)is int and pid>1,'Observed process PID')
    raw=Path('/proc',str(pid),'stat').read_text();tail=raw.rsplit(') ',1)[1].split()
    return dict(pid=pid,start_ticks=int(tail[19]),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip()),int(tail[1])


def _proc(pid,name):
    path=Path('/proc',str(pid),name)
    with path.open('rb')as f:raw=f.read(MAX_PROC+1)
    require(len(raw)<=MAX_PROC,'Bounded complete raw process observation')
    return raw


def capture(value,expected_binding,pid):
    """Read the live process only. Missing access or a racing exit is a refusal."""
    checked_bound(value,expected_binding);require(value['format']==CONTRACT,'Compiler observation contract role')
    require(value['test_only']is False,'Synthetic contracts cannot observe live compiler processes')
    _factory(value['command'],False)
    require(value['observer_sources']==source_manifest(),'Observer code changed since freeze')
    owner,_=_process(value['owner']['pid']);require(owner==value['owner'],'Original owner no longer live')
    owner_namespaces={n:os.readlink('/proc/%d/ns/%s'%(owner['pid'],n))for n in NAMESPACES}
    require(owner_namespaces==value['owner_namespaces'],'Actual owner namespaces changed since freeze')
    start,_=_process(pid);require(start['boot_id']==owner['boot_id'],'Same actual boot')
    executable=_file_identity(Path('/proc',str(pid),'exe'))
    require(executable=={k:v for k,v in value['executable'].items()if k!='path'},'Actual target executable differs')
    argv=_proc(pid,'cmdline').rstrip(b'\0').decode().split('\0')
    require(argv==value['argv'],'Actual frontend/compiler command changed')
    env={}
    for part in _proc(pid,'environ').rstrip(b'\0').split(b'\0'):
        key,sep,entry=part.partition(b'=');require(sep,'Raw process environment field')
        name=key.decode();require(name not in env,'Duplicate observed environment key');env[name]=entry
    allowed=set(value['environment'])|{'PWD'}
    # Unknown values never enter the serialized observation.
    require(set(env)<=allowed,'Unexpected environment names: '+','.join(sorted(set(env)-allowed)))
    public={k:v.decode()for k,v in env.items()}
    namespaces={n:os.readlink('/proc/%d/ns/%s'%(pid,n))for n in NAMESPACES}
    ancestors=[];cursor=pid
    for _ in range(64):
        entry,parent=_process(cursor);ancestors.append(dict(process=entry,parent_pid=parent))
        if entry==owner:break
        require(parent>1 and parent!=cursor,'Original owner is not a live ancestor');cursor=parent
    else:raise ValueError('Bounded original-owner ancestry')
    result=bound(dict(format=RECORD,contract_binding=expected_binding,nonce=value['nonce'],role=value['role'],
        owner=owner,process=start,ancestry=ancestors,observed_at_ns=time.time_ns(),
        argv=argv,cwd=os.readlink('/proc/%d/cwd'%pid),environment_names=sorted(public),public_environment=public,
        executable=executable,namespaces=namespaces,owner_namespaces=owner_namespaces,actual_mountinfo=_proc(pid,'mountinfo').decode(),
        actual_maps=_proc(pid,'maps').decode(),actual_limits=_proc(pid,'limits').decode(),
        bindings=[dict(destination=b['destination'],identity=_identity('/proc/%d/root%s'%(pid,b['destination'])))for b in value['bindings']],
        private_file_reads=0,private_key_hashes=0,process_memory_reads=0,processes_started=0,
        compilation_completed_or_source_semantics_or_native_or_first_delivery_certified=False))
    end,_=_process(pid);require(end==start,'Observed process exited or PID was reused during capture')
    last_owner,_=_process(owner['pid']);require(last_owner==owner
        and all(os.readlink('/proc/%d/ns/%s'%(owner['pid'],n))==owner_namespaces[n]for n in NAMESPACES),
        'Original owner or namespaces changed during capture')
    verify(value,expected_binding,result,result['binding'])
    return result


def verify(value,expected_binding,record,expected_record_binding):
    """Verify a complete raw snapshot against an independently selected contract."""
    checked_bound(value,expected_binding);checked_bound(record,expected_record_binding)
    require(type(value.get('test_only'))is bool and value.get('execution_or_approval_authority')is False,
        'Explicit observation scope, without execution authority')
    require(value['format']==CONTRACT and record['format']==RECORD
        and record['contract_binding']==expected_binding and record['nonce']==value['nonce']
        and record['role']==value['role'],'Exact stage/nonce/contract binding')
    require(value['observer_sources']==source_manifest(),'Frozen current observation implementation')
    binds,environment,_,argv,role,executable=_factory(value['command'],value['test_only'])
    require(value['argv']==argv and value['role']==role and value['executable']['path']==executable
        and value['environment']==environment and len(value['bindings'])==len(binds)
        and all({k:v for k,v in b.items()if k!='identity'}==p for b,p in zip(value['bindings'],binds)),
        'Contract metadata must describe the original command itself')
    _owner(record['process']);_namespace(record['namespaces'])
    require(record['owner']==value['owner']and record['process']['boot_id']==value['owner']['boot_id'],
        'Original shared owner and same boot')
    chain=record['ancestry'];require(type(chain)is list and 2<=len(chain)<=64
        and all(type(x)is dict and set(x)=={'process','parent_pid'}and type(x['parent_pid'])is int and x['parent_pid']>=1 for x in chain)
        and chain[0]['process']==record['process']and chain[-1]['process']==record['owner']
        and len({x['process']['pid']for x in chain})==len(chain)
        and all(_owner(x['process'])['boot_id']==record['owner']['boot_id']for x in chain)
        and all(a['parent_pid']==b['process']['pid']for a,b in zip(chain,chain[1:])),
        'Complete distinct process-to-owner ancestry')
    require(type(record['observed_at_ns'])is int and record['observed_at_ns']>0,'Original raw observation timestamp')
    require(record['argv']==value['argv']and record['cwd']==value['cwd'], 'Actual command and cwd, not requested values')
    require(record['executable']=={k:v for k,v in value['executable'].items()if k!='path'},'Exact actual executable inode and bytes')
    environment=dict(value['environment'])
    require(type(record['public_environment'])is dict and record['environment_names']==sorted(record['public_environment']),
        'Complete unique observed environment names')
    if 'PWD'in record['public_environment']:environment['PWD']=value['cwd']
    require(record['public_environment']==environment,'Actual cleared public environment')
    _namespace(record['owner_namespaces'])
    require(record['owner_namespaces']==value['owner_namespaces']
        and all(record['namespaces'][n]!=record['owner_namespaces'][n]for n in NAMESPACES),
        'Actual independent pid/user/mount/network/ipc/uts namespaces')
    actual=_mounts(record['actual_mountinfo']);expected={b['destination']:b for b in value['bindings']}
    seen=record['bindings'];require(type(seen)is list and len(seen)==len(expected)
        and len({b['destination']for b in seen})==len(seen),'Every original actual bind exactly once')
    for b in seen:
        require(b['destination']in expected and b['identity']==expected[b['destination']]['identity'],
            'Actual mount source device/inode/type differs')
    for destination,b in expected.items():
        require(destination in actual and b['mode']in actual[destination]['options'], 'Actual read-only/read-write bind mode')
    require(all(d in actual for d in('/proc','/dev','/tmp'))and actual['/proc']['filesystem']=='proc'
        and actual['/tmp']['filesystem']=='tmpfs','Actual required proc/dev/tmp sandbox mounts')
    allowed=set(expected)|{'/','/proc','/dev','/tmp','/dev/pts','/dev/shm','/dev/mqueue',
        '/dev/null','/dev/zero','/dev/random','/dev/urandom','/dev/full','/dev/tty',
        '/proc/sys','/proc/sysrq-trigger','/proc/irq','/proc/bus'}
    require(all(d in allowed for d in actual),
        'Unexpected reachable mount outside original compiler contract')
    raw=record['actual_limits'];require(type(raw)is str and len(raw)<MAX_PROC,'Complete raw limits')
    for prefix,limits in LIMITS.items():
        rows=[line for line in raw.splitlines()if line.startswith(prefix+' ')]
        require(len(rows)==1 and list(map(int,rows[0].split()[len(prefix.split()):][:2]))==limits,
            'Original actual process limit: '+prefix)
    require(type(record['actual_maps'])is str and len(record['actual_maps'])<=MAX_PROC
        and 'libseal'not in record['actual_maps'].lower(),'No alternate SEAL execution mapping')
    mapped=False
    for line in record['actual_maps'].splitlines():
        fields=line.split(maxsplit=5)
        if len(fields)<5:continue
        major,minor=(int(n,16)for n in fields[3].split(':'))
        if int(fields[4])==record['executable']['inode']and os.makedev(major,minor)==record['executable']['device']and 'x'in fields[1]:mapped=True
    require(mapped,'Actual executable mapping required, not only an executable path')
    require(all(record[k]==0 for k in('private_file_reads','private_key_hashes','process_memory_reads','processes_started'))
        and record['compilation_completed_or_source_semantics_or_native_or_first_delivery_certified']is False,
        'Passive snapshot cannot grant execution, compilation or numerical authority')
    return bound(dict(format='poseidon-original-compiler-process-snapshot-check-v1',contract_binding=expected_binding,
        observation_binding=expected_record_binding,role=value['role'],test_only=value['test_only'],raw_snapshot_matches_contract=True,
        compilation_completed=False,compiler_source_build_origin_certified=False,source_to_lowered_semantics_certified=False,
        native_input_same_key_or_numerical_or_first_delivery_certified=False,execution_authority=False))
