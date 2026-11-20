"""Prepared passive compiler snapshots with deferred public-executable hashing.

Preparation verifies the original approved CPU factory before process launch.
Capture reads real /proc fields while the target is alive, persists them outside
its mounts, then hashes an executable descriptor opened from that live target.
An exit after those reads does not invalidate the held public file descriptor.
No process is launched, stopped, repaired, or given compilation/numeric credit.
"""
import copy
import hashlib
import os
from pathlib import Path
import stat
import time

import approved_cpu_compiler_observation as base
from delivery_component import bound, canonical, checked_bound, read_bytes, require, sha, write_once

CONTRACT='poseidon-prepared-compiler-process-contract-v1'
RAW='poseidon-prepared-compiler-raw-fields-v1'
RECORD='poseidon-prepared-compiler-process-observation-v1'


def source_manifest():
    return dict(base.source_manifest(),**{str(Path(__file__).absolute()):sha(read_bytes(__file__))})


def _host_directory(folder,base_contract):
    folder=base._path(folder)
    require(folder.resolve()==folder and folder.is_dir() and not folder.is_symlink(),
            'Existing direct host evidence directory required')
    for item in base_contract['bindings']:
        source=base._path(item['source']).resolve(strict=True)
        require(not folder.is_relative_to(source), 'Raw records must be absent from every candidate mount')
    return folder


def contract(command,owner,owner_namespaces,nonce,selection,*,evidence_directory,test_only=False):
    original=base.contract(command,owner,owner_namespaces,nonce,selection,test_only=test_only)
    directory=_host_directory(evidence_directory,original)
    return bound(dict(format=CONTRACT,base_contract=original,host_evidence_directory=str(directory),
        observer_sources=source_manifest(),test_only=test_only,execution_or_approval_authority=False))


def _validate(value,expected_binding):
    checked_bound(value,expected_binding)
    require(set(value)=={'format','base_contract','host_evidence_directory','observer_sources',
            'test_only','execution_or_approval_authority','binding'}
        and value['format']==CONTRACT and type(value['test_only'])is bool
        and value['execution_or_approval_authority']is False,'Prepared passive contract scope')
    original=checked_bound(value['base_contract'])
    require(original['test_only']is value['test_only']and value['observer_sources']==source_manifest(),
            'Exact frozen preparation implementation and fixture scope')
    actual=base.contract(original['command'],original['owner'],original['owner_namespaces'],
        original['nonce'],original['selection'],test_only=original['test_only'])
    require(actual==original,'Prepared canonical SDK/command/mount/executable changed')
    return _host_directory(value['host_evidence_directory'],original)


def _signature(st):
    require(stat.S_ISREG(st.st_mode)and 0<=st.st_size<=512*1024**2,'Bounded public executable descriptor')
    return dict(device=st.st_dev,inode=st.st_ino,kind=stat.S_IFREG,bytes=st.st_size,
                mtime_ns=st.st_mtime_ns,ctime_ns=st.st_ctime_ns)


def _check_opened(opened,expected):
    require(type(opened)is dict and set(opened)=={'device','inode','kind','bytes','mtime_ns','ctime_ns'}
        and all(type(v)is int and v>=0 for v in opened.values()),'Complete raw executable stat fields')
    require({k:opened[k]for k in('device','inode','kind','bytes')}==
            {k:expected[k]for k in('device','inode','kind','bytes')},
            'Held descriptor must be the independently frozen executable inode and size')


def _hash_open_descriptor(fd,opened):
    """Hash the already held public inode, independent of a later /proc exit."""
    require(_signature(os.fstat(fd))==opened,'Executable descriptor changed before hashing')
    os.lseek(fd,0,os.SEEK_SET);digest=hashlib.sha256();size=0
    while True:
        piece=os.read(fd,1024**2)
        if not piece:break
        digest.update(piece);size+=len(piece)
        require(size<=opened['bytes'],'Executable grew while hashing')
    require(_signature(os.fstat(fd))==opened and size==opened['bytes'],
            'Executable descriptor changed during hashing')
    return dict(**{k:opened[k]for k in('device','inode','kind','bytes')},sha256=digest.hexdigest())


def _record(value,raw_path,raw,executable,hash_begin,hash_end):
    original=value['base_contract']
    snapshot=bound(dict(raw['snapshot_fields'],executable=executable))
    return bound(dict(format=RECORD,contract_binding=value['binding'],base_record=snapshot,
        raw_record=dict(path=str(raw_path),sha256=sha(read_bytes(raw_path)),binding=raw['binding']),
        executable_hash_begin_monotonic_ns=hash_begin,executable_hash_end_monotonic_ns=hash_end,
        test_only=value['test_only'],raw_saved_before_hash_and_validation=True,
        executable_opened_from_live_process=not value['test_only'],
        compiler_source_build_terminal_or_native_or_first_certified=False,
        execution_or_approval_authority=False))


class PreparedCollector:
    """Instantiate before launch; each instance permits one capture attempt."""
    def __init__(self,value,expected_binding):
        _validate(value,expected_binding)
        self._value=copy.deepcopy(value);self._binding=expected_binding;self._used=False

    def capture(self,pid):
        value=self._value;original=value['base_contract']
        require(not self._used,'The prepared capture attempt has already been consumed')
        self._used=True
        # Reconstructing the SDK or hashing executables is intentionally deferred
        # until the live reads are complete. The immutable prepared binding is
        # checked now; full source/SDK/command validation is repeated afterward.
        checked_bound(value,self._binding);checked_bound(original)
        require(value['test_only']is False,'Fixture contracts cannot read live processes')
        owner,_=base._process(original['owner']['pid'])
        require(owner==original['owner'],'Original shared owner is no longer live')
        owner_ns={n:os.readlink('/proc/%d/ns/%s'%(owner['pid'],n))for n in base.NAMESPACES}
        require(owner_ns==original['owner_namespaces'],'Owner namespace changed after preparation')
        begin=time.monotonic_ns();observed_at=time.time_ns()
        target,_=base._process(pid);require(target['boot_id']==owner['boot_id'],'Same actual boot')
        fd=os.open('/proc/%d/exe'%pid,os.O_RDONLY|os.O_NONBLOCK|os.O_CLOEXEC)
        try:
            opened=_signature(os.fstat(fd));_check_opened(opened,original['executable'])
            argv=base._proc(pid,'cmdline').rstrip(b'\0').decode().split('\0')
            require(argv==original['argv'],'Actual compiler/frontend command differs')
            environment={}
            for part in base._proc(pid,'environ').rstrip(b'\0').split(b'\0'):
                key,sep,entry=part.partition(b'=');require(sep,'Raw process environment field')
                name=key.decode();require(name not in environment,'Duplicate environment key')
                environment[name]=entry
            allowed=set(original['environment'])|{'PWD'}
            require(set(environment)<=allowed,
                'Unexpected environment names: '+','.join(sorted(set(environment)-allowed)))
            public={key:entry.decode()for key,entry in environment.items()}
            namespaces={n:os.readlink('/proc/%d/ns/%s'%(pid,n))for n in base.NAMESPACES}
            ancestors=[];cursor=pid
            for _ in range(64):
                entry,parent=base._process(cursor);ancestors.append(dict(process=entry,parent_pid=parent))
                if entry==owner:break
                require(parent>1 and parent!=cursor,'Actual owner is not a live ancestor');cursor=parent
            else:raise ValueError('Bounded process-to-owner ancestry')
            fields=dict(format=base.RECORD,contract_binding=original['binding'],nonce=original['nonce'],
                role=original['role'],owner=owner,process=target,ancestry=ancestors,observed_at_ns=observed_at,
                argv=argv,cwd=os.readlink('/proc/%d/cwd'%pid),environment_names=sorted(public),public_environment=public,
                namespaces=namespaces,owner_namespaces=owner_ns,
                actual_mountinfo=base._proc(pid,'mountinfo').decode(),actual_maps=base._proc(pid,'maps').decode(),
                actual_limits=base._proc(pid,'limits').decode(),
                bindings=[dict(destination=b['destination'],identity=base._identity('/proc/%d/root%s'%(pid,b['destination'])))for b in original['bindings']],
                private_file_reads=0,private_key_hashes=0,process_memory_reads=0,processes_started=0,
                compilation_completed_or_source_semantics_or_native_or_first_delivery_certified=False)
            end_target,_=base._process(pid)
            end_executable=_signature(os.stat('/proc/%d/exe'%pid))
            end_owner,_=base._process(owner['pid'])
            end_owner_ns={n:os.readlink('/proc/%d/ns/%s'%(owner['pid'],n))for n in base.NAMESPACES}
            require(target==end_target and opened==end_executable,'Target PID/start or executable changed during raw capture')
            require(owner==end_owner and owner_ns==end_owner_ns,'Owner changed during raw capture')
            end=time.monotonic_ns()
            raw=bound(dict(format=RAW,contract_binding=self._binding,snapshot_fields=fields,
                opened_executable=opened,end_executable=end_executable,end_target=end_target,
                end_owner=end_owner,end_owner_namespaces=end_owner_ns,
                raw_begin_monotonic_ns=begin,raw_end_monotonic_ns=end,test_only=False,
                execution_or_approval_authority=False))
            raw_path=Path(value['host_evidence_directory'])/'RAW_SNAPSHOT.json'
            raw_bytes=canonical(raw);require(len(raw_bytes)<=4*1024**2,'Bounded complete raw snapshot')
            write_once(raw_path,raw_bytes)
            hash_begin=time.monotonic_ns();executable=_hash_open_descriptor(fd,opened);hash_end=time.monotonic_ns()
            require(executable=={k:v for k,v in original['executable'].items()if k!='path'},'Actual executable bytes differ')
            _validate(value,self._binding)
            result=_record(value,raw_path,raw,executable,hash_begin,hash_end)
            verify(value,self._binding,result,result['binding'])
            return result
        finally:os.close(fd)


def verify(value,expected_binding,record,expected_record_binding):
    directory=_validate(value,expected_binding);checked_bound(record,expected_record_binding)
    require(set(record)=={'format','contract_binding','base_record','raw_record','executable_hash_begin_monotonic_ns',
        'executable_hash_end_monotonic_ns','test_only','raw_saved_before_hash_and_validation',
        'executable_opened_from_live_process','compiler_source_build_terminal_or_native_or_first_certified',
        'execution_or_approval_authority','binding'},'Closed prepared observation fields')
    require(record['format']==RECORD and record['contract_binding']==expected_binding
        and record['test_only']is value['test_only'],'Exact prepared observation binding and fixture scope')
    original=value['base_contract'];item=record['raw_record']
    require(set(item)=={'path','sha256','binding'}and Path(item['path'])==directory/'RAW_SNAPSHOT.json',
            'Exact independent raw record location')
    raw_bytes=read_bytes(item['path'],4*1024**2);require(sha(raw_bytes)==item['sha256'],'Durable raw bytes changed')
    import json
    raw=checked_bound(json.loads(raw_bytes),item['binding'])
    require(set(raw)=={'format','contract_binding','snapshot_fields','opened_executable','end_executable','end_target',
        'end_owner','end_owner_namespaces','raw_begin_monotonic_ns','raw_end_monotonic_ns','test_only',
        'execution_or_approval_authority','binding'},'Closed durable raw observation fields')
    require(raw['format']==RAW and raw['contract_binding']==expected_binding
        and raw['test_only']is value['test_only']and raw['execution_or_approval_authority']is False,
        'Raw observation scope and original prepared contract')
    fields=raw['snapshot_fields'];snapshot=checked_bound(record['base_record'])
    require({k:v for k,v in snapshot.items()if k not in('binding','executable')}==fields,
            'Snapshot raw fields cannot be invented or substituted after capture')
    opened=raw['opened_executable'];_check_opened(opened,original['executable'])
    require(opened==raw['end_executable']and fields['process']==raw['end_target']
        and fields['owner']==raw['end_owner']and fields['owner_namespaces']==raw['end_owner_namespaces'],
        'Actual target/owner/executable must remain stable throughout live raw reads')
    stamps=[raw['raw_begin_monotonic_ns'],raw['raw_end_monotonic_ns'],
            record['executable_hash_begin_monotonic_ns'],record['executable_hash_end_monotonic_ns']]
    require(all(type(t)is int and t>0 for t in stamps)and stamps==sorted(stamps),
            'Complete raw observations must precede deferred executable hashing')
    require(record['raw_saved_before_hash_and_validation']is True
        and record['executable_opened_from_live_process']is(not value['test_only'])
        and record['compiler_source_build_terminal_or_native_or_first_certified']is False
        and record['execution_or_approval_authority']is False,'Passive collector cannot certify completion or numerics')
    check=base.verify(original,original['binding'],snapshot,snapshot['binding'])
    return bound(dict(format='poseidon-prepared-compiler-process-snapshot-check-v1',
        contract_binding=expected_binding,observation_binding=expected_record_binding,raw_record_binding=raw['binding'],
        base_check_binding=check['binding'],test_only=value['test_only'],raw_snapshot_matches_contract=True,
        actual_live_capture_validated=not value['test_only'],
        compiler_source_build_terminal_or_native_or_first_certified=False,execution_or_approval_authority=False))
