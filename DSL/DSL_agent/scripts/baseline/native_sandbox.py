"""Fixed Native client/evaluator processes under the original shared lease.

Only the final client receives the original private directory. The Python host
checks directory existence and public descriptors; it never opens private
metadata or secret files. Native PID/limits/mount checks precede its owner ACK.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from hecate_python_env import ROOT,WORK
from native_execution_slots import native_slot
from process_resource_limits import limits
from shared_execution import Coordinator,process
import native_sdk_manifest as sdk_records

IDENTITY_FIELDS=('public_manifest_sha256','runtime_plan_sha256','operator_spec_sha256',
                 'source_plan_sha256','link_record_sha256','task_binding')
WANT_LIMITS={'Max cpu time':(150,155),'Max file size':(16777216,16777216),
             'Max core file size':(0,0),'Max open files':(128,128),
             'Max address space':(4294967296,4294967296)}


def require(condition,reason):
    if not condition:raise ValueError('Native sandbox: '+reason)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as stream:
        for part in iter(lambda:stream.read(1024**2),b''):h.update(part)
    return h.hexdigest()


def read(path):
    return sdk_records.base_sdk.read_json(path)


def json_sha(path,within):
    path=sdk_records._public_file(str(path),within)
    require(path.suffix=='.json'and path.stat().st_size<=256*1024,'bounded direct public protocol JSON')
    return sha(path)


def save(path,value):
    with Path(path).open('x')as stream:
        json.dump(value,stream,sort_keys=True,indent=2,allow_nan=False);stream.write('\n')


def directory(path,within):
    result=sdk_records.base_sdk.path(str(path),within=within,directory=True)
    info=result.stat()
    require(info.st_uid==os.getuid()and info.st_mode&0o077==0,'private owned execution directory')
    return result


def _shared_lease():
    coordinator=Coordinator('/home/lhohy/poseidon-work/coordination/v1',[WORK/'results'])
    state,_=coordinator.snapshot()
    owner=process()
    rows=[row for row in state['batches'].values()
          if row.get('owner')==owner and row.get('finished')is None and
          row.get('status')not in ('succeeded','failed','cancelled','interrupted')and
          type(row.get('active'))is dict and row['active'].get('kind')in ('qualify','encrypted')]
    require(len(rows)==1,'exact active original shared qualification/encrypted lease')
    return copy.deepcopy(rows[0]['active'])


def prove_process(child,executable,expected_sha,control,role,epoch,driver):
    row=read(control/'native-ready.json')
    require(set(row)=={'role','nonce','pid_in_namespace'}and row['role']==role and
            row['nonce']==epoch and type(row['pid_in_namespace'])is int,'original native ready record')
    pending=[child.pid];seen=set();matches=[];expected=Path(executable).stat()
    while pending:
        pid=pending.pop()
        if pid in seen:continue
        require(len(seen)<64,'native descendant inventory');seen.add(pid)
        try:
            actual=Path('/proc/'+str(pid)+'/exe').stat()
            if(actual.st_dev,actual.st_ino)==(expected.st_dev,expected.st_ino):matches.append(pid)
            pending.extend(int(x)for x in Path('/proc/'+str(pid)+'/task/'+str(pid)+'/children').read_text().split())
        except(FileNotFoundError,ProcessLookupError):
            pass
    require(len(matches)==1,'unique original executable descendant')
    pid=matches[0];owner=process(pid)
    require(owner is not None and sha('/proc/'+str(pid)+'/exe')==expected_sha,'actual executable bytes')
    status=Path('/proc/'+str(pid)+'/status').read_text()
    namespace_pids=next(line for line in status.splitlines()if line.startswith('NSpid:')).split()[1:]
    require(int(namespace_pids[-1])==row['pid_in_namespace'],'actual namespace PID')
    text=Path('/proc/'+str(pid)+'/limits').read_text()
    for prefix,wanted in WANT_LIMITS.items():
        line=next(line for line in text.splitlines()if line.startswith(prefix+' '))
        parts=line.split()[len(prefix.split()):]
        require(tuple(map(int,parts[:2]))==wanted,'unchanged native limits')
    ancestry=[];walk=pid
    for _ in range(64):
        ancestry.append(walk)
        if walk==driver['pid']:break
        raw=Path('/proc/'+str(walk)+'/stat').read_text()
        walk=int(raw[raw.rfind(')')+2:].split()[1])
        require(walk>1,'native ancestry remains below original driver')
    else:raise ValueError('Native sandbox: native ancestry depth')
    root=Path('/proc/'+str(pid)+'/root')
    wanted={'client-private':role=='validator','plain-input':role=='encryptor',
            'inputs':role in ('encryptor','runtime'),'evaluated':role in ('runtime','validator'),
            'validation':role=='validator','runtime-input':True,'public':True,
            'client-evidence':False,'home/lhohy':False}
    for name,present in wanted.items():
        require((root/name).exists()==present,'native role mount boundary: '+name)
    network=os.readlink('/proc/'+str(pid)+'/ns/net')
    require(network!=os.readlink('/proc/self/ns/net'),'separate evaluator/client network namespace')
    require(process(pid)==owner and process()==driver,'stable host process identities')
    return dict(format='poseidon-native-host-process-proof-v1',owner=owner,driver=driver,
        launcher=process(child.pid),ancestry=ancestry,namespace_pid=int(namespace_pids[-1]),
        checked_outside_native_PID_namespace=True,actual_executable_sha256=expected_sha,
        unchanged_native_limits=WANT_LIMITS,role_mount_existence_verified=wanted,
        isolated_network_namespace_verified=True,network_namespace=network,role=role,epoch=epoch,
        host_private_file_reads=0)


class NativeSandbox:
    def __init__(self,sdk_file,result,*,max_processes,verify_frozen):
        require(type(max_processes)is int and 1<=max_processes<=12,'finite native invocation budget')
        require(callable(verify_frozen),'original frozen-context verifier')
        self.filename=Path(sdk_file)
        self.sdk=sdk_records.attest(sdk_records.load(self.filename,WORK,'aarch64-linux',ROOT),WORK,ROOT)
        require(sdk_records.ready_for_agent_qualification(self.sdk),'measured fresh input precondition')
        self.result=directory(result,WORK/'results')
        self.max_processes=max_processes;self.used=0;self.verify_frozen=verify_frozen
        self.driver=process()
        self.metadata_used=0

    def check(self):
        self.verify_frozen()
        require(process()==self.driver,'original Native host driver')
        require(sdk_records.load(self.filename,WORK,'aarch64-linux',ROOT)==self.sdk,'frozen SDK descriptor')
        _shared_lease()

    def run(self,role,runtime_input,identity,payload_sha,*,plain_input=None,incoming=None,
            evaluated=None,validation=None):
        self.check()
        require(role in ('encryptor','runtime','validator'),'fixed Native process role')
        require(type(identity)is dict and set(identity)==set(IDENTITY_FIELDS),'exact six-field artifact identity')
        require(all(sdk_records._hash(identity[key])for key in IDENTITY_FIELDS)and
                sdk_records._hash(payload_sha),'exact handoff digests')
        native=self.sdk['native'];assets=native['assets']
        require(identity['public_manifest_sha256']==assets['public_manifest_sha256']and
                identity['operator_spec_sha256']==native['operator_spec']['sha256'],'selected public context/spec')
        runtime_input=directory(runtime_input,self.result)
        require({p.name for p in runtime_input.iterdir()}==
                {'plan.json','operator-spec.json','source-plan.json','link-record.json'},'exact runtime metadata inventory')
        for name,key in [('plan.json','runtime_plan_sha256'),('operator-spec.json','operator_spec_sha256'),
                         ('source-plan.json','source_plan_sha256'),('link-record.json','link_record_sha256')]:
            path=sdk_records.base_sdk.path(str(runtime_input/name),within=self.result)
            require(path.stat().st_size<=256*1024 and sha(path)==identity[key],'unchanged runtime input bytes')
        require(self.used<self.max_processes,'native physical-process budget exhausted')
        self.used+=1
        control=self.result/('native-'+str(self.used)+'-'+role);control.mkdir(mode=0o700)
        public=sdk_records.base_sdk.path(assets['public_directory'],within=WORK,directory=True)
        mounts=[('--ro-bind',public,'/public'),('--ro-bind',runtime_input,'/runtime-input')]
        fields=[identity[key]for key in IDENTITY_FIELDS]
        executable=Path(native['runtime'if role=='runtime'else'client']['path'])
        expected_sha=native['runtime'if role=='runtime'else'client']['sha256']
        from native_sdk_manifest import EPOCH,ORIGIN
        if role=='encryptor':
            require(plain_input is not None and incoming is not None and
                    evaluated is None and validation is None,'public encryptor directories')
            plain_input=directory(plain_input,self.result);incoming=directory(incoming,self.result)
            require(not list(incoming.iterdir())and {p.name for p in plain_input.iterdir()}=={'input.json'},
                    'fresh encryption output and exact plaintext payload')
            require(json_sha(plain_input/'input.json',self.result)==payload_sha,'frozen private input bytes')
            mounts += [('--ro-bind',plain_input,'/plain-input'),('--bind',incoming,'/inputs')]
            arguments=['encrypt',EPOCH,ORIGIN,*fields,payload_sha,'public-only']
            report=control/'encrypt-report.json'
        elif role=='runtime':
            require(incoming is not None and evaluated is not None and
                    plain_input is None and validation is None,'public evaluator directories')
            incoming=directory(incoming,self.result);evaluated=directory(evaluated,self.result)
            require(not list(evaluated.iterdir())and json_sha(incoming/'manifest.json',self.result)==payload_sha,
                    'fresh evaluator output and bound input manifest')
            mounts += [('--ro-bind',incoming,'/inputs'),('--bind',evaluated,'/evaluated')]
            arguments=[EPOCH,ORIGIN,*fields,payload_sha];report=control/'runtime-report.json'
        else:
            require(evaluated is not None and validation is not None and
                    plain_input is None and incoming is None,'final-only client directories')
            evaluated=directory(evaluated,self.result);validation=directory(validation,self.result)
            require(not list(validation.iterdir())and json_sha(evaluated/'manifest.json',self.result)==payload_sha,
                    'fresh final decode output and bound ciphertext manifest')
            # No private manifest/key bytes are inspected by this host.
            private=sdk_records.base_sdk.path(assets['client_private_directory'],within=WORK,directory=True)
            mounts += [('--ro-bind',evaluated,'/evaluated'),('--ro-bind',private,'/client-private'),
                       ('--bind',validation,'/validation')]
            arguments=['decrypt-final',EPOCH,ORIGIN,*fields,payload_sha,assets['client_private_manifest_sha256']]
            report=validation/'report.json'
        require(sha(executable)==expected_sha,'selected Native executable unchanged')
        library=Path(self.sdk['poseidon_build'])/'libposeidon_shared.so'
        require(sha(library)==self.sdk['artifacts']['poseidon_library'],'original Poseidon library unchanged')
        search=os.environ['HECATE_PYTHON_LIBRARY_PATH']
        require(all(Path(item).is_absolute()and Path(item).is_relative_to('/nix/store')
                    for item in search.split(':')),'original Nix library closure')
        command=['/usr/bin/bwrap','--unshare-all','--die-with-parent','--new-session','--cap-drop','ALL',
            '--clearenv','--tmpfs','/','--ro-bind','/nix/store','/nix/store','--ro-bind',str(executable),'/bridge',
            '--ro-bind',str(library),'/library/libposeidon_shared.so','--proc','/proc','--dev','/dev',
            '--tmpfs','/tmp','--bind',str(control),'/control']
        for mode,path,target in mounts:command.extend([mode,str(path),target])
        command += ['--setenv','PATH','/nonexistent','--setenv','HOME','/nonexistent',
            '--setenv','LANG','C.UTF-8','--setenv','LD_LIBRARY_PATH','/library:'+search,
            '--setenv','OMP_NUM_THREADS','2','--setenv','OPENBLAS_NUM_THREADS','2',
            '--chdir','/control','--','/bridge',*arguments]
        save(control/'invocation.json',dict(role=role,argv=command,driver=self.driver,
            executable_sha256=expected_sha,private_file_reads_by_host=0))
        proof=None;started=time.monotonic()
        with native_slot(WORK/'cache/agent-native-slots'):
            with(control/'native.log').open('x')as stream:
                child=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,
                    env={'PATH':'/usr/bin:/bin'},close_fds=True,preexec_fn=limits,start_new_session=True)
                try:
                    while child.poll()is None:
                        require(time.monotonic()-started<240,'original finite native process timeout')
                        if(control/'native-ready.json').exists()and proof is None:
                            proof=prove_process(child,executable,expected_sha,control,role,EPOCH,self.driver)
                            save(control/'host-native-proof.json',proof)
                            save(control/'owner-ack.json',dict(role=role,nonce=EPOCH,owner_verified=True))
                        time.sleep(.05)
                    code=child.wait()
                except BaseException:
                    if child.poll()is None:
                        os.killpg(child.pid,signal.SIGTERM)
                        try:child.wait(timeout=5)
                        except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=5)
                    raise
        self.check()
        save(control/'outcome.json',dict(returncode=code,elapsed_seconds=time.monotonic()-started,
            native_process_number=self.used,owner_proof=proof is not None,no_retry=True))
        require(code==0 and proof is not None,'Native process failed; preserve outcome, never replay automatically')
        return read(report)

    def verify_plan(self,runtime_input,identity):
        """Original C++ verifier and complete linker binding, without a context."""
        self.check()
        require(self.metadata_used==0,'one bounded Runtime metadata check')
        require(type(identity)is dict and set(identity)==set(IDENTITY_FIELDS)and
                all(sdk_records._hash(identity[key])for key in IDENTITY_FIELDS),'metadata identity')
        runtime_input=directory(runtime_input,self.result)
        require({p.name for p in runtime_input.iterdir()}==
            {'plan.json','operator-spec.json','source-plan.json','link-record.json'},'metadata inventory')
        for name,key in [('plan.json','runtime_plan_sha256'),('operator-spec.json','operator_spec_sha256'),
                         ('source-plan.json','source_plan_sha256'),('link-record.json','link_record_sha256')]:
            require(json_sha(runtime_input/name,self.result)==identity[key],'bound metadata bytes')
        self.metadata_used+=1
        control=self.result/'native-metadata';control.mkdir(mode=0o700)
        executable=self.sdk['native']['runtime']
        require(sha(executable['path'])==executable['sha256'],'actual metadata verifier binary')
        library=Path(self.sdk['poseidon_build'])/'libposeidon_shared.so'
        require(sha(library)==self.sdk['artifacts']['poseidon_library'],'original metadata library')
        search=os.environ['HECATE_PYTHON_LIBRARY_PATH']
        require(all(Path(p).is_absolute()and Path(p).is_relative_to('/nix/store')for p in search.split(':')),
                'original Nix metadata library closure')
        command=['/usr/bin/bwrap','--unshare-all','--die-with-parent','--new-session','--cap-drop','ALL',
            '--clearenv','--tmpfs','/','--ro-bind','/nix/store','/nix/store',
            '--ro-bind',executable['path'],'/bridge','--ro-bind',str(library),'/library/libposeidon_shared.so',
            '--ro-bind',str(runtime_input),'/runtime-input','--proc','/proc','--dev','/dev','--tmpfs','/tmp',
            '--setenv','PATH','/nonexistent','--setenv','HOME','/nonexistent',
            '--setenv','LD_LIBRARY_PATH','/library:'+search,
            '--','/bridge','--check-bound-plan','/runtime-input',*[identity[key]for key in IDENTITY_FIELDS]]
        save(control/'invocation.json',dict(argv=command,encrypted_execution=False,new_keys=0,
            public_or_private_key_assets_mounted=False))
        with native_slot(WORK/'cache/agent-native-slots'):
            completed=subprocess.run(command,capture_output=True,text=True,timeout=30,
                env={'PATH':'/usr/bin:/bin'},close_fds=True,preexec_fn=limits)
        save(control/'result.json',dict(returncode=completed.returncode,stdout=completed.stdout,
            stderr=completed.stderr,encrypted_execution=False))
        require(completed.returncode==0,'Original Runtime metadata/final-output binding rejected')
        result=json.loads(completed.stdout)
        require(result.get('complete_derived_plan_and_ordered_outputs_verified')is True and
                result.get('encrypted_execution')is False,'Pure complete Runtime metadata verification')
        self.check()
        return result
