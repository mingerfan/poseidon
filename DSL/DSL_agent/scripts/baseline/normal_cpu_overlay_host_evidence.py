"""Offline verification of the actual host witness for an ordinary CPU canary.

The failed old canary has no such witness. Numerical values or a successful
ledger row alone cannot replace raw process, mount, resource and DSO evidence.
No native process, key, model, or arbitrary evidence-supplied Python is executed.
"""
import os
from pathlib import Path
import re
import stat

from delivery_component import bound, canonical, checked_bound, read_bytes, require, sha
from first_output_backend import _json

LIMITS={'Max cpu time':[150,155],'Max file size':[16*1024**2]*2,
        'Max core file size':[0,0],'Max open files':[128,128],
        'Max address space':[4*1024**3]*2}
NAMESPACES={'mnt','net','pid','ipc','uts'}


def _mounts(text):
    require(type(text)is str and len(text)<1024**2,'Bounded raw mount observations')
    rows={}
    for line in text.splitlines():
        left,right=line.split(' - ',1);a,b=left.split(),right.split()
        require(len(a)>=6 and len(b)>=3,'Raw mountinfo fields')
        destination=re.sub(r'\\([0-7]{3})',lambda m:chr(int(m[1],8)),a[4])
        require(destination not in rows,'Duplicate raw mount destination')
        rows[destination]=dict(options=a[5].split(','),filesystem=b[0])
    return rows


def _command_parts(command):
    require(type(command)is list and command[0]=='/usr/bin/bwrap' and '--'in command,'Original frozen sandbox command')
    cut=command.index('--');binds=[];environment={};special={};at=1
    widths={'--cap-drop':1,'--ro-bind':2,'--bind':2,'--setenv':2,'--dir':1,'--proc':1,'--dev':1,'--tmpfs':1,'--chdir':1}
    bare={'--unshare-all','--die-with-parent','--new-session','--clearenv'};flags=set()
    while at<cut:
        key=command[at]
        if key in bare:require(key not in flags,'Repeated original sandbox flag');flags.add(key);at+=1;continue
        require(key in widths,'Unexpected sandbox command option')
        values=command[at+1:at+1+widths[key]];require(len(values)==widths[key],'Missing sandbox operands')
        if key in ('--ro-bind','--bind'):
            binds.append(dict(source=values[0],destination=values[1],mode='ro'if key=='--ro-bind'else'rw'))
        elif key=='--setenv':
            require(values[0]not in environment,'Duplicate sandbox environment name');environment[values[0]]=values[1]
        elif key in ('--proc','--dev','--tmpfs'):special[values[0]]=key
        at+=1+widths[key]
    require(flags==bare and special=={'/proc':'--proc','/dev':'--dev','/tmp':'--tmpfs'},'Original complete isolation flags')
    require(len({b['destination']for b in binds})==len(binds),'Duplicate original bind destination')
    return binds,environment,special,command[cut+1:]


def verify_records(contract,raw,host,*,expected_command,context):
    """Pure comparison with original command and SDK; inert controls use this too."""
    for value in (contract,raw,host):checked_bound(value)
    require(contract['format']=='poseidon-B-original-ordinary-CPU-exact-mount-contract-v1'
            and raw['format']=='poseidon-B-ordinary-CPU-native-raw-observation-v1'
            and host['format']=='poseidon-B-actual-ordinary-CPU-exact-mount-host-proof-v2','Complete exact-mount host witness versions required')
    require(contract['command']==expected_command,'Host contract differs from original selected-DSO sandbox builder')
    binds,declared_environment,special,argv=_command_parts(expected_command)
    # bwrap's --chdir sets PWD in the effective child environment even after
    # --clearenv. This describes the existing command; no variable is added to it.
    require(expected_command.count('--chdir')==1,'One original client working directory')
    environment=dict(declared_environment,PWD=expected_command[expected_command.index('--chdir')+1])
    expected={b['destination']:b for b in binds};declared={b['destination']:b for b in contract['binds']}
    require(len(declared)==len(contract['binds']) and set(declared)==set(expected)
            and contract['environment']==declared_environment and contract['special']==special,'Frozen command mount/environment contract')
    identities=raw['actual_expected_bind_identities']
    require(type(identities)is dict and set(identities)==set(expected),'Every actual bind source identity required')
    for destination,want in expected.items():
        saved=declared[destination];identity=saved['identity']
        require({k:saved[k]for k in ('source','destination','mode')}==want
                and type(identity)is dict and set(identity)=={'device','inode','kind'}
                and all(type(v)is int and v>=0 for v in identity.values())
                and identities[destination]==identity,'Actual source device/inode/type differs from frozen bind')
    require(host['command_contract_binding']==contract['binding'] and host['raw_observation_binding']==raw['binding']
            and host['raw_observations_saved_before_validation'] is True,'Durable raw observations must precede host acceptance')
    require(host['actual_mountinfo']==raw['actual_mountinfo']
            and host['actual_native_limits_text']==raw['actual_limits'],'Raw mount/resource data changed between observation and acceptance')
    rows=_mounts(raw['actual_mountinfo']);require(set(expected)<=set(rows),'Missing actual original bind mounts')
    for destination,want in expected.items():
        options=rows[destination]['options']
        require(want['mode']in options and not(want['mode']=='ro'and'rw'in options),'Actual mount read/write boundary')
    allowed={'/':'tmpfs','/proc':'proc','/tmp':'tmpfs','/dev':'tmpfs','/dev/pts':'devpts','/dev/shm':'tmpfs','/dev/mqueue':'mqueue'}
    devices={'/dev/null','/dev/zero','/dev/full','/dev/random','/dev/urandom','/dev/tty'}
    for destination,row in rows.items():
        if destination in expected:continue
        if destination in allowed:require(row['filesystem']==allowed[destination],'Original private filesystem type')
        else:require(destination in devices,'Unexpected actual home/private/reference/extra mount')
    require({'/',*special}<=set(rows),'Private namespace roots missing')
    actual_binds=[dict(expected[d],identity=declared[d]['identity'])for d in sorted(expected)]
    require(host['actual_bind_mounts']==actual_binds,'Accepted mounts differ from raw/frozen source identities')
    require(raw['actual_native_argv']==argv and raw['actual_environment_names']==sorted(environment)
            and raw['actual_environment_equals_frozen'] is True and host['cleared_native_environment_verified'] is True,'Actual cleared environment and original client arguments')
    require(raw['actual_environment_per_expected_key_matches']=={k:True for k in environment}
            and raw['actual_PWD_equals_original_chdir_out'] is True,'Every measured original environment value and bwrap PWD must match')
    namespaces=raw['actual_namespaces'];require(set(namespaces)==NAMESPACES and host['namespaces']==namespaces,'Complete actual namespace observations')
    for name,pair in namespaces.items():
        require(set(pair)=={'native','driver'} and all(type(v)is str and re.fullmatch(name+r':\[[0-9]+\]',v)for v in pair.values())
                and pair['native']!=pair['driver'],'Actual independent namespace required')
    for name,pair in LIMITS.items():
        line=next((v for v in raw['actual_limits'].splitlines()if v.startswith(name+' ')),None)
        require(line is not None and list(map(int,line.split()[len(name.split()):][:2]))==pair,'Actual original native resource limits')
    require(host['actual_native_limits_verified']==LIMITS,'Host resource summary differs from raw limits')
    require(raw['actual_executable_sha256']==host['executable_sha256']==context['inner']['artifacts']['executor']
            and raw['actual_DSO_sha256']==host['selected_library_sha256']==context['inner']['artifacts']['poseidon_library']
            and raw['actual_profile_sha256']==context['outer']['profile']['sha256']
            and raw['actual_operator_spec_sha256']==context['outer']['operator_spec']['sha256'],'Actual original executor, repaired DSO and parameter bytes')
    library=raw['actual_DSO_identity'];target='/selected-poseidon/libposeidon_shared.so'
    require(library==declared[target]['identity'] and library['kind']==stat.S_IFREG,'Actual mapped selected DSO identity')
    mapped=[]
    for line in raw['actual_maps'].splitlines():
        fields=line.split(maxsplit=5)
        if len(fields)<5:continue
        major,minor=[int(v,16)for v in fields[3].split(':')]
        if int(fields[4])==library['inode'] and os.makedev(major,minor)==library['device']:mapped.append(line)
    require(mapped and any('x'in line.split()[1]for line in mapped)
            and all('libposeidon'not in line or line in mapped for line in raw['actual_maps'].splitlines())
            and host['selected_library_actual_maps']==mapped and host['actual_mapped_selected_DSO_verified'] is True,'Actual executable repaired-DSO mapping only')
    for flag in ('original_parameter_file_bytes_verified','selected_library_readonly_mount_verified','isolated_network_namespace_verified','broad_host_home_bind_absent','key_memory_not_read'):
        require(host[flag]is True,'Incomplete host privacy/isolation witness')
    require(host['project_home_mounted'] is host['reference_mounted'] is host['persistent_private_directory_mounted'] is False
            and host['private_file_content_reads']==host['private_file_hashes']==raw['private_file_content_reads']==raw['private_file_hashes']==0
            and raw['key_memory_read'] is False and host['process_local_ephemeral_key_sets_maximum']==1
            and host['persistent_key_epochs']==0,'Original temporary-key and private data boundary')
    return dict(actual_bind_count=len(binds),raw_observation_binding=raw['binding'],host_proof_binding=host['binding'],command_contract_binding=contract['binding'])


def _record(path):return checked_bound(_json(path))


def _expected_command(root,context):
    from normal_cpu_overlay_sandbox import command
    client=Path(root)/'client'
    return command(client/'payload.json',client,
        ['/poseidon-agent-cpu','/out/bound/plan.json','/operator-spec.json','/out/bound/bundle','/out/inputs.json','/out/execution.json'],
        selected_library=context['library'],trace_loader=True)


def verify(root,context,canary,result,row):
    """Bind the completed B controller, host observer and immutable public records."""
    root=Path(root);paths=[root/name for name in ['NATIVE_COMMAND_CONTRACT.json','host-native-raw-observation.json','host-native-proof.json',
        'B_SOURCE_FREEZE.json','host-owner-proof.json','owner-ready.json','owner-ack.json','host-plan.json','B_REVIEW.json','RESOURCE_TERMINAL.json']]
    pins={str(p):sha(read_bytes(p,4*1024**2))for p in paths}
    contract,raw,host=(_record(root/name)for name in ['NATIVE_COMMAND_CONTRACT.json','host-native-raw-observation.json','host-native-proof.json'])
    summary=verify_records(contract,raw,host,expected_command=_expected_command(root,context),context=context)
    freeze=_record(root/'B_SOURCE_FREEZE.json');owner=_record(root/'host-owner-proof.json');ready=_record(root/'owner-ready.json')
    host_plan=_json(root/'host-plan.json');ack=_json(root/'owner-ack.json')
    require(host['canary_binding']==freeze['canary_binding']==ready['canary_binding']==host_plan['canary_binding']==canary['binding']
            and host['driver']==row['owner']==owner['owner']==ready['owner'] and host['owner']['pid']==raw['host_pid']
            and host['driver']['pid']==raw['driver_host_pid'] and host['owner']['boot_id']==host['driver']['boot_id'], 'Actual native/driver/shared-owner lineage')
    require(owner['verified_from_host'] is True and owner['nonce']==ready['nonce']==host_plan['nonce']==ack['nonce']
            and ack['owner_verified'] is True and host['ancestry'][0]==host['owner']['pid'] and host['ancestry'][-1]==host['driver']['pid'], 'Actual host observation owner acknowledgement')
    require(freeze['host_plan_sha256']==pins[str(root/'host-plan.json')] and freeze['all_sources_checked'] is True
            and freeze['selected_inner_sdk']==canary['selected_inner_sdk']
            and freeze['selected_library_sha256']==context['inner']['artifacts']['poseidon_library']
            and freeze['original_Boot_profile_modified'] is False and freeze['persistent_key_epochs']==0,'Original source/SDK/key scope')
    manifests=[]
    source_root=Path(context['inner']['backend_source_root'])/'DSL/DSL_agent/docs/whole_models'
    for name,digest in host_plan['source'].items():
        path=Path(name)
        if path.name!='manifest.json' or not path.is_relative_to(source_root):continue
        value=_json(path)
        if value.get('binding')!=freeze['manifest_binding']:continue
        checked_bound(value);require(sha(read_bytes(path))==digest,'Frozen B controller manifest changed');manifests.append((path,value))
    require(len(manifests)==1,'One source-bound B host controller required')
    manifest_path,manifest=manifests[0];directory=manifest_path.parent
    require(manifest['result_root']==str(root) and all(host_plan['source'].get(p)==h for p,h in canary['source_files'].items()), 'Actual B output and original canary adapters')
    # B host/driver sources remain frozen. A evaluation sources can acquire a new
    # version after the run; their historical hashes remain in the host plan.
    for name,digest in host_plan['source'].items():
        path=Path(name)
        if not path.is_relative_to(directory):continue
        require('.env'not in path.parts and 'client-private'not in path.parts,'Public controller source only')
        require(sha(read_bytes(path,16*1024**2))==digest,'Immutable B host/controller source changed');pins[str(path)]=digest
    terminal_path=directory/'actual-supervisor-terminal.json';terminal=_record(terminal_path);pins[str(terminal_path)]=sha(read_bytes(terminal_path))
    require(terminal['returncode']==0 and terminal['owner_verified'] is terminal['native_host_verified'] is True
            and terminal['native_clients_observed']==1 and terminal['canary_binding']==canary['binding'],'Complete actual host supervisor terminal')
    review=_record(root/'B_REVIEW.json');resources=_record(root/'RESOURCE_TERMINAL.json')
    require(review['A_primitive_report_binding']==result['binding'] and review['canary_binding']==canary['binding']
            and review['B_source_manifest_binding']==freeze['manifest_binding']
            and review['actual_host_native_proof_binding']==host['binding']
            and review['actual_DSO_loader_receipt_binding']==result['actual_DSO_binding']
            and review['resource_terminal_binding']==resources['binding'], 'B numerical/source/host/resource review bindings')
    require(review['input_groups']==result['input_groups'] and review['compared_real_values']==result['compared_values']
            and review['failed_real_values']==result['failed_values'] and review['max_real_absolute_error']==result['max_absolute_error']
            and review['complete_normal_SDK_qualification'] is False,'Scoped real-output numerical review only')
    require(resources['CPU_soft']==150 and resources['CPU_hard']==155 and resources['AS_bytes']==4*1024**3
            and resources['FSIZE_bytes']==16*1024**2 and resources['NOFILE']==128 and resources['CORE']==0
            and resources['native_wall_limit_seconds']==150 and resources['batch_wall_limit_seconds']==row['plan']['wall_seconds']
            and resources['evidence_bytes']==row['plan']['reserve_bytes']
            and resources['actual_native_exit']==0 and resources['actual_native_processes_observed']==1
            and resources['actual_native_limits_verified']==LIMITS
            and resources['API']==resources['compilation']==resources['reference_model_forwards']==resources['downloads']==resources['Boot']==0
            and resources['ephemeral_key_sets']==1 and resources['persistent_key_epochs']==0,'Actual original finite native resources')
    require(all(sha(read_bytes(p,16*1024**2))==h for p,h in pins.items()),'Host evidence changed while inspected')
    return bound(dict(format='poseidon-normal-CPU-complete-host-evidence-v1',canary_binding=canary['binding'],**summary,
        B_manifest_binding=manifest['binding'],B_review_binding=review['binding'],resource_terminal_binding=resources['binding'],
        supervisor_terminal_binding=terminal['binding'],evidence_sha256=pins,
        output_scope='original real-output client only; imaginary values and other models are not certified'))
