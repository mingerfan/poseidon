"""Pure offline consistency checks for frozen native-role host records.

Expected source types, identities and mount origins come from the independently
frozen prelaunch contract and parent FD/mount record. No declared source path,
/proc path, key or native file is opened or resolved. Actual collector trust,
source/build/terminal and input/same-key/final-only provenance remain separate.
"""
import hashlib
import json
import os
from pathlib import PurePosixPath
import re
import stat

CONTRACT='poseidon-B-native-role-host-contract-v3'
RAW='poseidon-B-native-role-durable-raw-v3'
CONSISTENCY='poseidon-portable-native-role-record-consistency-v1'
NS=('pid','user','mnt','net','ipc','uts')
ROLES={'trusted_zero_HE_sentinel','public_encryptor','public_evaluator','original_final_client'}
GENERATED={'/':('tmpfs','/'),'/tmp':('tmpfs','/'),'/proc':('proc','/'),'/dev':('tmpfs','/'),'/dev/pts':('devpts','/')}
PROTECTED=('/home/lhohy','/home/lhohy/Code/Poseidon')
LIMITS={'Max cpu time':(150,155),'Max file size':(16777216,16777216),'Max core file size':(0,0),'Max open files':(128,128),'Max address space':(4294967296,4294967296)}
LIMIT_UNITS={'Max cpu time':'seconds','Max file size':'bytes','Max core file size':'bytes','Max open files':'files','Max address space':'bytes'}

SOURCE_ORIGINS={'/home/lhohy/Code/Poseidon/DSL/DSL_agent/docs/whole_models/proposals/bootstrap-evalmod-output-diagnostic-r1929/original-remaining43-domain-campaign-r3105/lossless-resource64-v3-r3127/host-record-controls-r3130/launch/program_actual_host_record_command_complete_draft_r3115.py': '6e1459fdbaa3b3c6842c22fb452c484cf2fb40a8191fe342885920789d2788d3', '/home/lhohy/Code/Poseidon/DSL/DSL_agent/docs/whole_models/proposals/bootstrap-evalmod-output-diagnostic-r1929/live-native-host-observer-v4-r3138/launch/native_observer.py': 'c5aec9a2768ec018b3340b96013f074151f7787bb8111653018eebb4200b6f6d', '/home/lhohy/Code/Poseidon/DSL/DSL_agent/scripts/baseline/normal_cpu_overlay_host_evidence.py': '6d6fddb2e19a220e9f2a6e3032149c4fa34c869df6c8eb80d4fc22eba7323575'}

def need(ok, message):
    if not ok:
        raise ValueError('Program host record: ' + message)

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, allow_nan=False).encode('utf-8')

def bound(value):
    unsigned = {k: v for k, v in value.items() if k != 'binding'}
    return dict(unsigned, binding=hashlib.sha256(canonical(unsigned)).hexdigest())

def fields(value, names):
    need(type(value) is dict and set(value) == set(names), 'closed record fields')

def identifier(value, width=64):
    need(type(value) is str and re.fullmatch('[0-9a-f]{%d}' % width, value), 'hex identity')
    return value

def checked(value, expected):
    identifier(expected)
    need(type(value) is dict and value.get('binding') == expected
         and bound(value)['binding'] == expected, 'externally expected document binding')
    return value

def integer(value, minimum=0):
    need(type(value) is int and value >= minimum, 'nonnegative integer')
    return value

def path(value):
    need(type(value) is str and 0 < len(value) <= 4096 and value.startswith('/')
         and '\x00' not in value and '\n' not in value, 'absolute public metadata path')
    parts = value.split('/')[1:]
    need(value == '/' or all(p not in ('', '.', '..') for p in parts), 'normalized public metadata path')
    return value

def raw_limits(text):
    need(type(text) is str and len(text.encode()) <= 65536, 'bounded original raw limits')
    result = {}
    for prefix, expected in LIMITS.items():
        rows = [s.split() for s in text.splitlines() if s.startswith(prefix+' ')]
        need(len(rows) == 1, 'exact original limit row '+prefix)
        cells = rows[0][len(prefix.split()):]
        need(len(cells) == 3 and all(re.fullmatch('[0-9]+', x) for x in cells[:2]), 'numeric original limits')
        need(cells[2] == LIMIT_UNITS[prefix], 'original limit units '+prefix)
        actual = tuple(map(int, cells[:2]))
        need(actual == expected, 'original limit '+prefix)
        result[prefix] = list(actual)
    return result

def unescape(value):
    return re.sub(r'\\([0-7]{3})', lambda m: chr(int(m[1], 8)), value)

def mounts(text):
    need(type(text) is str and 0 < len(text.encode()) <= 1048576, 'bounded original mountinfo')
    result = {}
    for line in text.splitlines():
        cols = line.split()
        need(cols.count('-') == 1, 'mountinfo separator')
        split = cols.index('-')
        need(split >= 6 and len(cols) == split+4, 'complete mountinfo columns')
        destination = path(unescape(cols[4]))
        need(destination not in result, 'one actual mount per destination')
        need(re.fullmatch('[0-9]+:[0-9]+', cols[2]), 'actual mount device')
        options = cols[5].split(',')
        need(('ro' in options) != ('rw' in options), 'unambiguous actual mount access')
        result[destination] = dict(device=cols[2], root=unescape(cols[3]),
            readonly='ro' in options, filesystem=cols[split+1], source=unescape(cols[split+2]))
    return result

def device(value):
    need(type(value) is str and re.fullmatch('[0-9a-f]+:[0-9a-f]+', value), 'mmap device')
    a, b = value.split(':')
    return '%d:%d' % (int(a, 16), int(b, 16))

def mapped_library(text, expected):
    fields(expected, {'path', 'device', 'inode', 'sha256'})
    path(expected['path']); identifier(expected['sha256']); integer(expected['inode'], 1)
    need(type(text) is str and len(text.encode()) <= 1048576, 'bounded original mmap observation')
    actual = []
    for line in text.splitlines():
        cols = line.split(None, 5)
        need(len(cols) >= 5, 'original mmap columns')
        filename = cols[5] if len(cols) == 6 else ''
        if 'libposeidon' not in filename and filename != expected['path']:
            continue
        need(filename == expected['path'], 'alternate or deleted Poseidon mapping')
        need(device(cols[3]) == expected['device'] and int(cols[4]) == expected['inode'], 'actual mapped DSO inode/device')
        ends = cols[0].split('-')
        need(len(ends) == 2 and all(re.fullmatch('[0-9a-f]+', v) for v in ends), 'mapping address range')
        first, last = (int(v, 16) for v in ends)
        need(first < last and re.fullmatch('[r-][w-][x-][ps]', cols[1]), 'mapping permissions')
        need(not ('w' in cols[1] and 'x' in cols[1]), 'no writable executable DSO mapping')
        actual.append(dict(first=first, last=last, permissions=cols[1], offset=int(cols[2], 16)))
    need(actual and any('x' in v['permissions'] for v in actual), 'actual executable DSO mapping')
    ordered = sorted(actual, key=lambda x: x['first'])
    need(all(a['last'] <= b['first'] for a, b in zip(ordered, ordered[1:])), 'nonoverlapping DSO mappings')
    return actual

def _command_parts(command):
    need(type(command)is list and command[0]=='/usr/bin/bwrap' and '--'in command,'Original frozen sandbox command')
    cut=command.index('--');binds=[];environment={};special={};at=1
    widths={'--cap-drop':1,'--ro-bind':2,'--bind':2,'--setenv':2,'--dir':1,'--proc':1,'--dev':1,'--tmpfs':1,'--chdir':1}
    bare={'--unshare-all','--die-with-parent','--new-session','--clearenv'};flags=set()
    while at<cut:
        key=command[at]
        if key in bare:need(key not in flags,'Repeated original sandbox flag');flags.add(key);at+=1;continue
        need(key in widths,'Unexpected sandbox command option')
        values=command[at+1:at+1+widths[key]];need(len(values)==widths[key],'Missing sandbox operands')
        if key in ('--ro-bind','--bind'):
            binds.append(dict(source=values[0],destination=values[1],mode='ro'if key=='--ro-bind'else'rw'))
        elif key=='--setenv':
            need(values[0]not in environment,'Duplicate sandbox environment name');environment[values[0]]=values[1]
        elif key in ('--proc','--dev','--tmpfs'):special[values[0]]=key
        at+=1+widths[key]
    need(flags==bare and special=={'/proc':'--proc','/dev':'--dev','/tmp':'--tmpfs'},'Original complete isolation flags')
    need(len({b['destination']for b in binds})==len(binds),'Duplicate original bind destination')
    return binds,environment,special,command[cut+1:]

def parse_command(command):
    # The original native ABI has an explicit empty root. The A parser is
    # reused for all remaining arguments without changing the actual command.
    need(command[:10] == ['/usr/bin/bwrap', '--unshare-all', '--die-with-parent',
         '--new-session', '--cap-drop', 'ALL', '--clearenv', '--tmpfs', '/', '--ro-bind'],
         'original complete native isolation prefix')
    need(command.count('--chdir') == 1 and command[command.index('--chdir')+1] == '/control',
         'original native working directory')
    parsed = command[:7] + command[9:]
    binds, env, special, argv = _command_parts(parsed)
    need(command[command.index('--cap-drop')+1] == 'ALL' and argv and argv[0] == '/bridge',
         'original native executable and dropped capabilities')
    by = {v['destination']: v for v in binds}
    need(by.get('/nix/store') == dict(source='/nix/store', destination='/nix/store', mode='ro')
         and by.get('/bridge', {}).get('mode') == 'ro'
         and by.get('/library/libposeidon_shared.so', {}).get('mode') == 'ro'
         and by.get('/control', {}).get('mode') == 'rw', 'original native base bindings')
    need(set(env) == {'PATH', 'HOME', 'LANG', 'LD_LIBRARY_PATH', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS'}
         and env['PATH'] == env['HOME'] == '/nonexistent' and env['LANG'] == 'C.UTF-8'
         and env['OMP_NUM_THREADS'] == env['OPENBLAS_NUM_THREADS'] == '2'
         and env['LD_LIBRARY_PATH'].startswith('/library:'), 'original native public environment')
    return binds, dict(env, PWD='/control'), argv

def _owner(value):
    fields(value,{'pid','start_ticks','boot_id'})
    integer(value['pid'],2);integer(value['start_ticks'],1)
    need(type(value['boot_id'])is str and re.fullmatch('[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}',value['boot_id']),'original owner boot')
    return value


def _namespace(value):
    fields(value,set(NS))
    need(all(type(value[n])is str and re.fullmatch(n+r':\[[0-9]+\]',value[n])for n in NS),'all six raw namespace identities')
    return value


def _identity(value,kinds):
    fields(value,{'device','inode','kind'})
    integer(value['device']);integer(value['inode'],1)
    need(type(value['kind'])is int and value['kind']in kinds,'independently frozen source type')
    return value


def _parent_mounts(text):
    need(type(text)is str and 0<len(text.encode())<=1048576,'bounded original parent mount record')
    result={}
    for line in text.splitlines():
        cols=line.split();need(cols.count('-')==1,'parent mount separator');cut=cols.index('-')
        need(cut>=6 and len(cols)==cut+4 and cols[0].isdecimal(),'complete parent mount fields')
        mount_id=int(cols[0]);need(mount_id>0 and mount_id not in result,'distinct parent mount IDs, including overmounts')
        need(re.fullmatch('[0-9]+:[0-9]+',cols[2]),'parent mount device')
        result[mount_id]=dict(destination=path(unescape(cols[4])),device=cols[2],root=path(unescape(cols[3])),
            filesystem=cols[cut+1],source=unescape(cols[cut+2]))
    return result


def _origin(source,identity,origin,parent_mounts):
    path(source)
    fields(origin,{'device','root','filesystem','source','parent_mount_id','source_fdinfo'})
    integer(origin['parent_mount_id'],1)
    text=origin['source_fdinfo'];need(type(text)is str and len(text)<=65536,'bounded original source FD info')
    def number(name):
        rows=[line.split()[1:]for line in text.splitlines()if line.startswith(name+':')]
        need(len(rows)==1 and len(rows[0])==1 and rows[0][0].isdecimal(),'one independently recorded FD '+name)
        return int(rows[0][0])
    mount_id=number('mnt_id');need(mount_id==origin['parent_mount_id']and mount_id in parent_mounts,'actual prelaunch source FD selects its mount')
    need(number('ino')==identity['inode'],'actual source FD inode, not a same-path substitute')
    row=parent_mounts[mount_id]
    expected_device='%d:%d'%(os.major(identity['device']),os.minor(identity['device']))
    need(origin['device']==row['device']==expected_device
        and origin['filesystem']==row['filesystem']and origin['source']==row['source'],'source FD/mount filesystem identity')
    root=PurePosixPath(path(origin['root']));mount_root=PurePosixPath(row['root'])
    need(root==mount_root or root.is_relative_to(mount_root),'original opened source lies under selected mount root')
    resolved=PurePosixPath(row['destination'])/root.relative_to(mount_root)
    path(str(resolved));return resolved,root


def _public_paths(paths):
    forbidden={'.env','.ssh','.aws','secret.bin','secret.key'}
    for value in paths:
        need(not forbidden.intersection(value.parts)and 'reference'not in str(value).lower(),'no key/reference alias in frozen path')


def verify_source_scope(contract,expected_binding,parent_bytes,expected_parent_sha256):
    checked(contract,expected_binding);identifier(expected_parent_sha256)
    need(contract['format']==CONTRACT and contract['role']in ROLES,'versioned native-role source contract')
    need(type(parent_bytes)is bytes and len(parent_bytes)<=1048576
        and hashlib.sha256(parent_bytes).hexdigest()==expected_parent_sha256==contract['parent_mount_source']['sha256'],
        'independently selected prelaunch parent bytes')
    def pairs(rows):
        result={}
        for key,value in rows:
            need(key not in result,'duplicate parent JSON field');result[key]=value
        return result
    def bad(value):raise ValueError('Nonfinite parent mount record')
    parent=json.loads(parent_bytes,object_pairs_hook=pairs,parse_constant=bad)
    checked(parent,contract['parent_mount_source']['binding'])
    need(parent['format']=='poseidon-B-prelaunch-parent-mount-origin-v1'and parent['owner']==contract['owner'],
        'prelaunch parent and original shared owner')
    need(type(parent['at_ns'])is int and type(contract['frozen_at_ns'])is int
        and 0<parent['at_ns']<contract['frozen_at_ns'],'parent identities measured before contract freeze')
    _owner(contract['owner'])
    mount_rows=_parent_mounts(parent['raw_mountinfo'])
    binds,environment,argv=parse_command(contract['command'])
    need(contract['argv']==argv and contract['cwd']=='/control'and contract['environment']==environment,
        'complete original argv and effective environment')
    need(hashlib.sha256(canonical(contract['command'])).hexdigest()==contract['command_sha256'],'original command hash')
    records=contract['bindings'];need(type(records)is list and len(records)==len(binds),'every frozen binding')
    need([{k:b[k]for k in('source','destination','mode')}for b in records]==binds,'exact original command bind order and identity')
    protected=[PurePosixPath(p)for p in PROTECTED]
    host=PurePosixPath(path(contract['raw_root']['path']));_identity({k:contract['raw_root'][k]for k in('device','inode','kind')},{stat.S_IFDIR})
    summaries=[]
    for b in records:
        _identity(b['identity'],{stat.S_IFREG,stat.S_IFDIR})
        lexical=PurePosixPath(path(b['source']));destination=path(b['destination'])
        resolved,physical=_origin(str(lexical),b['identity'],b['source_origin'],mount_rows)
        _public_paths([lexical,resolved,physical,PurePosixPath(destination)])
        need(not(host==resolved or host.is_relative_to(resolved))and not(host==lexical or host.is_relative_to(lexical)),
            'trusted raw evidence absent from every frozen native source range')
        need(destination not in('/','/home','/keys')and not destination.startswith('/proc/'),'no broad or proc override destination')
        if b['identity']['kind']==stat.S_IFDIR:
            need(not any(p==source or p.is_relative_to(source)for source in(lexical,resolved,physical)for p in protected),
                'no virtual or underlying broad home/checkout directory bind')
            need('public_file'not in b,'directory cannot substitute for a measured public file')
        else:
            public=b.get('public_file');fields(public,{'device','inode','kind','bytes','sha256'})
            need({k:public[k]for k in('device','inode','kind')}==b['identity'],'frozen public file identity')
            integer(public['bytes'],1);need(public['bytes']<=512*1024**2,'bounded frozen public file');identifier(public['sha256'])
        category=b['category']
        if category=='original_private_view':
            need(contract['role']=='original_final_client'and destination=='/client-private'and b['mode']=='ro'
                and b['identity']['kind']==stat.S_IFDIR and resolved.name=='client-private',
                'unchanged original final-client private directory ABI only')
        else:
            need(all('private'not in str(p).lower()for p in(lexical,resolved,physical))and 'private'not in destination.lower(),
                'no hidden private source aliases')
            need(category in('system','public_code','public_runtime','public_parameters','public_ciphertext','public_plaintext','control','public_output'),
                'explicit frozen public binding category')
        if contract['role']in('public_evaluator','original_final_client'):
            need(category!='public_plaintext','evaluator/client cannot mount original plaintext')
        need(b['mode']=='ro'or category in('control','public_output'),'only declared control/output writable')
        summaries.append(dict(destination=destination,resolved_source=str(resolved),physical_root=str(physical),kind=b['identity']['kind'],category=category))
    need(set(contract['device_origins'])=={'/dev/'+name for name in('null','zero','full','random','urandom','tty')},'original native device set')
    for destination,item in contract['device_origins'].items():
        _identity(item['identity'],{stat.S_IFCHR});_origin(destination,item['identity'],item['source_origin'],mount_rows)
    return bound(dict(format='poseidon-portable-native-source-scope-consistency-v1',contract_binding=expected_binding,
        prelaunch_parent_binding=parent['binding'],source_ranges=summaries,protected_roots=list(PROTECTED),
        no_declared_source_path_opened_or_resolved=True,actual_prelaunch_source_collector_authentication=False,
        private_file_content_or_hash_reads=0,whole_program_or_numeric_or_first_or_Bootstrap_qualification=False,execution_authority=False))


def verify_record(contract,raw,expected_binding,expected_raw_binding,expected_sources,*,parent_mount_bytes,expected_parent_sha256,test_only):
    need(type(test_only)is bool,'explicit retained or fixture scope')
    checked(contract, expected_binding); checked(raw,expected_raw_binding)
    need(contract['format']==CONTRACT and raw['format']==RAW and contract['role'] in ROLES, 'versioned role record')
    need(contract['observer_sources']==expected_sources==raw['observer_sources'], 'external observer-source freeze')
    need(raw['contract_binding']==expected_binding and raw['run_nonce']==contract['run_nonce'] and not raw['errors'], 'complete source-bound observation')
    need(raw['owner_start']==raw['owner_end']==raw['observer_owner']==contract['owner'], 'original live owner unchanged')
    _owner(raw['process_start']); _owner(contract['owner'])
    need(raw['process_start']==raw['process_end'] and type(raw['uid']) is int and raw['uid']==contract['owner_uid'], 'stable native owner and uid')
    need(type(raw['started_at_ns']) is int and type(raw['finished_at_ns']) is int
         and contract['frozen_at_ns'] < raw['started_at_ns'] <= raw['finished_at_ns'], 'contract before actual observation')
    need(raw['argv']==contract['argv'] and raw['cwd']==contract['cwd'], 'actual unchanged native argv/cwd')
    env=raw['environment']; names=env['names']
    need(env['invalid_fields']==0 and len(names)==len(set(names)) and set(names)==set(contract['environment'])
         and canonical(env['public_values'])==canonical(contract['environment']), 'exact effective public environment including PWD')
    _namespace(raw['namespaces']); _namespace(raw['owner_namespaces'])
    need(raw['owner_namespaces']==contract['owner_namespaces'] and all(raw['namespaces'][n]!=raw['owner_namespaces'][n] for n in NS), 'all six native namespaces independent')
    ancestry=raw['ancestry']; need(1<len(ancestry)<=64 and ancestry[0]==raw['process_start'] and ancestry[-1]==contract['owner'], 'live original ancestry')
    need(len({(v['pid'],v['start_ticks']) for v in ancestry})==len(ancestry), 'no ancestry cycle')
    for node in ancestry: _owner(node); need(node['boot_id']==contract['owner']['boot_id'], 'same host boot')
    source_scope=verify_source_scope(contract,expected_binding,parent_mount_bytes,expected_parent_sha256)
    need(hashlib.sha256(canonical(contract['command'])).hexdigest()==contract['command_sha256'], 'original full command bytes')
    actual=mounts(raw['mountinfo']); allowed=set(GENERATED)|set(contract['device_origins'])|{b['destination'] for b in contract['bindings']}
    need(set(actual)==allowed, 'complete exact actual mount destinations')
    for dest, policy in contract['generated_mount_policy'].items():
        m=actual[dest]; need(m['filesystem']==policy['filesystem'] and m['root']==policy['root'] and m['readonly']==policy['readonly'], 'predeclared generated system filesystem '+dest)
    for b in contract['bindings']:
        dest=b['destination']; m=actual[dest]
        need(canonical(raw['binding_destinations'][dest])==canonical(b['identity']) and canonical(raw['source_identities'][b['source']])==canonical(b['identity']), 'source and actual destination identity '+dest)
        need(m['readonly']==(b['mode']=='ro') and all(m[k]==b['source_origin'][k] for k in ('device','root','filesystem','source')), 'prelaunch-derived bind range/mode '+dest)
    for dest, expected in contract['device_origins'].items():
        need(canonical(raw['device_destinations'][dest])==canonical(expected['identity']), 'actual device destination '+dest)
        need(not actual[dest]['readonly'] and all(actual[dest][k]==expected['source_origin'][k] for k in ('device','root','filesystem','source')), 'predeclared device bind range '+dest)
    by={b['destination']:b for b in contract['bindings']}
    need(canonical(raw['executable'])==canonical(by['/bridge']['public_file']), 'actual unchanged executor inode and SHA')
    lib=by['/library/libposeidon_shared.so']['public_file']
    need(canonical(raw['mapped_library_file'])==canonical(lib), 'actual DSO backing-file inode and SHA')
    ranges=mapped_library(raw['maps'],dict(path='/library/libposeidon_shared.so',device='%d:%d'%(os.major(lib['device']),os.minor(lib['device'])),inode=lib['inode'],sha256=lib['sha256']))
    need('libseal' not in raw['maps'].lower(), 'no alternate HE library mapping')
    raw_limits(raw['limits'])
    return bound(dict(format=CONSISTENCY,test_only=test_only,portable_scope_binding=source_scope['binding'],live_source_path_resolution_or_contents_read=False,actual_collector_authentication=False,execution_authority=False, contract_binding=expected_binding, raw_binding=raw['binding'], role=contract['role'],
        run_nonce=contract['run_nonce'], actual_mapped_DSO_ranges=ranges, all_six_namespaces=True, effective_environment_includes_PWD=True,
        source_derived_mount_expectations=True, actual_destination_stat=True, private_content_reads=0,
        raw_record_consistent=True, trusted_collector_and_terminal_provenance_independently_required=True,
        whole_program_input_or_numerical_or_Bootstrap_qualification=False))

