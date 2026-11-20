"""Native raw-field projection into the original passive compiler verifier.

No process is launched or observed here. The trusted external launcher must
authenticate its collector binary, owner and config. Data consistency is not
source/build/terminal, compiler success, SDK or numerical certification.
"""
import copy
import json
from pathlib import Path

from delivery_component import bound,canonical,checked_bound,read_bytes,require,sha,write_once
import fast_compiler_process_observation as original

CONFIG='poseidon-native-fast-compiler-config-v1'
RAW='poseidon-native-fast-compiler-raw-v1'
HASH='poseidon-native-held-executable-hash-v1'


def make_config(prepared,expected_prepared,collector,collector_source_sha256):
    directory=original._validate(prepared,expected_prepared)
    require(type(collector)is dict and set(collector)=={'path','sha256','device','inode','bytes'},
            'Externally frozen trusted collector image')
    for value in [collector['sha256'],collector_source_sha256]:
        require(type(value)is str and len(value)==64 and all(c in '0123456789abcdef'for c in value),'Frozen collector source/image hash')
    require(Path(collector['path']).is_absolute()and all(type(collector[k])is int and collector[k]>0 for k in ['device','inode','bytes']),
            'Exact positive collector inode/size')
    return bound(dict(format=CONFIG,base_contract=copy.deepcopy(prepared['base_contract']),prepared_binding=prepared['binding'],
        collector=copy.deepcopy(collector),collector_source_sha256=collector_source_sha256,evidence_directory=str(directory),
        maximum_attempts=4,wall_seconds=90,test_only=prepared['test_only'],execution_or_approval_authority=False))


def parse(raw):
    require(type(raw)is bytes and len(raw)<=4*1024**2,'Complete bounded native metadata bytes')
    def pairs(rows):
        result={}
        for key,value in rows:
            require(key not in result,'Duplicate native JSON key');result[key]=value
        return result
    def invalid(value):raise ValueError('Nonfinite native metadata')
    return json.loads(raw,object_pairs_hook=pairs,parse_constant=invalid)


def project(prepared,expected_prepared,config,expected_config,raw_bytes,expected_raw_sha256,hash_bytes,expected_hash_sha256,*,test_only):
    original._validate(prepared,expected_prepared);checked_bound(config,expected_config)
    require(type(test_only)is bool and test_only is prepared['test_only']is config['test_only'],
            'Fixture records can never become actual process evidence')
    require(config==make_config(prepared,expected_prepared,config['collector'],config['collector_source_sha256']),
            'Original prepared command, collector and host-only evidence path')
    require(sha(raw_bytes)==expected_raw_sha256 and sha(hash_bytes)==expected_hash_sha256,'Independently pinned native raw/hash bytes')
    raw=parse(raw_bytes);hashed=parse(hash_bytes)
    require(set(raw)=={'format','config_binding','snapshot_fields','opened_executable','end_executable','end_target',
        'end_owner','end_owner_namespaces','raw_begin_monotonic_ns','raw_end_monotonic_ns','attempt','test_only'},'Closed native raw fields')
    require(set(hashed)=={'format','config_binding','raw_sha256','opened_executable','executable','hash_begin_monotonic_ns',
        'hash_end_monotonic_ns','raw_saved_before_hash','test_only'},'Closed native executable hash fields')
    require(raw['format']==RAW and hashed['format']==HASH and raw['config_binding']==hashed['config_binding']==config['binding']
        and raw['test_only']is hashed['test_only']is test_only and type(raw['attempt'])is int and 1<=raw['attempt']<=4,
        'Exact native source/role/attempt identity')
    fields=raw['snapshot_fields'];required={'format','contract_binding','nonce','role','owner','process','ancestry','observed_at_ns',
        'argv','cwd','environment_names','public_environment','namespaces','owner_namespaces','actual_mountinfo','actual_maps',
        'actual_limits','bindings','private_file_reads','private_key_hashes','process_memory_reads','processes_started',
        'compilation_completed_or_source_semantics_or_native_or_first_delivery_certified'}
    require(type(fields)is dict and set(fields)==required,'Exact complete original raw compiler fields')
    original._check_opened(raw['opened_executable'],prepared['base_contract']['executable'])
    require(raw['opened_executable']==raw['end_executable']==hashed['opened_executable']and
        raw['end_target']==fields['process']and raw['end_owner']==fields['owner']and raw['end_owner_namespaces']==fields['owner_namespaces'],
        'Target, owner, namespace and executable stability during actual raw reads')
    require(hashed['raw_sha256']==expected_raw_sha256 and hashed['raw_saved_before_hash']is True,
            'Native raw bytes durably saved before held-executable hashing')
    stamps=[raw['raw_begin_monotonic_ns'],raw['raw_end_monotonic_ns'],hashed['hash_begin_monotonic_ns'],hashed['hash_end_monotonic_ns']]
    require(all(type(value)is int and value>0 for value in stamps)and stamps==sorted(stamps),'Raw-read and held-hash order')
    require(hashed['executable']=={k:v for k,v in prepared['base_contract']['executable'].items()if k!='path'},
            'Held public inode bytes match original selected executable')
    projection=bound(dict(format=original.RAW,contract_binding=prepared['binding'],snapshot_fields=copy.deepcopy(fields),
        opened_executable=copy.deepcopy(raw['opened_executable']),end_executable=copy.deepcopy(raw['end_executable']),
        end_target=copy.deepcopy(raw['end_target']),end_owner=copy.deepcopy(raw['end_owner']),
        end_owner_namespaces=copy.deepcopy(raw['end_owner_namespaces']),raw_begin_monotonic_ns=stamps[0],raw_end_monotonic_ns=stamps[1],
        test_only=test_only,execution_or_approval_authority=False))
    snapshot=bound(dict(copy.deepcopy(fields),executable=copy.deepcopy(hashed['executable'])))
    check=original.base.verify(prepared['base_contract'],prepared['base_contract']['binding'],snapshot,snapshot['binding'])
    provenance=bound(dict(format='poseidon-native-compiler-raw-projection-v1',prepared_binding=prepared['binding'],config_binding=config['binding'],
        native_raw_sha256=expected_raw_sha256,native_hash_record_sha256=expected_hash_sha256,projected_raw_binding=projection['binding'],
        snapshot_binding=snapshot['binding'],original_base_check_binding=check['binding'],test_only=test_only,
        snapshot_fields_copied_without_invention_or_retagging=True,collector_execution_authentication=False,
        compiler_source_build_terminal_or_native_or_first_certified=False,execution_or_approval_authority=False))
    return projection,snapshot,provenance


def persist_and_verify(prepared,expected_prepared,config,expected_config,raw_path,raw_sha256,hash_path,hash_sha256):
    raw_bytes=read_bytes(raw_path,4*1024**2);hash_bytes=read_bytes(hash_path,4*1024**2)
    projection,snapshot,provenance=project(prepared,expected_prepared,config,expected_config,raw_bytes,raw_sha256,hash_bytes,hash_sha256,test_only=False)
    directory=Path(prepared['host_evidence_directory']);path=directory/'RAW_SNAPSHOT.json'
    write_once(path,canonical(projection));hashed=parse(hash_bytes)
    record=original._record(prepared,path,projection,hashed['executable'],hashed['hash_begin_monotonic_ns'],hashed['hash_end_monotonic_ns'])
    check=original.verify(prepared,expected_prepared,record,record['binding'])
    write_once(directory/'NATIVE_SOURCE_PROJECTION.json',canonical(provenance))
    write_once(directory/'COMPLETE_RECORD.json',canonical(record));write_once(directory/'SNAPSHOT_CHECK.json',canonical(check))
    return record,check,provenance
