"""Read-only resource/source identity after the exact separately approved move.

Prospective records carry no authority. The committed reader requires the real
approval, original three-file transaction and complete retained live history.
It never writes, applies a migration, or grants compiler/numerical/SDK gates.
"""
import importlib.util
from pathlib import Path
import re

from delivery_component import bound, canonical, read_bytes, require, sha

RESULTS=Path('/home/lhohy/poseidon-work/platforms/aarch64-linux/results')
PROPOSAL=RESULTS/'agent-evidence64-peer-proof-revision-r2812'
PARENT='f3e7cba3abe91ef830ad33dfbbfffe82d1b449893e8ee2fb9d56ecd75b1927e0'
CORE_SHA='21efe3dd1fcb5c8a6c06691cc5765f43441f5fece60a53e99eecb7b6a35c1f1f'
REFRESH_SHA='66ca9ddc70aec30091cc93540512e7a816ad8dd6f5a6c417e203a6fa471093ef'
OLD_RECORD=RESULTS/'agent-approved60-resource-transition-r2654/source46-approved60-r2655.json'
OLD_SHA='fc02affe7b9cd78a41783d3f9f87aa1663cb3a04872b263275c6545424691cd3'
BACKUP=RESULTS/'agent-evidence-60-to-64gib-migration-backup-v1'


def _module(name,expected):
    path=PROPOSAL/name;code=read_bytes(path);require(sha(code)==expected,'Reviewed resource helper identity')
    spec=importlib.util.spec_from_file_location('approved64_'+path.stem,str(path));module=importlib.util.module_from_spec(spec)
    exec(compile(code,str(path),'exec'),module.__dict__);return module


def _proposal():
    core=_module('migration_core.py',CORE_SHA);manifest=core.checked(core.file_read(PROPOSAL/'manifest.json')[0])
    require(core.digest(manifest)==PARENT and manifest['resource_transition']==dict(old_bytes=60*1024**3,new_bytes=64*1024**3),
        'Exact pending64 proposal; another cap or proposal is not interchangeable')
    require(manifest['fixed_after_sha256']['source']=='b60443d71c9ad1a70da8425e5d0272e17711e06036eef449cb7c6c3dcd555eab'
        and core.NEW==dict(core.OLD,evidence_bytes=64*1024**3),'Only shared evidence limit changes')
    return core,manifest


def _derive(old,manifest,before,after):
    shared=manifest['targets']['source'];files=old['files']
    require(type(files)is dict and len(files)==46 and shared in files,'Original46-file closure')
    require(files[shared]==sha(before) and sha(after)==manifest['fixed_after_sha256']['source']
        and before.count(b'evidence_bytes=60*GIB')==1
        and after==before.replace(b'evidence_bytes=60*GIB',b'evidence_bytes=64*GIB',1),'Exact single resource literal; no unrelated source repair')
    return dict(files,**{shared:sha(after)})


def prospective():
    core,manifest=_proposal();raw=core.file_read(OLD_RECORD)[0];require(sha(raw)==OLD_SHA,'Original source46 record retained')
    old=core.checked(raw);blobs=manifest['source_blobs']
    before=core.file_read(blobs['before-shared_execution.py']['path'])[0]
    after=core.file_read(blobs['after-shared_execution.py']['path'])[0]
    require(sha(before)==blobs['before-shared_execution.py']['sha256'] and sha(after)==blobs['after-shared_execution.py']['sha256'],'Frozen actual before/after source bytes')
    files=_derive(old,manifest,before,after);shared=manifest['targets']['source']
    require(all(sha(read_bytes(p))==v for p,v in old['files'].items()if p!=shared),'All unchanged45 actual source files required')
    actual=sha(read_bytes(shared));require(actual in (sha(before),sha(after)),'Live source outside the exact resource transition')
    return bound(dict(format='poseidon-prospective-approved64-source46-identity-v1',proposal_binding=PARENT,
        previous_source_record=str(OLD_RECORD),previous_source_record_sha256=OLD_SHA,previous_source_binding=core.digest(old),
        files=files,current_source_file_count=46,unchanged_previous45_files_verified=True,
        changed_production_source_files=[shared],current_shared_source_matches='before60'if actual==sha(before)else'after64',
        resource_approval_or_commit_verified=False,normal_Application_execution_ready=False,
        native_numerical_qualification=False,full_model_accepted=False,execution_authority=False))


def _history(core,migrated,live):
    require(set(migrated)==set(live)and migrated['policy']==live['policy']
        and core.canonical(migrated.get('store_accounting'))==core.canonical(live.get('store_accounting')),
        'Only post-migration batch additions; no policy/store/schema reinterpretation')
    require(set(migrated['batches'])<=set(live['batches'])and all(core.canonical(live['batches'][k])==core.canonical(v)
        for k,v in migrated['batches'].items()),'Every migrated batch and consumed counter must remain unchanged')
    return len(live['batches'])-len(migrated['batches'])


def committed_identity(receipt_descriptor,approval_descriptor):
    """Actual committed transaction only; call in a fresh post-64 process."""
    core,manifest=_proposal();recipe=prospective()
    for d in (receipt_descriptor,approval_descriptor):
        require(type(d)is dict and set(d)=={'path','sha256'},'Independently selected receipt descriptor')
    receipt_path=Path(receipt_descriptor['path'])
    require(receipt_path.is_absolute()and receipt_path.parent.parent==BACKUP
        and re.fullmatch(r'[a-z0-9][a-z0-9_-]{0,47}',receipt_path.parent.name)
        and re.fullmatch(r'result-[0-9]+\.json',receipt_path.name),'Original counted migration transaction receipt')
    approval_path=Path(approval_descriptor['path'])
    require(approval_path.is_absolute()and approval_path.is_relative_to(RESULTS)and approval_path.suffix=='.json',
        'Private counted explicit approval record')
    approval_raw=core.file_read(approval_path)[0]
    core.approval_check(approval_raw,approval_descriptor['sha256'],PARENT)
    receipt_raw=core.file_read(receipt_path)[0];require(sha(receipt_raw)==receipt_descriptor['sha256'],'Committed receipt bytes')
    receipt=core.checked(receipt_raw)
    require(receipt.get('format')=='poseidon-evidence-migration-receipt-v1'and receipt.get('proposal_binding')==PARENT
        and receipt.get('approval_sha256')==approval_descriptor['sha256']and receipt.get('outcome')=='committed'
        and receipt.get('old_bytes')==60*1024**3 and receipt.get('new_bytes')==64*1024**3
        and receipt.get('history_preserved')is True and receipt.get('SDK_or_B_certification_updated')is False,
        'Actual exact resource-only commit, not an approval flag or recovery rollback')
    transaction=receipt_path.parent;attempt=core.checked(core.file_read(transaction/'bounded-refresh.json')[0])
    require(attempt['proposal_binding']==PARENT,'Transaction proposal')
    before={k:core.file_read(transaction/(k+'.before'))[0]for k in core.LABELS}
    after={k:core.file_read(transaction/(k+'.after'))[0]for k in core.LABELS}
    refresh=_module('bounded_ledger_refresh.py',REFRESH_SHA);archive=manifest['source_blobs']['before-ledger.json.gz']
    frozen=core.checked(core.file_read(PROPOSAL/'frozen-inputs.json')[0])
    actual_frozen,computed,observed=refresh.images(core,before,frozen,core.file_read(archive['path'])[0],{k:v for k,v in archive.items()if k!='path'})
    require(actual_frozen==attempt['fresh_frozen_inputs']and observed==attempt['refresh_receipt']and after==computed
        and {k:sha(v)for k,v in after.items()}==attempt['after_sha256']==receipt['final_sha256'],
        'Exact before/after images and complete bounded terminal refresh')
    targets={k:Path(v)for k,v in manifest['targets'].items()};journal=core.checked(core.file_read(transaction/'journal.json')[0])
    core.journal_check(journal,targets,PARENT,actual_frozen,attempt['after_sha256']);require(journal['outcome']=='committed','Committed original journal')
    require(sha(read_bytes(targets['source']))==sha(after['source'])and sha(read_bytes(targets['policy']))==sha(after['policy']),
        'Current source/policy differ from actual commit')
    live=core.checked(core.file_read(targets['ledger'])[0]);migrated=core.checked(after['ledger']);core.policy_check(live['policy'],core.NEW)
    added=_history(core,migrated,live)
    require(receipt['batch_count']==len(migrated['batches'])and all(sha(read_bytes(p))==v for p,v in recipe['files'].items()),
        'Full current source46/64 and original commit batch count')
    return bound(dict(format='poseidon-agent-approved64-resource-source46-v1',proposal_binding=PARENT,
        previous_source_record=str(OLD_RECORD),previous_source_record_sha256=OLD_SHA,previous_source_binding=recipe['previous_source_binding'],
        files=recipe['files'],current_source_file_count=46,unchanged_previous45_files=True,
        changed_production_source_files=recipe['changed_production_source_files'],changed_source_scope='shared evidence60GiB to64GiB only',
        resource_transition=dict(old_GiB=60,new_GiB=64,normal_protected_GiB=4,parent_protected_MiB=256),
        migration_receipt=receipt_descriptor,explicit_approval=approval_descriptor,migration_batch_count=len(migrated['batches']),
        actual_history_rows=len(live['batches']),post_commit_additional_batches=added,all_migrated_rows_preserved=True,
        normal_Application_metadata_preflight='pending_fresh64_source_and_owner',normal_Application_execution_ready=False,
        native_numerical_qualification=False,full_model_accepted=False,execution_authority=False))
