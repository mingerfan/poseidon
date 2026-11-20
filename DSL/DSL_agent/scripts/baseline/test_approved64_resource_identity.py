"""Six new identity checks; no real approval, migration or native execution."""
import copy
import unittest
from pathlib import Path

from delivery_component import read_bytes,sha
import approved64_resource_identity as identity


class Approved64IdentityControls(unittest.TestCase):
    def test_01_actual45_unchanged_sources_and_frozen_transition_are_prospective_only(self):
        r=identity.prospective();self.assertEqual(r['current_source_file_count'],46)
        self.assertTrue(r['unchanged_previous45_files_verified']);self.assertEqual(r['current_shared_source_matches'],'before60')
        self.assertFalse(any(r[k]for k in ['resource_approval_or_commit_verified','normal_Application_execution_ready',
            'native_numerical_qualification','full_model_accepted','execution_authority']))

    def test_02_other_source_changes_or_forged46_closure_reject(self):
        core,manifest=identity._proposal();old=core.checked(core.file_read(identity.OLD_RECORD)[0]);blobs=manifest['source_blobs']
        before=core.file_read(blobs['before-shared_execution.py']['path'])[0];after=core.file_read(blobs['after-shared_execution.py']['path'])[0]
        self.assertEqual(len(identity._derive(old,manifest,before,after)),46)
        with self.assertRaises(ValueError):identity._derive(old,manifest,before,after+b'\n')
        bad=copy.deepcopy(old);bad['files'].pop(next(p for p in bad['files']if p!=manifest['targets']['source']))
        with self.assertRaises(ValueError):identity._derive(bad,manifest,before,after)

    def test_03_arbitrary_or_rollback_receipt_paths_cannot_be_live_commit(self):
        valid_approval=dict(path=str(identity.RESULTS/'synthetic-approval.json'),sha256='a'*64)
        for path in ['/tmp/synthetic.json',str(identity.BACKUP/'result-1.json'),str(identity.BACKUP/'attempt'/'rollback.json')]:
            with self.assertRaises(ValueError):identity.committed_identity(dict(path=path,sha256='b'*64),valid_approval)

    def test_04_old_consumed_rows_and_store_inventory_cannot_be_reinterpreted(self):
        core,_=identity._proposal();old=dict(policy=dict(limits=dict(evidence_bytes=64*1024**3)),batches=dict(original=dict(used=dict(API=3),signed=-0.0)),
            store_accounting=dict(roots=[dict(path='/retained',inode=3)]))
        new=copy.deepcopy(old);new['batches']['new_terminal']=dict(used=dict(API=0))
        self.assertEqual(identity._history(core,old,new),1)
        for modify in [lambda d:d['batches']['original']['used'].update(API=0),lambda d:d['batches'].pop('original'),
                lambda d:d['store_accounting']['roots'].append(dict(path='/unreviewed')),lambda d:d['policy']['limits'].update(evidence_bytes=65*1024**3)]:
            bad=copy.deepcopy(new);modify(bad)
            with self.assertRaises(ValueError):identity._history(core,old,bad)

    def test_05_future_receipt_without_explicit_approval_is_not_accepted(self):
        receipt=dict(path=str(identity.BACKUP/'not-created-control'/'result-1.json'),sha256='b'*64)
        with self.assertRaises((ValueError,FileNotFoundError)):
            identity.committed_identity(receipt,dict(path='/tmp/standing-plan-approval.json',sha256='a'*64))
        with self.assertRaises((ValueError,FileNotFoundError)):
            identity.committed_identity(receipt,dict(path=str(identity.RESULTS/'not-created-control-approval.json'),sha256='a'*64))

    def test_06_old_source_record_live60_and_all_original_bytes_remain(self):
        before=read_bytes(identity.OLD_RECORD);self.assertEqual(sha(before),identity.OLD_SHA)
        r=identity.prospective();shared=r['changed_production_source_files'][0]
        self.assertEqual(sha(read_bytes(shared)),'528e700035618921f4c6420b37e79bcdd01ee6164f064ac9b872f647a75127e4')
        self.assertEqual(read_bytes(identity.OLD_RECORD),before)
        self.assertEqual(r['files'][shared],'b60443d71c9ad1a70da8425e5d0272e17711e06036eef449cb7c6c3dcd555eab')
