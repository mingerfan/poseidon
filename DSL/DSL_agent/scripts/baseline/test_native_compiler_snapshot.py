"""Four new inert native-projection controls; do not run the old TestCases."""
import copy
import json
from pathlib import Path
import unittest

from delivery_component import canonical,sha
import native_compiler_snapshot as target
import test_fast_compiler_process_observation as fixtures


class NativeCompilerSnapshotControls(unittest.TestCase):
    def setUp(self):
        self.old=fixtures.FastCompilerControls('test_01_prepared_inert_records_preserve_original_snapshot_checks')
        self.old.setUp();self.addCleanup(self.old.doCleanups)

    def fixture(self):
        prepared,record=self.old.fixture();raw=json.loads(Path(record['raw_record']['path']).read_bytes())
        executable=record['base_record']['executable']
        collector=dict(path=str(self.old.home/'compiler'),sha256=executable['sha256'],device=executable['device'],inode=executable['inode'],bytes=executable['bytes'])
        config=target.make_config(prepared,prepared['binding'],collector,'c'*64)
        native=dict(format=target.RAW,config_binding=config['binding'],
            **{k:copy.deepcopy(raw[k])for k in ['snapshot_fields','opened_executable','end_executable','end_target','end_owner',
                'end_owner_namespaces','raw_begin_monotonic_ns','raw_end_monotonic_ns']},attempt=1,test_only=True)
        hashed=dict(format=target.HASH,config_binding=config['binding'],raw_sha256=sha(canonical(native)),
            opened_executable=copy.deepcopy(native['opened_executable']),executable=copy.deepcopy(executable),
            hash_begin_monotonic_ns=30,hash_end_monotonic_ns=40,raw_saved_before_hash=True,test_only=True)
        return prepared,config,native,hashed,raw

    def project(self,data,*,expected_raw=None,test_only=True):
        prepared,config,native,hashed,_=data;raw=canonical(native);hashed=copy.deepcopy(hashed);hashed['raw_sha256']=sha(raw)
        actual_hash=canonical(hashed)
        return target.project(prepared,prepared['binding'],config,config['binding'],raw,sha(raw)if expected_raw is None else expected_raw,
                              actual_hash,sha(actual_hash),test_only=test_only)

    def test_01_exact_projection_preserves_every_original_raw_field(self):
        data=self.fixture();projection,snapshot,proof=self.project(data)
        self.assertEqual(projection,data[-1]);self.assertEqual(snapshot['argv'],data[0]['base_contract']['argv'])
        self.assertTrue(proof['snapshot_fields_copied_without_invention_or_retagging'])
        self.assertFalse(proof['collector_execution_authentication']);self.assertFalse(proof['compiler_source_build_terminal_or_native_or_first_certified'])

    def test_02_external_raw_hash_and_closed_native_fields_required(self):
        data=self.fixture()
        with self.assertRaises(ValueError):self.project(data,expected_raw='0'*64)
        data=self.fixture();data[2]['invented_extra_snapshot']=True
        with self.assertRaises(ValueError):self.project(data)

    def test_03_live_interval_and_hash_order_cannot_be_reconstructed(self):
        data=self.fixture();data[2]['end_target']['start_ticks']+=1
        with self.assertRaises(ValueError):self.project(data)
        data=self.fixture();data[3]['hash_begin_monotonic_ns']=19
        with self.assertRaises(ValueError):self.project(data)

    def test_04_fixture_argv_and_public_environment_cannot_gain_live_scope(self):
        data=self.fixture()
        with self.assertRaises(ValueError):self.project(data,test_only=False)
        data=self.fixture();data[2]['snapshot_fields']['argv'].append('--changed')
        with self.assertRaises(ValueError):self.project(data)
        data=self.fixture();fields=data[2]['snapshot_fields'];fields['public_environment']['UNEXPECTED_FIXTURE_NAME']='inert';fields['environment_names']=sorted(fields['public_environment'])
        with self.assertRaises(ValueError):self.project(data)
