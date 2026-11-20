"""Eight inert controls for deferred hashing and durable raw-record origin."""
import copy
import os
from pathlib import Path
import unittest

from delivery_component import bound,canonical,write_once
import fast_compiler_process_observation as fast
import test_approved_cpu_compiler_observation as old

def rebind(value):return bound({k:v for k,v in value.items()if k!='binding'})

class FastCompilerControls(unittest.TestCase):
    def setUp(self):
        self.fixtures=old.ApprovedCpuCompilerControls()
        self.fixtures.setUp();self.addCleanup(self.fixtures.doCleanups)
        self.home=self.fixtures.home;self.number=0

    def fixture(self,change=None):
        original,snapshot=self.fixtures.fixture();self.number+=1
        folder=self.home/('host-%02d'%self.number);folder.mkdir()
        value=fast.contract(original['command'],original['owner'],original['owner_namespaces'],
            original['nonce'],original['selection'],evidence_directory=folder,test_only=True)
        self.assertEqual(value['base_contract'],original)
        fields={k:copy.deepcopy(v)for k,v in snapshot.items()if k not in('binding','executable')}
        signature=fast._signature(os.stat(self.home/'compiler'))
        raw=dict(format=fast.RAW,contract_binding=value['binding'],snapshot_fields=fields,
            opened_executable=signature,end_executable=copy.deepcopy(signature),
            end_target=copy.deepcopy(fields['process']),end_owner=copy.deepcopy(fields['owner']),
            end_owner_namespaces=copy.deepcopy(fields['owner_namespaces']),
            raw_begin_monotonic_ns=10,raw_end_monotonic_ns=20,test_only=True,
            execution_or_approval_authority=False)
        if change is not None:change(raw)
        raw=bound(raw);path=folder/'RAW_SNAPSHOT.json';write_once(path,canonical(raw))
        record=fast._record(value,path,raw,snapshot['executable'],30,40)
        return value,record

    def check(self,c,r):return fast.verify(c,c['binding'],r,r['binding'])

    def test_01_prepared_inert_records_preserve_original_snapshot_checks(self):
        c,r=self.fixture();check=self.check(c,r)
        self.assertTrue(check['raw_snapshot_matches_contract'])
        self.assertTrue(check['test_only'])
        self.assertFalse(check['actual_live_capture_validated'])
        self.assertFalse(check['compiler_source_build_terminal_or_native_or_first_certified'])
        self.assertFalse(check['execution_or_approval_authority'])

    def test_02_held_public_inode_hash_survives_disappearance_of_proc_view(self):
        c,r=self.fixture();link=self.home/'inert-proc-exe-view';link.symlink_to(self.home/'compiler')
        fd=os.open(link,os.O_RDONLY|os.O_CLOEXEC)
        try:
            opened=fast._signature(os.fstat(fd));link.unlink()
            self.assertFalse(link.exists())
            result=fast._hash_open_descriptor(fd,opened)
            self.assertEqual(result,r['base_record']['executable'])
            self.assertTrue((self.home/'compiler').is_file())
        finally:os.close(fd)

    def test_03_underlying_executable_mutation_is_refused_after_open(self):
        c,r=self.fixture();fd=os.open(self.home/'compiler',os.O_RDONLY|os.O_CLOEXEC)
        try:
            opened=fast._signature(os.fstat(fd))
            with(self.home/'compiler').open('ab')as stream:stream.write(b'changed')
            with self.assertRaises(ValueError):fast._hash_open_descriptor(fd,opened)
        finally:os.close(fd)

    def test_04_equal_bytes_from_another_inode_cannot_replace_actual_executable(self):
        c,r=self.fixture();other=self.home/'another-inert-compiler'
        other.write_bytes((self.home/'compiler').read_bytes())
        opened=fast._signature(os.stat(other))
        with self.assertRaises(ValueError):fast._check_opened(opened,c['base_contract']['executable'])

    def test_05_target_owner_and_executable_must_survive_the_live_read_interval(self):
        changes=[lambda raw:raw['end_target'].update(start_ticks=9999),
                 lambda raw:raw['end_owner_namespaces'].update(mnt='mnt:[999]'),
                 lambda raw:raw['end_executable'].update(inode=0)]
        for change in changes:
            c,r=self.fixture(change)
            with self.assertRaises(ValueError):self.check(c,r)

    def test_06_order_raw_bytes_and_original_record_location_are_independent(self):
        c,r=self.fixture();r['executable_hash_begin_monotonic_ns']=19
        with self.assertRaises(ValueError):self.check(c,rebind(r))
        c,r=self.fixture();path=Path(r['raw_record']['path']);os.chmod(path,0o600);path.write_bytes(b'changed')
        with self.assertRaises(ValueError):self.check(c,r)
        c,r=self.fixture();other=Path(c['host_evidence_directory'])/'other.json'
        other.write_bytes(Path(r['raw_record']['path']).read_bytes());r['raw_record']['path']=str(other)
        with self.assertRaises(ValueError):self.check(c,rebind(r))
        c,r=self.fixture();r['base_record']['cwd']='/different';r['base_record']=rebind(r['base_record'])
        with self.assertRaises(ValueError):self.check(c,rebind(r))

    def test_07_raw_record_directories_inside_native_mounts_are_refused(self):
        original,_=self.fixtures.fixture()
        for folder in(self.home/'out',self.home/'nix'):
            with self.assertRaises(ValueError):
                fast.contract(original['command'],original['owner'],original['owner_namespaces'],
                    original['nonce'],original['selection'],evidence_directory=folder,test_only=True)

    def test_08_fixture_capture_and_changed_collector_source_never_gain_live_authority(self):
        c,r=self.fixture();prepared=fast.PreparedCollector(c,c['binding'])
        with self.assertRaises(ValueError):prepared.capture(99999999)
        with self.assertRaises(ValueError):prepared.capture(99999999)
        c,r=self.fixture();key=next(iter(c['observer_sources']));c['observer_sources'][key]='0'*64;c=rebind(c)
        with self.assertRaises(ValueError):fast.PreparedCollector(c,c['binding'])

if __name__=='__main__':unittest.main()
