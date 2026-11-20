"""Eight pure checks against retained prelaunch metadata, no native rerun."""
import copy
import json
from pathlib import Path
import re
import unittest
from unittest import mock

import portable_native_host_record as reader

ROOT=Path('/home/lhohy/poseidon-work/platforms/aarch64-linux/results/agent-b-native-observer-v4-zero-HE-sentinel-r3138/trusted-host-evidence')
MANIFEST=Path('/home/lhohy/Code/Poseidon/DSL/DSL_agent/docs/whole_models/proposals/bootstrap-evalmod-output-diagnostic-r1929/live-native-host-observer-v4-r3138/launch/manifest.json')

class PortableNativeControls(unittest.TestCase):
    def setUp(self):
        self.contract=json.loads((ROOT/'PRELAUNCH_CONTRACT.json').read_bytes())
        self.raw=json.loads((ROOT/'raw-observation.json').read_bytes())
        self.parent=json.loads((ROOT/'prelaunch-parent-mounts.json').read_bytes())
        self.parent_bytes=(ROOT/'prelaunch-parent-mounts.json').read_bytes()
        self.expected_sources=json.loads(MANIFEST.read_bytes())['observer_sources']

    def full(self,contract=None,raw=None,parent=None,test_only=True):
        c=self.contract if contract is None else contract;r=self.raw if raw is None else raw
        return reader.verify_record(c,r,c['binding'],r['binding'],self.expected_sources,
            parent_mount_bytes=self.parent_bytes if parent is None else parent,
            expected_parent_sha256=c['parent_mount_source']['sha256'],test_only=test_only)

    def scoped(self,c,p):
        p=reader.bound(p);data=reader.canonical(p);c=copy.deepcopy(c)
        c['parent_mount_source'].update(binding=p['binding'],sha256=reader.hashlib.sha256(data).hexdigest())
        c['command_sha256']=reader.hashlib.sha256(reader.canonical(c['command'])).hexdigest();c=reader.bound(c)
        return reader.verify_source_scope(c,c['binding'],data,c['parent_mount_source']['sha256'])

    def test_01_retained_zero_HE_sentinel_record_is_consistent_only(self):
        result=self.full(test_only=False)
        self.assertTrue(result['raw_record_consistent'])
        self.assertFalse(result['live_source_path_resolution_or_contents_read'])
        self.assertFalse(result['actual_collector_authentication'])
        self.assertFalse(result['execution_authority'])
        self.assertFalse(result['whole_program_input_or_numerical_or_Bootstrap_qualification'])

    def test_02_missing_live_Nix_paths_require_no_filesystem_lookup(self):
        with mock.patch('pathlib.Path.resolve',side_effect=AssertionError('No source resolve')),\
             mock.patch('pathlib.Path.stat',side_effect=AssertionError('No source stat')),\
             mock.patch('pathlib.Path.is_dir',side_effect=AssertionError('No source is_dir')),\
             mock.patch('os.stat',side_effect=AssertionError('No source stat')),\
             mock.patch('os.open',side_effect=AssertionError('No source open')),\
             mock.patch('os.readlink',side_effect=AssertionError('No live process lookup')),\
             mock.patch('builtins.open',side_effect=AssertionError('No hidden file read')):
            result=self.full()
        self.assertTrue(result['raw_record_consistent'])

    def test_03_source_FD_mount_and_inode_must_select_the_frozen_parent(self):
        for change in('fdmount','mount','inode'):
            c=copy.deepcopy(self.contract);b=c['bindings'][1]
            if change=='fdmount':b['source_origin']['source_fdinfo']=re.sub(r'mnt_id:\s*[0-9]+','mnt_id:\t999999',b['source_origin']['source_fdinfo'])
            elif change=='mount':b['source_origin']['parent_mount_id']=999999
            else:b['identity']['inode']+=1
            with self.assertRaises(ValueError):self.scoped(c,self.parent)

    def test_04_broad_underlying_home_cannot_hide_under_a_public_Nix_mount(self):
        c=copy.deepcopy(self.contract);p=copy.deepcopy(self.parent);b=c['bindings'][0]
        p['raw_mountinfo']+='990001 1 252:1 / /nix/store rw - ext4 /dev/vda1 rw\n'
        b['source_origin'].update(parent_mount_id=990001,root='/home/lhohy',
            source_fdinfo='pos:\t0\nflags:\t012000000\nmnt_id:\t990001\nino:\t%d\n'%b['identity']['inode'])
        with self.assertRaises(ValueError):self.scoped(c,p)

    def test_05_private_directory_and_plaintext_do_not_enter_evaluator_scope(self):
        for category in('original_private_view','public_plaintext'):
            c=copy.deepcopy(self.contract);c['role']='public_evaluator';c['bindings'][2]['category']=category
            with self.assertRaises(ValueError):self.scoped(c,self.parent)

    def test_06_parent_origin_must_precede_freeze_and_belong_to_same_owner(self):
        for change in('time','owner'):
            p=copy.deepcopy(self.parent)
            if change=='time':p['at_ns']=self.contract['frozen_at_ns']+1
            else:p['owner']['start_ticks']+=1
            with self.assertRaises(ValueError):self.scoped(self.contract,p)

    def test_07_public_file_identity_and_writable_scope_are_checked(self):
        c=copy.deepcopy(self.contract);c['bindings'][1]['public_file']['inode']+=1
        with self.assertRaises(ValueError):self.scoped(c,self.parent)
        c=copy.deepcopy(self.contract);next(b for b in c['bindings']if b['destination']=='/control')['category']='public_code'
        with self.assertRaises(ValueError):self.scoped(c,self.parent)

    def test_08_forbidden_lexical_aliases_and_observation_rebinding_are_refused(self):
        for source in('/home/lhohy/.env','/nix/store/../hidden'):
            c=copy.deepcopy(self.contract);b=c['bindings'][-1];old=b['source'];b['source']=source
            c['command']=[source if arg==old else arg for arg in c['command']]
            with self.assertRaises(ValueError):self.scoped(c,self.parent)
        with self.assertRaises(ValueError):
            reader.verify_record(self.contract,self.raw,self.contract['binding'],'0'*64,self.expected_sources,
                parent_mount_bytes=self.parent_bytes,expected_parent_sha256=self.contract['parent_mount_source']['sha256'],test_only=True)

if __name__=='__main__':unittest.main()
