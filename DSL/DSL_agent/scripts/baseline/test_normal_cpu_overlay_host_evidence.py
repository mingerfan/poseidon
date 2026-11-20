"""Inert retained-witness counterexamples; no /proc, key or HE process exists."""
import copy
import stat
import unittest

from delivery_component import bound
from normal_cpu_overlay_host_evidence import LIMITS,NAMESPACES,verify_records


class RawHostWitnessTests(unittest.TestCase):
    def setUp(self):
        target='/selected-poseidon/libposeidon_shared.so'
        self.target=target
        self.command=['/usr/bin/bwrap','--unshare-all','--die-with-parent','--new-session','--cap-drop','ALL','--clearenv',
            '--ro-bind','/inert/library/libposeidon_shared.so',target,'--bind','/inert/client','/out',
            '--ro-bind','/inert/venv','/home/lhohy/poseidon-work/venv',
            '--proc','/proc','--dev','/dev','--tmpfs','/tmp',
            '--setenv','HOME','/nonexistent','--setenv','PATH','/nonexistent','--setenv','LD_DEBUG','libs',
            '--chdir','/out','--','/poseidon-agent-cpu','/out/bound/plan.json']
        self.binds=[dict(source='/inert/library/libposeidon_shared.so',destination=target,mode='ro',identity=dict(device=1,inode=50,kind=stat.S_IFREG)),
            dict(source='/inert/client',destination='/out',mode='rw',identity=dict(device=1,inode=51,kind=stat.S_IFDIR)),
            dict(source='/inert/venv',destination='/home/lhohy/poseidon-work/venv',mode='ro',identity=dict(device=1,inode=52,kind=stat.S_IFDIR))]
        self.contract=dict(format='poseidon-B-original-ordinary-CPU-exact-mount-contract-v1',command=copy.deepcopy(self.command),binds=copy.deepcopy(self.binds),
            environment=dict(HOME='/nonexistent',PATH='/nonexistent',LD_DEBUG='libs'),special={'/proc':'--proc','/dev':'--dev','/tmp':'--tmpfs'})
        self.context=dict(inner=dict(artifacts=dict(executor='a'*64,poseidon_library='b'*64)),outer=dict(profile=dict(sha256='c'*64),operator_spec=dict(sha256='d'*64)))
        mounts=['1 0 0:1 / / rw - tmpfs tmpfs rw','2 1 0:2 / /proc rw - proc proc rw',
            '3 1 0:3 / /dev rw - tmpfs tmpfs rw','4 1 0:4 / /tmp rw - tmpfs tmpfs rw']
        for i,b in enumerate(self.binds):mounts.append(f"{10+i} 1 0:1 / {b['destination']} {b['mode']} - ext4 /dev/fixture {b['mode']}")
        self.raw=dict(format='poseidon-B-ordinary-CPU-native-raw-observation-v1',actual_mountinfo='\n'.join(mounts),
            actual_expected_bind_identities={b['destination']:copy.deepcopy(b['identity'])for b in self.binds},
            actual_native_argv=self.command[self.command.index('--')+1:],actual_environment_names=sorted([*self.contract['environment'],'PWD']),actual_environment_equals_frozen=True,
            actual_environment_per_expected_key_matches={k:True for k in [*self.contract['environment'],'PWD']},actual_PWD_equals_original_chdir_out=True,
            actual_namespaces={n:dict(native=n+':[11]',driver=n+':[22]')for n in NAMESPACES},
            actual_limits='\n'.join(name+' '+str(pair[0])+' '+str(pair[1])for name,pair in LIMITS.items()),
            actual_executable_sha256='a'*64,actual_DSO_sha256='b'*64,actual_profile_sha256='c'*64,actual_operator_spec_sha256='d'*64,
            actual_DSO_identity=copy.deepcopy(self.binds[0]['identity']),
            actual_maps='1000-2000 r-xp 00000000 00:01 50 '+target,
            private_file_content_reads=0,private_file_hashes=0,key_memory_read=False)
        self.host=dict(format='poseidon-B-actual-ordinary-CPU-exact-mount-host-proof-v2',raw_observations_saved_before_validation=True,
            actual_native_limits_verified=copy.deepcopy(LIMITS),cleared_native_environment_verified=True,
            executable_sha256='a'*64,selected_library_sha256='b'*64,actual_mapped_selected_DSO_verified=True,
            original_parameter_file_bytes_verified=True,selected_library_readonly_mount_verified=True,isolated_network_namespace_verified=True,
            broad_host_home_bind_absent=True,key_memory_not_read=True,project_home_mounted=False,reference_mounted=False,persistent_private_directory_mounted=False,
            private_file_content_reads=0,private_file_hashes=0,process_local_ephemeral_key_sets_maximum=1,persistent_key_epochs=0,
            allowed_empty_home_ancestor_exists=True)

    def verify(self,*,mismatched_mount_summary=False):
        contract=bound(copy.deepcopy(self.contract));raw=bound(copy.deepcopy(self.raw));host=copy.deepcopy(self.host)
        host.update(command_contract_binding=contract['binding'],raw_observation_binding=raw['binding'],
            actual_mountinfo=raw['actual_mountinfo'],actual_native_limits_text=raw['actual_limits'],
            actual_bind_mounts=[copy.deepcopy(b)for b in sorted(contract['binds'],key=lambda b:b['destination'])],
            namespaces=copy.deepcopy(raw['actual_namespaces']),selected_library_actual_maps=raw['actual_maps'].splitlines())
        if mismatched_mount_summary:host['actual_mountinfo']+='\nchanged summary'
        return verify_records(contract,raw,bound(host),expected_command=self.command,context=self.context)

    def test_allowed_home_ancestors_do_not_become_broad_home_mount_claims(self):
        result=self.verify();self.assertEqual(result['actual_bind_count'],3)

    def test_extra_broad_home_mount_cannot_be_resigned_as_safe(self):
        self.raw['actual_mountinfo']+='\n99 1 0:1 / /home/lhohy ro - ext4 /dev/fixture ro'
        with self.assertRaises(ValueError):self.verify()

    def test_extra_private_or_reference_mount_cannot_be_resigned_as_safe(self):
        self.raw['actual_mountinfo']+='\n99 1 0:1 / /client-private ro - ext4 /dev/fixture ro'
        with self.assertRaises(ValueError):self.verify()

    def test_writable_selected_DSO_cannot_be_hidden_by_readonly_success_flag(self):
        self.raw['actual_mountinfo']=self.raw['actual_mountinfo'].replace(self.target+' ro',self.target+' rw')
        with self.assertRaises(ValueError):self.verify()

    def test_source_identity_must_equal_actual_inode_and_device(self):
        self.raw['actual_expected_bind_identities'][self.target]['inode']=99
        with self.assertRaises(ValueError):self.verify()

    def test_accepted_mount_summary_cannot_differ_from_raw_observation(self):
        with self.assertRaises(ValueError):self.verify(mismatched_mount_summary=True)

    def test_shared_native_network_namespace_cannot_use_true_isolation_flag(self):
        self.raw['actual_namespaces']['net']['native']=self.raw['actual_namespaces']['net']['driver']
        with self.assertRaises(ValueError):self.verify()

    def test_expanded_actual_CPU_limit_cannot_be_resigned_as_original_limits(self):
        self.raw['actual_limits']=self.raw['actual_limits'].replace('Max cpu time 150 155','Max cpu time 300 305')
        self.host['actual_native_limits_verified']['Max cpu time']=[300,305]
        with self.assertRaises(ValueError):self.verify()

    def test_actual_executor_or_library_bytes_must_equal_selected_SDK(self):
        self.raw['actual_DSO_sha256']='e'*64;self.host['selected_library_sha256']='e'*64
        with self.assertRaises(ValueError):self.verify()

    def test_other_Poseidon_mapping_cannot_be_claimed_as_selected_library(self):
        self.raw['actual_maps']+='\n3000-4000 r-xp 00000000 00:01 99 /old/libposeidon_shared.so'
        with self.assertRaises(ValueError):self.verify()

    def test_actual_cleared_environment_and_arguments_must_match_original(self):
        self.raw['actual_environment_names'].append('EXTRA_AMBIENT')
        with self.assertRaises(ValueError):self.verify()

    def test_overall_success_cannot_hide_a_different_measured_environment_value(self):
        self.raw['actual_environment_per_expected_key_matches']['PATH']=False
        with self.assertRaises(ValueError):self.verify()

    def test_PWD_name_without_measured_value_cannot_establish_the_working_directory_model(self):
        self.raw['actual_PWD_equals_original_chdir_out']=False
        with self.assertRaises(ValueError):self.verify()

    def test_contract_cannot_substitute_an_extra_writable_destination(self):
        self.contract['command']=self.contract['command'][:-1]+['/out/other.json']
        with self.assertRaises(ValueError):self.verify()

    def test_missing_mount_or_private_namespace_root_is_rejected(self):
        self.raw['actual_mountinfo']='\n'.join(line for line in self.raw['actual_mountinfo'].splitlines()if' /proc 'not in line)
        with self.assertRaises(ValueError):self.verify()

    def test_observations_after_validation_cannot_be_presented_as_original_raw_witness(self):
        self.host['raw_observations_saved_before_validation']=False
        with self.assertRaises(ValueError):self.verify()


if __name__=='__main__':unittest.main()
