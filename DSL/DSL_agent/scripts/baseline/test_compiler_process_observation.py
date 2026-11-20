"""Twelve inert counterexample controls; never execute the fixture tool bytes."""
import copy
import os
from pathlib import Path
import stat
import tempfile
import unittest

import compiler_process_observation as observer
from delivery_component import bound


def rebind(value):return bound({k:v for k,v in value.items()if k!='binding'})


class CompilerProcessControls(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);home=Path(self.temp.name)
        for n in('nix','out'):(home/n).mkdir()
        for n in('payload.json','compiler','python'):(home/n).write_bytes(b'Inert process-observation fixture; never executed.\n')
        self.home=home
        self.owner=dict(pid=100,start_ticks=1000,boot_id='6c7a04be-6625-49bd-a03b-f439e5bc29d1')
        self.namespaces={n:n+':[10]'for n in observer.NAMESPACES}

    def fixture(self,role='compiler'):
        h=self.home
        command=['/usr/bin/bwrap','--unshare-all','--die-with-parent','--new-session','--clearenv','--cap-drop','ALL',
            '--ro-bind',str(h/'nix'),'/nix/store','--ro-bind',str(h/'payload.json'),'/payload.json',
            '--bind',str(h/'out'),'/out','--ro-bind',str(h/'compiler'),'/hecate-opt',
            '--proc','/proc','--dev','/dev','--tmpfs','/tmp','--setenv','HOME','/nonexistent',
            '--setenv','PATH','/nonexistent','--chdir','/out','--']
        command+=['/hecate-opt','/out/candidate_trace.mlir','--verify-each']if role=='compiler'else[str(h/'python'),'/app/candidate_trace.py']
        contract=observer.contract(command,self.owner,self.namespaces,'a'*32,test_only=True)
        exe={k:v for k,v in contract['executable'].items()if k!='path'}
        major,minor=os.major(exe['device']),os.minor(exe['device'])
        limits='\n'.join(f'{name} {a} {b} units'for name,(a,b)in observer.LIMITS.items())+'\n'
        mounts=['1 0 0:0 / / ro - tmpfs tmpfs ro']
        for i,b in enumerate(contract['bindings'],2):mounts.append(f'{i} 1 0:0 / {b["destination"]} {b["mode"]} - ext4 /dev/test {b["mode"]}')
        for i,d in enumerate(['/proc','/dev','/tmp'],20):
            fs='proc'if d=='/proc'else'tmpfs';mounts.append(f'{i} 1 0:0 / {d} rw - {fs} {fs} rw')
        target=dict(pid=101,start_ticks=1100,boot_id=self.owner['boot_id'])
        record=bound(dict(format=observer.RECORD,contract_binding=contract['binding'],nonce=contract['nonce'],role=role,
            owner=self.owner,process=target,ancestry=[dict(process=target,parent_pid=100),dict(process=self.owner,parent_pid=1)],
            observed_at_ns=1000000,argv=contract['argv'],cwd='/out',environment_names=['HOME','PATH','PWD'],
            public_environment=dict(HOME='/nonexistent',PATH='/nonexistent',PWD='/out'),executable=exe,
            namespaces={n:n+':[11]'for n in observer.NAMESPACES},owner_namespaces=dict(self.namespaces),actual_mountinfo='\n'.join(mounts)+'\n',
            actual_maps=f'00100000-00200000 r-xp 00000000 {major:02x}:{minor:02x} {exe["inode"]} /hecate-opt\n',
            actual_limits=limits,bindings=[dict(destination=b['destination'],identity=b['identity'])for b in contract['bindings']],
            private_file_reads=0,private_key_hashes=0,process_memory_reads=0,processes_started=0,
            compilation_completed_or_source_semantics_or_native_or_first_delivery_certified=False))
        return contract,record

    def check(self,c,r):return observer.verify(c,c['binding'],r,r['binding'])
    def rejected(self,c,r):
        with self.assertRaises((ValueError,KeyError,TypeError)):self.check(rebind(c),rebind(r))

    def test_01_roles_are_only_inert_snapshots_without_compilation_credit(self):
        for role in('compiler','frontend'):
            c,r=self.fixture(role);v=self.check(c,r)
            self.assertTrue(v['raw_snapshot_matches_contract']);self.assertTrue(v['test_only'])
            for k in('compilation_completed','compiler_source_build_origin_certified','source_to_lowered_semantics_certified',
                'native_input_same_key_or_numerical_or_first_delivery_certified','execution_authority'):self.assertIs(v[k],False)
            with self.assertRaises(ValueError):observer.capture(c,c['binding'],99999999)

    def test_02_role_nonce_and_independent_binding_cannot_be_relabelled(self):
        for key,value in [('role','frontend'),('nonce','b'*32),('contract_binding','0'*64)]:
            c,r=self.fixture();r[key]=value;self.rejected(c,r)
        c,r=self.fixture();c['nonce']='b'*32;c=rebind(c)
        with self.assertRaises(ValueError):observer.verify(c,'0'*64,r,r['binding'])

    def test_03_requested_command_or_cwd_cannot_replace_actual_values(self):
        for key,value in [('argv',['/hecate-opt','/out/repaired.mlir']),('cwd','/app')]:
            c,r=self.fixture();r[key]=value;self.rejected(c,r)

    def test_04_cleared_environment_requires_all_original_names_and_values(self):
        for change in('unknown','missing','cwd','duplicate'):
            c,r=self.fixture()
            if change=='unknown':r['public_environment']['UNEXPECTED']='redacted';r['environment_names'].append('UNEXPECTED')
            elif change=='missing':del r['public_environment']['HOME'];r['environment_names'].remove('HOME')
            elif change=='cwd':r['public_environment']['PWD']='/app'
            else:r['environment_names'].append('HOME')
            self.rejected(c,r)

    def test_05_contract_metadata_must_match_its_actual_factory_command(self):
        c,r=self.fixture();c['bindings'][0]['source']=str(self.home/'out');c=rebind(c);r['contract_binding']=c['binding'];self.rejected(c,r)

    def test_06_executable_file_and_executable_mapping_are_both_required(self):
        for change in('inode','hash','mapping'):
            c,r=self.fixture()
            if change=='inode':r['executable']['inode']+=1
            elif change=='hash':r['executable']['sha256']='0'*64
            else:r['actual_maps']=r['actual_maps'].replace('r-xp','r--p')
            self.rejected(c,r)

    def test_07_requested_mounts_cannot_stand_in_for_actual_bind_sources(self):
        for change in('inode','missing','duplicate'):
            c,r=self.fixture()
            if change=='inode':r['bindings'][0]['identity']=dict(r['bindings'][0]['identity'],inode=0)
            elif change=='missing':r['bindings'].pop()
            else:r['bindings'].append(copy.deepcopy(r['bindings'][0]))
            self.rejected(c,r)

    def test_08_mount_modes_and_unexpected_home_mount_are_rejected(self):
        c,r=self.fixture();r['actual_mountinfo']=r['actual_mountinfo'].replace('/hecate-opt ro','/hecate-opt rw');self.rejected(c,r)
        c,r=self.fixture();r['actual_mountinfo']+='30 1 0:0 / /home rw - ext4 /dev/test rw\n';self.rejected(c,r)

    def test_09_all_six_actual_namespace_relations_are_required(self):
        for name in observer.NAMESPACES:
            c,r=self.fixture();r['namespaces'][name]=c['owner_namespaces'][name];self.rejected(c,r)
        c,r=self.fixture();r['namespaces'].pop('user');self.rejected(c,r)

    def test_10_original_cpu_memory_file_descriptors_and_core_limits(self):
        for name,values in observer.LIMITS.items():
            c,r=self.fixture();a,b=values
            r['actual_limits']=r['actual_limits'].replace(f'{name} {a} {b} units',f'{name} {a+1} {b+1} units')
            self.rejected(c,r)

    def test_11_owner_boot_pid_reuse_and_real_parent_edges_are_checked(self):
        for change in('owner','boot','start','edge','cycle'):
            c,r=self.fixture()
            if change=='owner':r['owner']=dict(r['owner'],pid=90)
            elif change=='boot':r['process']=dict(r['process'],boot_id='0'*36)
            elif change=='start':r['process']=dict(r['process'],start_ticks=1101)
            elif change=='edge':r['ancestry'][0]['parent_pid']=99
            else:r['ancestry'][1]=copy.deepcopy(r['ancestry'][0])
            self.rejected(c,r)

    def test_12_snapshot_has_no_completed_compile_native_or_private_authority(self):
        for key,value in [('compilation_completed_or_source_semantics_or_native_or_first_delivery_certified',True),
            ('private_file_reads',1),('private_key_hashes',1),('process_memory_reads',1),('processes_started',1),
            ('actual_maps','libseal is forbidden')]:
            c,r=self.fixture();r[key]=value;self.rejected(c,r)

    def test_13_observed_owner_namespaces_cannot_be_replaced_by_requested_ones(self):
        c,r=self.fixture();r['owner_namespaces']['mnt']='mnt:[99]';self.rejected(c,r)
        c,r=self.fixture();r.pop('owner_namespaces');self.rejected(c,r)

    def test_14_special_mount_absence_and_hidden_store_mount_are_rejected(self):
        c,r=self.fixture();r['actual_mountinfo']='\n'.join(x for x in r['actual_mountinfo'].splitlines()if' /proc 'not in x)+'\n';self.rejected(c,r)
        c,r=self.fixture();r['actual_mountinfo']=r['actual_mountinfo'].replace(' - proc proc rw',' - tmpfs tmpfs rw');self.rejected(c,r)
        c,r=self.fixture();r['actual_mountinfo']+='31 1 0:0 / /nix/store/hidden-private rw - ext4 /dev/test rw\n';self.rejected(c,r)


if __name__=='__main__':unittest.main()
