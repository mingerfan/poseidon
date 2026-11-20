"""New overlay profile counterexamples; fixture tools are never executed."""
from pathlib import Path
import copy, hashlib, json
import test_compiler_process_observation as original_controls
import approved_cpu_compiler_observation as observer
from delivery_component import bound, canonical
from normal_cpu_overlay_sandbox import rewrite_command

def descriptor(path):
    raw=path.read_bytes()
    return dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest(),binding=json.loads(raw)['binding'])

class ApprovedCpuCompilerControls(original_controls.CompilerProcessControls):
    def fixture(self,role='compiler'):
        old,record=super().fixture(role)
        home=self.home
        library=home/'libposeidon_shared.so'
        library.write_bytes(b'Inert selected-library fixture; never executed.\n')
        library_sha=hashlib.sha256(library.read_bytes()).hexdigest()
        selector=home/'selector.json'
        selector.write_bytes(canonical(bound(dict(poseidon_build=str(home),
            artifacts=dict(poseidon_library=library_sha)))))
        overlay=home/'overlay.json'
        overlay.write_bytes(canonical(bound(dict(format='poseidon-normal-cpu-approved-overlay-v1',
            inner_selector=descriptor(selector)))))
        selection=bound(dict(format='poseidon-compiler-observer-approved-cpu-selection-v1',
            overlay=descriptor(overlay),selector=descriptor(selector),
            library=dict(path=str(library),sha256=library_sha),test_only=True))
        command=old['command'][:]
        at=command.index('--chdir')
        command[at:at]=['--ro-bind',str(library),str(library),'--setenv','LD_LIBRARY_PATH','/nix/store/inert/lib']
        command=rewrite_command(command,library)
        contract=observer.contract(command,self.owner,self.namespaces,'a'*32,selection,test_only=True)
        record.pop('binding')
        record.update(format=observer.RECORD,contract_binding=contract['binding'],
            environment_names=sorted([*record['environment_names'],'LD_LIBRARY_PATH']),
            public_environment=dict(record['public_environment'],LD_LIBRARY_PATH='/selected-poseidon:/nix/store/inert/lib'),
            bindings=[dict(destination=b['destination'],identity=b['identity'])for b in contract['bindings']])
        record['actual_mountinfo']+='40 1 0:0 / /selected-poseidon/libposeidon_shared.so ro - ext4 /dev/test ro\n'
        return contract,bound(record)

    def check(self,c,r):return observer.verify(c,c['binding'],r,r['binding'])

    def test_15_selected_library_bytes_changed_are_refused(self):
        c,r=self.fixture()
        Path(c['selection']['library']['path']).write_bytes(b'Changed inert library')
        self.rejected(c,r)

    def test_16_overlay_and_selector_binding_mismatch_are_refused(self):
        c,r=self.fixture()
        c['selection']['selector']['binding']='0'*64
        c['selection']=original_controls.rebind(c['selection'])
        self.rejected(c,r)

    def test_17_duplicate_host_path_library_mount_is_refused(self):
        c,r=self.fixture()
        at=c['command'].index('--chdir');library=c['selection']['library']['path']
        c['command'][at:at]=['--ro-bind',library,library]
        self.rejected(c,r)

    def test_18_library_search_cannot_escape_the_approved_prefix(self):
        c,r=self.fixture()
        at=c['command'].index('LD_LIBRARY_PATH')+1;c['command'][at]='/private:/nix/store/inert/lib'
        self.rejected(c,r)

    def test_19_selected_library_actual_mount_identity_is_required(self):
        c,r=self.fixture()
        row=next(v for v in r['bindings']if v['destination']=='/selected-poseidon/libposeidon_shared.so')
        row['identity']=dict(row['identity'],inode=0)
        self.rejected(c,r)

    def test_20_inert_selection_never_becomes_a_live_contract(self):
        c,r=self.fixture()
        with self.assertRaises(ValueError):observer.capture(c,c['binding'],99999999)
        with self.assertRaises(ValueError):observer.contract(c['command'],self.owner,self.namespaces,'b'*32,c['selection'])

if __name__=='__main__':
    import unittest
    unittest.main()
