"""Inert provenance counterexamples; no compiler, HE or key material exists."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from delivery_component import bound,canonical,sha
from first_output_assessment import compare
from normal_cpu_overlay_sandbox import loaded_library,TARGET
from poseidon_artifact_gate import PRIMES
from shared_execution import KINDS
import normal_cpu_overlay_evidence as evidence
import normal_cpu_overlay_probe as probe
import normal_cpu_overlay_sdk as sdk
import normal_cpu_overlay_sandbox as sandbox
import normal_cpu_overlay_host_evidence as host_evidence


class CanaryEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);base=Path(self.tmp.name)
        self.work=base/'work/platforms/aarch64-linux';self.root=self.work/'results/canary';self.root.mkdir(parents=True)
        self.client=self.root/'client';self.client.mkdir();self.original=self.work/'results/original';self.original.mkdir()
        self.source=b'inert source fixture, never parsed or executed'
        self.cfg=dict(test_fixture='inert ordinary CPU configuration')
        self.request=dict(request_id='1'*64,compiler_configuration=self.cfg,layout=dict(output_selectors=[[0,0],[0,1]]))
        self.library=base/'libposeidon_shared.so';self.library.write_bytes(b'inert non-executable DSO fixture')
        self.overlay=dict(path=str(base/'overlay.json'),sha256='a'*64,binding='b'*64)
        self.context=dict(outer=dict(binding=self.overlay['binding'],ordinary_compiler_configuration=self.cfg,
                inner_selector=dict(path=str(base/'selector.json'))),
            inner=dict(artifacts=dict(poseidon_library=sha(self.library.read_bytes()))),library=self.library)
        self.proof=dict(binding='c'*64)
        (self.original/'attempt-00/output/bound').mkdir(parents=True)
        (self.client/'bound').mkdir()
        runtime=dict(execution=[dict(kind='compute',op='mul_cc')])
        for p in [self.original/'attempt-00/output/bound/plan.json',self.client/'bound/plan.json']:self.save(p,runtime)
        inputs=[[[1.,2.]],[[3.,4.]]];self.reference=[[1.,4.],[9.,16.]]
        for p in [self.original/'attempt-00/output/inputs.json',self.client/'inputs.json']:self.save(p,inputs)
        self.save(self.original/'report.json',dict(attempts=[dict(comparison=dict(reference=self.reference))]))
        self.plan=bound(dict(format=probe.FORMAT,overlay=self.overlay,retained_compilation=dict(path='inert-proof-fixture',sha256='0'*64),
            request_id=self.request['request_id'],source_sha256=sha(self.source),output=str(self.root),
            source_files={str(Path(p).absolute()):sha(Path(p).read_bytes())for p in (probe.__file__,sdk.__file__,sandbox.__file__)},
            selected_inner_sdk=self.context['outer']['inner_selector']['path'],
            original_inputs_sha256=sha(canonical(inputs)),original_reference_sha256=sha(canonical(self.reference)),
            limits=dict(runtime_client_invocations=1,ephemeral_key_sets_maximum=1,persistent_key_epochs=0,
                API=0,compilation=0,reference_model_forwards=0,downloads=0,CPU=[150,155],AS=4*1024**3,FSIZE=16*1024**2,FD=128,CORE=0),
            CKKS=dict(atol=1e-5,rtol=1e-4),bootstrap=False,Agent_generated=False,first_output_acceptance=False))
        self.plan_path=base/'canary-plan.json';self.save(self.plan_path,self.plan)
        self.row=dict(plan=dict(source_binding=self.plan['binding'],stages={k:int(k=='encrypted')for k in KINDS}),
            status='succeeded',active=None,used={k:int(k=='encrypted')for k in KINDS},artifacts=[dict(path=str(self.root))])
        self.save(self.root/'probe-start.json',bound(dict(plan_binding=self.plan['binding'],SDK_binding=self.overlay['binding'],runtime_client_attempts=1,persistent_key_epochs=0)))
        self.save(self.root/'native-terminal.json',bound(dict(exit_code=0,runtime_client_attempts=1,persistent_key_epochs=0)))
        self.execution=dict(backend='poseidon_cpu',encrypted_execution=True,executor_secret_key=False,bootstrap=False,
            security='tc128',degree=32768,q=list(PRIMES[:13]),p=list(PRIMES[13:]),outputs=[[[1.,4.]],[[9.,16.]]])
        self.save(self.client/'execution.json',self.execution)
        (self.client/'execute.log').write_text(' 123: calling init: '+TARGET+'\n')
        dso=loaded_library(self.client/'execute.log',self.library);self.save(self.client/'normal-cpu-overlay-dso.json',dso)
        comparisons=[compare(r,r,atol=1e-5,rtol=1e-4)for r in self.reference]
        self.result=bound(dict(format='poseidon-normal-cpu-overlay-original-client-result-v1',plan_binding=self.plan['binding'],SDK_binding=self.overlay['binding'],
            request_id=self.request['request_id'],source_sha256=sha(self.source),original_compilation_binding=self.proof['binding'],actual_DSO_binding=dso['binding'],
            input_groups=2,compared_values=4,failed_values=0,max_absolute_error=0.,comparisons=comparisons,
            ordinary_CPU_canary_passed=True,real_encrypted_execution=True,runtime_client_invocations=1,ephemeral_key_sets=1,persistent_key_epochs=0,
            API=0,compilation=0,reference_model_forwards=0,Agent_generated=False,first_output_acceptance=False,complete_network_acceptance=False,
            complete_normal_SDK_qualification=False))
        self.result_path=self.root/'report.json';self.save(self.result_path,self.result)

    def save(self,path,value):path.write_bytes(canonical(value))
    def descriptor(self,path):
        value=json.loads(path.read_bytes());return dict(path=str(path),sha256=sha(path.read_bytes()),binding=value['binding'])
    def resign_result(self):
        self.result.pop('binding');self.result=bound(self.result);self.save(self.result_path,self.result)
    def verify(self, *, host_missing=False):
        host_result=bound(dict(format='inert-host-witness-fixture',evidence_sha256={}))
        with patch.object(host_evidence,'verify',return_value=host_result,side_effect=ValueError('missing complete actual host witness')if host_missing else None),patch.object(sdk,'verify',return_value=self.context),patch.object(probe,'_proof',return_value=(self.proof,self.original,self.request,self.source)),patch('shared_execution.Coordinator')as coordinator:
            coordinator.return_value.snapshot.return_value=(dict(batches=dict(fixture=self.row)),{})
            return evidence.verify(self.overlay,self.descriptor(self.plan_path),self.descriptor(self.result_path),
                work=self.work,platform='fixture',agent_root=self.work)

    def test_positive_numbers_and_ledger_cannot_replace_missing_actual_host_witness(self):
        with self.assertRaises(ValueError):self.verify(host_missing=True)

    def test_actual_numbers_and_primitive_scope_are_recomputed_from_inert_records(self):
        value=self.verify();self.assertEqual(value['counts']['compared_values'],4)
        self.assertTrue(value['ordinary_CPU_test_admission']);self.assertEqual(value['actual_covered_runtime_ops'],['mul_cc'])
        self.assertFalse(value['complete_normal_SDK_qualification']);self.assertFalse(value['bootstrap_certified'])

    def test_resigned_success_cannot_hide_changed_actual_decoded_values(self):
        self.execution['outputs'][0][0][0]+=1.;self.save(self.client/'execution.json',self.execution)
        with self.assertRaises(ValueError):self.verify()

    def test_changed_retained_runtime_plan_cannot_use_original_compilation_receipt(self):
        self.save(self.client/'bound/plan.json',dict(execution=[dict(kind='compute',op='add_cc')]))
        with self.assertRaises(ValueError):self.verify()

    def test_changed_client_inputs_cannot_use_original_reference(self):
        self.save(self.client/'inputs.json',[[[9.,9.]],[[9.,9.]]])
        with self.assertRaises(ValueError):self.verify()

    def test_missing_or_wrong_actual_loaded_DSO_is_rejected(self):
        (self.client/'execute.log').write_text(' 123: calling init: /old/libposeidon_shared.so\n')
        with self.assertRaises(ValueError):self.verify()

    def test_canary_cannot_be_resigned_as_whole_SDK_or_first_output_certification(self):
        for key in ['complete_normal_SDK_qualification','first_output_acceptance','Agent_generated','complete_network_acceptance']:
            self.result[key]=True;self.resign_result()
            with self.assertRaises(ValueError):self.verify()
            self.result[key]=False;self.resign_result()

    def test_nonterminal_or_duplicate_native_ledger_does_not_grant_admission(self):
        self.row['status']='running'
        with self.assertRaises(ValueError):self.verify()
        self.row['status']='succeeded';self.row['used']['encrypted']=2
        with self.assertRaises(ValueError):self.verify()

    def test_changed_actual_parameter_or_executor_secret_key_boundary_is_rejected(self):
        self.execution['executor_secret_key']=True;self.save(self.client/'execution.json',self.execution)
        with self.assertRaises(ValueError):self.verify()

    def test_reference_substitution_is_rejected_even_when_decodes_are_unchanged(self):
        self.save(self.original/'report.json',dict(attempts=[dict(comparison=dict(reference=[[0.,0.],[0.,0.]]))]))
        with self.assertRaises(ValueError):self.verify()

    def test_claimed_error_numbers_must_match_fixed_threshold_comparison(self):
        self.result['failed_values']=3;self.resign_result()
        with self.assertRaises(ValueError):self.verify()


class BackendAuthorityTests(unittest.TestCase):
    def test_ordinary_base_adapter_cannot_admit_the_overlay_route(self):
        from application_backend import PoseidonBackend
        from normal_cpu_overlay_backend import trusted_backend
        with self.assertRaises(ValueError):trusted_backend(PoseidonBackend(),approved_overlay_required=True)

    def test_arbitrary_backend_subclasses_cannot_admit_live_delivery(self):
        from application_backend import PoseidonBackend
        from normal_cpu_overlay_backend import trusted_backend
        class Untrusted(PoseidonBackend):pass
        self.assertIsInstance(trusted_backend(PoseidonBackend()),PoseidonBackend)
        with self.assertRaises(ValueError):trusted_backend(Untrusted())

    def test_approved_overlay_class_requires_actual_verified_canary_before_trust(self):
        from normal_cpu_overlay_backend import NormalCpuOverlayBackend,trusted_backend
        backend=NormalCpuOverlayBackend({}, {}, {})
        with patch.object(backend,'_evidence',side_effect=ValueError('no actual canary')):
            with self.assertRaises(ValueError):trusted_backend(backend)


if __name__=='__main__':unittest.main()
