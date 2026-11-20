"""Source-binding controls with inert artifacts; not compiler/CKKS evidence."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from benchmark_suite import Builder
from component_contract import prepare_task
from delivery_component import canonical, read_bytes, sha
from first_output_backend import bind_qualification, verify_qualification_binding, FrozenSourceBackendEvaluator


class BackendSourceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.evidence = self.root/'evidence';self.evidence.mkdir()
        self.attempt = self.evidence/'attempt-00';self.attempt.mkdir()
        self.output = self.attempt/'output';self.output.mkdir()
        b=Builder([(2,)]);self.request=prepare_task(b.finish(b.node('square',['input0'])))
        self.source=b'@hc.func("c,c")\r\ndef golden(x, zero_ct):\r\n    return x*x\r\n\r\n'
        self.identity=dict(test_fixture=True,version=1)
        self.write(self.evidence/'model.json',self.request['model'])
        self.write(self.evidence/'request.json',self.request)
        self.write(self.evidence/'logical-bindings.json',self.request['layout'])
        (self.evidence/'arrays.npz').write_bytes(b'inert numeric fixture; not evaluated')
        (self.evidence/'weights.npz').write_bytes(b'inert public-weight fixture')
        (self.attempt/'candidate.py').write_bytes(self.source)
        self.payload=dict(request=self.request,candidate=dict(schema=1,request_id=self.request['request_id'],hecate_source=self.source.decode()))
        self.write(self.attempt/'trace-payload.json',self.payload)
        for name in ['candidate_trace.mlir','candidate_trace.entry.mlir','lowered.earth.mlir','lowered.ckks.mlir','lowered._hecate_golden.runtime-plan.json']:
            (self.output/name).write_bytes(b'inert compiler fixture')
        self.write(self.output/'trace-evidence.json',dict(frontend='real_Hecate',candidate_python_executed=False,request_id=self.request['request_id']))
        self.write(self.output/'execution.json',dict(backend='poseidon_cpu',encrypted_execution=True,executor_secret_key=False,security='tc128',outputs=[[1.,4.]]))
        self.write(self.output/'inputs.json',[[1.,2.]])
        (self.output/'decrypted.npy').write_bytes(b'inert decrypted fixture')
        self.report=dict(backend='poseidon_cpu',request_id=self.request['request_id'],status='passed',
            qualification=dict(generation_performed=False,format='poseidon-qualification-session-v1'),
            tolerance=dict(atol=1e-5,rtol=1e-4),backend_identity=self.identity,
            compiler_sha256='a'*64,runtime_sha256='b'*64,
            attempts=[dict(index=0,status='passed',source_parsed=True,checked=True,compiled=True,executed=True,
                compile_command=['/hecate-opt','/out/candidate_trace.entry.mlir','--eva'],comparison=dict(passed=True))])
        self.refresh()
    def tearDown(self):self.tmp.cleanup()
    def write(self,path,value):path.write_bytes(canonical(value))
    def refresh(self):
        names=['model.json','request.json','logical-bindings.json','arrays.npz','weights.npz','attempt-00/trace-payload.json']
        self.report['frozen_hashes']={name:sha((self.evidence/name).read_bytes())for name in names}
        self.report['attempts'][0]['artifact_hashes']={p.name:sha(p.read_bytes())for p in self.output.iterdir()if p.is_file()}
        self.write(self.evidence/'report.json',self.report)
    def bind(self,source=None):
        return bind_qualification(self.evidence,self.request,self.source if source is None else source,
            expected_report_sha256=sha((self.evidence/'report.json').read_bytes()),expected_backend_identity=self.identity,evidence_root=self.root)

    def test_exact_original_bytes_and_actual_compiler_input_are_bound(self):
        proof=self.bind()
        self.assertEqual(proof['source_sha256'],sha(self.source));self.assertTrue(proof['compiler_input_bound'])
        self.assertEqual(proof['compiler_input'],'/out/candidate_trace.entry.mlir')
        self.assertEqual(read_bytes(self.attempt/'candidate.py'),self.source)
        self.assertEqual(verify_qualification_binding(proof,self.request,self.source),proof)
        self.assertFalse(proof['numerical_evidence_complete'])
        self.assertEqual(proof['origin'],'retained_qualification')

    def test_repaired_source_cannot_replace_first_output(self):
        with self.assertRaises(ValueError):self.bind(b'first bad answer\n')

    def test_trace_payload_source_must_match_even_when_candidate_file_matches(self):
        self.payload['candidate']['hecate_source']='replacement';self.write(self.attempt/'trace-payload.json',self.payload);self.refresh()
        with self.assertRaises(ValueError):self.bind()

    def test_later_success_is_never_selected(self):
        self.report['attempts']=[dict(self.report['attempts'][0],compiled=False,executed=False,status='failed'),dict(self.report['attempts'][0],index=1)]
        self.write(self.evidence/'report.json',self.report)
        with self.assertRaises(ValueError):self.bind()

    def test_nonzero_attempt_and_changed_request_are_rejected(self):
        self.report['attempts'][0]['index']=1;self.refresh()
        with self.assertRaises(ValueError):self.bind()
        self.report['attempts'][0]['index']=0;self.report['request_id']='0'*64;self.refresh()
        with self.assertRaises(ValueError):self.bind()

    def test_hash_change_after_inspection_invalidates_binding(self):
        proof=self.bind();(self.output/'lowered.ckks.mlir').write_bytes(b'changed')
        with self.assertRaises(ValueError):verify_qualification_binding(proof,self.request,self.source)

    def test_unfrozen_payload_or_modified_reference_is_rejected(self):
        (self.evidence/'arrays.npz').write_bytes(b'reference changed')
        with self.assertRaises(ValueError):self.bind()
        self.refresh();self.report['frozen_hashes'].pop('attempt-00/trace-payload.json');self.write(self.evidence/'report.json',self.report)
        with self.assertRaises(ValueError):self.bind()

    def test_symbolic_success_without_bound_compiler_artifacts_is_rejected(self):
        self.report['attempts'][0]['artifact_hashes'].pop('candidate_trace.entry.mlir');self.write(self.evidence/'report.json',self.report)
        with self.assertRaises(ValueError):self.bind()

    def test_backend_fallback_or_qualifier_generation_is_rejected(self):
        for field,value in [('backend','seal'),('qualification',dict(format='poseidon-qualification-session-v1',generation_performed=True))]:
            original=copy.deepcopy(self.report);self.report[field]=value;self.refresh()
            with self.assertRaises(ValueError):self.bind()
            self.report=original

    def test_tolerance_relaxation_and_integrity_failure_are_rejected(self):
        self.report['tolerance']['atol']=1.;self.refresh()
        with self.assertRaises(ValueError):self.bind()
        self.report['tolerance']['atol']=1e-5;self.report['attempts'][0]['failure_layer']='integrity';self.refresh()
        with self.assertRaises(ValueError):self.bind()

    def test_numerical_failure_keeps_real_compile_observation_separate(self):
        self.report['status']='failed';self.report['attempts'][0].update(status='failed',failure_layer='numerical_comparison',comparison=dict(passed=False));self.refresh()
        proof=self.bind();self.assertTrue(proof['compiler_input_bound']);self.assertTrue(proof['real_encrypted_execution_observed'])
        self.assertFalse(proof['numerical_evidence_complete'])

    def test_secret_key_in_executor_or_wrong_sdk_identity_is_rejected(self):
        self.write(self.output/'execution.json',dict(backend='poseidon_cpu',encrypted_execution=True,executor_secret_key=True,security='tc128'));self.refresh()
        with self.assertRaises(ValueError):self.bind()
        self.report['backend_identity']=dict(version=99);self.refresh()
        with self.assertRaises(ValueError):self.bind()

    def test_symlink_and_path_escape_are_rejected(self):
        proof=self.bind();p=self.attempt/'candidate.py';p.unlink();p.symlink_to(self.root/'outside.py');(self.root/'outside.py').write_bytes(self.source)
        with self.assertRaises(ValueError):verify_qualification_binding(proof,self.request,self.source)
        p.unlink();p.write_bytes(self.source);self.report['frozen_hashes']['../outside.py']=sha(self.source);self.write(self.evidence/'report.json',self.report)
        with self.assertRaises(ValueError):self.bind()

    def backend(self,*,blocked=False,drift=False):
        case=self
        class Backend:
            def __init__(self):self.calls=0;self.checks=0
            def identity_manifest(self):return dict(v=2 if drift and self.calls else 1)
            def check_request(self,request,level):
                self.checks+=1
                if blocked:raise ValueError('SDK not ready')
            def qualify(self,request,candidate,level):
                self.calls+=1
                case.assertEqual(candidate['hecate_source'].encode(),case.source)
                return dict(request_id=request['request_id'],status='passed',evidence=str(case.evidence))
        return Backend()

    def test_external_service_calls_original_qualifier_once_with_exact_source(self):
        backend=self.backend();checks=[]
        evaluator=FrozenSourceBackendEvaluator(backend,batch=None,checkpoint=lambda:checks.append(1),frozen_identity=dict(v=1),output=self.root/'call',test_only=True)
        witness=evaluator(self.request,self.source)
        self.assertEqual(backend.calls,1);self.assertEqual(len(checks),2)
        self.assertTrue(witness['compiled_validated']);self.assertTrue(witness['encrypted_execution'])
        self.assertEqual(witness['inputs'],[]);self.assertEqual(witness['failure_category'],'backend_blocked')
        self.assertEqual(witness['failure_code'],'independent_numerical_witness_required')
        self.assertTrue(witness['test_only'])
        with self.assertRaises(ValueError):evaluator(self.request,self.source)
        self.assertEqual(backend.calls,1)

    def test_backend_gate_failure_consumes_no_qualifier_or_repair(self):
        backend=self.backend(blocked=True)
        evaluator=FrozenSourceBackendEvaluator(backend,batch=None,checkpoint=lambda:None,frozen_identity=dict(v=1),output=self.root/'call',test_only=True)
        result=evaluator(self.request,self.source)
        self.assertEqual(result['failure_category'],'backend_blocked');self.assertEqual(backend.calls,0)
        self.assertFalse(result['compiled_validated'])

    def test_identity_drift_cannot_publish_success(self):
        backend=self.backend(drift=True)
        evaluator=FrozenSourceBackendEvaluator(backend,batch=None,checkpoint=lambda:None,frozen_identity=dict(v=1),output=self.root/'call',test_only=True)
        with self.assertRaises(ValueError):evaluator(self.request,self.source)
        self.assertEqual(backend.calls,1)

    def test_missing_mathematical_layer_is_never_copied_from_native_output(self):
        evaluator=FrozenSourceBackendEvaluator(self.backend(),batch=None,checkpoint=lambda:None,frozen_identity=dict(v=1),output=self.root/'call',numerical_reader=lambda *_:{'inputs':[],'intermediate_decryptions':0,'reference_or_tolerance_changed':False},test_only=True)
        result=evaluator(self.request,self.source)
        self.assertEqual(result['inputs'],[])
        self.assertFalse(result['backend_binding']['numerical_evidence_complete'])


if __name__=='__main__':unittest.main()
