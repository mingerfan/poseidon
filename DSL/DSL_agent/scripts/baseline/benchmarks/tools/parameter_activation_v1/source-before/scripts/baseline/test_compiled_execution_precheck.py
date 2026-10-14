"""Resource execution controls; mocked paths are not encrypted acceptance."""
import copy
import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest import mock
from application_planning import compiled_resources as cr
from application_planning.tests.test_compiled_resources import plan

def checked(status="assessed", ordinal=4):
    failure = None
    if status == "blocked":
        failure = dict(layer="resource", code="compiled_retained_payload_exceeds_budget",
            location=dict(phase="execution", ordinal=ordinal, value="private-value-name"),
            context=dict(required_lower_bytes=9872, budget_bytes=9871))
    result = dict(format=cr.ARTIFACT_FORMAT, status=status, failure=failure,
                  execution_readiness="unknown", encrypted_execution=False)
    result["binding"] = cr.sha(cr.canonical(result))
    return result

class BoundArtifactControls(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.output = Path(self.tmp.name)
        (self.output/"bound").mkdir()
        self.p = plan()
        (self.output/"bound/plan.json").write_bytes(cr.canonical(self.p))
        self.request = dict(request_id="test-public-request")
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        for name, value in (
            ("application_planning.compiled_resources.source_authority", dict(sources={})),
            ("application_planning.compiled_resources._basis", dict(policy=cr.POLICY)),
            ("poseidon_backend_identity.identity", dict(binding="backend")),
            ("application_planning.resources.native_address_limit", 16384),
            ("application_planning.resources.parameters", dict(poly_degree=8)),
        ):
            self.stack.enter_context(mock.patch(name, return_value=value))
        self.gate = dict(plan_sha256="sha256:"+cr.sha(cr.canonical(self.p)))
        self.inspect = self.stack.enter_context(
            mock.patch("poseidon_artifact_gate.inspect", return_value=self.gate))
    def analyze(self, budget=None):
        return cr.analyze_bound_artifacts(self.output, self.request, memory_budget_bytes=budget)
    def test_exact_boundary_and_unknown_peak(self):
        equal = self.analyze(9872)
        self.assertEqual(equal["status"], "assessed")
        self.assertEqual(equal["facts"]["known_retained_payload_lower_bytes"],9872)
        self.assertEqual(equal["resources"]["peak_memory_bytes"],dict(value=None,certainty="unknown"))
        self.assertEqual(equal["execution_readiness"],"unknown")
        self.assertFalse(equal["limits"]["requested_budget_applied_to_runtime"])
        self.assertEqual(self.analyze(9873)["status"],"assessed")
        blocked = self.analyze(9871)
        self.assertEqual(blocked["status"],"blocked")
        self.assertEqual(blocked["failure"]["location"]["ordinal"],4)
        self.assertEqual(blocked["failure"]["context"],dict(required_lower_bytes=9872,budget_bytes=9871))
    def test_custom_analysis_cannot_expand_execution_limit(self):
        for budget in (True,0,16385,1.5,"16384"):
            with self.subTest(budget=budget), self.assertRaises(ValueError):
                self.analyze(budget)
        self.assertEqual(self.analyze()["limits"]["memory_budget_bytes"],16384)
    def test_bound_plan_hash_and_artifact_gate_cannot_be_bypassed(self):
        (self.output/"bound/plan.json").write_bytes(cr.canonical(dict(self.p, final_outputs=["x"])))
        with self.assertRaises(ValueError):
            self.analyze()
        self.assertEqual(self.inspect.call_count,1)
    def test_second_artifact_audit_detects_changed_gate(self):
        self.inspect.side_effect=[self.gate, dict(self.gate, plan_sha256="changed")]
        with self.assertRaisesRegex(ValueError,"identity_drift"):
            self.analyze()
    def test_source_policy_and_backend_identity_drift(self):
        for name in ("source_authority","_basis"):
            with self.subTest(name=name), mock.patch.object(cr,name,side_effect=[{},{"changed":True}]):
                with self.assertRaisesRegex(ValueError,"identity_drift"): self.analyze()
        with mock.patch("poseidon_backend_identity.identity",side_effect=[{},{"changed":True}]):
            with self.assertRaisesRegex(ValueError,"identity_drift"): self.analyze()
    def test_checked_bundle_length_is_used(self):
        self.p["initialization"][0]["payload"]=dict(kind="bundle",content="constant-content")
        raw=cr.canonical(self.p);(self.output/"bound/plan.json").write_bytes(raw)
        self.inspect.return_value=dict(plan_sha256="sha256:"+cr.sha(raw))
        with mock.patch("poseidon_artifact_gate.payload_values",return_value=[1.,2.,3.,4.]) as reader:
            self.assertEqual(self.analyze()["facts"]["retained_plain_slot_payload_bytes"],32)
            reader.assert_called_once()
    def test_native_file_fresh_check_is_stable_and_tamper_rejected(self):
        value=cr.execution_storage_precheck(self.output,self.request)
        self.assertEqual(cr.execution_storage_precheck(self.output,self.request),value)
        file=self.output/cr.PRECHECK_FILE
        altered=copy.deepcopy(value);altered["execution_readiness"]="ready"
        altered["binding"]=cr.sha(cr.canonical({k:v for k,v in altered.items() if k!="binding"}))
        file.write_bytes(cr.canonical(altered))
        with self.assertRaises(cr.CompiledResourceIntegrityRejected):
            cr.execution_storage_precheck(self.output,self.request)
    def test_known_over_limit_records_analysis_and_rejects_without_execution(self):
        with mock.patch("application_planning.resources.native_address_limit",return_value=9871):
            with self.assertRaises(cr.CompiledStorageRejected) as raised:
                cr.execution_storage_precheck(self.output,self.request)
        self.assertTrue((self.output/cr.PRECHECK_FILE).is_file())
        diagnostic=raised.exception.structured_diagnostic()
        self.assertNotIn("private-value-name",json.dumps(diagnostic))
        self.assertEqual(diagnostic["context"]["required_lower_bytes"],9872)
    def test_precheck_symlink_and_invalid_policy_are_integrity_errors(self):
        external=self.output/"external.json";external.write_bytes(cr.canonical(checked()))
        (self.output/cr.PRECHECK_FILE).symlink_to(external)
        with self.assertRaises(cr.CompiledResourceIntegrityRejected):
            cr.execution_storage_precheck(self.output,self.request)
        with mock.patch.object(cr,"_basis",side_effect=ValueError("private path/policy")):
            with self.assertRaises(cr.CompiledResourceIntegrityRejected) as raised:
                cr.execution_storage_precheck(self.output,self.request)
        self.assertNotIn("private path",str(raised.exception))
    def test_io_failure_remains_service_environment_not_candidate(self):
        with mock.patch.object(cr,"analyze_bound_artifacts",side_effect=OSError("device unavailable")):
            with self.assertRaises(OSError): cr.execution_storage_precheck(self.output,self.request)

class TypedResourceControls(unittest.TestCase):
    def test_binding_and_exact_integer_context_are_required(self):
        value=checked("blocked")
        error=cr.CompiledStorageRejected(value)
        value["failure"]["context"]["budget_bytes"]=1
        self.assertEqual(error.analysis["failure"]["context"]["budget_bytes"],9871)
        for field,bad in (("budget_bytes",True),("required_lower_bytes",1.5),
                          ("required_lower_bytes",9871),("required_lower_bytes",2**53+1)):
            candidate=checked("blocked");candidate["failure"]["context"][field]=bad
            candidate["binding"]=cr.sha(cr.canonical({k:v for k,v in candidate.items() if k!="binding"}))
            with self.subTest(field=field,bad=bad),self.assertRaises(ValueError):
                cr.CompiledStorageRejected(candidate)
        with self.assertRaises(ValueError):cr.CompiledStorageRejected(value)
    def test_external_input_location_is_public_candidate(self):
        value=checked("blocked",None)
        self.assertEqual(cr.CompiledStorageRejected(value).structured_diagnostic()["location"],dict(kind="candidate"))

class NativeExecutionControls(unittest.TestCase):
    def request(self):
        return dict(layout=dict(output_ciphertexts=1,output_selectors=[[0,0]],output_shape=[1]))
    def test_central_execute_blocks_before_native_runner(self):
        from poseidon_validation import execute
        for rejection in (cr.CompiledStorageRejected(checked("blocked")),cr.CompiledResourceIntegrityRejected()):
            with self.subTest(error=type(rejection).__name__),tempfile.TemporaryDirectory() as folder:
                native=mock.Mock()
                with mock.patch.object(cr,"execution_storage_precheck",side_effect=rejection):
                    with self.assertRaises(type(rejection)):
                        execute(Path(folder)/"payload",Path(folder),self.request(),native,Path(folder)/"log")
                native.assert_not_called()
    def successful(self, mutate=False, private_key=False):
        from poseidon_validation import execute
        from poseidon_artifact_gate import PRIMES
        with tempfile.TemporaryDirectory() as folder:
            output=Path(folder);value=checked();(output/cr.PRECHECK_FILE).write_bytes(cr.canonical(value))
            def runner(payload,out,argv,log,seconds):
                self.assertEqual(seconds,150)
                self.assertEqual(argv[0],"/poseidon-agent-cpu")
                if mutate=="remove":(out/cr.PRECHECK_FILE).unlink()
                elif mutate=="symlink":
                    original=out/"unchanged-resource.json"
                    (out/cr.PRECHECK_FILE).rename(original)
                    (out/cr.PRECHECK_FILE).symlink_to(original)
                elif mutate:(out/cr.PRECHECK_FILE).write_text("{}")
                (out/"execution.json").write_text(json.dumps(dict(
                    backend="poseidon_cpu",encrypted_execution=True,bootstrap=False,security="tc128",
                    executor_secret_key=private_key,q=list(PRIMES[:13]),p=list(PRIMES[13:]),
                    outputs=[[[.25]*256]])))
                return 0
            with mock.patch.object(cr,"execution_storage_precheck",return_value=value):
                report=execute(output/"payload",output,self.request(),runner,output/"log")
                self.assertEqual(json.loads((output/"execution-metadata.json").read_text()),report)
                return report
    def test_checked_return_includes_lower_bound_not_peak_claim(self):
        report=self.successful()
        self.assertEqual(report["resource_precheck"]["execution_readiness"],"unknown")
        self.assertFalse(report["resource_precheck"]["encrypted_execution"])
    def test_precheck_mutation_and_private_key_executor_are_rejected(self):
        for change in (True,"remove","symlink"):
            with self.subTest(change=change),self.assertRaises(cr.CompiledResourceIntegrityRejected):
                self.successful(mutate=change)
        with self.assertRaisesRegex(ValueError,"key boundary"):self.successful(private_key=True)

class CandidateAdapterControls(unittest.TestCase):
    def evaluate(self,rejection,grade="numerical"):
        from validation_adapter import ValidationContext,CandidateValidationAdapter
        folder=tempfile.TemporaryDirectory();self.addCleanup(folder.cleanup);result=Path(folder.name)
        request=dict(request_id="fixture",task="fixture",compiler_configuration=dict(backend="poseidon_cpu"))
        calls=[]
        def native(payload,out,argv,*args):
            calls.append(argv)
            (out/"trace-evidence.json").write_text(json.dumps(dict(frontend="real_Hecate")))
            return 0
        context=ValidationContext(result,request,dict(attempts=[]),{},dict(atol=1e-5,rtol=1e-4),45,result/"keys",
                                  False,native,lambda:None,validation_level=grade)
        with ExitStack() as stack:
            for name,value in (
                ("validation_adapter.validate_candidate",{}),
                ("validation_adapter.trace_payload_json","{}"),
                ("poseidon_runtime_compile.prepare_entrypoint",{}),
                ("poseidon_runtime_compile.command",["compiler"]),
                ("poseidon_validation.gate",{}),
            ):stack.enter_context(mock.patch(name,return_value=value))
            pre=stack.enter_context(mock.patch.object(cr,"execution_storage_precheck",side_effect=rejection))
            prepare=stack.enter_context(mock.patch("poseidon_validation.prepare_inputs"))
            execute=stack.enter_context(mock.patch("poseidon_validation.execute"))
            feedback=CandidateValidationAdapter(context).evaluate(json.dumps(dict(hecate_source="x=1")),0)
        self.assertEqual(len(calls),2)
        prepare.assert_not_called();execute.assert_not_called()
        return feedback,context.report["attempts"][0],pre.call_count
    def test_resource_rejection_preserves_real_stage_facts(self):
        feedback,attempt,count=self.evaluate(cr.CompiledStorageRejected(checked("blocked")))
        self.assertTrue(attempt["compiled"])
        self.assertFalse(attempt["executed"])
        self.assertFalse(attempt["numerically_correct"])
        self.assertEqual(feedback["layer"],"resource_gate")
        self.assertEqual(feedback["category"],"resource")
        self.assertIn("structured_diagnostic",attempt)
        self.assertEqual(count,1)
    def test_integrity_rejection_does_not_trigger_candidate_repair(self):
        feedback,attempt,count=self.evaluate(cr.CompiledResourceIntegrityRejected())
        self.assertEqual(feedback["layer"],"integrity")
        self.assertEqual(feedback["category"],"integrity")
        self.assertFalse(attempt["executed"])
    def test_explicit_compile_grade_keeps_original_contract(self):
        feedback,attempt,count=self.evaluate(AssertionError("must not apply execution gate"),"compiled")
        self.assertEqual(feedback["status"],"passed")
        self.assertEqual(attempt["validation_level"],"compiled")
        self.assertFalse(attempt["executed"])
        self.assertEqual(count,0)

class OutputProvenanceControls(unittest.TestCase):
    def test_ordered_package_tree_is_preserved_v12_through_v15(self):
        from program_execution import package_execution_metadata
        from pytorch_model_intake import SOURCE_FORMATS
        tree=dict(kind="tuple",items=[dict(kind="tensor",name="output0")])
        model=dict()
        for fmt in SOURCE_FORMATS:
            if fmt not in ("poseidon-trusted-pytorch-v12","poseidon-trusted-pytorch-v13",
                           "poseidon-trusted-pytorch-v14","poseidon-trusted-pytorch-v15"):
                continue
            source=dict(original=model,trusted_pytorch=dict(manifest=dict(format=fmt)),
                        reference_report=dict(capture_metadata=dict(output_structure=tree)))
            raw=cr.canonical(source)
            manifest=dict(format="poseidon-dsl-bundle-v6",model_provenance="included",
                          files={"model-provenance.json":cr.sha(raw)})
            from candidate_bundle import PROVENANCE
            manifest["files"]={PROVENANCE:cr.sha(raw)}
            with self.subTest(fmt=fmt), mock.patch("model_weight_package.regular_bytes",return_value=raw),\
                 mock.patch("candidate_bundle.check_provenance"):
                result=package_execution_metadata("package",manifest,model)
                self.assertEqual(result["output_structure"],tree)
                self.assertIsNot(result["output_structure"],tree)

if __name__=="__main__":unittest.main()
