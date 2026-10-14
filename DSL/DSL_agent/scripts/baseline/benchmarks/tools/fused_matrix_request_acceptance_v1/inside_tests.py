"""End-to-end source-to-generation-context checks using retained real captures."""
import base64, copy, hashlib, importlib.abc, io, json, os, sys, time, unittest
from pathlib import Path
import numpy, torch
from benchmark_graph import canonical, digest
from candidate_bundle import check_provenance, audit_provenance
from component_contract import prepare_task, reconstruct_request, runner_options
from application_capabilities import assess_model
from public_weight_context import verify_binding, verify_context
from pytorch_weight_package import read, binding
from pytorch_ordered_contract import encode_weight_files, source_identity
import pytorch_model_intake as intake
from unified_graph_contract import validate_request
from public_weight_context import POLICY
PROFILE = "general-tensors-v1"
HERE = Path("/tests")
SOURCE = Path("/captures")
OUT = Path("/out")
PLAN = json.loads((HERE / "plan.json").read_text())
ROWS, RECORDS, PREPARATION_FAILURES = [], {}, []

class DenySourceImport(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in ("source_models", "source_helpers"):
            raise AssertionError("Request preparation imported model source")
sys.meta_path.insert(0, DenySourceImport())

def prepare_one(key):
    fixture, captured = SOURCE / "source-fixtures" / key, SOURCE / ("source-worker-" + key)
    spec, paths = intake.prepare(fixture / "manifest.json")
    high = json.loads((captured / "original.json").read_text())
    ref = json.loads((captured / "capture-reference.json").read_text())
    arrays, raw = read(fixture, spec["weights"], resource_profile=PROFILE)
    expected = binding(arrays, spec["weights"])
    low, identity = intake.capture_binding(high, spec, resource_profile=PROFILE,
        capture_metadata=ref["capture_metadata"], expected_weight_binding=expected)
    intake.check_reference(ref, high, low, spec, resource_profile=PROFILE,
                           expected_weight_binding=expected)
    provenance = dict(original=high, decomposition=identity, reference_report=ref,
        trusted_pytorch=dict(format="poseidon-trusted-pytorch-provenance-v14",
                             manifest=spec, files={n:p.read_text() for n,p in paths.items()},
                             weight_files=encode_weight_files(spec, raw, resource_profile=PROFILE)))
    check_provenance(provenance, low)
    audit_provenance(provenance, low)
    options = intake.preparation_options(provenance, low, dict(
        compiler_configuration="poseidon-cpu-eva-w45-v1",
        generation_guidance="explicit-v11", public_weight_policy=POLICY))
    request = prepare_task(low, options)
    validate_request(request)
    context = verify_binding(request)
    verify_context(context)
    folder = OUT / key
    folder.mkdir()
    for name, value in (("model.json",low), ("request.json",request),
                        ("generation-context.json",context), ("model-provenance.json",provenance)):
        (folder / name).write_bytes(canonical(value))
    assessment = assess_model(low, options=options, resource_profile=PROFILE)
    (folder / "preflight.json").write_bytes(canonical(assessment))
    RECORDS[key] = (low, provenance, options, request, context, assessment)
    ROWS.append(dict(case=key, original_model_sha256=digest(high), model_sha256=digest(low),
        source_manifest_sha256=digest(spec), provenance_sha256=digest(provenance),
        capture_binding_sha256=identity["binding_sha256"], request_id=request["request_id"],
        generation_context_bytes=len(canonical(context)), trusted_request_bytes=len(canonical(request)),
        public_constant_count=len(context["public_constants"]),
        output_structure=ref["capture_metadata"]["ordered_alias_capture"].get("output_structure"),
        layout_sha256=digest(request["layout"]), period=request["layout"]["input_slot_period"],
        input_ciphertexts=len(request["layout"]["inputs"]), output_ciphertexts=request["layout"]["output_ciphertexts"],
        preparation_options=options, admission=assessment["admission"],
        execution_readiness=assessment["execution_readiness"],
        compiler=False, encrypted_execution=False, provider_calls=0))

def remember():
    for key in PLAN["source_cases"]:
        try:
            prepare_one(key)
        except (ValueError, TypeError, KeyError, IndexError, AttributeError) as error:
            PREPARATION_FAILURES.append(dict(case=key, error_type=type(error).__name__,
                                             diagnostic=str(error)[:1000]))

def load_prepared():
    prior = Path("/prepared")
    old = json.loads((prior / "report.json").read_text())
    if old["source_models"] != 9 or old["preparation_failures"]:
        raise ValueError("Nine retained prepared sources")
    for row in old["request_rows"]:
        key = row["case"]
        folder = prior / key
        low = json.loads((folder / "model.json").read_text())
        provenance = json.loads((folder / "model-provenance.json").read_text())
        request = json.loads((folder / "request.json").read_text())
        context = json.loads((folder / "generation-context.json").read_text())
        assessment = json.loads((folder / "preflight.json").read_text())
        if (digest(low) != row["model_sha256"] or digest(provenance) != row["provenance_sha256"] or
            request["request_id"] != row["request_id"]):
            raise ValueError("Retained source/request binding")
        validate_request(request)
        if canonical(verify_binding(request)) != canonical(context):
            raise ValueError("Retained projected request binding")
        verify_context(context)
        RECORDS[key] = (low, provenance, row["preparation_options"], request, context, assessment)
        current = copy.deepcopy(row)
        current.pop("output_structure", None)
        current["named_outputs"] = [item["name"] for item in low["outputs"]]
        ROWS.append(current)

selected = sys.argv[1] if len(sys.argv)==2 else "all"
if selected=="privacy":load_prepared()
elif selected=="all":remember()
else:raise ValueError("Explicit developer test selection")

class RequestTests(unittest.TestCase):
    def test_source_revalidation_and_request_identity_are_deterministic(self):
        for key,(low, provenance, options, request, context, assessment) in RECORDS.items():
            with self.subTest(case=key):
                saved = canonical(low), canonical(provenance)
                self.assertEqual(canonical(reconstruct_request(request)), canonical(request))
                self.assertEqual(canonical(prepare_task(low, options)), canonical(request))
                self.assertEqual(request["request_id"], context["request_id"])
                self.assertEqual((canonical(low),canonical(provenance)),saved)
                self.assertEqual(request["compiler_configuration"]["backend"],"poseidon_cpu")
                self.assertIn("--unified-packing",runner_options(request))
                self.assertEqual(options["capacity_packing_version"],"capacity-public-tensors-v9")

    def test_public_context_contains_descriptors_and_keeps_private_evidence_local(self):
        forbidden = {"reference_report","capture_metadata","finite_reference","test_inputs",
                     "arrays","hecate_source","model_provenance","decomposition"}
        def keys(value):
            if isinstance(value,dict):
                for key,item in value.items():
                    yield key
                    yield from keys(item)
            elif isinstance(value,list):
                for item in value:yield from keys(item)
        for key,(low,provenance,options,request,context,assessment) in RECORDS.items():
            with self.subTest(case=key):
                from candidate_contract import RESPONSE_SCHEMA
                self.assertEqual(canonical(context["response_schema"]),canonical(RESPONSE_SCHEMA))
                payload={k:v for k,v in context.items() if k!="response_schema"}
                self.assertFalse(forbidden.intersection(keys(payload)))
                self.assertNotIn("hecate_source",context)
                self.assertNotEqual(context["public_constants"],request["public_constants"])
                self.assertLessEqual(len(canonical(context)),131072)
                self.assertTrue(context["public_constants"])
                for descriptor in context["public_constants"].values():
                    self.assertEqual(set(descriptor),{"kind","shape","elements","content_sha256"})
                self.assertEqual(verify_context(context),context)

    def test_preflight_does_not_promote_binding_to_executable_success(self):
        for key,(_,_,_,request,_,assessment) in RECORDS.items():
            with self.subTest(case=key):
                self.assertEqual(assessment["admission"],"accepted_pending_validation")
                self.assertEqual(assessment["execution_readiness"],"unknown")
                self.assertEqual(assessment["identity"]["request_id"],request["request_id"])
                for stage in ("compiler","encrypted","agent"):
                    self.assertEqual(assessment["stages"][stage]["status"],"unverified")
                for item in ("intermediate_ciphertexts","multiplicative_depth","peak_memory_bytes","execution_seconds"):
                    self.assertEqual(assessment["resources"][item]["certainty"],"unknown")
                self.assertFalse(assessment["encrypted_execution"])
                self.assertFalse(assessment["numerically_validated"])
                self.assertEqual(assessment["paid_calls"],0)

    def test_context_corruption_and_private_fields_are_rejected(self):
        for key,(_,_,_,request,context,_) in RECORDS.items():
            with self.subTest(case=key):
                bad = copy.deepcopy(context)
                bad["reference_report"] = {"private":"synthetic marker"}
                with self.assertRaises(ValueError):verify_context(bad)
                bad = copy.deepcopy(context)
                constant = next(iter(bad["public_constants"]))
                bad["public_constants"][constant]["content_sha256"] = "f"*64
                with self.assertRaises(ValueError):verify_context(bad)
                bad = copy.deepcopy(request)
                bad["request_id"] = "0"*64
                with self.assertRaises(ValueError):validate_request(bad)

    def test_weight_content_change_creates_new_request_binding(self):
        for key,(low,_,options,request,context,_) in RECORDS.items():
            with self.subTest(case=key):
                self.assertTrue(low["constants"])
                modified = copy.deepcopy(low)
                name = next(iter(modified["constants"]))
                def change_first(value):
                    if isinstance(value,list):
                        result=copy.deepcopy(value)
                        result[0]=change_first(result[0])
                        return result
                    return value+.125
                modified["constants"][name] = change_first(modified["constants"][name])
                other = prepare_task(modified,options)
                validate_request(other)
                self.assertNotEqual(other["request_id"],request["request_id"])
                self.assertNotEqual(verify_binding(other)["public_weight_binding"],
                                    context["public_weight_binding"])

    def test_no_source_reimport_provider_or_compiler(self):
        self.assertEqual(len(RECORDS),9)
        self.assertFalse(any(n.split(".")[0] in ("source_models","source_helpers") for n in sys.modules))
        self.assertNotIn("POSEIDON_PROBE_SECRET",os.environ)
        for name in ("deepseek_provider","component_backend","validation_adapter","candidate_worker"):
            self.assertNotIn(name,sys.modules)

stream=io.StringIO();started=time.monotonic()
suite = (unittest.defaultTestLoader.loadTestsFromTestCase(RequestTests) if selected=="all" else
         unittest.defaultTestLoader.loadTestsFromName(
             "__main__.RequestTests.test_public_context_contains_descriptors_and_keeps_private_evidence_local"))
result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
runtime=dict(python=list(sys.version_info[:3]),numpy=numpy.__version__,torch=torch.__version__,cuda=torch.version.cuda)
report=dict(format="poseidon-fused-matrix-generation-context-tests-v1",status="passed" if result.wasSuccessful() else "failed",
 tests=result.testsRun,failed=len(result.failures),errors=len(result.errors),skipped=len(result.skipped),
 suite=selected,source_models=len(RECORDS),planned_source_models=9,preparation_failures=PREPARATION_FAILURES,request_rows=ROWS,runtime=runtime,
 namespace=str(Path("/proc/self/ns/net").readlink()),
 secret_probe_removed="POSEIDON_PROBE_SECRET" not in os.environ,
 source_module_imports=0,source_capture_replays=0,compiler_calls=0,encrypted_executions=0,
 provider_calls=0,new_dependencies=0,production_applied=False,seconds=time.monotonic()-started)
(OUT/"tests.log").write_text(stream.getvalue())
(OUT/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+"\n")
print(stream.getvalue()[-20000:] if not result.wasSuccessful() else stream.getvalue()[-350:])
print(json.dumps({k:report[k] for k in ("status","tests","failed","errors","skipped","source_models")}))
raise SystemExit(0 if result.wasSuccessful() else 1)
