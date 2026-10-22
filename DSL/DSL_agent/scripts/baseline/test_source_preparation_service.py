"""Application/CLI preparation boundary controls; no source, API or FHE calls."""
import copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from benchmark_graph import digest
from source_preparation_service import SourcePreparation,REQUEST,checked,public_summary
from source_calibration_contract import CalibrationError,FX_FORMAT
from source_approximation_preparation import PreparationError
from source_preparation_entry import parser,main

class PreparationServiceTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        self.request=dict(format=REQUEST,action="inspect",
            manifest=dict(path=str(self.root/"manifest.json"),sha256="a"*64),
            source_signature=dict(format="poseidon-source-calibration-signature-v1",outputs=[dict(name="out",shape=[2])]),
            samples=[dict(x=[.5,-.25])],input_dtypes={"x":"float64"},
            output_structure=dict(kind="tensor",name="out"),resource_profile="general-tensors-v1")
    def receipt(self,status="described"):
        value=dict(status=status,fx_graph=dict(graph_sha256="b"*64,nodes=[
            dict(name="x",op="placeholder",target="x",args="PRIVATE",kwargs={})]),
            observations={},api_calls=0,encrypted_execution=False)
        value["binding"]=digest(value);return value
    def run_action(self,request=None,**changes):
        return SourcePreparation().run(request or self.request,self.root/"out",trust_local_code=changes.get("trusted",True))
    def test_inspect_dispatch_and_request_are_bound(self):
        before=copy.deepcopy(self.request)
        with patch("source_preparation_service.inspect_source",return_value=self.receipt())as worker:
            result=self.run_action()
        self.assertEqual(result["status"],"described");self.assertEqual(result["request_sha256"],digest(before))
        self.assertEqual(self.request,before);self.assertEqual(worker.call_count,1)
        self.assertFalse(result["application_accuracy_certified"]);self.assertFalse(result["ckks_numerically_validated"])
    def test_observe_selection_is_hashed_and_dispatches_once(self):
        value=dict(self.request,action="observe",selection=dict(format=FX_FORMAT,purpose="calibration",
            mode="observe",graph_sha256="b"*64,selectors=[]))
        with patch("source_preparation_service.observe_source",return_value=self.receipt("observed"))as worker:
            result=self.run_action(value)
        self.assertEqual(result["status"],"observed")
        self.assertEqual(worker.call_args.kwargs["expected_selection_sha256"],digest(value["selection"]))
    def test_freeze_dispatch_returns_existing_source_handoff(self):
        value={k:v for k,v in self.request.items()if k in("format","manifest","source_signature")}
        value.update(action="freeze",calibration_evidence=[],configuration={})
        receipt=self.receipt("frozen");receipt["configured_manifest"]=str(self.root/"source/manifest.json")
        receipt["binding"]=digest({k:v for k,v in receipt.items()if k!="binding"})
        with patch("source_preparation_service.freeze",return_value=receipt)as worker:
            result=self.run_action(value)
        self.assertEqual(result["configured_manifest"],receipt["configured_manifest"]);self.assertEqual(worker.call_count,1)
    def test_explicit_true_trust_is_required(self):
        for trust in(False,None,1,"yes"):
            with self.subTest(trust=trust),patch("source_preparation_service.inspect_source")as worker:
                result=self.run_action(trusted=trust)
                self.assertEqual(result["status"],"unsupported");worker.assert_not_called()
    def test_extra_fields_wrong_action_and_callbacks_rejected(self):
        for value in(dict(self.request,callback="bad"),dict(self.request,action="generate"),
                     dict(self.request,extra="PRIVATE"),dict(self.request,format="old")):
            with patch("source_preparation_service.inspect_source")as worker:
                result=self.run_action(value);self.assertEqual(result["status"],"validation_failed");worker.assert_not_called()
    def test_manifest_hash_and_paths_rejected(self):
        for manifest in(dict(path="relative/manifest.json",sha256="a"*64),
                        dict(path=str(self.root/"manifest.json"),sha256="bad"),
                        dict(path=str(self.root/"file.py"),sha256="a"*64)):
            result=self.run_action(dict(self.request,manifest=manifest))
            self.assertEqual(result["failure"]["code"],"source_preparation_input_rejected")
    def test_known_failures_are_classified(self):
        cases=[(CalibrationError("calibration_fx_trace_rejected"),"unsupported"),
               (CalibrationError("calibration_timeout"),"timed_out"),
               (CalibrationError("calibration_binding_changed"),"service_error"),
               (PreparationError("approximation_calibration_domain_exceeded"),"validation_failed")]
        for error,status in cases:
            with self.subTest(error=error.code),patch("source_preparation_service.inspect_source",side_effect=error):
                result=self.run_action();self.assertEqual(result["status"],status)
                self.assertEqual(result["failure"]["code"],error.code)
    def test_private_exception_text_is_not_returned(self):
        for error in(ValueError("SECRET_ENV_VALUE"),OSError("PRIVATE_PATH"),RuntimeError("PRIVATE_SOURCE")):
            with patch("source_preparation_service.inspect_source",side_effect=error):
                result=self.run_action()
            encoded=json.dumps(result);self.assertNotIn(str(error),encoded)
    def test_summary_drops_inputs_source_reference_and_node_args(self):
        receipt=self.receipt();receipt.update(inputs="SECRET_INPUTS",source="PRIVATE_SOURCE",reference="PRIVATE_REFERENCE")
        with patch("source_preparation_service.inspect_source",return_value=receipt):result=self.run_action()
        summary=public_summary(result);encoded=json.dumps(summary)
        for word in("SECRET_INPUTS","PRIVATE_SOURCE","PRIVATE_REFERENCE","PRIVATE"):self.assertNotIn(word,encoded)
        self.assertEqual(summary["node_inventory"],[dict(name="x",op="placeholder",target="x")])
    def test_summary_and_result_have_separate_bindings(self):
        with patch("source_preparation_service.inspect_source",return_value=self.receipt()):result=self.run_action()
        summary=public_summary(result)
        self.assertEqual(result["binding"],digest({k:v for k,v in result.items()if k!="binding"}))
        self.assertEqual(summary["binding"],digest({k:v for k,v in summary.items()if k!="binding"}))
        self.assertNotEqual(summary["binding"],result["binding"])
    def test_cli_parser_and_timeout_bounds(self):
        args=parser().parse_args(["inspect","--request",str(self.root/"request.json"),"--output",str(self.root/"out")])
        self.assertEqual(args.timeout,240);self.assertFalse(args.trust_local_code)
        with self.assertRaises(SystemExit):
            main(["inspect","--request",str(self.root/"request.json"),"--output",str(self.root/"out"),"--timeout","301"])
    def test_cli_rejects_action_mismatch_and_duplicate_json_fields(self):
        path=self.root/"request.json";path.write_text(json.dumps(dict(self.request,action="observe")))
        with patch("builtins.print")as printed:
            result=main(["inspect","--request",str(path),"--output",str(self.root/"out"),"--inside","--trust-local-code"])
        self.assertEqual(result,2);self.assertEqual(json.loads(printed.call_args.args[0])["status"],"validation_failed")
        path.write_text('{"action":"inspect","action":"freeze"}')
        with patch("builtins.print")as printed:
            result=main(["inspect","--request",str(path),"--output",str(self.root/"out"),"--inside","--trust-local-code"])
        self.assertEqual(result,2)
    def test_cli_success_returns_only_public_summary(self):
        path=self.root/"request.json";path.write_text(json.dumps(self.request))
        with patch("source_preparation_service.inspect_source",return_value=self.receipt()),patch("builtins.print")as printed:
            result=main(["inspect","--request",str(path),"--output",str(self.root/"out"),"--inside","--trust-local-code"])
        self.assertEqual(result,0);self.assertNotIn("receipt",json.loads(printed.call_args.args[0]))
    def test_cli_without_trust_never_executes_worker(self):
        path=self.root/"request.json";path.write_text(json.dumps(self.request))
        with patch("source_preparation_service.inspect_source")as worker,patch("builtins.print"):
            result=main(["inspect","--request",str(path),"--output",str(self.root/"out"),"--inside"])
        self.assertEqual(result,2);worker.assert_not_called()

if __name__=="__main__":unittest.main()
