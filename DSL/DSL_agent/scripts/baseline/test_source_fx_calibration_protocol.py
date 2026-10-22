"""Synthetic closed FX receipts; no source inputs or reference answers are embedded."""
import copy,unittest
from benchmark_graph import digest
from source_calibration_contract import FX_FORMAT,CalibrationError
from source_calibration import FX_REPORT,check_report

class FXProtocolTests(unittest.TestCase):
    def setUp(self):
        node=lambda name,op,target,args:dict(name=name,op=op,target=target,args=args,kwargs={"dict":{}})
        graph=dict(format="poseidon-source-calibration-fx-graph-v1",nodes=[
            node("x","placeholder","x",{"tuple":[]}),
            node("gelu","call_function","torch.nn.functional.gelu",{"tuple":[{"node":"x"}]}),
            node("output","output","output",{"tuple":[{"node":"gelu"}]})])
        graph["graph_sha256"]=digest(graph)
        selection=dict(format=FX_FORMAT,purpose="calibration",mode="observe",
            graph_sha256=graph["graph_sha256"],selectors=[
                dict(id="input",node="gelu",op="call_function",target="torch.nn.functional.gelu",
                     point="input",argument=0,projection=dict(kind="tensor"),domain=None)])
        self.request=dict(selection=selection,manifest={"unit_source_sha256":"a"*64},
            implementation={"unit-observer":"b"*64},runtime={"unit":"runtime"},input_dtypes={"x":"float64"},
            samples=[{}],host_net_namespace="unit-host")
        self.report=dict(format=FX_REPORT,status="observed",purpose="calibration",request_sha256=digest(self.request),
            implementation=self.request["implementation"],selection_sha256=digest(selection),
            manifest_sha256=digest(self.request["manifest"]),runtime=self.request["runtime"],input_dtypes={"x":"float64"},
            samples=1,forward_output_sha256=["0"*64],fx_graph=graph,native_dtype_preserved=True,
            model_imported_in_host=False,encrypted_execution=False,approximation_modified=False,
            reference_modified=False,application_accuracy_certified=False,api_calls=0,
            observations={"input":dict(**{k:selection["selectors"][0][k]for k in
                ("node","op","target","point","argument","projection")},samples=1,calls=1,values=2,
                minimum=0.,maximum=0.,mean=0.,std_population=0.,dtypes=["float64"],shapes=[[2]],
                declared_domain=None,outside_domain_calls=0,domain_status="not_declared")},
            checks=["registered_state_unchanged","rng_unchanged","input_storage_isolated",
                "fx_forward_equal_to_original","repeat_observations_equal","trace_state_unchanged"],
            sandbox=dict(network_isolated=True,network_namespace="unit-worker",address_space_bytes=4*1024**3,
                cpu_seconds=150,file_bytes=16*1024**2,open_files=128))
        self.rebind()
    def rebind(self):
        self.report["binding"]=digest({k:v for k,v in self.report.items()if k!="binding"})
    def rejected(self):
        self.rebind()
        with self.assertRaises(CalibrationError):check_report(self.report,self.request)
    def test_synthetic_closed_receipt_and_inventory_are_accepted(self):
        self.assertEqual(check_report(self.report,self.request),self.report)
        self.request["selection"].update(mode="describe",graph_sha256=None,selectors=[])
        self.report.update(status="described",observations={},request_sha256=digest(self.request),
            selection_sha256=digest(self.request["selection"]))
        self.rebind();self.assertEqual(check_report(self.report,self.request),self.report)
    def test_extra_fields_cannot_export_raw_answers(self):
        for key in ("outputs","inputs","credentials","raw_environment"):
            value=copy.deepcopy(self.report);self.report[key]=["private-placeholder"]
            self.rejected();self.report=value
    def test_wrong_graph_identity_or_operator_is_rejected(self):
        self.report["fx_graph"]["nodes"][1]["target"]="torch.rsqrt"
        graph=self.report["fx_graph"];graph["graph_sha256"]=digest({k:v for k,v in graph.items()if k!="graph_sha256"})
        self.rejected()
    def test_node_descriptor_and_boolean_count_mismatch_are_rejected(self):
        before=copy.deepcopy(self.report)
        self.report["observations"]["input"]["point"]="output";self.rejected()
        self.report=before;self.report["observations"]["input"]["samples"]=True;self.rejected()
    def test_false_network_or_forward_checks_are_rejected(self):
        before=copy.deepcopy(self.report)
        self.report["sandbox"]["network_namespace"]="unit-host";self.rejected()
        self.report=before;self.report["checks"].remove("fx_forward_equal_to_original");self.rejected()
    def test_missing_or_duplicate_graph_nodes_are_rejected(self):
        graph=self.report["fx_graph"];graph["nodes"].append(copy.deepcopy(graph["nodes"][1]))
        graph["graph_sha256"]=digest({k:v for k,v in graph.items()if k!="graph_sha256"})
        self.rejected()
    def test_wrong_sample_hash_and_forged_numerical_grade_are_rejected(self):
        before=copy.deepcopy(self.report)
        self.report["forward_output_sha256"]=["bad"];self.rejected()
        self.report=before;self.report["application_accuracy_certified"]=True;self.rejected()

    def test_forward_equality_requires_exact_native_dtype_and_shape(self):
        import numpy as np
        from source_calibration import same_forward
        x=np.asarray([.5,.25],dtype=np.float32)
        self.assertFalse(same_forward({"out":x},{"out":x.astype(np.float64)}))
        self.assertFalse(same_forward({"out":x},{"out":x.reshape(1,2)}))
        self.assertFalse(same_forward({"out":x},{"other":x}))

    def test_forward_equality_accepts_detached_copies_with_identical_dtype(self):
        import numpy as np
        from source_calibration import same_forward
        x=np.asarray([.5,.25],dtype=np.float64)
        self.assertTrue(same_forward({"out":x},{"out":x.copy()},{"out":x.copy()}))

if __name__=="__main__":unittest.main()
