"""Calibration selector and passive observation controls; no FHE/API."""
import copy,unittest
import numpy as np
import torch
from source_calibration_contract import checked,CalibrationError,FORMAT
from source_calibration_observer import ModuleObserver,summarize

def selection(module="activation",*,argument=0,projection=None,domain=None):
    return dict(format=FORMAT,purpose="calibration",selectors=[dict(id="observed",module=module,
        argument=argument,projection=projection or dict(kind="tensor"),domain=domain)])
class Model(torch.nn.Module):
    def __init__(self,dtype=torch.float64):
        super().__init__();self.linear=torch.nn.Linear(2,2,dtype=dtype)
        self.activation=torch.nn.GELU()
        with torch.no_grad():
            self.linear.weight.copy_(torch.tensor([[.5,-.25],[.75,.375]],dtype=dtype))
            self.linear.bias.copy_(torch.tensor([.1,-.2],dtype=dtype))
    def forward(self,x):return self.activation(self.linear(x))

class SelectionTests(unittest.TestCase):
    def rejected(self,value):
        with self.assertRaises(CalibrationError):checked(value)
    def test_selector_detached_and_exact_shape(self):
        value=selection(domain=[-.5,.5]);before=copy.deepcopy(value)
        clean=checked(value);clean["selectors"][0]["module"]="changed"
        self.assertEqual(value,before)
    def test_role_and_unknown_fields_are_rejected(self):
        for value in (dict(selection(),purpose="test"),dict(selection(),purpose="validation"),
                      dict(selection(),format="unknown"),dict(selection(),extra=True)):
            self.rejected(value)
    def test_boolean_argument_epsilon_and_domain_rejected(self):
        self.rejected(selection(argument=True))
        self.rejected(selection(projection=dict(kind="rms_mean_square",epsilon=True)))
        self.rejected(selection(domain=[False,1.]))
    def test_nonfinite_and_invalid_domain_rejected(self):
        for domain in ([0.,0.],[2.,1.],[-1025.,0.],[0.,float("inf")]):
            self.rejected(selection(domain=domain))
    def test_duplicate_ids_targets_and_budget_rejected(self):
        value=selection();value["selectors"]*=2;self.rejected(value)
        value=selection();value["selectors"].append(dict(value["selectors"][0],id="another"));self.rejected(value)
        value=selection();value["selectors"]=[dict(value["selectors"][0],id="s"+str(i),module="m"+str(i))for i in range(33)]
        self.rejected(value)
    def test_path_traversal_dynamic_selector_and_unknown_projection_rejected(self):
        for module in ("../secret","activation[0]","a/b","a..b"):
            self.rejected(selection(module))
        self.rejected(selection(projection=dict(kind="custom")))
        self.rejected(selection(projection=dict(kind="tensor",function="callback")))

class ObserverTests(unittest.TestCase):
    def observe(self,module,x,config,**kwargs):
        baseline=module(x.clone())
        observer=ModuleObserver(module,config,tensor_elements=65536)
        with observer:
            observer.begin();first=module(x.clone());events=observer.observations()
            observer.begin();second=module(x.clone());other=observer.observations()
        torch.testing.assert_close(first,baseline,rtol=0,atol=0)
        torch.testing.assert_close(second,baseline,rtol=0,atol=0)
        self.assertEqual(events,other)
        return events,summarize(config,[events])
    def test_float32_and_float64_input_precision_and_output_unchanged(self):
        for dtype in (torch.float32,torch.float64):
            with self.subTest(dtype=dtype):
                module=Model(dtype).eval();x=torch.tensor([-.7,.3],dtype=dtype)
                events,report=self.observe(module,x,selection())
                expected=module.linear(x).detach().numpy()
                self.assertEqual(report["observed"]["dtypes"],[str(dtype).removeprefix("torch.")])
                self.assertEqual(report["observed"]["minimum"],float(expected.min()))
                self.assertEqual(report["observed"]["maximum"],float(expected.max()))
                self.assertFalse(module.activation._forward_pre_hooks)
    def test_outside_domain_is_observed_without_altering_model(self):
        module=Model().eval();x=torch.tensor([2.,-2.],dtype=torch.float64)
        _,report=self.observe(module,x,selection(domain=[-.1,.1]))
        self.assertEqual(report["observed"]["domain_status"],"outside")
        self.assertEqual(report["observed"]["outside_domain_calls"],1)
    def test_layer_norm_variance_matches_native_dtype(self):
        class Norm(torch.nn.Module):
            def __init__(self):
                super().__init__();self.norm=torch.nn.LayerNorm(2,eps=1e-4,dtype=torch.float32)
            def forward(self,x):return self.norm(x)
        module=Norm().eval();x=torch.tensor([[.1,.6],[-.3,.2]],dtype=torch.float32)
        _,report=self.observe(module,x,selection("norm",projection=dict(kind="layer_norm_variance")))
        expected=x.var(dim=-1,unbiased=False,keepdim=True)+module.norm.eps
        self.assertEqual(report["observed"]["minimum"],float(expected.min()))
        self.assertEqual(report["observed"]["maximum"],float(expected.max()))
        self.assertEqual(report["observed"]["dtypes"],["float32"])
    def test_rms_projection_is_explicit(self):
        module=torch.nn.Sequential(torch.nn.Identity()).eval()
        x=torch.tensor([[.1,.2],[.3,.4]],dtype=torch.float64)
        _,report=self.observe(module,x,selection("0",projection=dict(kind="rms_mean_square",epsilon=1e-5)))
        expected=(x*x).mean(dim=-1,keepdim=True)+1e-5
        self.assertEqual(report["observed"]["minimum"],float(expected.min()))
        self.assertEqual(report["observed"]["maximum"],float(expected.max()))
    def test_keyword_input_and_root_selection(self):
        class Keyword(torch.nn.Module):
            def __init__(self):super().__init__();self.norm=torch.nn.LayerNorm(2,dtype=torch.float64)
            def forward(self,x):return self.norm(input=x)
        module=Keyword().eval();x=torch.tensor([.1,.2],dtype=torch.float64)
        _,report=self.observe(module,x,selection("norm",argument="input"))
        self.assertEqual(report["observed"]["values"],2)
        _,report=self.observe(module,x,selection(""))
        self.assertEqual(report["observed"]["minimum"],.1)
    def test_reused_module_records_each_invocation(self):
        class Reused(torch.nn.Module):
            def __init__(self):super().__init__();self.activation=torch.nn.ReLU()
            def forward(self,x):return self.activation(x)+self.activation(x*2)
        module=Reused().eval();x=torch.tensor([-.25,.5],dtype=torch.float64)
        _,report=self.observe(module,x,selection())
        self.assertEqual(report["observed"]["calls"],2);self.assertEqual(report["observed"]["values"],4)
        self.assertEqual(report["observed"]["minimum"],-.5);self.assertEqual(report["observed"]["maximum"],1.)
    def test_stats_are_weighted_by_values(self):
        config=selection("");module=torch.nn.Identity().eval()
        observations=[]
        for x in (torch.tensor([0.,2.],dtype=torch.float64),torch.tensor([4.],dtype=torch.float64)):
            events,_=self.observe(module,x,config);observations.append(events)
        report=summarize(config,observations)["observed"]
        self.assertEqual(report["values"],3);self.assertEqual(report["mean"],2.)
        self.assertAlmostEqual(report["std_population"],np.std([0.,2.,4.]))
    def test_missing_and_unused_selector_rejected(self):
        with self.assertRaises(CalibrationError):ModuleObserver(Model(),selection("missing"),tensor_elements=16)
        class Unused(torch.nn.Module):
            def __init__(self):super().__init__();self.activation=torch.nn.ReLU()
            def forward(self,x):return x
        observer=ModuleObserver(Unused(),selection(),tensor_elements=16)
        with observer:
            observer.begin();observer.module(torch.tensor([.1],dtype=torch.float64))
            with self.assertRaises(CalibrationError):observer.observations()
        self.assertFalse(observer.module.activation._forward_pre_hooks)
    def test_projection_dtype_nonfinite_and_tensor_budget_rejected(self):
        module=torch.nn.Identity()
        for value,config,limit in (
            (torch.tensor([1],dtype=torch.int64),selection(""),16),
            (torch.tensor([float("nan")]),selection(""),16),
            (torch.tensor([.1,.2]),selection(""),1)):
            with self.subTest(value=str(value)),self.assertRaises(CalibrationError):
                with ModuleObserver(module,config,tensor_elements=limit)as observer:
                    observer.begin();module(value)
            self.assertFalse(module._forward_pre_hooks)
        with self.assertRaises(CalibrationError):
            ModuleObserver(Model(),selection(projection=dict(kind="layer_norm_variance")),tensor_elements=16)
    def test_event_budget_and_cleanup_on_failure(self):
        module=torch.nn.Identity();observer=ModuleObserver(module,selection(""),tensor_elements=16)
        with self.assertRaises(CalibrationError):
            with observer:
                observer.begin()
                for _ in range(257):module(torch.tensor([.1]))
        self.assertFalse(module._forward_pre_hooks)
    def test_clone_isolated_from_later_inplace_changes(self):
        class Changed(torch.nn.Module):
            def __init__(self):super().__init__();self.activation=torch.nn.Identity()
            def forward(self,x):y=self.activation(x);y.add_(1);return y
        module=Changed();x=torch.tensor([.1,.2],dtype=torch.float64)
        _,report=self.observe(module,x,selection())
        self.assertEqual(report["observed"]["minimum"],.1)
    def test_existing_hooks_are_restored(self):
        module=Model();handle=module.activation.register_forward_pre_hook(lambda m,args:None)
        before=dict(module.activation._forward_pre_hooks)
        self.observe(module,torch.tensor([.1,.2],dtype=torch.float64),selection())
        self.assertEqual(dict(module.activation._forward_pre_hooks),before)
        handle.remove()
if __name__=="__main__":unittest.main()


class CalibrationReportTests(unittest.TestCase):
    def setUp(self):
        from pathlib import Path
        from benchmark_graph import digest
        import source_calibration as calibration
        # Real isolated report from the previous bound fixture; tests mutate
        # detached copies and do not launch any worker.
        folder=Path("/home/lhohy/poseidon-work/platforms/aarch64-linux/results/agent-source-calibration-acceptance-r1622/gelu_inputs_f32/observation")
        import json
        if folder.exists():
            self.report=json.loads((folder/"calibration.json").read_bytes())
            self.request=json.loads((folder/"request.json").read_bytes())
        else:
            self.skipTest("Bound isolated calibration fixture is unavailable")
        self.calibration=calibration;self.digest=digest
    def rejected(self,value):
        value["binding"]=self.digest({k:v for k,v in value.items()if k!="binding"})
        with self.assertRaises(CalibrationError):self.calibration.check_report(value,self.request)
    def test_original_real_report_protocol_is_valid(self):
        self.calibration.check_report(self.report,self.request)
    def test_output_hash_count_and_type_are_bound(self):
        for hashes in ([],["bad"]*3,["0"*64]*2):
            value=copy.deepcopy(self.report);value["forward_output_sha256"]=hashes;self.rejected(value)
    def test_domain_status_cannot_hide_observed_excess(self):
        value=copy.deepcopy(self.report);item=value["observations"]["domain_values"]
        item["outside_domain_calls"]=0;item["domain_status"]="inside";self.rejected(value)
    def test_unbounded_counts_and_dtype_are_rejected(self):
        for changes in (dict(calls=True),dict(calls=10**9),dict(values=10**9),dict(dtypes=["int64"])):
            value=copy.deepcopy(self.report);value["observations"]["domain_values"].update(changes);self.rejected(value)
    def test_modified_reference_or_fhe_claim_is_rejected(self):
        for field in ("reference_modified","approximation_modified","encrypted_execution","model_imported_in_host"):
            value=copy.deepcopy(self.report);value[field]=True;self.rejected(value)
    def test_implementation_selector_and_runtime_mismatch_are_rejected(self):
        for field,value in (("selection_sha256","0"*64),("manifest_sha256","0"*64),("implementation",{}),("runtime",{})):
            record=copy.deepcopy(self.report);record[field]=value;self.rejected(record)
