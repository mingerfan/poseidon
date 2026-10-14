"""Host-verified public context controls; no provider or FHE execution."""
import copy,json,unittest
from benchmark_graph import digest
from model_decomposition import FORMAT
from operator_arithmetic_graph import decompose
from generation_model_origin import build,verify
from compiler_configuration import configuration
from unified_graph_contract import prepare

def fixture():
    high=dict(format=FORMAT,id="source",inputs=[dict(name="x",shape=[2])],constants={},
        nodes=[dict(id="activation",op="sigmoid",inputs=["x"],
                    attrs=dict(approximation=dict(method="polynomial",basis="power",domain=[-1.,1.],coefficients=[.5,.25])),
                    outputs=["y"])],outputs=[dict(name="out",value="y")])
    model,binding=decompose(high)
    provenance=dict(original=high,decomposition=binding,reference_report=dict(
        original_sha256=digest(high),model_sha256=digest(model),reference_status="passed",
        approximation_accuracy_certified=False))
    cfg=configuration("poseidon-cpu-eva-w45-v1")
    request=prepare(model,cfg["ckks_config_sha256"],cfg,generation_guidance="explicit-v20")
    return provenance,request

class ModelOriginTests(unittest.TestCase):
    def test_verified_math_and_source_map_are_public(self):
        p,r=fixture();c=build(p,r)
        self.assertEqual(c["nodes"][0]["op"],"sigmoid")
        self.assertEqual(c["target_model_sha256"],digest(r["model"]))
        self.assertEqual(c["original_model_sha256"],digest(p["original"]))
        self.assertEqual(c["source_map"],p["decomposition"]["source_map"])
        self.assertFalse(c["rule_dsl_included"]);self.assertFalse(c["source_code_included"])
        self.assertFalse(c["reference_answers_included"])
        self.assertEqual(verify(c,p,r),c)
    def test_request_and_provenance_stay_unchanged(self):
        p,r=fixture();before=copy.deepcopy((p,r));build(p,r)
        self.assertEqual(before,(p,r))
    def test_private_reference_fields_are_not_emitted(self):
        p,r=fixture();p["reference_report"].update(test_inputs="PRIVATE INPUT",actual="PRIVATE ANSWER")
        c=build(p,r);text=json.dumps(c)
        self.assertNotIn("PRIVATE",text);self.assertNotIn("reference_report",text)
    def test_source_model_name_does_not_dispatch_mathematical_nodes(self):
        p,r=fixture();a=build(p,r)
        p["original"]["id"]="another_source"
        model,binding=decompose(p["original"]);p["decomposition"]=binding
        p["reference_report"]["original_sha256"]=digest(p["original"])
        b=build(p,r)
        self.assertEqual(a["nodes"],b["nodes"]);self.assertEqual(a["source_map"],b["source_map"])
    def test_changed_model_or_profile_is_rejected(self):
        for field in("model","profile"):
            p,r=fixture()
            if field=="model":r["model"]["nodes"][0]["op"]="negate"
            else:p["original"]["nodes"][0]["attrs"]["approximation"]["coefficients"]=[.5,.5]
            with self.assertRaises(ValueError):build(p,r)
    def test_changed_source_map_is_rejected(self):
        p,r=fixture();key=next(iter(p["decomposition"]["source_map"]))
        p["decomposition"]["source_map"][key]="unknown"
        with self.assertRaises(ValueError):build(p,r)
    def test_context_tampering_requires_host_reverification(self):
        p,r=fixture();c=build(p,r);c["nodes"][0]["op"]="tanh"
        with self.assertRaises(ValueError):verify(c,p,r)
    def test_reference_status_not_promoted_to_application_accuracy(self):
        p,r=fixture();c=build(p,r)
        self.assertFalse(c["original_function_accuracy_certified"])
        p["reference_report"]["reference_status"]="skipped"
        with self.assertRaises(ValueError):build(p,r)
if __name__=="__main__":unittest.main()
