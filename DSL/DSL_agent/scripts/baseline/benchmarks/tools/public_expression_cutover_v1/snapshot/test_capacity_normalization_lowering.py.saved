"""Bounded capacity representation algebra checked against independent references."""
import copy,json,math,unittest
from pathlib import Path
import numpy as np
from benchmark_graph import samples,canonical,digest,CAPACITY_FORMAT
from benchmark_suite import Builder
from benchmark_math import evaluate as math_reference
from benchmark_torch import evaluate as torch_reference
from operator_reference import evaluate as operator_reference
from compiler_configuration import configuration
from unified_graph_contract import prepare,validate_candidate
from unified_graph_lowering import candidate_source
from unified_capacity_lowering import lower
from unified_capacity_constants import CHANNEL_VERSION,VERSION
from capacity_algebraic_schedule import scalar_helpers
from test_unified_capacity_linear import interpreted_output

ROOT=Path(__file__).parent
CASES=ROOT/"cases/capacity-normalization-regression-v1"

def request(model):
    cfg=configuration("poseidon-cpu-eva-w45-v1")
    return prepare(model,cfg["ckks_config_sha256"],cfg,chunk_period=4,generation_guidance="explicit-v10",capacity_packing_version=CHANNEL_VERSION)

def capacity(model):
    model.update(format=CAPACITY_FORMAT,resource_profile="general-tensors-v1")
    return model

class CapacityNormalizationTests(unittest.TestCase):
    def check(self,model,expected_pairs=None):
        r=request(model);before=copy.deepcopy(r);source,record=candidate_source(r)
        self.assertEqual(r,before)
        validate_candidate(dict(schema=1,request_id=r["request_id"],hecate_source=source),r)
        if expected_pairs is not None:self.assertEqual(record["centered_pairs"],expected_pairs)
        original=lower(r,semantic_pairs=False)[0]
        for inputs in samples(model,16):
            out=interpreted_output(source,r,inputs);old=interpreted_output(original,r,inputs)
            np.testing.assert_allclose(out,old,atol=2e-12,rtol=2e-12)
            for evaluate in (math_reference,torch_reference):
                refs=evaluate(model,inputs)
                ref=np.concatenate([refs[o["name"]].reshape(-1) for o in model["outputs"]])
                np.testing.assert_allclose(out,ref,atol=2e-12,rtol=2e-12)
        return source,record

    def test_original_failed_model_is_unchanged_and_plaintext_equivalent(self):
        model=json.loads((CASES/"original-norms.json").read_text())
        self.assertEqual(digest(model),"9514be3fc3a0a854896c171798feb8f2469ba3788c4e0a436f26148ebfbd674d")
        _,record=self.check(model,2)
        self.assertTrue(record["scheduling"]["applied"])
        self.assertFalse(record["scheduling"]["provider_candidate_rewritten"])
        self.assertFalse(record["scheduling"]["depth_is_capacity_proof"])

    def test_norm_contexts_with_affine_weights_and_different_shapes(self):
        manifest=json.loads((CASES/"manifest.json").read_text())
        for entry in manifest["cases"]:
            if "source" not in entry:continue
            high=json.loads((CASES/entry["source"]).read_text());model=json.loads((CASES/entry["file"]).read_text())
            with self.subTest(case=entry["id"]):
                self.check(model,0 if entry["id"]=="rms_pair" else 1)
                for inputs in samples(model,16):
                    a=math_reference(model,inputs);b=operator_reference(high,inputs)
                    for name in a:np.testing.assert_allclose(a[name],b[name],atol=2e-12,rtol=2e-12)

    def test_centering_axes_with_singleton_dimensions(self):
        for shape in ((2,),(1,2),(2,1),(1,2,1)):
            b=Builder([shape]);x="input0"
            mean=b.node("mean",[x],axes=list(range(len(shape))),keepdims=True)
            centered=b.node("subtract",[x,mean])
            squared=b.node("square",[centered])
            variance=b.node("mean",[squared],axes=list(range(len(shape))),keepdims=False)
            self.check(capacity(b.finish(centered,variance)),1)

    def test_dirty_tail_and_named_multiple_outputs(self):
        self.check(json.loads((CASES/"split_center_variance.json").read_text()),1)

    def test_two_inputs_shared_centering_and_zero_result(self):
        b=Builder([(2,),(2,)]);results=[]
        for x in ("input0","input1"):
            mean=b.node("mean",[x],axes=[0],keepdims=True)
            results.append(b.node("subtract",[x,mean]))
        y=b.node("add",results)
        squared=b.node("square",[y])
        mean=b.node("mean",[squared],axes=[0],keepdims=True)
        zero=b.node("subtract",[y,y])
        self.check(capacity(b.finish(y,mean,zero)),2)

    def test_unproved_or_different_mean_is_not_centered_pair(self):
        for width,wrong in ((3,False),(2,True)):
            b=Builder([(width,),(width,)])
            mean=b.node("mean",["input1" if wrong else "input0"],axes=[0],keepdims=True)
            y=b.node("subtract",["input0",mean])
            self.check(capacity(b.finish(y)),0)

    def test_sum_and_fourth_power_of_centered_values(self):
        b=Builder([(2,)]);mean=b.node("mean",["input0"],axes=[0],keepdims=True)
        y=b.node("subtract",["input0",mean]);fourth=b.node("power",[y],exponent=4)
        total=b.node("sum",[fourth],axes=[0],keepdims=False)
        self.check(capacity(b.finish(fourth,total)),1)

    def test_default_direct_lowering_preserves_disabled_representation(self):
        model=json.loads((CASES/"original-norms.json").read_text());r=request(model)
        source,record=lower(r)
        self.assertEqual(source,lower(r,semantic_pairs=False)[0])
        self.assertFalse(record["semantic_pairs"])
        self.assertEqual(record["centered_pairs"],0)

    def test_model_name_and_node_names_do_not_dispatch_lowering(self):
        model=json.loads((CASES/"original-norms.json").read_text())
        first,_=candidate_source(request(model))
        changed=copy.deepcopy(model);changed["id"]="unseen_normalization_identity"
        for i,n in enumerate(changed["nodes"]):n["id"]="unseen_node_"+str(i)
        second,_=candidate_source(request(changed))
        self.assertEqual(first,second)

    def test_prior_packing_keeps_ordinary_source(self):
        model=json.loads((CASES/"original-norms.json").read_text())
        cfg=configuration("poseidon-cpu-eva-w45-v1")
        r=prepare(model,cfg["ckks_config_sha256"],cfg,chunk_period=4,
                  generation_guidance="explicit-v10",capacity_packing_version=VERSION)
        source,record=candidate_source(r)
        self.assertEqual(source,lower(r,semantic_pairs=False)[0])
        self.assertFalse(record["semantic_pairs"])

    def test_scalar_helper_recursion_and_expansion_are_bounded(self):
        source='@hc.func("c")\ndef f(a):\n    v0 = f(a)\n    return v0\n@hc.func("c,c")\ndef golden(x0, zero_ct):\n    v0 = f(x0)\n    return [v0]\n'
        with self.assertRaisesRegex(ValueError,"recursive"):
            scalar_helpers(source)
        source='@hc.func("c,c")\ndef golden(x0, zero_ct):\n'
        source+=''.join("    v%d = x0 + zero_ct\n"%i for i in range(1025))
        source+='    return [v1024]\n'
        with self.assertRaisesRegex(ValueError,"expansion budget"):
            scalar_helpers(source)

if __name__=="__main__":unittest.main()
