import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
from benchmark_suite import Builder
from benchmark_graph import samples
from benchmark_math import evaluate as mathematical
from benchmark_torch import evaluate as torch_reference
from compiler_configuration import configuration
from unified_graph_contract import prepare, validate_candidate
from unified_graph_lowering import lower, candidate_source
from unified_graph_scheduling import Schedule
from unified_graph_quadratic import Quadratic, Budget, schedule_quadratic_source
from test_packed_prefix_lowering import public_execute

def request_for(model):
    config=configuration("poseidon-cpu-eva-w45-v1")
    return prepare(model,config["ckks_config_sha256"],config)

class PublicQuadraticTests(unittest.TestCase):
    def assert_reference(self,model):
        request=request_for(model);before=copy.deepcopy(request)
        original=lower(request,balanced_power=True)
        source,record=schedule_quadratic_source(original,request)
        self.assertEqual(request,before)
        self.assertEqual((source,record),schedule_quadratic_source(original,request))
        validate_candidate(dict(schema=1,request_id=request["request_id"],hecate_source=source),request)
        for inputs in samples(model,16):
            values=public_execute(source,request,inputs)
            for reference in (mathematical,torch_reference):
                expected=reference(model,inputs)
                for i,out in enumerate(model["outputs"]):
                    target=expected[out["name"]].reshape(-1)
                    np.testing.assert_allclose(values[i][:len(target)],target,atol=1e-12,rtol=1e-12)
        self.assertFalse(record["provider_candidate_rewritten"])
        return source,record

    def test_dot_and_norm_in_three_shapes(self):
        for shape in ((2,),(2,2),(1,2,3)):
            for kind in ("dot","norm"):
                b=Builder([shape,shape] if kind=="dot" else [shape])
                value=b.node("multiply",["input0","input1"]) if kind=="dot" else b.node("square",["input0"])
                reduced=b.node("sum",[value],axes=list(range(len(shape))),keepdims=True)
                self.assert_reference(b.finish(reduced))

    def test_nonuniform_weights_and_partial_sum_keep_math(self):
        for size in (2,3,6):
            b=Builder([(size,),(size,)])
            product=b.node("multiply",["input0","input1"])
            weighted=b.node("multiply",[product,b.const([.25*(i+1) for i in range(size)])])
            value=b.node("sum",[weighted],axes=[0],keepdims=True)
            self.assert_reference(b.finish(value))

    def test_rotated_self_product_half_period_multiplicity(self):
        for size in (2,4,8):
            b=Builder([(size,)])
            shifted=b.node("rotate",["input0"],step=size//2)
            value=b.node("multiply",["input0",shifted])
            total=b.node("sum",[value],axes=[0],keepdims=True)
            self.assert_reference(b.finish(total))

    def test_shared_full_product_and_sign(self):
        for size in (2,3,4):
            b=Builder([(size,),(size,),(size,)])
            u=b.node("square",["input1"])
            v=b.node("power",["input2"],exponent=4)
            common=b.node("multiply",[u,v])
            left=b.node("multiply",["input0",common])
            right=b.node("multiply",["input1",common])
            value=b.node("subtract",[left,right])
            self.assert_reference(b.finish(value,common))

    def test_public_only_remainder_is_not_emitted(self):
        b=Builder([(4,)])
        request=request_for(b.finish(b.node("square",["input0"])))
        name=request["layout"]["inputs"][0]["dsl_name"]
        source='@hc.func("c,c")\ndef golden('+name+', zero_ct):\n'
        source+='    v0 = '+name+' * mask0\n    v1 = '+name+' * mask1\n    v2 = v0 + v1\n    return [v2]\n'
        result,record=schedule_quadratic_source(source,request)
        self.assertEqual(result,source)
        self.assertFalse(record["applied"])
        validate_candidate(dict(schema=1,request_id=request["request_id"],hecate_source=result),request)

    def test_nonzero_underflow_is_not_discarded(self):
        b=Builder([(2,)])
        request=request_for(b.finish(b.node("square",["input0"])))
        s=Schedule(request);q=Quadratic(s,request)
        a=s.node("name",value="1e-200",plain=True)
        product=s.node("multiply",(a,a),plain=True)
        self.assertIsNone(q.state(product))

    def test_budget_and_outside_scope_preserve_source(self):
        b=Builder([(2,)])
        request=request_for(b.finish(b.node("sum",[b.node("square",["input0"])],axes=[0],keepdims=True)))
        original=lower(request)
        with patch.object(Quadratic,"spend",side_effect=Budget("test budget")):
            source,record=schedule_quadratic_source(original,request)
        self.assertEqual(source,original);self.assertFalse(record["applied"])
        altered=copy.deepcopy(request);altered["layout"]["input_slot_period"]=32
        source,record=schedule_quadratic_source(original,altered)
        self.assertEqual(source,original);self.assertFalse(record["applied"])

    def test_original_whole_requests_and_model_identity_are_preserved(self):
        root=(Path(__file__).resolve().parents[4] if Path(__file__).resolve().parents[2].name == "DSL_agent" and Path(__file__).resolve().parents[3].name == "DSL" else Path(__file__).resolve().parents[2])/"scripts/baseline/cases/poseidon-whole-models-v1"
        for name in ("mlp","rmsnorm","attention_decode"):
            model=json.loads((root/(name+".json")).read_text())
            request=request_for(model);before=copy.deepcopy(request)
            source,record=candidate_source(request)
            self.assertEqual(request,before)
            renamed=copy.deepcopy(model);renamed["id"]="different_public_model_name"
            self.assertEqual(source,candidate_source(request_for(renamed))[0])
            for inputs in samples(model,16):
                actual=public_execute(source,request,inputs)
                for ref in (mathematical,torch_reference):
                    expected=ref(model,inputs)
                    for i,out in enumerate(model["outputs"]):
                        target=expected[out["name"]].reshape(-1)
                        np.testing.assert_allclose(actual[i][:len(target)],target,atol=1e-12,rtol=1e-12)

if __name__=="__main__":unittest.main()
