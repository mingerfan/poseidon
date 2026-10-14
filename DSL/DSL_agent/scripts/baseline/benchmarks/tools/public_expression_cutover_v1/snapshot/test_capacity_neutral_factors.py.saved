"""Exact public identities, scope safety, tail masks and reference differences."""
import ast,copy,json,unittest
import numpy as np
from benchmark_graph import samples,CAPACITY_FORMAT
from benchmark_suite import Builder
from benchmark_math import evaluate as mathematical
from benchmark_torch import evaluate as tensor_reference
from compiler_configuration import configuration
from unified_graph_contract import prepare,validate_candidate
from unified_graph_lowering import candidate_source
from unified_capacity_lowering import lower
from unified_capacity_constants import INDEX_VERSION
from capacity_algebraic_schedule import remove_neutral_factors
from test_unified_capacity_linear import interpreted_output

def request_for(model,period=4):
    cfg=configuration("poseidon-cpu-eva-w45-v1")
    return prepare(model,cfg["ckks_config_sha256"],cfg,chunk_period=period,
        generation_guidance="explicit-v10",capacity_packing_version=INDEX_VERSION)

def graph(width=4):
    b=Builder([(width,)])
    x=b.node("polynomial",["input0",b.const([1.,1.,1.])],basis="power")
    x=b.node("multiply",[x,b.const([1.])])
    x=b.node("slice",[x],axis=0,start=0,stop=width,step=1)
    g=b.finish(x)
    g.update(format=CAPACITY_FORMAT,resource_profile="general-tensors-v1")
    return g

def simple_request(constants):
    return dict(layout=dict(input_slot_period=4),public_constants=constants)

def source(expression,helper=False):
    if helper:
        return ('@hc.func("c")\ndef f(a0):\n    v0 = '+expression.replace("x0","a0")+
            '\n    return v0\n@hc.func("c,c")\ndef golden(x0, zero_ct):\n'
            '    v0 = f(x0)\n    return [v0]\n')
    return '@hc.func("c,c")\ndef golden(x0, zero_ct):\n    v0 = '+expression+'\n    return [v0]\n'


def straight_line(text,request,x):
    # Independent finite ring evaluation, without exec/eval or backend helpers.
    env={name:np.asarray(value,dtype=np.float64) for name,value in request["public_constants"].items()}
    env.update(x0=x,zero_ct=np.zeros(4))
    def expression(node):
        if isinstance(node,ast.Name):return env[node.id]
        if isinstance(node,ast.BinOp):
            a,b=expression(node.left),expression(node.right)
            return {ast.Mult:np.multiply,ast.Add:np.add,ast.Sub:np.subtract}[type(node.op)](a,b)
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=="rotate":
            return np.roll(expression(node.func.value),-node.args[0].value)
        raise AssertionError("Unexpected trusted test expression")
    fn=ast.parse(text).body[0]
    for statement in fn.body[:-1]:
        assert isinstance(statement,ast.Assign) and len(statement.targets)==1
        env[statement.targets[0].id]=expression(statement.value)
    return np.asarray([expression(v) for v in fn.body[-1].value.elts])

class CapacityNeutralTests(unittest.TestCase):
    def test_scalar_identity_and_complete_period_vector(self):
        r=simple_request(dict(one=1.,full=[1.,1.,1.,1.]))
        original=source("(x0 * one) * full")
        before=copy.deepcopy(r)
        result,record=remove_neutral_factors(original,r)
        self.assertEqual(r,before)
        self.assertTrue(record["applied"])
        self.assertEqual(record["removed_multiplications"],2)
        self.assertEqual(record["scalar_multiplications"],1)
        self.assertEqual(record["vector_multiplications"],1)
        self.assertEqual(record["bound_names"],["full","one"])
        self.assertEqual(sum(isinstance(n,ast.Mult) for n in ast.walk(ast.parse(result))),0)
        self.assertEqual((result,record),remove_neutral_factors(original,r))
        for key in ("request_changed","reference_changed","provider_candidate_rewritten"):
            self.assertFalse(record[key])

    def test_near_one_bool_nonidentity_and_wrong_length_are_retained(self):
        for value in (1.+1e-12,1.-1e-12,True,False,0.,-1.,[],
            [1.,1.,1.],[1.,1.,1.,1.,1.],[1.,1.,1.,0.],
            [1.,1.,1.,1.+1e-12],[True]*4):
            with self.subTest(value=value):
                original=source("x0 * p0")
                result,record=remove_neutral_factors(original,simple_request(dict(p0=value)))
                self.assertEqual(result,original)
                self.assertFalse(record["applied"])

    def test_no_plaintext_product_is_made_legal(self):
        original=source("weight * one")
        result,record=remove_neutral_factors(original,simple_request(dict(weight=.5,one=1.)))
        self.assertEqual(result,original)
        self.assertFalse(record["applied"])

    def test_helper_is_preserved_without_expansion(self):
        original=source("(x0 * one) * one",helper=True)
        result,record=remove_neutral_factors(original,simple_request(dict(one=1.)))
        self.assertEqual(record["removed_multiplications"],2)
        tree=ast.parse(result)
        self.assertEqual([f.name for f in tree.body],["f","golden"])
        self.assertEqual(sum(isinstance(n,ast.Call) and isinstance(n.func,ast.Name)
            and n.func.id=="f" for n in ast.walk(tree)),1)

    def test_bound_identity_cannot_be_shadowed(self):
        r=simple_request(dict(one=1.))
        originals=[
            source("x0 * one").replace("golden(x0, zero_ct)","golden(x0, one)"),
            source("x0 * one").replace("    v0 =","    one = x0\n    v0 ="),
            source("x0 * one",helper=True).replace("f(a0)","f(one)") ]
        for original in originals:
            with self.assertRaisesRegex(ValueError,"binding shadowed"):
                remove_neutral_factors(original,r)

    def test_array_identity_requires_full_period_not_logical_width(self):
        r=simple_request(dict(mask=[1.,1.,1.,0.],one=1.))
        original=source("(x0 * mask) * one")
        result,record=remove_neutral_factors(original,r)
        self.assertEqual(record["removed_multiplications"],1)
        self.assertIn("x0 * mask",result)
        self.assertEqual(record["bound_names"],["one"])

    def test_closed_scope_and_source_budget(self):
        r=simple_request(dict(one=1.))
        with self.assertRaisesRegex(ValueError,"Trusted scalar functions"):
            remove_neutral_factors("import os\n"+source("x0 * one"),r)
        with self.assertRaisesRegex(ValueError,"Rule source budget"):
            remove_neutral_factors(source("x0 * one")+"#"+("a"*65536),r)

    def test_real_request_static_contract_and_dual_reference(self):
        for width,period in ((4,4),(3,4),(6,4),(8,8)):
            model=graph(width);r=request_for(model,period);before=copy.deepcopy(r)
            original=lower(r,semantic_pairs=False)[0]
            result,record=candidate_source(r)
            self.assertEqual(r,before)
            self.assertIn("neutral_factors",record)
            validate_candidate(dict(schema=1,request_id=r["request_id"],hecate_source=result),r)
            for inputs in samples(model,16):
                actual=interpreted_output(result,r,inputs)
                old=interpreted_output(original,r,inputs)
                np.testing.assert_allclose(actual,old,atol=2e-12,rtol=2e-12)
                for evaluate in (mathematical,tensor_reference):
                    refs=evaluate(model,inputs)
                    expected=np.concatenate([refs[o["name"]].reshape(-1) for o in model["outputs"]])
                    np.testing.assert_allclose(actual,expected,atol=2e-12,rtol=2e-12)

    def test_direct_legacy_lowering_preserves_exact_source(self):
        r=request_for(graph())
        source0,record=lower(r)
        self.assertEqual(source0,lower(r,semantic_pairs=False)[0])
        self.assertNotIn("neutral_factors",record)
        transformed,_=candidate_source(r)
        self.assertEqual(source0,lower(r,semantic_pairs=False)[0])
        self.assertNotEqual(len(source0),0)
        self.assertNotEqual(len(transformed),0)

    def test_capacity_scheduler_removes_one_introduced_by_public_reuse(self):
        from capacity_algebraic_schedule import schedule
        r=simple_request(dict(one=1.))
        r["layout"].update(inputs=[dict(dsl_name="x0")],auxiliary_ciphertexts=[
            dict(dsl_name="zero_ct",kind="encrypted_zero",source="trusted_client",slot_period=4)])
        original=source("x0 * (zero_ct + one)")
        result,record=schedule(original,r)
        self.assertTrue(record["applied"])
        self.assertEqual(record["reused_encrypted_public_values"],1)
        self.assertEqual(record["removed_identity_factors"],1)
        self.assertEqual(sum(isinstance(n,ast.Mult) for n in ast.walk(ast.parse(result))),0)

    def test_binary_masks_in_three_composition_contexts(self):
        from capacity_algebraic_schedule import schedule
        for values in ([1.,0.,1.,0.],[1.,1.,0.,0.],[0.,0.,0.,1.]):
            r=simple_request(dict(pv0=values,bias=.125))
            r["layout"].update(inputs=[dict(dsl_name="x0")],auxiliary_ciphertexts=[
                dict(dsl_name="zero_ct",kind="encrypted_zero",source="trusted_client",slot_period=4)])
            for expression in ("(x0 * pv0) * pv0","((x0 + bias) * pv0) * pv0",
                               "((x0 * pv0) + (x0 * pv0)) * pv0"):
                original=source(expression)
                result,record=schedule(original,r)
                self.assertTrue(record["applied"])
                self.assertGreater(record["removed_duplicate_masks"]+record.get("removed_supported_bound_masks",0),0)
                self.assertLessEqual(record["scheduled_depth_bound"],record["original_depth_bound"])
                for i in range(16):
                    x=np.random.default_rng(175+i).uniform(-2.,2.,4)
                    np.testing.assert_allclose(straight_line(result,r,x),
                        straight_line(original,r,x),atol=2e-12,rtol=2e-12)

    def test_nonbinary_different_and_rotated_masks_are_not_idempotent(self):
        from capacity_algebraic_schedule import schedule
        for expression,values in (
                ("(x0 * pv0) * pv0",[1.,.5,0.,1.]),
                ("(x0 * pv0) * pv1",[1.,0.,1.,0.]),
                ("(x0 * pv0).rotate(1) * pv0",[1.,0.,1.,0.])):
            r=simple_request(dict(pv0=values,pv1=[0.,1.,0.,1.]))
            r["layout"].update(inputs=[dict(dsl_name="x0")],auxiliary_ciphertexts=[
                dict(dsl_name="zero_ct",kind="encrypted_zero",source="trusted_client",slot_period=4)])
            original=source(expression)
            result,record=schedule(original,r)
            self.assertEqual(record["removed_duplicate_masks"],0)
            for i in range(16):
                x=np.random.default_rng(247+i).uniform(-2.,2.,4)
                np.testing.assert_allclose(straight_line(result,r,x),
                    straight_line(original,r,x),atol=2e-12,rtol=2e-12)

    def test_original_scalar_schedule_keeps_previous_mask_and_identity_rules(self):
        from unified_graph_scheduling import schedule_source
        for value in (1.,[1.,1.,0.,0.]):
            r=simple_request(dict(pv0=value))
            r["layout"].update(inputs=[dict(dsl_name="x0")],auxiliary_ciphertexts=[
                dict(dsl_name="zero_ct",kind="encrypted_zero",source="trusted_client",slot_period=4)])
            original=source("(x0 * pv0) * pv0")
            result,record=schedule_source(original,r)
            self.assertEqual(result,original)
            self.assertFalse(record["applied"])


if __name__=="__main__":unittest.main()
