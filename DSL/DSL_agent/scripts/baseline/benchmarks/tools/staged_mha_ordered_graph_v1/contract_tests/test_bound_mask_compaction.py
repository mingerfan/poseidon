"""Standard-library controls for the isolated, rule-only mask helper."""
import ast,copy,hashlib,importlib.util,math,random,unittest
from pathlib import Path
STAGE=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("trusted_bound_mask_compaction",STAGE/"trusted_bound_mask_compaction.py")
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

def request(mask=None):
    return dict(request_id="test-bound-mask",model={"unchanged":True},
      layout=dict(input_slot_period=4,inputs=[dict(dsl_name="x0")]),
      public_constants={"m":[1.,0.,1.,0.] if mask is None else mask,"other":[0.,1.,0.,1.],"b":.3})

def source(expression="(x0*m)*m",suffix=""):
    return "@hc.func('c,c')\ndef golden(x0,zero_ct):\n    v0 = "+expression+"\n"+suffix+"    return [v0]\n"

def apply(text,req=None):
    return module.compact(text,request() if req is None else req,source_kind=module.SOURCE_KIND)

def evaluate(text,inputs,public):
    """Independent lane interpreter; never Python eval/exec candidate code."""
    def lanes(v):return [float(v)]*4 if type(v) in (int,float) else v
    env={**copy.deepcopy(public),**copy.deepcopy(inputs)}
    def expression(node):
        if type(node) is ast.Name:return lanes(env[node.id])
        if type(node) is ast.BinOp:
            a,b=expression(node.left),expression(node.right)
            fn={ast.Add:lambda x,y:x+y,ast.Sub:lambda x,y:x-y,ast.Mult:lambda x,y:x*y}[type(node.op)]
            return [fn(x,y) for x,y in zip(a,b)]
        if type(node) is ast.Call and type(node.func) is ast.Attribute and node.func.attr=="rotate":
            a=expression(node.func.value);step=node.args[0].value
            return [a[(j+step)%len(a)] for j in range(len(a))]
        if type(node) is ast.Constant:return lanes(node.value)
        raise ValueError("Independent lane grammar")
    fn=ast.parse(text).body[0]
    for statement in fn.body[:-1]:env[statement.targets[0].id]=expression(statement.value)
    return [expression(v) for v in fn.body[-1].value.elts]

class BoundMaskCompaction(unittest.TestCase):
    def test_complete_bound_binary_mask(self):
        original=source();changed,record=apply(original)
        self.assertTrue(record["applied"]);self.assertEqual(record["removed"],1)
        self.assertEqual(record["bound_names"],["m"])
        self.assertEqual(sum(type(n) is ast.Mult for n in ast.walk(ast.parse(changed))),1)
    def test_independent_mathematical_lanes(self):
        original=source("((x0+b)*m)*m");changed,_=apply(original);rng=random.Random(861)
        for _ in range(48):
            inputs={"x0":[rng.uniform(-2.,2.) for _ in range(4)],"zero_ct":[0.]*4}
            self.assertEqual(evaluate(original,inputs,request()["public_constants"]),
                             evaluate(changed,inputs,request()["public_constants"]))
    def test_half_mask_counterexample_is_not_rewritten(self):
        req=request([.5]*4);original=source();changed,record=apply(original,req)
        self.assertFalse(record["applied"]);self.assertEqual(original,changed)
        inputs={"x0":[1.,2.,3.,4.],"zero_ct":[0.]*4}
        self.assertNotEqual(evaluate(original,inputs,req["public_constants"]),
                            evaluate(source("x0*m"),inputs,req["public_constants"]))
    def test_short_padded_mask_is_not_assumed_full_period(self):
        original=source();changed,record=apply(original,request([1.,0.]))
        self.assertEqual(changed,original);self.assertEqual(record["removed"],0)
    def test_distinct_masks_and_bound_names_are_not_merged(self):
        req=request();req["public_constants"]["other"]=req["public_constants"]["m"].copy()
        original=source("(x0*m)*other");changed,record=apply(original,req)
        self.assertFalse(record["applied"]);self.assertEqual(changed,original)
    def test_rotation_cannot_be_crossed(self):
        original=source("(x0*m).rotate(1)*m");changed,record=apply(original)
        self.assertEqual(changed,original);self.assertEqual(record["removed"],0)
    def test_multisuse_preserves_both_consumers(self):
        original="@hc.func('c,c')\ndef golden(x0,zero_ct):\n    v0=(x0*m)*m\n    v1=v0+x0\n    return [v0,v1]\n"
        changed,record=apply(original);self.assertEqual(record["removed"],1)
        inputs={"x0":[1.,2.,3.,4.],"zero_ct":[0.]*4}
        self.assertEqual(evaluate(original,inputs,request()["public_constants"]),
                         evaluate(changed,inputs,request()["public_constants"]))
    def test_separate_ssa_mask_products_are_not_rewritten(self):
        original="@hc.func('c,c')\ndef golden(x0,zero_ct):\n    v0=x0*m\n    v1=v0*m\n    return [v0,v1]\n"
        changed,record=apply(original);self.assertFalse(record["applied"]);self.assertEqual(changed,original)
    def test_assignment_shadow_is_rejected_even_before_usage(self):
        text=source().replace("    v0","    m=x0\n    v0")
        with self.assertRaisesRegex(ValueError,"shadow"):apply(text)
    def test_parameter_and_helper_function_shadow_are_rejected(self):
        parameter=source().replace("golden(x0,zero_ct)","golden(x0,m)")
        helper="@hc.func('c')\ndef m(a0):\n    return a0\n"+source()
        for text in (parameter,helper):
            with self.subTest(text=text),self.assertRaisesRegex(ValueError,"shadow"):apply(text)
    def test_global_nonlocal_and_subscript_stores_are_rejected(self):
        for statement in ("global m","nonlocal m","m[0]=x0"):
            text=source().replace("    v0","    "+statement+"\n    v0")
            with self.subTest(statement=statement),self.assertRaises(ValueError):apply(text)
    def test_nested_lambda_cannot_shadow_mask_or_add_dynamic_scope(self):
        for expression in ("(lambda m:(x0*m)*m)(.5)","(lambda p:(x0*m)*m)(.5)"):
            with self.subTest(expression=expression),self.assertRaises(ValueError):
                apply(source(expression))
    def test_provider_authority_rejected_without_changes(self):
        for origin in ("provider_candidate","agent_generated",None,""):
            with self.subTest(origin=origin),self.assertRaisesRegex(ValueError,"Rule-only"):
                module.compact(source(),request(),source_kind=origin)
    def test_record_hashes_bind_complete_request_and_source(self):
        req=request();before=copy.deepcopy(req);text=source();changed,record=apply(text,req)
        self.assertEqual(req,before)
        self.assertEqual(record["static_binding"]["request_sha256"],module.digest(req))
        self.assertEqual(record["static_binding"]["input_source_sha256"],hashlib.sha256(text.encode()).hexdigest())
        self.assertEqual(record["output_source_sha256"],hashlib.sha256(changed.encode()).hexdigest())
        self.assertEqual(record["static_binding_sha256"],module.digest(record["static_binding"]))
        self.assertEqual(record["binding"],module.digest({k:v for k,v in record.items() if k!="binding"}))
        self.assertFalse(record["request_changed"]);self.assertFalse(record["provider_candidate_rewritten"])
    def test_no_constant_rotation_or_zero_construction_added(self):
        text=source("(x0*m)*m");changed,record=apply(text)
        def selected(source):
            tree=ast.parse(source)
            return (sum(type(n) is ast.Call for n in ast.walk(tree)),
                    [n.value for n in ast.walk(tree) if type(n) is ast.Constant],
                    sum(type(n) is ast.Name and n.id=="zero_ct" for n in ast.walk(tree)))
        self.assertEqual(selected(text),selected(changed))
        self.assertEqual([record[n] for n in ("new_constants","new_rotations","new_zero_ciphertexts")],[0,0,0])
    def test_nonfinite_bool_and_nonnumeric_masks_are_not_identities(self):
        for mask in ([True,False,True,False],[float('nan')]*4,["1"]*4):
            with self.subTest(mask=mask):
                if type(mask[0]) is float and math.isnan(mask[0]):
                    with self.assertRaises(ValueError):apply(source(),request(mask))
                else:
                    original=source();changed,record=apply(original,request(mask))
                    self.assertEqual(changed,original);self.assertFalse(record["applied"])
    def test_same_mask_chain_compacts_all_duplicates(self):
        changed,record=apply(source("((x0*m)*m)*m"))
        self.assertEqual(record["removed"],2)
        self.assertEqual(sum(type(n) is ast.Mult for n in ast.walk(ast.parse(changed))),1)
    def test_changed_model_layout_or_public_value_changes_binding(self):
        _,first=apply(source())
        req=request();req["model"]["unchanged"]=False
        _,second=apply(source(),req)
        self.assertNotEqual(first["static_binding_sha256"],second["static_binding_sha256"])

if __name__=="__main__":unittest.main()
