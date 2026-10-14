"""Typed output/container interventions, with unchanged acceptance boundaries."""
import ast, unittest
from benchmark_suite import Builder
from compiler_configuration import PROFILE_SHA256
from unified_graph_contract import prepare, validate_candidate
from unified_public_contract import normalize
from unified_public_coverage import fingerprint

class TypedWitnessTests(unittest.TestCase):
    def request(self, feature=None, multiple=False):
        b=Builder([(2,)]);s=b.node("square",["input0"])
        return prepare(b.finish(s,"input0") if multiple else b.finish(s),PROFILE_SHA256,
            construction_profile="hecate-unified-public-v1",
            construction="unified-public-"+feature if feature else None)
    def source(self,body):
        return '@hc.func("c,c")\ndef golden(x, zero_ct):\n'+''.join('    '+line+'\n' for line in body.splitlines())
    def check(self,r,body):
        s=self.source(body);return validate_candidate(dict(schema=1,request_id=r["request_id"],hecate_source=s),r)
    def test_single_expr_same_slots_as_singleton_list(self):
        r=self.request("call-list")
        body='v=list([1.0])[0]\ny=x*x*v\nreturn '
        s1=self.source(body+'y');s2=self.source(body+'[y]')
        self.assertEqual(fingerprint(normalize(s1,r),r),fingerprint(normalize(s2,r),r))
        self.check(r,body+'y');self.check(r,body+'[y]')
    def test_single_expr_does_not_expand_into_multiple_outputs(self):
        r=self.request(multiple=True)
        with self.assertRaisesRegex(ValueError,"Return"):
            self.check(r,"return x")
        self.check(r,"return [x*x,x]")
        self.check(r,"return (x*x,x)")
    def test_public_scalar_nested_and_wrong_count_still_rejected(self):
        r=self.request()
        for body in ["return 1.0","return [[x*x]]","return [x*x,x]"]:
            with self.subTest(body=body),self.assertRaises(ValueError):self.check(r,body)
    def test_enumerate_cipher_results_have_typed_influence(self):
        r=self.request("call-enumerate")
        self.check(r,"out=[]\nfor i,v in enumerate([x*x]):\n    out.append(v)\nreturn out")
    def test_unused_enumerate_cannot_borrow_argument_side_effect(self):
        r=self.request("call-enumerate")
        body="a=[zero_ct]\ndef produce():\n    a[0]=x*x\n    return [x]\nunused=enumerate(produce())\nreturn a"
        with self.assertRaisesRegex(ValueError,"Missing contributing"):self.check(r,body)
    def test_dict_items_cipher_results_have_typed_influence(self):
        r=self.request("call-items")
        self.check(r,"a={'x':x*x}\nout=[]\nfor key,v in a.items():\n    out.append(v)\nreturn out")
        with self.assertRaisesRegex(ValueError,"Missing contributing"):
            self.check(r,"a={'x':x*x}\nunused=a.items()\nreturn [x*x]")
    def test_sorted_index_order_has_influence(self):
        r=self.request("counter-sorted_calls")
        self.check(r,"order=sorted([1,0])\na=[x*x,zero_ct]\nreturn [a[order[0]]]")
    def test_sorted_commutative_sum_is_not_order_influence(self):
        r=self.request("counter-sorted_calls")
        with self.assertRaisesRegex(ValueError,"Missing contributing"):
            self.check(r,"a=[('a',x*x),('b',zero_ct)]\nb=sorted(a,key=lambda v:v[0])\nreturn [b[0][1]+b[1][1]]")
    def test_typed_probe_evaluates_target_once(self):
        from unified_public_interventions import replacements
        for feature,expression in [("call.enumerate","enumerate(produce())"),("call.items","produce().items()"),("counter.sorted_calls","sorted(produce())")]:
            target=ast.parse(expression,mode="eval").body
            changed=replacements(target,feature,ast.parse("unused=1"))[0]
            self.assertEqual(sum(type(n) is ast.Call and type(n.func) is ast.Name and n.func.id=="produce" for n in ast.walk(changed)),1)
if __name__=="__main__":unittest.main()
