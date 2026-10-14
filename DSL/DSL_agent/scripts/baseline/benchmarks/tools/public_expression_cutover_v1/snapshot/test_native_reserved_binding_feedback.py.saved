"""Authoritative immutable-binding failures and controlled repair feedback."""
import ast,json,unittest
from native_public_loops import expand
from candidate_check_diagnostic import CandidateCheckError
from component_contract import failure
from application_diagnostics import attach,checked,make,repeat_key

def program(body):
    return '@hc.func("c,c")\ndef golden(x,zero_ct):\n'+''.join('    '+line+'\n'for line in body.splitlines())
class ReservedBindingFeedbackTests(unittest.TestCase):
    def rejected(self,body,constants=None,**options):
        with self.assertRaises(CandidateCheckError)as caught:
            expand(ast.parse(program(body)),{}if constants is None else constants,**options)
        error=caught.exception
        self.assertEqual(error.reason,"native_reserved_binding")
        self.assertIsInstance(error,ValueError)
        self.assertEqual(str(error),"Native loops cannot overwrite reserved bindings")
        return error
    def test_public_constant_assignment_has_exact_source_location(self):
        error=self.rejected("for i in range(1):\n    local=x\np0=x\nreturn x",{"p0":[1,0,0,0]})
        self.assertEqual(error.location,dict(kind="source",line=5,column=4))
    def test_zero_trip_loop_does_not_erase_binding_rejection(self):
        error=self.rejected("for i in range(0):\n    p0=x\nreturn x",{"p0":[1]})
        self.assertEqual(error.location["line"],4)
    def test_runtime_and_zero_bindings_remain_immutable(self):
        for name in("zero_ct","hc","np","object","range"):
            with self.subTest(name=name):
                self.rejected("for i in range(1):\n    "+name+"=x\nreturn x")
    def test_declared_function_binding_is_protected(self):
        helper='@hc.func("c")\ndef helper(x):\n    return x\n'
        source=helper+program("for i in range(1):\n    helper=x\nreturn x")
        with self.assertRaises(CandidateCheckError)as caught:expand(ast.parse(source),{})
        self.assertEqual(caught.exception.reason,"native_reserved_binding")
    def test_tuple_and_augmented_targets_still_reject(self):
        for target in("p0,local=x,x","p0+=x"):
            with self.subTest(target=target):
                self.rejected("for i in range(1):\n    "+target+"\nreturn x",{"p0":[1]},
                    scalar_augmented=True)
    def test_fresh_local_alias_and_accumulation_are_preserved(self):
        tree,info=expand(ast.parse(program("local=x\nfor i in range(2):\n    local=local+x\nreturn local")),{"p0":[1]})
        self.assertEqual(info["iterations"],2)
        self.assertFalse(any(isinstance(n,ast.For)for n in ast.walk(tree)))
        self.assertEqual(tree.body[0].body[-1].value.id,"local")
    def test_component_feedback_uses_witness_and_filters_private_text(self):
        error=self.rejected("for i in range(1):\n    p0=x\nreturn x",{"p0":[1]})
        raw={"status":"failed","failure":failure("static_check",error),
             "public_feedback":{"status":"failed","layer":"static_check","category":"candidate",
                "diagnostic":"PRIVATE_REFERENCE_987 and credentials"},
             "diagnostic":"PRIVATE_REFERENCE_987"}
        result=attach(raw,candidate={"hecate_source":program("return x")})
        typed=result["structured_diagnostic"];self.assertEqual(typed["reason_code"],"native_reserved_binding")
        self.assertEqual(typed["location"],error.location)
        self.assertEqual(typed["retry_policy"],"repair_within_budget")
        public=result["public_feedback"]
        self.assertIn("fresh local name",public["diagnostic"])
        self.assertNotIn("PRIVATE",json.dumps(public));self.assertNotIn("credentials",json.dumps(public))
    def test_missing_location_extra_context_and_forged_repeat_are_rejected(self):
        loc=dict(kind="source",line=4,column=8)
        with self.assertRaises(ValueError):make("native_reserved_binding")
        with self.assertRaises(ValueError):make("native_reserved_binding",location=loc,context={"observed":1})
        typed=make("native_reserved_binding",location=loc)
        self.assertIsNone(repeat_key(typed))
        self.assertEqual(checked(typed),typed)
        with self.assertRaises(ValueError):checked(dict(typed,repeat_key_available=True))
if __name__=="__main__":unittest.main(verbosity=2)
