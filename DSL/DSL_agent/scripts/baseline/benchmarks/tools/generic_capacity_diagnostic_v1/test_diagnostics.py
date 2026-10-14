"""Tests of isolated proposed Agent modules; no active-source monkeypatch."""
import ast,hashlib,importlib.util,json,sys,unittest
from pathlib import Path
HERE=Path(__file__).resolve().parent
BASE=HERE.parents[2]
sys.path[:0]=[str(HERE/"proposed"),str(BASE)]
from native_public_loops import expand,_located_nodes
from candidate_check_diagnostic import CandidateCheckError
from application_diagnostics import attach,checked,make,repeat_key
from component_contract import failure
from candidate_contract import validate_candidate

spec=importlib.util.spec_from_file_location("frozen_native_loop_comparison",BASE/"native_public_loops.py")
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)

def program(body):
    return '@hc.func("c,c")\ndef golden(x,zero_ct):\n'+''.join('    '+s+'\n'for s in body.splitlines())

class LocatedSyntaxTests(unittest.TestCase):
    def rejected(self,body,reason,line,column):
        tree=ast.parse(program(body))
        with self.assertRaises(ValueError)as before:
            old.expand(tree,{})
        self.assertEqual(str(before.exception),"Unsupported native loop syntax")
        with self.assertRaises(CandidateCheckError)as after:
            expand(tree,{})
        error=after.exception
        self.assertEqual(str(error),str(before.exception))
        self.assertEqual(error.reason,reason)
        self.assertEqual(error.location,dict(kind="source",line=line,column=column))
        self.assertEqual(error.context,{})
        return error

    def test_if_has_exact_statement_location(self):
        self.rejected("if True:\n    x=x+x\nreturn x","native_control_flow_not_enabled",3,4)

    def test_conditional_expression_location(self):
        self.rejected("value=x if True else zero_ct\nreturn value",
                      "native_control_flow_not_enabled",3,10)

    def test_while_remains_rejected(self):
        self.rejected("while True:\n    x=x+x\nreturn x","native_control_flow_not_enabled",3,4)

    def test_comparison_remains_rejected(self):
        self.rejected("value=x>zero_ct\nreturn value","native_control_flow_not_enabled",3,10)

    def test_boolean_expr_is_not_public_boolean(self):
        self.rejected("value=x and zero_ct\nreturn value","native_control_flow_not_enabled",3,10)

    def test_comprehension_has_exact_location(self):
        self.rejected("value=[x for i in range(2)]\nreturn x","native_public_syntax",3,10)

    def test_operator_uses_nearest_source_expression(self):
        self.rejected("value=x/2\nreturn value","native_public_syntax",3,10)

    def test_shared_operator_instances_keep_distinct_locations(self):
        tree=ast.parse(program("first=x/2\nsecond=x/3\nreturn first"))
        locations=[(n.lineno,n.col_offset)for node,n in _located_nodes(tree)if type(node)is ast.Div]
        self.assertEqual(locations,[(3,10),(4,11)])

    def test_positive_static_range_unchanged(self):
        tree=ast.parse(program("value=x\nfor i in range(2):\n    value=value+zero_ct\nreturn value"))
        after,meta=expand(tree,{});before,previous=old.expand(tree,{})
        self.assertEqual(ast.dump(after,include_attributes=True),ast.dump(before,include_attributes=True))
        self.assertEqual(meta,previous)

    def test_positive_object_array_unchanged(self):
        tree=ast.parse(program("arr=np.array([x,zero_ct],dtype=object)\nreturn arr[0]"))
        after,meta=expand(tree,{});before,previous=old.expand(tree,{})
        self.assertEqual(ast.dump(after,include_attributes=True),ast.dump(before,include_attributes=True))
        self.assertEqual(meta,previous)

    def test_existing_array_constructor_feedback_preserved(self):
        tree=ast.parse(program("arr=np.array([x],dtype=float)\nreturn arr[0]"))
        with self.assertRaises(CandidateCheckError)as caught:
            expand(tree,{})
        self.assertEqual(caught.exception.reason,"native_array_constructor")
        typed=failure("static_check",caught.exception)
        self.assertEqual(typed["location"],dict(kind="source",line=3,column=8))
        self.assertEqual(typed["reason_code"],"native_array_constructor")

    def test_control_failure_forwarded_and_private_text_dropped(self):
        error=self.rejected("if PRIVATE_INPUT_ARRAY:\n    x=x+x\nreturn x",
                            "native_control_flow_not_enabled",3,4)
        raw=dict(status="failed",failure=failure("static_check",error),
                 diagnostic="PRIVATE_REFERENCE_ARRAY /private/credentials",
                 public_feedback=dict(category="candidate",
                                      diagnostic="PRIVATE_REFERENCE_ARRAY and credentials"))
        result=attach(raw,candidate=dict(hecate_source=program("return x")))
        typed=result["failure"]
        self.assertEqual(checked(typed),typed)
        self.assertEqual(typed["retry_policy"],"repair_within_budget")
        self.assertIsNone(repeat_key(typed))
        text=json.dumps(result["public_feedback"])
        for forbidden in("PRIVATE","credentials","/private","PRIVATE_INPUT_ARRAY"):
            self.assertNotIn(forbidden,text)
        self.assertIn("does not enable Python conditional",text)
        self.assertIn("line",text)

    def test_new_reasons_require_bounded_static_witness(self):
        for reason in("native_control_flow_not_enabled","native_public_syntax"):
            for location in(None,dict(kind="source",line=True,column=0),
                            dict(kind="source",line=1,column=0,path="/private"),
                            dict(kind="instruction",index=0)):
                with self.subTest(reason=reason,location=location),self.assertRaises(ValueError):
                    make(reason,location=location)
            with self.assertRaises(ValueError):
                make(reason,location=dict(kind="source",line=1,column=0),context={"reference":42})

    def test_reconstructed_flags_cannot_be_forged(self):
        v=make("native_control_flow_not_enabled",location=dict(kind="source",line=3,column=4))
        for change in(dict(repeat_key_available=True),dict(owner="host"),dict(code="backend_rejected")):
            with self.subTest(change=change),self.assertRaises(ValueError):
                checked(dict(v,**change))

    def test_node_limit_unchanged(self):
        tree=ast.parse(program("\n".join("x=x+x"for _ in range(900))+"\nreturn x"))
        for expand_fn in(expand,old.expand):
            with self.subTest(variant=expand_fn.__module__),self.assertRaisesRegex(ValueError,"Native loop source node bound"):
                expand_fn(tree,{})

    def test_real_current_candidates_preserve_acceptance(self):
        audit=json.loads((BASE/"benchmarks/tools/generic_capacity_post_audit_v1/static-candidates-analysis.json").read_text())
        plan=json.loads((BASE/"benchmarks/tools/generic_capacity_backend_v10/live-plan.json").read_text())
        for row in audit["rows"]:
            with self.subTest(case=row["case"],response=row["response_file"]):
                scope=next(c for c in plan["cases"]if c["id"]==row["case"])
                file=Path(row["response_file"])
                self.assertEqual(hashlib.sha256(file.read_bytes()).hexdigest(),row["response_sha256"])
                candidate=json.loads(json.loads(file.read_text())["raw"])
                request=json.loads(Path(scope["request_file"]).read_text())
                if row["status"]=="static_passed":
                    validate_candidate(candidate,request)
                else:
                    with self.assertRaises(ValueError)as caught:
                        validate_candidate(candidate,request)
                    if row["error_message"]=="Unsupported native loop syntax":
                        self.assertIsInstance(caught.exception,CandidateCheckError)
                        self.assertEqual(caught.exception.reason,"native_control_flow_not_enabled")
                        # Explicit locations in the three preserved public sources:
                        # helper condition (response 0), and loop condition (response 2).
                        expected=(34,8)if Path(row["response_file"]).name=="response-2.json"else(3,4)
                        self.assertEqual(caught.exception.location,
                                         dict(kind="source",line=expected[0],column=expected[1]))

if __name__=="__main__":
    unittest.main()
