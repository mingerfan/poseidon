"""Authoritative fixed AST-budget diagnostics; static analysis only."""
import ast
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from application_diagnostics import attach, checked, make, render, repeat_key
from candidate_check_diagnostic import CandidateCheckError
from decorated_functions import validate, LIMIT


def source(n):
    return '@hc.func("c,c")\ndef golden(x,zero_ct):\n' + ''.join(
        '    a'+str(i)+'=x\n' for i in range(n)) + '    return x\n'


class NativeAstBudgetDiagnostics(unittest.TestCase):
    def check(self, raw):
        return validate(raw, {}, 1, input_names=('x','zero_ct'), slot_period=32)

    def test_budget_boundary_is_unchanged(self):
        n=0
        while len(list(ast.walk(ast.parse(source(n+1)))))<=LIMIT:
            n+=1
        self.assertLessEqual(len(list(ast.walk(ast.parse(source(n))))),4096)
        self.check(source(n))
        raw=source(n+1);count=len(list(ast.walk(ast.parse(raw))))
        with self.assertRaises(CandidateCheckError)as caught:self.check(raw)
        self.assertEqual(caught.exception.reason,'native_ast_node_budget')
        self.assertEqual(caught.exception.context,dict(observed=count,limit=4096))
        self.assertEqual(caught.exception.location,dict(kind='candidate'))

    def test_comment_only_oversize_module_has_a_valid_candidate_location(self):
        raw='# type: ignore\n'*4097
        with self.assertRaises(CandidateCheckError)as caught:self.check(raw)
        self.assertEqual(caught.exception.location,dict(kind='candidate'))
        self.assertGreater(caught.exception.context['observed'],4096)

    def test_diagnostic_context_types_threshold_and_location_are_strict(self):
        tree=ast.parse(source(1))
        for context in [dict(observed=True,limit=4096),dict(observed=5000,limit=True),
                        dict(observed=4096,limit=4096),dict(observed=5000,limit=5000),
                        dict(observed=5000,limit=4096,private_reference=[1]),{}]:
            with self.subTest(context=context),self.assertRaises(ValueError):
                CandidateCheckError('native_ast_node_budget',tree,context=context)
        with self.assertRaises(ValueError):
            CandidateCheckError('native_ast_node_budget',tree.body[0],context=dict(observed=5000,limit=4096))
        for location in [None,dict(kind='source',line=1,column=0),dict(kind='candidate',path='/private')]:
            with self.assertRaises(ValueError):
                make('native_ast_node_budget',location=location,context=dict(observed=5000,limit=4096))

    def test_public_feedback_is_fixed_and_retains_no_private_exception_text(self):
        value=make('native_ast_node_budget',location=dict(kind='candidate'),
                   context=dict(observed=4732,limit=4096))
        self.assertEqual(checked(value),value)
        self.assertIsNone(repeat_key(value))
        out=attach(dict(status='failed',failure=value,structured_diagnostic=value,
                        public_feedback=dict(status='failed',layer='static_check',category='candidate',
                                             diagnostic='PRIVATE /client/reference.npy')))
        text=json.dumps(out['public_feedback'])
        self.assertIn('4096',text)
        self.assertNotIn('PRIVATE',text)
        self.assertNotIn('/client',text)
        self.assertEqual(out['failure']['context'],dict(observed=4732,limit=4096))

    def test_malformed_context_and_forged_repeat_flags_are_rejected(self):
        for context in [{},dict(observed=4096,limit=4096),dict(observed=4732,limit=4095),
                        dict(observed=4732.,limit=4096),dict(observed=2**54,limit=4096)]:
            with self.assertRaises(ValueError):
                make('native_ast_node_budget',location=dict(kind='candidate'),context=context)
        value=make('native_ast_node_budget',location=dict(kind='candidate'),context=dict(observed=4732,limit=4096))
        value['repeat_key_available']=True
        with self.assertRaises(ValueError):checked(value)

    def test_original_retained_Conv_candidate_stays_rejected_with_exact_public_count(self):
        from unified_graph_contract import validate_candidate
        from component_contract import validate_program
        S=Path('/home/lhohy/poseidon-work/platforms/aarch64-linux/tasks/a6-source-contract-r2017-0/f31fb52c017f455489738f59d70578b8')
        request=json.loads((S/'task.json').read_bytes())['request']
        candidate=json.loads(json.loads((S/'generation-0.json').read_bytes())['raw'])
        with self.assertRaises(CandidateCheckError)as caught:validate_candidate(candidate,request)
        self.assertEqual(caught.exception.context,dict(observed=4732,limit=4096))
        result=attach(validate_program(candidate,request),candidate=candidate,request=request)
        self.assertEqual(result['failure']['reason_code'],'native_ast_node_budget')
        self.assertEqual(result['failure']['context'],dict(observed=4732,limit=4096))
        self.assertFalse(result.get('encrypted_execution',False))

    def test_Application_rejects_before_qualification(self):
        from application_component import Application
        from test_application_component import Backend
        from benchmark_suite import Builder
        b=Builder([(2,)]);model=b.finish(b.node('square',['input0']))
        class Large:
            kind='scripted_replay';agent_calls=0
            def generate(self,q,history):
                raw=source(1100).replace('return x\n','return x*x\n')
                return json.dumps(dict(schema=1,request_id=q['request_id'],hecate_source=raw))
        with tempfile.TemporaryDirectory()as folder:
            backend=Backend();app=Application(folder,backend)
            task=app.submit(model,options=dict(compiler_configuration='poseidon-cpu-eva-w45-current-v1'),
                limits=dict(generations=1,http_attempts=1,wall_seconds=120,api_timeout_seconds=120,max_tokens=8192),
                feedback_policy='precise-v2')
            result=app.run(task,Large())
            self.assertEqual(result['failure']['reason_code'],'native_ast_node_budget')
            self.assertEqual(backend.calls,0)


if __name__=='__main__':unittest.main()
