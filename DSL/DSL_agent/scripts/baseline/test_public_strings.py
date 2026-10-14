"""Public coefficient parsing: native string semantics, not FHE proof."""
import itertools
import os
from pathlib import Path
import unittest
import public_strings as strings
from function_construction import normalize
from candidate_trace import evaluate_tree
from test_function_construction import program
from test_closure_construction import Packed


class StringOperationTests(unittest.TestCase):
    def test_native_string_matrix(self):
        samples = ('',' ','  a  b  ','a,,b,','--1.25--','a\tb\nc','\u2003a\u2003','α,β,α')
        plans = []
        for method in ('strip','lstrip','rstrip'):
            plans += [(method,[chars],{}) for chars in (None,'',' -',',','α')]
        for method in ('split','rsplit'):
            plans += [(method,[sep,count],{}) for sep,count in itertools.product((None,' ',',','α'),(-2,0,1,4))]
        for method in ('partition','rpartition'):
            plans += [(method,[sep],{}) for sep in (',',' ','α')]
        plans += [('replace',[old,new,count],{}) for old,new,count in itertools.product(('',',','a'),('','X'),(-1,0,1,3))]
        for text,(method,args,kwargs) in itertools.product(samples,plans):
            with self.subTest(text=text,method=method,args=args):
                expected = strings.METHODS[method](text,*args,**kwargs)
                self.assertEqual(strings.invoke(method,text,args,kwargs),expected)
                call = repr(text)+'.'+method+'('+','.join(repr(v) for v in args)+')'
                source = program('','return x if '+call+' == '+repr(expected)+' else -x')
                expanded = normalize(source,{},public_strings=True)
                actual = evaluate_tree(expanded['source'],{},Packed([1,2,3,4]))
                self.assertEqual(actual.values,(1,2,3,4))
                self.assertEqual(expanded['construction']['public_string_calls'],1)

    def test_join_and_pinned_signatures_and_limits(self):
        for sep,items in itertools.product(('',',','::'),([],[''],['a','b'],['α','β'])):
            self.assertEqual(strings.invoke('join',sep,[items],{}),sep.join(items))
        self.assertEqual(strings.invoke('split','a,,b',[],dict(sep=',',maxsplit=1)),['a',',b'])
        bad = [('split','x',[''],{}),('split','x',[None,1.5],{}),
               ('split','x',[','],dict(sep=',')),('strip','x',[],dict(chars='x')),
               ('replace','a',['a','b'],dict(count=1)),('join','-',[[1]],{}),
               ('join','-'*128,[['a','b']],{}),('replace','a'*128,['a','b'*128],{}),
               ('split',','*128,[','],{}),('format','{}',[1],{})]
        for method,receiver,args,kwargs in bad:
            with self.subTest(method=method),self.assertRaises(ValueError):
                strings.invoke(method,receiver,args,kwargs)


class StringConstructionTests(unittest.TestCase):
    def run_source(self,body,helpers=''):
        expanded = normalize(program(helpers,body),{},public_strings=True)
        result = evaluate_tree(expanded['source'],{k:Packed(v) for k,v in expanded['constants'].items()},Packed([1,2,3,4]))
        return result.values, expanded

    def test_actual_coefficient_and_tree_parsing_patterns(self):
        helpers = '''def coefficients(rows, scale):
    return [float(token.strip()) / scale for token in rows]
'''
        values,expanded = self.run_source('tree = [[int(v) for v in line.strip().split(" ")] for line in [" 0 1 "]]\n'
            'coefs = coefficients([" 3.0 "," 0.75 "],2)\n'
            'return x*coefs[tree[0][0]] + coefs[tree[0][1]]',helpers)
        self.assertEqual(values,(1.875,3.375,4.875,6.375))
        self.assertEqual(expanded['construction']['public_string_calls'],4)

    def test_join_consumption_keyword_unpack_and_receiver_evaluation(self):
        values,_ = self.run_source('it = iter(["1",".","5"])\nw = float("".join(it))\n'
            'remaining = next(it,"done")\nreturn x*w if remaining == "done" else -x')
        self.assertEqual(values,(1.5,3,4.5,6))
        values,_ = self.run_source('words = "1.5,0.375".split(*[","],**{"maxsplit":1})\n'
                                  'return x*float(words[0])+float(words[1])')
        self.assertEqual(values,(1.875,3.375,4.875,6.375))
        helpers = '''def argument(box):
    box[0] = "-2"
    return "-"
'''
        values,_ = self.run_source('box = ["-1"]\nw = float(box[0].strip(argument(box)))\nreturn x*w',helpers)
        self.assertEqual(values,(1,2,3,4))

    def test_explicit_separator_empty_tokens_are_not_whitespace_split(self):
        values,_ = self.run_source('a = "1  2".split(" ")\nb = "1  2".split()\n'
                                  'return x if a == ["1","","2"] and b == ["1","2"] else -x')
        self.assertEqual(values,(1,2,3,4))

    def test_rejected_capabilities_types_and_unimplemented_bound_methods(self):
        bodies = ('return x.strip()', 'return x*float("{x}".format(x=x))',
                  'method = "x".strip\nreturn x', 'return x*float("".join([x]))',
                  'return x*float("1".replace("1",x))',
                  'return x*float("1".strip(**{0:"1"}))',
                  'return x*float("1".split(sep=",",**{"sep":";"})[0])',
                  'return x*float("1".encode())', 'return x*float("nan".strip())')
        for body in bodies:
            with self.subTest(body=body),self.assertRaises(ValueError): self.run_source(body)
        with self.assertRaises(ValueError):
            normalize(program('','return x*float(" 1 ".strip())'),{},public_control=True)

    def test_request_cli_coverage_and_batch_plan(self):
        from candidate_contract import make_request,validate_candidate,valid_semantic_guidance
        from run_candidate import parse_args,forward_options
        from dsl_grammar_coverage import analyze_source
        from run_agent_batch import main,construction_options,validate_construction_continuation
        from types import SimpleNamespace
        from unittest.mock import patch
        import contextlib,io,json
        payload = dict(fx_graph='x',public_constants={},constant_origins={},
            layout=dict(input_shape=[4],output_shape=[4],output_ciphertexts=1,output_selectors=[[0,i] for i in range(4)]))
        request = make_request(payload,dict(schema=2,id='string-unit'),'a'*64,public_strings=True)
        source = program('','return x*float(" 1.5 ".strip())')
        self.assertEqual(request['task'],'hecate-function-synthesis-v16')
        self.assertTrue(valid_semantic_guidance(request))
        checked = validate_candidate(dict(schema=1,request_id=request['request_id'],hecate_source=source),request)
        self.assertEqual(checked['contract'],'hecate-function-v15')
        self.assertEqual(analyze_source(source,{},contract='hecate-function-v15')['public_construction']['schema'],10)
        self.assertIn('--public-strings',forward_options(parse_args(['--case','case.json','--prepare','--public-strings'])))
        args = SimpleNamespace(public_strings=True)
        self.assertEqual(construction_options(args),['--public-strings'])
        validate_construction_continuation({'public_strings':True},args)
        with self.assertRaisesRegex(ValueError,'fresh batch'): validate_construction_continuation({},args)
        output = io.StringIO()
        with patch('sys.argv',['run_agent_batch.py','--plan','--public-strings']), \
             patch('agent_credentials.load_api_key',side_effect=AssertionError('credential access')), \
             contextlib.redirect_stdout(output):
            self.assertEqual(main(),0)
        report = json.loads(output.getvalue())
        self.assertTrue(report['public_strings'])
        self.assertEqual(report['agent_calls'],0)


class StringGoldenPlainTests(unittest.TestCase):
    def test_goldens_against_independent_formula(self):
        from run_public_string_goldens import PLANS
        root = Path(__file__).resolve().parent
        weights = [[.5,-.25,.125,.75],[-.375,.25,.5,-.125]]
        for case,golden,wrong in PLANS:
            source = (root/'golden_cases/public_strings'/(golden+'.py')).read_text()
            expanded = normalize(source,{},2 if case == 'construction-linear' else 1,public_strings=True)
            detected = False
            for x in ([0.]*4,[.5,-1.,.25,-.75],[-1.,1.,-1.,1.]):
                expected = ([sum(a*b for a,b in zip(x,row)) for row in weights] if case == 'construction-linear' else
                            [4*v**3-3*v for v in x] if case == 'construction-chebyshev3' else [1.5*v+.375 for v in x])
                value = evaluate_tree(expanded['source'],{k:Packed(v) for k,v in expanded['constants'].items()},Packed(x))
                actual = [v.values[0] for v in value] if type(value) is list else value.values
                equal = len(actual) == len(expected) and all(abs(a-b)<1e-12 for a,b in zip(actual,expected))
                detected |= not equal
                if not wrong: self.assertTrue(equal,(golden,actual,expected))
            if wrong: self.assertTrue(detected,golden)


@unittest.skipUnless(os.environ.get('POSEIDON_STRING_REPORT'),'requires actual string-parsing CPU evidence')
class StringEvidenceTests(unittest.TestCase):
    def test_actual_artifacts_and_frozen_reference(self):
        from test_closure_construction import ClosureEvidenceTests
        from run_public_string_goldens import PLANS
        # Preserve the actual producer; the auditor reproduces exact old
        # expansion and verifies all artifact/reference hashes independently.
        producers = {
            'function_construction.py':'6ec591c71a7f96bcdc42c2fb131db2799aac743359ab5c0ae1979685493cc9d0',
            'public_strings.py':'244dee49f23786b48a43c8e27c9d629a6d5a55a3ba8be7e85410f5a96cc3445c',
            'public_numeric.py':'9cc7895f04ac8fd921e5899f8102a2767ccefe0c19a8dae8f558ad7452eff780',
            'lexical_scope.py':'5f80b879132ca2974a20a48610d13f2037597e1c4aa847fe29037f464f4a7c70',
            'construction_calls.py':'5c9fc99056c364324be7796004a29e3e515c15991f95c656c589b5796d039539'}
        ClosureEvidenceTests.verify_evidence(self,os.environ['POSEIDON_STRING_REPORT'],PLANS,
            'public_strings','hecate-function-synthesis-v16',dict(public_strings=True),producers)


if __name__ == '__main__':
    unittest.main()
