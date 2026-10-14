"""Typed-core safety and resource gates; no model API or native execution here."""
import ast
import unittest

from decorated_functions import validate, register
from native_function_core_fixtures import PROGRAMS, declaration as fn, options


class TypedCoreTests(unittest.TestCase):
    def test_all_eleven_manual_programs_and_explicit_evidence_limits(self):
        for name, source in PROGRAMS.items():
            with self.subTest(name=name):
                plan = validate(source, {}, **options(name))
                self.assertEqual(plan['contract'], 'decorated-functions-core-v1')
                self.assertTrue(plan['agent_contract_enabled'])
                self.assertFalse(plan['agent_generation_validated'])
                self.assertFalse(plan['compilation_checked'])
                self.assertFalse(plan['encrypted_correctness_checked'])
                self.assertEqual(plan['functions']['golden']['kinds'], ('c',)*len(options(name)['input_names']))

    def test_forward_order_and_expanded_cost(self):
        p = validate(PROGRAMS['forward'], {})
        self.assertEqual(p['topological_order'], ['helper','golden'])
        p = validate(PROGRAMS['repeated'], {})
        self.assertEqual(p['functions']['golden']['calls'], ['helper','helper'])
        self.assertGreater(p['functions']['golden']['expanded_cost'], 2*p['functions']['helper']['expanded_cost'])

    def test_unpacking_indexing_negative_indices_and_rebinding(self):
        source = fn('helper','x','return [x+.25,x*.5]')+fn('golden','x',
            'pair=helper(x)\na,b=pair\na=a+b\nreturn a+pair[-1]')
        self.assertEqual(validate(source,{})['functions']['golden']['result'], 'c')
        for wrong in ('a=pair[2]','a=pair[True]','a=pair[x]','a=pair[:]','a,b,c=pair','a,a=pair'):
            with self.subTest(wrong=wrong),self.assertRaises(ValueError):
                validate(source.replace('a,b=pair',wrong),{})

    def test_public_constants_and_c_p_types(self):
        source = PROGRAMS['public_argument'].replace('helper(x,.5)','helper(x,w)')
        self.assertEqual(validate(source,{'w':[.5]})['functions']['helper']['kinds'], ('c','p'))
        for wrong in ('helper(x,x)','helper(.5,.5)','helper(x)','helper(x,.5,.5)', 'helper(x,weight=.5)'):
            with self.subTest(wrong=wrong),self.assertRaises(ValueError):
                validate(PROGRAMS['public_argument'].replace('helper(x,.5)',wrong),{})

    def test_rotation_requirements_propagate(self):
        source=fn('helper','x','return -x.rotate(-3)')+fn('golden','x','return helper(x).rotate(2)')
        self.assertEqual(validate(source,{})['rotation_steps'],[-3,2])
        for bad in ('0','4','True','x','1+1'):
            with self.assertRaises(ValueError):validate(source.replace('rotate(-3)','rotate('+bad+')'),{})

    def test_cycles_including_unused_helpers_rejected(self):
        for source in (fn('helper','x','return helper(x)')+fn('golden','x','return x'),
                       fn('first','x','return second(x)')+fn('second','x','return first(x)')+fn('golden','x','return x')):
            with self.assertRaisesRegex(ValueError,'Recursive'):validate(source,{})

    def test_invalid_code_rejected_before_frontend_registration(self):
        class Untouched:
            def __getattribute__(self,name): raise AssertionError('Frontend touched before validation')
        for source in ('import os\n'+PROGRAMS['scalar'],fn('golden','x','return open("secret")'),
                       fn('golden','x','return x.__class__'),fn('golden','x','return hc.bootstrap(x)'),
                       fn('golden','x','return eval("x")'),fn('golden','x','return getattr(x,"rotate")(1)'),
                       fn('golden','x','return (lambda y:y)(x)'),fn('golden','x','while True:\n    pass\nreturn x'),
                       fn('golden','x','return __import__("os")')):
            with self.subTest(source=source),self.assertRaises(ValueError):register(source,{},Untouched())

    def test_headers_and_global_collisions(self):
        base=PROGRAMS['scalar']
        for source in (base+base,base.replace('@hc.func(\'c\')','@evil()'),base.replace('def helper(x):','def helper(x=1):'),
                       base.replace('def helper(x):','def helper(x: int):'),base.replace('def helper(x):','def helper(x, /):'),
                       base.replace('def helper(x):','def helper(*x):'),base.replace('return helper(x)','helper=x\n    return x')):
            with self.subTest(source=source),self.assertRaises(ValueError):validate(source,{})
        for consts in ({'x':[1.]},{'helper':[1.]},{'w':[1.,2.]},{'w':True},{'w':float('nan')},{'w':float('inf')}):
            with self.subTest(constants=consts),self.assertRaises(ValueError):validate(base,consts)

    def test_result_and_unsupported_body_boundaries(self):
        for body in ('return .5','return [[x]]','return (x,x)','return None','return x/x',
                     'return x**2','return +x','return x[0]','x+=x\nreturn x','return np.array([x],dtype=object)',
                     'return True','return 1e309','return x+2048','return .5+.25',
                     'if x:\n    return x\nreturn x','x[0]=x\nreturn x'):
            with self.subTest(body=body),self.assertRaises(ValueError):validate(fn('golden','x',body),{})

    def test_expansion_bomb_rejected_before_native_work(self):
        source=fn('f0','x','return x+.25')
        for i in range(1,12):source+=fn('f'+str(i),'x',f'return f{i-1}(x)+f{i-1}(x)')
        source+=fn('golden','x','return f11(x)')
        with self.assertRaisesRegex(ValueError,'resource limit'):validate(source,{})

    def test_no_candidate_exec_or_compile_path(self):
        from pathlib import Path
        import decorated_functions
        tree=ast.parse(Path(decorated_functions.__file__).read_text())
        calls={n.func.id for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)}
        self.assertFalse(calls & {'eval','exec','compile','getattr','__import__'})

    def test_legacy_coverage_does_not_silently_inspect_first_helper(self):
        from dsl_grammar_coverage import analyze_source
        with self.assertRaisesRegex(ValueError, 'dedicated call-graph audit'):
            analyze_source(PROGRAMS['scalar'], {}, contract='hecate-native-functions-v1')


if __name__=='__main__':unittest.main()
