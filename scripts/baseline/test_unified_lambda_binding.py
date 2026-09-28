"""Lambda binding probes preserve evaluation and never perturb arbitrary bodies."""
import ast
import copy
import unittest
from benchmark_suite import Builder
from compiler_configuration import PROFILE_SHA256,configuration
from unified_graph_contract import prepare,validate_candidate
from unified_graph_lowering import lower
from unified_public_contract import CONTRACT,normalize,event_record
from unified_public_exercises import SPECS,LAMBDA_RECIPES,golden_variant
from unified_public_coverage import verify_trace_coverage,fingerprint,span

class LambdaBindingTests(unittest.TestCase):
    def request(self,feature):
        b=Builder([(2,),(2,)]);g=b.finish(b.node('add',['input0','input1']))
        name=next(k for k,v in SPECS.items() if v[0]==feature)
        return prepare(g,PROFILE_SHA256,configuration('seal-cpu-eva-w45-v1'),name,construction_profile=CONTRACT)
    def source(self,body):
        return '@hc.func("c,c,c")\ndef golden(x,y,zero_ct):\n'+''.join(' '+s+'\n' for s in body.splitlines())
    def checked(self,req,source):
        return validate_candidate(dict(schema=1,request_id=req['request_id'],hecate_source=source),req)['construction_exercise']
    def test_all_lambda_partitions_match_trace_and_reject_missing_events(self):
        for feature in LAMBDA_RECIPES:
            req=self.request(feature);source=golden_variant(lower(req),req['construction_exercise']['id'],req)
            checked=self.checked(req,source);events=[];expanded=normalize(source,req,lambda n:events.append(event_record(n)))
            record=dict(events=events,normalized_sha256=expanded['construction']['normalized_sha256'],candidate_python_executed=False)
            result=verify_trace_coverage(checked,record);self.assertTrue(result['finite_influence_checked'])
            self.assertEqual(result['numeric_features'],[feature])
            bad=copy.deepcopy(record);bad['events']=[]
            with self.assertRaises(ValueError):verify_trace_coverage(checked,bad)
    def test_ignored_parameters_length_keys_and_empty_expansions_do_not_count(self):
        cases={
            'lambda.posonly':['f=lambda v,/:x+0.5\nreturn [f(y)]','unused=lambda v,/:v+0.5\nreturn [x]'],
            'lambda.kwonly':['f=lambda *,v:x+0.5\nreturn [f(v=y)]','unused=lambda *,v:v+0.5\nreturn [x]'],
            'lambda.vararg':['f=lambda *v:x+len(v)*0.25\nreturn [f(x,y)]','f=lambda *v:x+0.5\nreturn [f()]'],
            'lambda.kwarg':["f=lambda **v:x+len(v)*0.25\nreturn [f(left=x,right=y)]",
                            "f=lambda **v:x+float('left' in v)\nreturn [f(left=y)]",
                            'f=lambda **v:x+0.5\nreturn [f()]'],
        }
        for feature,bodies in cases.items():
            req=self.request(feature)
            for body in bodies:
                with self.subTest(feature=feature,body=body),self.assertRaisesRegex(ValueError,'Missing contributing'):
                    self.checked(req,self.source(body))
    def test_argument_or_default_side_effect_alone_does_not_count(self):
        cases={
            'lambda.posonly':'f=lambda v,/:value\nreturn [f(produce())]',
            'lambda.kwonly':'f=lambda *,v:value\nreturn [f(v=produce())]',
            'lambda.vararg':'f=lambda *v:value\nreturn [f(produce())]',
            'lambda.kwarg':'f=lambda **v:value\nreturn [f(v=produce())]',
        }
        for feature,tail in cases.items():
            req=self.request(feature);body='value=x\ndef produce():\n nonlocal value\n value=value+0.5\n return y\n'+tail
            with self.subTest(feature=feature),self.assertRaisesRegex(ValueError,'Missing contributing'):
                self.checked(req,self.source(body))
        req=self.request('lambda.posonly')
        body='value=x\ndef produce():\n nonlocal value\n value=value+0.5\n return y\nf=lambda v=produce(),/:value\nreturn [f()]'
        with self.assertRaisesRegex(ValueError,'Missing contributing'):self.checked(req,self.source(body))
    def test_nonuniform_variadic_probes_detect_cancelled_uniform_offsets(self):
        for feature,body in [
            ('lambda.vararg','f=lambda *v:v[0]-v[1]\nreturn [f(x,y)]'),
            ('lambda.kwarg',"f=lambda **v:v['left']-v['right']\nreturn [f(left=x,right=y)]")]:
            req=self.request(feature);checked=self.checked(req,self.source(body))
            self.assertIn('enumerate',checked['witnesses'][feature]['replacement'])
            self.assertGreaterEqual(checked['intervention_attempts'],2)
    def test_scaling_detects_repeated_invocation_offset_cancellation(self):
        for feature,body in [
            ('lambda.posonly','f=lambda v,/:v+0.25\nreturn [f(x)-f(y)]'),
            ('lambda.kwonly','f=lambda *,v:v+0.25\nreturn [f(v=x)-f(v=y)]'),
            ('lambda.vararg','f=lambda *v:v[0]+0.25\nreturn [f(x)-f(y)]'),
            ('lambda.kwarg',"f=lambda **v:v['x']+0.25\nreturn [f(x=x)-f(x=y)]")]:
            checked=self.checked(self.request(feature),self.source(body))
            self.assertIn('* 2',checked['witnesses'][feature]['replacement'])

    def test_default_and_argument_evaluation_retained_once(self):
        req=self.request('lambda.posonly')
        body='calls=0\ndef produce():\n nonlocal calls\n calls+=1\n return x\nf=lambda v=produce(),/:v+0.25\nvalue=f()\nif calls!=1:\n return [zero_ct]\nreturn [value]'
        for variant in (body,body.replace('f=lambda v=produce(),/:v+0.25','f=lambda v,/:v+0.25').replace('value=f()','value=f(produce())')):
            source=self.source(variant);checked=self.checked(req,source);witness=checked['witnesses']['lambda.posonly']
            position=tuple(witness['span']);replacement=ast.parse(witness['replacement'],mode='eval').body
            class Change(ast.NodeTransformer):
                def visit_Lambda(self,node):
                    return copy.deepcopy(replacement) if span(node)==position else self.generic_visit(node)
            changed=ast.unparse(ast.fix_missing_locations(Change().visit(ast.parse(source))))
            before=fingerprint(normalize(source,req),req);after=fingerprint(normalize(changed,req),req)
            self.assertEqual([b-a for a,b in zip(before,after)],[1.]*len(before))
    def test_closure_capture_and_temporary_name_collision(self):
        req=self.request('lambda.posonly')
        self.checked(req,self.source('factory=lambda v,/:(lambda :v+0.25)\nf=factory(x)\nreturn [f()]'))
        req=self.request('lambda.vararg')
        checked=self.checked(req,self.source('discardProbe0=x\nf=lambda *v:v[0]+discardProbe0\nreturn [f(y)]'))
        self.assertIn('discardProbe1',checked['witnesses']['lambda.vararg']['replacement'])
    def test_named_and_lambda_exercises_remain_separate(self):
        pairs=[('posonly','def f(v,/):\n return v','f=lambda v,/:v','f(x)'),
               ('kwonly','def f(*,v):\n return v','f=lambda *,v:v','f(v=x)'),
               ('vararg','def f(*v):\n return v[0]','f=lambda *v:v[0]','f(x)'),
               ('kwarg',"def f(**v):\n return v['x']","f=lambda **v:v['x']",'f(x=x)')]
        for suffix,definition,lam,call in pairs:
            with self.subTest(suffix=suffix),self.assertRaises(ValueError):
                self.checked(self.request('lambda.'+suffix),self.source(definition+'\nreturn ['+call+']'))
            with self.subTest(suffix=suffix),self.assertRaises(ValueError):
                self.checked(self.request('signature.'+suffix),self.source(lam+'\nreturn ['+call+']'))

if __name__=='__main__':unittest.main()
