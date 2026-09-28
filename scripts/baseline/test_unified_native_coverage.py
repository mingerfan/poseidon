"""Contract witnesses must survive arbitrary packing and reject dead padding."""
import ast
import copy
import unittest
import numpy as np
from benchmark_suite import Builder
from benchmark_graph import samples
from benchmark_math import evaluate
from compiler_configuration import PROFILE_SHA256,configuration
from unified_graph_contract import prepare,validate_candidate
from unified_graph_exercises import SPECS,STRUCTURAL,golden_variant
from unified_graph_lowering import lower
from unified_native_coverage import verify_trace_coverage

class ProbeExpr:
    """Immutable whole-cipher vectors for the trusted AST interpreter, no FHE claim."""
    def __init__(self,v):
        self.v=np.asarray(v,dtype=np.float64);self.obj=object()
    def other(self,x):return x.v if isinstance(x,ProbeExpr) else x
    def __add__(self,x):return ProbeExpr(self.v+self.other(x))
    __radd__=__add__
    def __sub__(self,x):return ProbeExpr(self.v-self.other(x))
    def __rsub__(self,x):return ProbeExpr(self.other(x)-self.v)
    def __mul__(self,x):return ProbeExpr(self.v*self.other(x))
    __rmul__=__mul__
    def __neg__(self):return ProbeExpr(-self.v)
    def rotate(self,k):return ProbeExpr(np.roll(self.v,-k))

def probe(request,source,inputs):
    from types import SimpleNamespace
    from decorated_functions import register
    p=request['layout']['input_slot_period']
    frontend=SimpleNamespace(np=np,Expr=ProbeExpr,resolveType=ProbeExpr,
                             func=lambda signature:lambda body:body)
    names=tuple(x['dsl_name'] for x in request['layout']['inputs'])+('zero_ct',)
    functions,_=register(source,request['public_constants'],frontend,
                         expected_outputs=len(request['layout']['outputs']),input_names=names,
                         arrays=True,starred_calls=True,array_arithmetic=True,public_loops=True,
                         scalar_augmented=True,array_mutation=True,slot_period=p)
    args=[ProbeExpr(np.pad(inputs[x['name']].reshape(-1),(0,p-inputs[x['name']].size)))
          for x in request['layout']['inputs']]+[ProbeExpr(np.zeros(p))]
    result=functions['golden'](*args)
    out=list(result.flat) if type(result) is np.ndarray else list(result) if type(result) in (list,tuple) else [result]
    return np.asarray([out[i].v[j] for i,j in request['layout']['output_selectors']])


class UnifiedNativeCoverageTests(unittest.TestCase):
    def request(self,feature,multi=True):
        b=Builder([(2,3),(6,)])
        y=b.node('reshape',['input1'],shape=[2,3])
        z=b.node('subtract',['input0',y])
        avg=b.node('mean',[z],axes=[1],keepdims=True)
        g=b.finish(z,avg) if multi else b.finish(z)
        name=next(k for k,v in SPECS.items() if v[0]==feature)
        return prepare(g,PROFILE_SHA256,configuration('seal-cpu-eva-w45-v1'),name)
    def checked(self,req,source):
        return validate_candidate(dict(schema=1,request_id=req['request_id'],hecate_source=source),req)['construction_exercise']
    def test_all_native_witnesses_match_independent_math_with_multiple_inputs(self):
        for name,(feature,_) in SPECS.items():
            with self.subTest(feature=feature):
                req=self.request(feature,multi=feature!='golden.zero')
                source=golden_variant(lower(req),name);got=self.checked(req,source)
                self.assertEqual(got['slot_period'],8)
                self.assertEqual(got['structural_features'],[f for f in req['construction_exercise']['required_features'] if f in STRUCTURAL])
                for inputs in samples(req['model'],4):
                    expected=np.concatenate([v.reshape(-1) for v in evaluate(req['model'],inputs).values()])
                    np.testing.assert_allclose(probe(req,source,inputs),expected,atol=1e-12,rtol=1e-12)
    def candidate(self,req,helper,lines):
        tree=ast.parse(lower(req));g=tree.body[0];ret=g.body[-1]
        first=ast.unparse(ret.value.elts[0])
        g.body[-1:-1]=ast.parse('covered_out = '+first+'\n'+lines).body
        ret.value.elts[0]=ast.Name(id='covered_out',ctx=ast.Load())
        tree.body[:0]=ast.parse(helper).body
        return ast.unparse(ast.fix_missing_locations(tree))
    def test_dead_helper_and_unused_repeated_call_do_not_count(self):
        for feature in ('call.identity','call.repeated'):
            req=self.request(feature)
            helper='@hc.func("c")\ndef coverage_helper(a):\n    return a\n'
            lines='' if feature=='call.identity' else 'coverage_helper(covered_out)\ncovered_out = coverage_helper(covered_out)\n'
            with self.subTest(feature=feature),self.assertRaisesRegex(ValueError,'Missing contributing'):
                self.checked(req,self.candidate(req,helper,lines))
    def test_unused_public_and_starred_arguments_rejected(self):
        for feature,helper,lines in [
            ('call.public_argument','@hc.func("c,p")\ndef coverage_helper(a,p):\n    return a\n',
             'covered_out = coverage_helper(covered_out,0.5)\n'),
            ('star.tuple','@hc.func("c,c")\ndef coverage_helper(a,b):\n    return a\n',
             'covered_out = coverage_helper(*(covered_out,covered_out))\n')]:
            req=self.request(feature)
            with self.subTest(feature=feature),self.assertRaisesRegex(ValueError,'Missing contributing'):
                self.checked(req,self.candidate(req,helper,lines))
    def test_structural_empty_cannot_get_numeric_credit(self):
        for feature in STRUCTURAL:
            req=self.request(feature);name=req['construction_exercise']['id']
            checked=self.checked(req,golden_variant(lower(req),name))
            self.assertEqual(checked['numeric_features'],[])
            self.assertEqual(checked['witnesses'][feature]['influencing_cells'],[])
            records=dict(storage=[],star=[],calls=[],augmented=[],mutation=[])
            witness=copy.deepcopy(checked['witnesses'][feature]['trace']);kind=witness.pop('kind')
            records[{'storage':'storage','star':'star','call':'calls'}[kind]].append(witness)
            result=verify_trace_coverage(checked,records)
            self.assertTrue(result['structural_only'])
            self.assertFalse(result['finite_influence_checked'])
            records[{'storage':'storage','star':'star','call':'calls'}[kind]].clear()
            with self.assertRaisesRegex(ValueError,'Missing real'):
                verify_trace_coverage(checked,records)

    def test_repeated_trace_requires_observed_multiplicity(self):
        trace=dict(kind='call',caller='golden',callee='helper',span=[1,0,1,9])
        checked=dict(id='probe',witnesses={'call.repeated':dict(trace=trace,additional_traces=[trace])},
                     numeric_features=['call.repeated'],structural_features=[])
        event={k:v for k,v in trace.items() if k!='kind'}
        records=dict(storage=[],star=[],calls=[event],augmented=[],mutation=[])
        with self.assertRaisesRegex(ValueError,'Missing real'):
            verify_trace_coverage(checked,records)
        records['calls'].append(event)
        self.assertTrue(verify_trace_coverage(checked,records)['finite_influence_checked'])

if __name__=='__main__':unittest.main()
