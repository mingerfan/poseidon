"""Aggregate acceptance is an intersection, never any-child coverage."""
import ast
import copy
import unittest
from benchmark_suite import Builder
from compiler_configuration import PROFILE_SHA256,configuration
from unified_graph_contract import prepare,validate_candidate,validate_request
from unified_graph_exercises import SPECS,COMPOSITES,STRUCTURAL,golden_variant,spec
from unified_graph_lowering import lower
from unified_native_coverage import verify_trace_coverage
from audit_semantic_benchmark import verify_directed_coverage
from benchmark_graph import digest

class CompositeTests(unittest.TestCase):
    def fixture(self,feature):
        b=Builder([(2,),(2,)]);g=b.finish(b.node('add',['input0','input1']))
        name=next(k for k,v in SPECS.items() if v[0]==feature)
        req=prepare(g,PROFILE_SHA256,configuration('seal-cpu-eva-w45-v1'),name)
        return req,golden_variant(lower(req),name)
    def checked(self,req,source):
        return validate_candidate(dict(schema=1,request_id=req['request_id'],hecate_source=source),req)['construction_exercise']
    def records(self,checked):
        records=dict(storage=[],star=[],calls=[],augmented=[],mutation=[])
        groups={'storage':'storage','star':'star','call':'calls','augmented':'augmented','mutation':'mutation'}
        for witness in checked['witnesses'].values():
            for record in [witness['trace'],*witness.get('additional_traces',[])]:
                record=copy.deepcopy(record);kind=record.pop('kind');records[groups[kind]].append(record)
        return records
    def test_each_aggregate_requires_all_registered_children(self):
        for feature,children in COMPOSITES.items():
            req,source=self.fixture(feature);got=self.checked(req,source)
            with self.subTest(feature=feature):
                self.assertEqual(tuple(req['construction_exercise']['required_features']),children)
                self.assertEqual(tuple(got['witnesses']),children)
                self.assertEqual(got['structural_features'],[f for f in children if f in STRUCTURAL])
                for child in children:
                    name=next(k for k,v in SPECS.items() if v[0]==child)
                    with self.subTest(child=child),self.assertRaisesRegex(ValueError,'Missing'):
                        self.checked(req,golden_variant(lower(req),name))
    def test_live_but_unused_numerical_children_do_not_count(self):
        for feature in COMPOSITES:
            req,source=self.fixture(feature);tree=ast.parse(source)
            fn=next(n for n in tree.body if n.name=='golden')
            fn.body[-1]=ast.parse(lower(req)).body[0].body[-1]
            with self.subTest(feature=feature),self.assertRaisesRegex(ValueError,'Missing contributing'):
                self.checked(req,ast.unparse(ast.fix_missing_locations(tree)))
    def test_each_child_requires_a_matching_real_trace(self):
        for feature in COMPOSITES:
            req,source=self.fixture(feature);got=self.checked(req,source);records=self.records(got)
            coverage=verify_trace_coverage(got,records)
            self.assertTrue(coverage['finite_influence_checked'])
            for child,witness in got['witnesses'].items():
                bad=copy.deepcopy(records);trace=witness['trace'];plain={k:v for k,v in trace.items() if k!='kind'}
                for group in bad:bad[group]=[r for r in bad[group] if r!=plain]
                with self.subTest(feature=feature,child=child),self.assertRaisesRegex(ValueError,'Missing real'):
                    verify_trace_coverage(got,bad)
    def test_mixed_tasks_keep_structure_separate_and_reject_missing_children(self):
        for feature in COMPOSITES:
            req,source=self.fixture(feature);got=self.checked(req,source)
            coverage=verify_trace_coverage(got,self.records(got));expected=req['construction_exercise']
            task=dict(requirement=feature,required_features=expected['required_features'],
                      structural_features=got['structural_features'],
                      acceptance_kind='mixed' if got['structural_features'] else 'numeric_influence')
            verify_directed_coverage(coverage,task,expected)
            for field in ('numeric_features','structural_features'):
                if not coverage[field]:continue
                bad=copy.deepcopy(coverage);bad[field]=bad[field][1:]
                with self.assertRaisesRegex(ValueError,'child witnesses'):verify_directed_coverage(bad,task,expected)
            bad=copy.deepcopy(coverage);bad['actual_frontend_checked']=False
            with self.assertRaisesRegex(ValueError,'No real'):verify_directed_coverage(bad,task,expected)
            if got['structural_features']:
                bad=copy.deepcopy(coverage);bad['numeric_features']+=bad['structural_features'];bad['structural_features']=[]
                with self.assertRaisesRegex(ValueError,'child witnesses'):verify_directed_coverage(bad,task,expected)
    def test_subrequirements_are_request_hash_bound(self):
        req,_=self.fixture('native.call');req['construction_exercise']['required_features'].pop()
        req['request_id']=digest({k:v for k,v in req.items() if k!='request_id'})
        with self.assertRaises(ValueError):validate_request(req)
    def test_old_single_feature_requests_keep_exact_metadata(self):
        for name,(feature,instruction) in SPECS.items():
            if feature not in COMPOSITES:
                self.assertEqual(spec(name),dict(id=name,required_features=[feature],instruction=instruction))
    def test_composite_helper_names_are_disjoint_and_forward_order_is_retained(self):
        req,source=self.fixture('native.call');tree=ast.parse(source)
        names=[n.name for n in tree.body];self.assertEqual(len(names),len(set(names)))
        golden=names.index('golden');self.assertGreater(golden,0);self.assertLess(golden,len(names)-1)
        checked=self.checked(req,source)
        self.assertIn('call.forward',checked['witnesses'])

if __name__=='__main__':unittest.main()
