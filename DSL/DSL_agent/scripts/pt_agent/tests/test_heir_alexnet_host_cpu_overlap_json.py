"""Small synthetic31-row literal fixtures and fake original Alex supervision."""
import ast
from contextlib import ExitStack
import inspect
import json
import textwrap
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from pt_agent.experiments import heir_alexnet_host_cpu_overlap_json as entry
from pt_agent.experiments import heir_cnn_export as exporter
from pt_agent.experiments import heir_common_models
from pt_agent.tests import test_heir_alexnet_host_cpu_overlap as fixtures


def tiny_inventory():
    nodes=[];weights={};emitter=exporter.Emitter()
    for i in range(29):
        name='derived-exact-local-avgpool:n'+str(i)if i in(3,7)else'W'+str(i)
        weights[name]=np.array([[i/8]],dtype=np.float64)
        nodes.append(SimpleNamespace(id='n'+str(i),op='linear',attrs=dict(weight=name)))
        emitter.constant(weights[name])
    pairs=[];masks=[]
    for i in(3,7):
        pair=dict(pool_id='n'+str(i),mask_name='derived-wide-pool-mask:n'+str(i))
        mask=np.array([[0.,1.]],dtype=np.float64);value=emitter.constant(mask)
        pairs.append(pair);masks.append(dict(name=pair['mask_name'],ssa=value.name))
    return SimpleNamespace(nodes=nodes,weights=weights), '\n'.join(emitter.lines),dict(mask_constants=masks),pairs


class AlexJSONChecks(unittest.TestCase):
    def scope(self,pairs):
        stack=ExitStack();self.addCleanup(stack.close)
        stack.enter_context(patch.object(entry.alex,'find_pairs',return_value=pairs))
        stack.enter_context(patch.object(entry.alex,'pair_constants',return_value=(None,np.array([[0.,1.]],dtype=np.float64))))
        # The tiny inventory has29 scalar-sized tensors, rather than importing a
        # model. Only architecture metadata validation is replaced in this fixture.
        stack.enter_context(patch.object(heir_common_models,'graph_contract',return_value={}))
        return stack

    def test_complete_AST_identical_after_exact_approved_import_binding_reversal(self):
        original=ast.parse(textwrap.dedent(inspect.getsource(entry.alex.check_constants)))
        changed=entry.checker_tree();body=changed.body[0].body
        assigns=[i for i,node in enumerate(body)if isinstance(node,ast.Assign)
            and isinstance(node.value,ast.Name)and node.value.id=='_json_common_checker']
        self.assertEqual(len(assigns),1);i=assigns[0]
        self.assertIsInstance(body[i-1],ast.ImportFrom)
        self.assertEqual([(a.name,a.asname)for a in body[i-1].names],[('check_math',None)])
        original_import=next(n for n in original.body[0].body if isinstance(n,ast.ImportFrom)and n.module=='heir_cnn_math_check')
        body[i-1:i+1]=[original_import]
        self.assertEqual(ast.dump(changed,include_attributes=False),ast.dump(original,include_attributes=False))

    def test_two_real_private_literal_paths_match_original31_rows_and_exact_order(self):
        graph,text,exported,pairs=tiny_inventory();self.scope(pairs)
        original=entry.alex.check_constants(graph,text,exported)
        observed=entry.constant_checker()(graph,text,exported)
        self.assertEqual(observed,original);self.assertEqual(len(observed),31)
        self.assertEqual(observed[4]['name'],'derived-wide-pool-mask:n3')
        self.assertEqual(observed[9]['name'],'derived-wide-pool-mask:n7')
        self.assertTrue(all(row['passed']and row['numerically_equal']for row in observed))

    def test_both_mask_and_common_paths_work_with_original_AST_literal_parser_forbidden(self):
        graph,text,exported,pairs=tiny_inventory();self.scope(pairs)
        with patch.object(ast,'literal_eval',side_effect=AssertionError('No AST literal path allowed')):
            rows=entry.constant_checker()(graph,text,exported)
        self.assertEqual(len(rows),31)

    def test_mask_wrong_name_duplicate_missing_and_subtolerance_value_still_reject(self):
        graph,text,exported,pairs=tiny_inventory();self.scope(pairs);checker=entry.constant_checker()
        bad=json.loads(json.dumps(exported));bad['mask_constants'][0]['name']='foreign-mask'
        with self.assertRaises(ValueError):checker(graph,text,bad)
        mask_line=next(line for line in text.splitlines()if exported['mask_constants'][0]['ssa']+' ='in line)
        with self.assertRaises(ValueError):checker(graph,text+'\n'+mask_line,exported)
        with self.assertRaises(ValueError):checker(graph,text.replace(mask_line,''),exported)
        changed=text.replace(mask_line,mask_line.replace('0.00000000000000000e+00','1.00000000000000003e-13',1))
        with self.assertRaises(ValueError):checker(graph,changed,exported)

    def test_mask_illegal_grammar_and_overflow_and_common_inventory_still_reject(self):
        graph,text,exported,pairs=tiny_inventory();self.scope(pairs);checker=entry.constant_checker()
        mask_line=next(line for line in text.splitlines()if exported['mask_constants'][0]['ssa']+' ='in line)
        for token in('true','null','"0"','NaN','1.0e400'):
            with self.subTest(token=token):
                bad=text.replace(mask_line,mask_line.replace('0.00000000000000000e+00',token,1))
                with self.assertRaises(ValueError):checker(graph,bad,exported)
        with self.assertRaises(ValueError):checker(graph,'\n'.join(text.splitlines()[1:]),exported)

    def test_checker_original_global_references_except_AST_and_one_added_common_binding(self):
        checker=entry.constant_checker();original=vars(entry.alex)
        self.assertEqual(set(checker.__globals__)-set(original),{'_json_common_checker'})
        changed=[name for name,value in original.items()if checker.__globals__[name]is not value]
        self.assertEqual(changed,['ast'])
        self.assertIs(checker.__globals__['ast'].literal_eval,entry.common_json.numeric_json_literal)
        self.assertIs(checker.__globals__['_json_common_checker'].__code__,entry.common_json.math_check.check_constants.__code__)

    def test_runtime_worker_only_changes_constant_checker_and_runtime_binding(self):
        sentinel=object();api=dict(check_constants=entry.alex.check_constants,other=sentinel)
        with patch.object(entry.alex,'_runtime',return_value=api):result=entry._runtime()
        self.assertIs(result['other'],sentinel);self.assertEqual(set(result),{'check_constants','other'})
        worker=entry.numeric_worker();self.assertIs(worker.__code__,entry.alex.worker.__code__)
        changed=[name for name,value in vars(entry.alex).items()if worker.__globals__[name]is not value]
        self.assertEqual(changed,['_runtime'])


class AlexJSONSupervisionChecks(unittest.TestCase):
    def setUp(self):
        fixtures.AlexHostChecks.setUp(self);self.args.out=self.root/(entry.OUT_PREFIX+'fixture')

    def fake(self,**options):
        return entry.host.overlap._clone_plain(fixtures.AlexHostChecks.fake,dict(vars(fixtures),entry=entry))(self,**options)

    def test_original_Alex_numeric_module_source_native_profile_and_resources_preserved(self):
        result=self.fake();self.assertEqual(result['status'],'ok',result['error'])
        self.assertEqual(self.calls[0][0][:4],[str(entry.host.host_python()),'-B','-m',entry.MODULE])
        self.assertEqual(self.calls[1][0][0],str(self.args.heir_opt));self.assertEqual(len(self.math_calls),1)
        self.assertEqual(result['limits']['total_seconds'],10800);self.assertEqual(result['limits']['subtree_RSS_gib'],10)
        self.assertEqual(result['limits']['AS_gib'],24);self.assertEqual(result['limits']['minimum_available_gib'],2)
        self.assertTrue(result['original_math_array_equality_and_31_constant_order_unchanged'])
        self.assertTrue(result['model_compiled_accepted']);self.assertIn('common-numeric-JSON',result['constant_parser_variant'])
        self.assertTrue((self.args.out/'source/private-common-numeric-json-checker.py').is_file())

    def test_math_rejection_keeps_original_closure_semantics(self):
        result=self.fake(math_bad=True);self.assertEqual(len(self.calls),1);self.assertFalse(result['model_compiled_accepted'])
        self.assertEqual(result['attempted']['native_compile'],0);self.assertTrue((self.args.out/'completion.json').is_file())

    def test_terminal_resource_rejection_revokes_acceptance_retains_actual_native_fact(self):
        result=self.fake(late=True);self.assertNotEqual(result['status'],'ok')
        self.assertTrue(result['model_compiled']);self.assertFalse(result['model_compiled_accepted'])
        self.assertEqual(result['completed'],dict(export_math=1,native_compile=1))

    def test_pure_worker_dispatch_delegates_original_finite_worker_protocol_without_actual_runtime(self):
        argv=['--worker','--out','/tmp/Alex-synthetic','--memory-gib','24']
        for name in entry.host.original.SOURCE_FIELDS:argv+=['--'+name.replace('_','-'),'/tmp/original-'+name]
        with(patch.object(entry.common_json.resource,'setrlimit')as limits,
                patch.object(entry.common_json.signal,'alarm')as alarm,
                patch.object(entry,'worker',return_value=dict(status='ok'))as worker):
            self.assertEqual(entry.main(argv),0)
        self.assertEqual(worker.call_count,1);self.assertEqual([x.args[0]for x in alarm.call_args_list],[1800,0])
        self.assertEqual(limits.call_count,2)


if __name__=='__main__':unittest.main()
