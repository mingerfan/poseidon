"""Trusted tiny emitter literals and metadata-only worker/supervision fixtures."""
import ast
from contextlib import redirect_stderr
import io
import json
import math
import os
from pathlib import Path
import re
import resource
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from pt_agent.experiments import heir_cnn_export as exporter
from pt_agent.experiments import heir_vgg_host_cpu_overlap_rss12_json as entry
from pt_agent.tests import test_heir_vgg_host_cpu_overlap_rss12 as fixtures


def graph_and_text(shape=(2,3)):
    weight=np.arange(math.prod(shape),dtype=np.float64).reshape(shape)/8
    bias=np.zeros(shape[0],dtype=np.float64)
    graph=SimpleNamespace(nodes=[SimpleNamespace(id='linear',op='linear',attrs=dict(weight='W',bias='b'))],
        weights=dict(W=weight,b=bias))
    emitter=exporter.Emitter();emitter.constant(weight);emitter.constant(bias)
    return graph,'\n'.join(emitter.lines)


class TinyLiteralChecks(unittest.TestCase):
    def test_original_scalar_literals_float64_negative_zero_tiny_and_large_roundtrip(self):
        values=[0.,-0.,1.,-1.,np.nextafter(0.,1.),np.finfo(np.float64).tiny,
            np.finfo(np.float64).max,-np.finfo(np.float64).max,1e-200,-1e200]
        for value in values:
            with self.subTest(value=float(value)):
                literal=exporter.scalar_literal(value);decoded=entry.numeric_json_literal(literal)
                self.assertEqual(decoded,ast.literal_eval(literal));self.assertEqual(decoded,float(value))
                self.assertEqual(np.signbit(decoded),np.signbit(value))

    def test_actual_original_nested_emitter_rectangular_constants_match_original_checker_rows(self):
        graph,text=graph_and_text();old=entry.math_check.check_constants(graph,text)
        new=entry.constant_checker()(graph,text)
        self.assertEqual(new,old);self.assertEqual([row['name']for row in new],['W','b'])
        self.assertTrue(all(row['numerically_equal']and row['passed']for row in new))

    def test_boolean_null_string_object_nonfinite_and_python_expressions_rejected(self):
        for text in('true','[true]','[false]','null','[null]','"1.0"','["1"]','{"x":1}',
                'NaN','[NaN]','Infinity','[-Infinity]','None','[1]*2','(1,2)','__import__("os")'):
            with self.subTest(text=text):
                with self.assertRaises(ValueError):entry.numeric_json_literal(text)

    def test_original_checker_rejects_overflow_nonfinite_literal(self):
        graph,text=graph_and_text();bad=text.replace('0.00000000000000000e+00','1.00000000000000000e+400',1)
        with self.assertRaises(ValueError):entry.constant_checker()(graph,bad)

    def test_original_checker_rejects_wrong_shape_inventory_and_even_subtolerance_value_change(self):
        graph,text=graph_and_text();checker=entry.constant_checker()
        with self.assertRaises(ValueError):checker(graph,text.splitlines()[0])
        emitter=exporter.Emitter();emitter.constant(graph.weights['W'].reshape(-1));emitter.constant(graph.weights['b'])
        with self.assertRaises(ValueError):checker(graph,'\n'.join(emitter.lines))
        changed=text.replace('0.00000000000000000e+00','1.00000000000000003e-13',1)
        with self.assertRaises(ValueError):checker(graph,changed)

    def test_original_check_function_code_defaults_and_all_other_globals_unchanged(self):
        original=entry.math_check.check_constants;checker=entry.constant_checker()
        self.assertIs(checker.__code__,original.__code__);self.assertEqual(checker.__defaults__,original.__defaults__)
        self.assertEqual(checker.__kwdefaults__,original.__kwdefaults__)
        self.assertEqual(set(checker.__globals__),set(original.__globals__))
        changed=[name for name,value in original.__globals__.items()if checker.__globals__[name]is not value]
        self.assertEqual(changed,['ast']);self.assertIs(checker.__globals__['check_math'],entry.math_check.check_math)
        self.assertIs(checker.__globals__['ast'].literal_eval,entry.numeric_json_literal)

    def test_runtime_only_replaces_checker_and_numeric_worker_only_runtime_global(self):
        original_checker=entry.math_check.check_constants;sentinel=object();api=dict(check_constants=original_checker,other=sentinel)
        with patch.object(entry.original,'_runtime',return_value=api):observed=entry._runtime()
        self.assertIs(observed,api);self.assertIs(observed['other'],sentinel)
        self.assertIs(observed['check_constants'].__code__,original_checker.__code__)
        worker=entry.numeric_worker();self.assertIs(worker.__code__,entry.original.worker.__code__)
        changes=[name for name,value in vars(entry.original).items()if worker.__globals__[name]is not value]
        self.assertEqual(changes,['_runtime']);self.assertIs(worker.__globals__['_runtime'],entry._runtime)

    def test_pure_worker_finite_limits_alarm_cleanup_and_explicit_paths_without_runtime_call(self):
        args=['--worker','--out','/tmp/synthetic-worker','--memory-gib','24']
        for name in entry.original.SOURCE_FIELDS:args+=['--'+name.replace('_','-'),'/tmp/original-'+name]
        with(patch.object(entry.resource,'setrlimit')as limit,patch.object(entry.signal,'alarm')as alarm,
                patch.object(entry,'worker',side_effect=ValueError('synthetic worker failure'))as worker):
            with self.assertRaisesRegex(ValueError,'synthetic'):entry.worker_main(args)
        self.assertEqual(limit.call_args_list[0].args,(resource.RLIMIT_AS,(24*entry.original.GIB,24*entry.original.GIB)))
        self.assertEqual(limit.call_args_list[1].args,(resource.RLIMIT_FSIZE,(16*entry.original.GIB,16*entry.original.GIB)))
        self.assertEqual([call.args[0]for call in alarm.call_args_list],[1800,0])
        self.assertEqual(worker.call_args.args[0].memory_gib,24)
        with redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):entry.worker_parser().parse_args(args+['--registry','/tmp/disallowed'])


class JSONSupervisionChecks(unittest.TestCase):
    def setUp(self):
        fixtures.RSS12Checks.setUp(self);self.args.out=self.root/(entry.OUT_PREFIX+'fixture')

    def fake(self,**options):
        namespace=dict(vars(fixtures),entry=entry)
        return entry.host.overlap._clone_plain(fixtures.RSS12Checks.fake,namespace)(self,**options)

    def test_real_worker_command_targets_private_JSON_module_original_fields_and_RSS12_scope(self):
        result=self.fake();self.assertEqual(result['status'],'ok',result['error'])
        command=self.calls[0][0];self.assertEqual(command[:4],[str(entry.host.host_python()),'-B','-m',entry.MODULE])
        for name,path in self.source.items():self.assertEqual(command[command.index('--'+name.replace('_','-'))+1],str(path))
        self.assertEqual(result['limits']['subtree_RSS_gib'],12);self.assertEqual(result['limits']['AS_gib'],24)
        self.assertEqual(result['limits']['minimum_available_gib'],2);self.assertEqual(result['limits']['total_seconds'],9600)
        self.assertTrue(result['original_constant_check_code_unchanged']);self.assertTrue(result['original_math_and_array_equality_unchanged'])
        self.assertTrue(result['model_compiled_accepted']);self.assertIn('JSON',result['constant_parser_variant'])
        self.assertTrue((self.args.out/'source/heir-vgg-host-cpu-overlap-rss12-json.py').is_file())

    def test_late_RSS_rejects_acceptance_with_original_native_fact_and_completion(self):
        result=self.fake(terminal_rss=True);self.assertNotEqual(result['status'],'ok')
        self.assertTrue(result['model_compiled']);self.assertFalse(result['model_compiled_accepted'])
        self.assertEqual(result['completed'],dict(export_math=1,native_compile=1))
        self.assertTrue((self.args.out/'completion.json').is_file())


class MediumLiteralOnly(unittest.TestCase):
    def test_synthetic_256x256_JSON_checker_without_any_AST_literal_parse(self):
        graph,text=graph_and_text((256,256))
        with patch.object(ast,'literal_eval',side_effect=AssertionError('No AST parser in JSON variant')):
            rows=entry.constant_checker()(graph,text)
        self.assertTrue(all(row['passed']and row['numerically_equal']for row in rows))
        print(json.dumps(dict(synthetic_shape=[256,256],synthetic_weight_elements=65536,
            literal_text_bytes=len(text.encode()),checked_constants=len(rows),
            process_ru_maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            model_reads=0,AST_literal_parse_calls=0)))


if __name__=='__main__':unittest.main()
