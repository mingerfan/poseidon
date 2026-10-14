"""Exercise trusted upstream metaclass dispatch with a recording C-ABI double.

This is an operand/alias unit test, not a real tracing or encrypted execution.
Only selected definitions from the repository dependency are executed; candidate
Python, keys and provider APIs are never involved.
"""
import ast
import inspect
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest

SOURCE=Path(__file__).resolve().parents[2]/'third_party/dacapo/python/hecate/hecate/expr.py'


def frontend_double():
    tree=ast.parse(SOURCE.read_text())
    selected=[]
    for node in tree.body:
        if isinstance(node,ast.ClassDef) and node.name in ('hecateMetaBase','hecateMetaBinary','Expr'):
            selected.append(node)
        elif isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id in
                ('toBinary','toInnerUnary') for t in node.targets):
            selected.append(node)
    namespace=dict(inspect=inspect,ctxt=None)
    namespace['lt']=SimpleNamespace(
        createBinary=lambda ctxt,op,left,right,*loc: ('binary',op,left,right),
        createUnary=lambda ctxt,op,value,*loc: ('unary',op,value),
        createRotation=lambda ctxt,value,step,*loc: ('rotate',value,step))
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(SOURCE),'exec'),namespace)
    expr=namespace['Expr']
    namespace['resolveType']=lambda value: value if isinstance(value,expr) else expr(('plain',value))
    return expr


class AugmentedFrontendTests(unittest.TestCase):
    def test_forward_and_reverse_subtraction_have_distinct_order(self):
        expr=frontend_double()
        x,y=expr('x'),expr('y')
        self.assertEqual((x-y).obj,('binary',7,'x','y'))
        self.assertEqual((.25-x).obj,('binary',7,('plain',.25),'x'))

    def test_augmented_subtraction_preserves_left_right_order(self):
        expr=frontend_double()
        x,y=expr('x'),expr('y')
        x-=y
        self.assertEqual(x.obj,('binary',7,'x','y'))

    def test_augmented_public_subtraction_does_not_reverse(self):
        expr=frontend_double()
        x=expr('x')
        x-=.25
        self.assertEqual(x.obj,('binary',7,'x',('plain',.25)))

    def test_augmented_rebinding_does_not_mutate_an_existing_alias(self):
        expr=frontend_double()
        x=expr('x'); alias=x
        x-=.25
        self.assertIsNot(x,alias)
        self.assertEqual(alias.obj,'x')

    def test_add_multiply_and_chains_preserve_python_operand_order(self):
        expr=frontend_double()
        x=expr('x'); y=expr('y')
        x+=y
        self.assertEqual(x.obj,('binary',6,'x','y'))
        x*=.5
        self.assertEqual(x.obj,('binary',8,('binary',6,'x','y'),('plain',.5)))

    def test_same_operand_and_negate_are_not_rewritten(self):
        expr=frontend_double(); x=expr('x')
        x-=x
        self.assertEqual(x.obj,('binary',7,'x','x'))
        self.assertEqual((-x).obj,('unary',13,('binary',7,'x','x')))

    def test_legacy_agent_contract_is_not_silently_opened(self):
        from hecate_contract import validate_function
        source='@hc.func("c")\ndef golden(x):\n    x -= c0\n    return x\n'
        with self.assertRaises(ValueError):
            validate_function(source,{'c0':.25},contract='hecate-function-v1')


@unittest.skipUnless(os.environ.get('POSEIDON_AUGMENTED_BEFORE') and os.environ.get('POSEIDON_AUGMENTED_AFTER'),
                     'requires actual before/after frontend compiler probes')
class AugmentedCompilerEvidenceTests(unittest.TestCase):
    def test_actual_compiles_hashes_and_fixed_equivalence_without_execution_claim(self):
        for variable,expected in (('POSEIDON_AUGMENTED_BEFORE','failed'),('POSEIDON_AUGMENTED_AFTER','passed')):
            root=Path(os.environ[variable])
            report=json.loads((root/'report.json').read_text())
            self.assertEqual(report['status'],expected)
            self.assertEqual(report['agent_calls'],0)
            self.assertFalse(report['encrypted_execution'])
            self.assertEqual([c['operation'] for c in report['cases']],['add','subtract','multiply'])
            for case in report['cases']:
                for style in ('direct','augmented'):
                    row=case[style]
                    self.assertEqual((row['trace_exit'],row['compile_exit']),(0,0))
                    self.assertIn('--verify-each',row['compile_command'])
                    for filename,key in (('lowered._hecate_golden.hevm','hevm_sha256'),
                                         ('_hecate_golden.cst','cst_sha256')):
                        raw=(root/(case['operation']+'-'+style)/filename).read_bytes()
                        self.assertEqual(hashlib.sha256(raw).hexdigest(),row[key])
                if expected=='passed':
                    self.assertTrue(case['byte_identical_artifacts'])
                    self.assertEqual(case['direct']['hevm_sha256'],case['augmented']['hevm_sha256'])
                    self.assertEqual(case['direct']['cst_sha256'],case['augmented']['cst_sha256'])
            if expected=='failed':
                self.assertFalse(report['cases'][1]['byte_identical_artifacts'])
            else:
                self.assertEqual(report['frontend_source_sha256'],hashlib.sha256(SOURCE.read_bytes()).hexdigest())


if __name__=='__main__':
    unittest.main()
