import ast
import unittest
from types import SimpleNamespace

from pt_agent.experiments.s3_bert_components import aggregate, category, softmax_only


class ComponentSourceTests(unittest.TestCase):
    def test_only_softmax_calls_change_and_keep_original_references(self):
        rule = ('def model(x, mask):\n'
                '    q = linear(x, w=W["same"])\n'
                '    a = softmax(q, mask=mask, steps=6)\n'
                '    z = act(a, fn="gelu", deg=63)\n'
                '    b = softmax(z, mask=mask, steps=5)\n'
                '    return b\n')
        frozen = ('def model(other, renamed):\n'
                  '    p = softmax(other, mask=renamed, steps=4, inv_method="newton")\n'
                  '    v = softmax(other, mask=renamed, steps=3, inv_method="newton")\n')
        result = softmax_only(rule, frozen)
        original = ast.parse(rule).body[0].body
        changed = ast.parse(result).body[0].body
        for i in (0, 2, 4):
            self.assertEqual(ast.dump(original[i]), ast.dump(changed[i]))
        for i, steps in ((1, 4), (3, 3)):
            before, after = original[i], changed[i]
            self.assertEqual(ast.dump(before.targets[0]), ast.dump(after.targets[0]))
            self.assertEqual(ast.dump(before.value.args[0]), ast.dump(after.value.args[0]))
            keywords = {k.arg: k.value for k in after.value.keywords}
            self.assertEqual(keywords['mask'].id, 'mask')
            self.assertEqual(keywords['steps'].value, steps)
            self.assertEqual(keywords['inv_method'].value, 'newton')

    def test_different_statement_counts_are_rejected(self):
        with self.assertRaises(ValueError):
            softmax_only('def f(x):\n    y = softmax(x)\n', 'def f(x):\n    return x\n')

    def test_mask_cannot_introduce_a_frozen_program_variable(self):
        with self.assertRaises(ValueError):
            softmax_only('def f(x):\n    y = softmax(x)\n',
                         'def f(x, unknown):\n    y = softmax(x, mask=unknown)\n')

    def test_categories_cover_both_matmuls_and_all_normalizations(self):
        self.assertEqual([category(k) for k in ('act','softmax','matmul','layernorm','rmsnorm','linear','pack')],
                         ['act','softmax','attention','norm','norm','other','other'])

    def test_each_bootstrap_is_assigned_once_and_unknown_owners_are_rejected(self):
        program = SimpleNamespace(ops=[SimpleNamespace(id=i, kind=k, lineno=i+3)
                                      for i, k in enumerate(('act','layernorm','matmul'))])
        totals, statements, inputs = aggregate(program, [None, 0, 1, 2], [0, 1, 2, 3])
        self.assertEqual(totals, dict(act=1, softmax=0, attention=1, norm=1, other=1))
        self.assertEqual(sum(row['boots'] for row in statements), 4)
        self.assertEqual(inputs, 1)
        with self.assertRaises(ValueError):
            aggregate(program, [99], [0])


if __name__ == '__main__':
    unittest.main()
