"""R8h2 confirmation deduplication with fixed fake numeric results."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pt_agent.dsl.parser import parse
from pt_agent.dsl.printer import print_program
from pt_agent.planners.confirm import Candidate, MAX_CONFIRM_EVALS, confirm_select


def source(name='net'):
    return ('@fhe.program(slots=16384)\ndef '+name+'(v: Cipher[1, 2]):\n'
            '    x = pack(v, layout="flat")\n'
            '    y = act(x, fn="relu", method="cheb", deg=15, domain=(-8.0, 8.0))\n'
            '    return y\n')


def validation(drop=1.):
    return dict(top1=90.-drop, exact_top1=90., acc_drop_pp=drop, max_u=.8,
                diverged=False, n_diverged=0, boot_input_bound=None, boot_input_max_abs=0.)


class ConfirmDedupTests(unittest.TestCase):
    def confirm(self, pool, results=None):
        with patch('pt_agent.planners.confirm.validate',
                   side_effect=results if results is not None else lambda *_args, **_kwargs: validation()) as validate:
            result = confirm_select(pool, SimpleNamespace(val_split='test[8000:9000]'))
        for call in validate.call_args_list:
            self.assertEqual(call.kwargs, dict(stop_on_divergence=False))
        return result, validate

    def test_01_duplicate_origins_keep_cheapest_and_pool_order_ties(self):
        original = Candidate(source(), 'llm', 1, 20., 2)
        cheaper = Candidate(source(), 'tuned', 2, 10., 1)
        other = Candidate(source('other'), 'llm', 3, 15., 2)
        result, validate = self.confirm([original, cheaper, other])
        self.assertEqual(validate.call_count, 2)
        self.assertEqual(result['deduplicated'], dict(before=3, after=2))
        self.assertEqual(result['duplicates'], [dict(iteration=1, origin='llm', kept_iteration=2, kept_origin='tuned')])
        self.assertEqual([(r['iteration'], r['origin']) for r in result['evals']], [(2, 'tuned'), (3, 'llm')])
        first_in_pool = Candidate(source(), 'tuned', 9, 10., 1)
        later = Candidate(source(), 'llm', 1, 10., 1)
        result, validate = self.confirm([first_in_pool, later], [validation(0.)])
        self.assertIs(result['selected'], first_in_pool)
        self.assertEqual(validate.call_count, 1)
        self.assertEqual(result['duplicates'][0]['iteration'], 1)

    def test_02_every_distinct_program_below_limit_is_checked(self):
        self.assertEqual(MAX_CONFIRM_EVALS, 6)
        pool = [Candidate(source('net'+str(i)), 'llm', i+1, float(i), i) for i in range(5)]
        pool.insert(1, Candidate(pool[0].source, 'tuned', 8, 1., 1))
        pool.insert(3, Candidate(pool[2].source, 'tuned', 9, 2., 2))
        result, validate = self.confirm(pool)
        self.assertEqual(validate.call_count, 5)
        self.assertEqual(result['deduplicated'], dict(before=7, after=5))
        self.assertEqual([call.args[0].name for call in validate.call_args_list], ['net'+str(i) for i in range(5)])
        self.assertIsNone(result['selected'])
        self.assertEqual(len(result['duplicates']), 2)

    def test_03_distinct_pool_preserves_original_order_and_results(self):
        slow = Candidate(source('slow'), 'llm', 1, 30., 3)
        tied_later = Candidate(source('later'), 'llm', 3, 10., 1)
        tied_tuned = Candidate(source('tuned'), 'tuned', 2, 10., 1)
        tied_llm = Candidate(source('llm'), 'llm', 2, 10., 1)
        result, validate = self.confirm([slow, tied_later, tied_tuned, tied_llm],
                                        [validation(1.), validation(0.)])
        self.assertIs(result['selected'], tied_tuned)
        self.assertEqual([call.args[0].name for call in validate.call_args_list], ['llm', 'tuned'])
        self.assertEqual(result['deduplicated'], dict(before=4, after=4))
        self.assertEqual(result['duplicates'], [])
        expected = [dict(iteration=2, origin='llm', latency_pred_us=10., boots_pred=1,
                         **{k: validation(1.)[k] for k in ('top1', 'exact_top1', 'acc_drop_pp', 'max_u', 'diverged', 'n_diverged')}, passed=False),
                    dict(iteration=2, origin='tuned', latency_pred_us=10., boots_pred=1,
                         **{k: validation(0.)[k] for k in ('top1', 'exact_top1', 'acc_drop_pp', 'max_u', 'diverged', 'n_diverged')}, passed=True)]
        self.assertEqual(result['evals'], expected)

    def test_04_whitespace_and_indentation_share_a_key(self):
        original = source()
        changed = '\n\n'+original.replace('    ', '        ')+'\n'
        self.assertEqual(print_program(parse(original)), print_program(parse(changed)))
        first = Candidate(original, 'llm', 1, 10., 1)
        second = Candidate(changed, 'tuned', 2, 10., 1)
        result, validate = self.confirm([first, second], [validation(0.)])
        self.assertEqual(validate.call_count, 1)
        self.assertIs(result['selected'], first)
        self.assertEqual(first.source, original)
        self.assertEqual(second.source, changed)
        self.assertEqual(result['deduplicated'], dict(before=2, after=1))
        self.assertEqual(result['duplicates'], [dict(iteration=2, origin='tuned', kept_iteration=1, kept_origin='llm')])


if __name__ == '__main__':
    unittest.main()
