"""Pure metadata exact-key/order checks; no numerical imports."""
import unittest

from pt_agent.experiments.qwen_grouped_ids import GroupedIds


class GroupedIdsChecks(unittest.TestCase):
    def test_shared_operands_distinct_groups_and_unbounded_integer_keys(self):
        operands = (('cipher', 2**100), ('plain', 'k123'))
        keys = [('add', operands, None), ('sub', operands, None),
                ('rot', operands, 0), ('rot', operands, 2**100), ('rot', operands, None)]
        original, grouped = {}, GroupedIds()
        for index, key in enumerate(keys):
            original[key] = index; grouped[key] = index
        equal_operands = tuple(list(operands))
        grouped[('add', equal_operands, None)] = 99
        original[('add', equal_operands, None)] = 99
        self.assertEqual(list(grouped.items()), list(original.items()))
        self.assertEqual(len(grouped._groups), 5)
        self.assertTrue(all(row is operands for row in grouped._order_operands))

    def test_interleaved_key_domain_matches_dict_and_borrows_operands(self):
        rows = [(('cipher', 3), ('plain', 'k0')), (('cipher', 4),),
                (('plain', 'k1'), ('cipher', 3)), (), (('cipher', 3), ('cipher', 3))]
        keys = [('add', rows[0], None), ('rot', rows[1], 7), ('add', rows[2], None),
                ('input', rows[3], None), ('mul_cc', rows[4], None)]
        expected, grouped = {}, GroupedIds()
        for index, key in enumerate(keys):
            expected[key] = index; grouped[key] = index
        self.assertEqual(list(grouped), list(expected))
        self.assertEqual(list(grouped.items()), list(expected.items()))
        self.assertEqual(dict(grouped), expected); self.assertEqual(len(grouped), len(expected))
        for key, saved in zip(keys, grouped._order_operands):
            self.assertIs(saved, key[1]); self.assertEqual(grouped[key], expected[key])
            self.assertIn(key, grouped)
        self.assertIs(grouped._order_operands[0], next(iter(grouped._groups[('add', None)])))

    def test_update_delete_reinsert_and_missing_match_dict(self):
        keys = [('add', (('cipher', 2),), None), ('rot', (('cipher', 7),), 3),
                ('add', (('cipher', 9),), None)]
        old, new = dict.fromkeys(keys, 1), GroupedIds()
        new.update(old)
        for mapping in (old, new):
            mapping[keys[0]] = 5; del mapping[keys[1]]; mapping[keys[1]] = 6
        self.assertEqual(list(new.items()), list(old.items()))
        self.assertEqual(new._order_operands[1], None)
        missing = ('sub', (('plain', 'k404'),), None)
        self.assertNotIn(missing, new)
        with self.assertRaises(KeyError) as read:
            new[missing]
        self.assertEqual(read.exception.args, (missing,))
        with self.assertRaises(KeyError):
            del new[missing]
        self.assertEqual(new.pop(keys[0]), old.pop(keys[0]))
        self.assertEqual(dict(new), old)

    def test_order_iterator_update_and_structural_fail_fast(self):
        a, b = ('add', (('cipher', 1),), None), ('rot', (('cipher', 2),), 1)
        grouped = GroupedIds(); grouped[a] = 1
        iterator = iter(grouped); grouped[a] = 2
        self.assertEqual(next(iterator), a)
        iterator = iter(grouped); grouped[b] = 3
        with self.assertRaises(RuntimeError):
            next(iterator)
        iterator = iter(grouped); del grouped[a]
        with self.assertRaises(RuntimeError):
            next(iterator)

    def test_clear_releases_all_owned_containers_and_retained_bucket(self):
        grouped = GroupedIds(); grouped[('add', (('cipher', 12345),), None)] = 2
        bucket = next(iter(grouped._groups.values()))
        operands, groups, keys = grouped._order_operands, grouped._order_groups, grouped._group_keys
        iterator = iter(grouped); grouped.clear()
        self.assertEqual((len(grouped), len(bucket), len(operands), len(groups), len(keys)), (0,0,0,0,0))
        self.assertEqual(grouped._groups, {})
        with self.assertRaises(RuntimeError):
            next(iterator)
        grouped[('rot', (('cipher', 3),), 2)] = 8
        self.assertEqual(list(grouped.values()), [8])


if __name__ == '__main__':
    unittest.main()
