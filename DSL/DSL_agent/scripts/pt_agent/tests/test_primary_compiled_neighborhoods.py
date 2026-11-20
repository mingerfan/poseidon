import copy
import unittest

from pt_agent.experiments.primary_compiled_neighborhoods import neighborhoods


def fixture():
    values = [dict(id=str(i), kind='ciphertext', level=5, scale_log2=43,
        context='fixture', components=2, ntt=True)for i in range(8)]
    operations = [dict(kind='compute', ordinal=i+20, op='boot' if i in (2, 5) else 'add_cc',
        inputs=[str(i)]if i in (2, 5)else[str(i), str(i)], output=str(i+1), attrs={})for i in range(7)]
    return dict(values=values, external_inputs=['0'], initialization=[], execution=operations, final_outputs=['7'])


class NativeNeighborhoodTest(unittest.TestCase):
    def test_original_indices_metadata_and_operand_multiplicity(self):
        got = neighborhoods(fixture(), 2)
        self.assertEqual(got['bootstrap_execution_indices'], [2, 5])
        middle = got['windows'][1]
        self.assertEqual(middle['execution_index'], 5)
        self.assertEqual([r['execution_index']for r in middle['rows']], [3, 4, 5, 6])
        repeated = [e for e in middle['actual_SSA_edges']if e['consumer_index']==4]
        self.assertEqual([e['operand_index']for e in repeated], [0, 1])
        self.assertEqual(middle['rows'][0]['ordinal'], 23)
        self.assertFalse(got['semantic_layer_alignment'])

    def test_wrong_count_duplicate_output_or_forward_dependency_rejected(self):
        with self.assertRaises(ValueError):neighborhoods(fixture(), 3)
        for key, value in [('output', '0'), ('inputs', ['7'])]:
            p=fixture();p['execution'][0][key]=value
            with self.assertRaises(ValueError):neighborhoods(p, 2)

    def test_duplicate_value_and_missing_final_definition_rejected(self):
        p=fixture();p['values'].append(copy.deepcopy(p['values'][0]))
        with self.assertRaises(ValueError):neighborhoods(p, 2)
        p=fixture();p['final_outputs']=['999']
        with self.assertRaises(ValueError):neighborhoods(p, 2)

    def test_window_clips_at_actual_plan_boundaries(self):
        p=fixture();p['execution'][0]['op']='boot';p['execution'][-1]['op']='boot'
        got=neighborhoods(p, 4)
        self.assertEqual(len(got['windows'][0]['rows']), 3)
        self.assertEqual(len(got['windows'][-1]['rows']), 3)


if __name__ == '__main__':unittest.main()
