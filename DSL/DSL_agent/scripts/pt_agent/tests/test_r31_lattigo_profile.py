"""Metadata-only profile construction checks; no numeric/crypto dependencies."""
import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from pt_agent.experiments.r31_lattigo_profile import make_profile_plan,profile_cells,rotation_cache_inventory
from pt_agent.experiments.r31_lattigo_runtime_plan import make_regression,plan_inventory
from pt_agent.experiments.r31_lattigo_binding import NativeBinding,map_precision_plan


class ProfileConstructionChecks(unittest.TestCase):
    def test_generated_targets_cover_cells_with_separate_setup_and_one_output(self):
        with TemporaryDirectory() as temp:
            source,bundle,_,_=make_regression(Path(temp));original=json.loads(source.read_text())
            before=copy.deepcopy(original);plan,manifest,metadata=make_profile_plan(original)
            checked=plan_inventory(plan,manifest)
            self.assertEqual(original,before)
            self.assertEqual(metadata['cell_count'],len(profile_cells(original)))
            targets={row['target_ordinal'] for row in metadata['target_operations']}
            self.assertEqual(len(targets),metadata['cell_count'])
            self.assertEqual(len(plan['execution']),len(targets)+metadata['setup_and_aggregation_count'])
            self.assertEqual(checked['terminal_ciphertext_ids'],plan['final_outputs'])
            self.assertTrue(any(row['op']=='relinearize' for row in metadata['cells']))
            self.assertEqual(metadata['actual_native_executions'],0)

    def test_distinct_modulus_switch_targets_have_distinct_cells(self):
        place=dict(kind='host',rank=0)
        def value(name,level):return dict(id=name,kind='ciphertext',level=level,scale_log2=43,components=2)
        plan=dict(values=[value('x',5),value('a',3),value('b',4)],execution=[
            dict(op='mod_switch',inputs=['x'],output='a',ordinal=1,attrs=dict(target_level=3)),
            dict(op='mod_switch',inputs=['x'],output='b',ordinal=2,attrs=dict(target_level=4))])
        cells=profile_cells(plan)
        self.assertEqual(len(cells),2)
        self.assertEqual({row['attrs']['target_level'] for row in cells},{3,4})

    def test_rotation_schedule_keeps_real_order_eviction_and_logical_period(self):
        p=dict(execution=[dict(op='rotate',attrs=dict(steps=n)) for n in (1,1,-1,16383,0,16384,1)])
        one=rotation_cache_inventory(p)
        self.assertEqual((one['nonzero_rotation_calls'],one['hits'],one['misses'],one['evictions']),(5,2,3,2))
        self.assertEqual(one['zero_rotation_copies'],2)
        self.assertEqual(one['unique_normalized_steps'],2)
        self.assertIsNone(one['predicted_keygen_seconds'])
        two=rotation_cache_inventory(p,capacity=2)
        self.assertEqual((two['hits'],two['misses'],two['evictions']),(3,2,0))

    def test_explicit50_binding_preserves_default43_and_separates_profile_identity(self):
        with TemporaryDirectory() as temp:
            source,bundle,_,_=make_regression(Path(temp));original=json.loads(source.read_text())
            original_profile,_,original_meta=make_profile_plan(original)
            mapped,_=map_precision_plan(original,NativeBinding(50))
            plan,manifest,meta=make_profile_plan(mapped,binding=NativeBinding(50))
            checked=plan_inventory(plan,manifest,binding=NativeBinding(50))
            self.assertEqual(meta['cell_count'],original_meta['cell_count'])
            self.assertEqual(checked['execution_count'],len(plan['execution']))
            self.assertEqual({v['scale_log2'] for v in plan['values']},{50,100})
            self.assertEqual({v['scale_log2'] for v in original_profile['values']},{43,86})
            self.assertEqual(original_meta['native_binding']['log_default_scale'],43)
            self.assertEqual(meta['native_binding']['log_default_scale'],50)
            self.assertEqual(original_profile['plaintext_bundle']['id'],'r31-independent-native-cells')
            self.assertNotEqual(plan['plaintext_bundle']['id'],original_profile['plaintext_bundle']['id'])
            with self.assertRaises(ValueError):make_profile_plan(mapped)


if __name__=='__main__':unittest.main()
