"""Precision mapping semantics; no model/constant/native execution."""
import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from pt_agent.experiments.r31_lattigo_binding import NativeBinding,map_precision_plan,dft_static_inventory
from pt_agent.experiments.r31_lattigo_runtime_plan import VARIANT,make_regression,plan_inventory,runtime_variant


class PrecisionBindingChecks(unittest.TestCase):
    def test_original_default_variant_and_metadata_contract_stay43(self):
        self.assertEqual(NativeBinding().precision_bits,43)
        self.assertEqual(runtime_variant(1),VARIANT)
        self.assertIn('q55-43x8',runtime_variant(48))
        for value in (44,49,51,True):
            with self.assertRaises(ValueError):NativeBinding(value)

    def test50_mapping_changes_only_precision_metadata_and_explicit_target(self):
        with TemporaryDirectory() as temp:
            source,bundle,_,_=make_regression(Path(temp));original=json.loads(source.read_text());before=copy.deepcopy(original)
            original['execution'][0]['attrs']['steps']=43;before=copy.deepcopy(original)
            mapped,report=map_precision_plan(original,NativeBinding(50))
            manifest=json.loads((bundle/'manifest.json').read_text())
            checked=plan_inventory(mapped,manifest,binding=NativeBinding(50))
            self.assertEqual(original,before);self.assertEqual(checked['execution_count'],11)
            self.assertEqual({v['scale_log2'] for v in mapped['values']},{50,100})
            self.assertEqual(mapped['execution'][0]['attrs']['steps'],43)
            self.assertEqual(mapped['plaintext_bundle'],original['plaintext_bundle'])
            self.assertEqual(mapped['initialization'],original['initialization'])
            self.assertEqual(mapped['target']['source_original_target'],original['target'])
            for left,right in zip(original['values'],mapped['values']):
                self.assertEqual({k:v for k,v in left.items() if k!='scale_log2'},
                                 {k:v for k,v in right.items() if k!='scale_log2'})
            with self.assertRaises(ValueError):plan_inventory(mapped,manifest)
            self.assertFalse(report['raw_data_modified']);self.assertFalse(report['levels_modified'])
            self.assertEqual(report['native_binding']['log_q'],[55]+[50]*8)

    def test_optional_s2c50_is_separate_from_original39_and_keeps_plan_arithmetic(self):
        default=NativeBinding();old50=NativeBinding(50);new=NativeBinding(50,50)
        self.assertNotIn('s2c50',default.variant(48))
        self.assertNotIn('s2c50',old50.variant(48))
        self.assertEqual(new.variant(48),old50.variant(48)+'-s2c50x3')
        field='bootstrap_slots_to_coeffs_factorization_depth_and_log_scales'
        self.assertNotIn(field,old50.metadata())
        self.assertEqual(new.metadata()[field],[[50],[50],[50]])
        for args in ((43,50),(50,39),(50,True)):
            with self.assertRaises(ValueError):NativeBinding(*args)
        with TemporaryDirectory() as temp:
            path,bundle,_,_=make_regression(Path(temp));original=json.loads(path.read_text())
            mapped_old,_=map_precision_plan(original,old50)
            mapped_new,report=map_precision_plan(original,new)
            self.assertEqual({k:v for k,v in mapped_old.items() if k!='target'},
                             {k:v for k,v in mapped_new.items() if k!='target'})
            manifest=json.loads((bundle/'manifest.json').read_text())
            self.assertEqual(plan_inventory(mapped_new,manifest,binding=new)['execution_count'],11)
            self.assertEqual(report['native_binding'][field],[[50],[50],[50]])

    def test_inverse7_changes_one_policy_and_keeps_original_default_bindings(self):
        inverse=NativeBinding(50,mod1_inv_degree=7)
        self.assertEqual(inverse.variant(32),NativeBinding(50).variant(32)+'-mod1inv7')
        self.assertEqual(inverse.metadata()['bootstrap_mod1_inv_degree'],7)
        self.assertNotIn('bootstrap_mod1_inv_degree',NativeBinding(50).metadata())
        for args in ((43,None,7),(50,50,7),(50,None,5),(50,None,True)):
            with self.assertRaises(ValueError):NativeBinding(*args)
        with TemporaryDirectory() as temp:
            path,_,_,_=make_regression(Path(temp));original=json.loads(path.read_text())
            previous,_=map_precision_plan(original,NativeBinding(50))
            candidate,_=map_precision_plan(original,inverse,bootstrap_cache=32)
            self.assertEqual({k:v for k,v in previous.items() if k!='target'},
                             {k:v for k,v in candidate.items() if k!='target'})

    def test_static_dft_inventory_counts_diagonals_without_allocating_vectors(self):
        literal=dict(BootstrappingParameters=dict(LogN=16),
            CoeffsToSlotsParameters=dict(Type=0,LogSlots=14,Format=2,BitReversed=False,
                Levels=[1,1,1,1],LevelQ=26,LevelP=3),
            SlotsToCoeffsParameters=dict(Type=1,LogSlots=14,Format=2,BitReversed=False,
                Levels=[1,1,1],LevelQ=11,LevelP=3))
        report=dft_static_inventory(literal);c2s,s2c=report['matrices']
        self.assertEqual(c2s['diagonals_per_matrix'],[16,31,15,15])
        self.assertEqual(s2c['diagonals_per_matrix'],[62,63,32])
        self.assertEqual(c2s['coefficient_bytes_per_diagonal'],65536*31*8)
        self.assertEqual(s2c['coefficient_bytes_per_diagonal'],65536*16*8)
        self.assertEqual(report['total_rns_coefficient_payload_bytes'],(77*31+157*16)*65536*8)

    def test_degree63_is_single_optional_policy_with_original30_retained(self):
        selected=NativeBinding(50,mod1_degree=63)
        self.assertEqual(selected.variant(32),NativeBinding(50).variant(32)+'-mod1degree63')
        self.assertEqual(selected.metadata()['bootstrap_mod1_degree'],63)
        self.assertNotIn('bootstrap_mod1_degree',NativeBinding(50).metadata())
        for args in ((43,None,None,63),(50,50,None,63),(50,None,7,63),(50,None,None,True),(50,None,None,62)):
            with self.assertRaises(ValueError):NativeBinding(*args)
        with TemporaryDirectory() as temp:
            path,_,_,_=make_regression(Path(temp));original=json.loads(path.read_text())
            previous,_=map_precision_plan(original,NativeBinding(50),bootstrap_cache=32)
            candidate,report=map_precision_plan(original,selected,bootstrap_cache=32)
            self.assertEqual({k:v for k,v in previous.items() if k!='target'},
                             {k:v for k,v in candidate.items() if k!='target'})
            self.assertIn('degree63',report['mathematical_equations']['bootstrap'])

    def test_message_ratio2_only_extends_inverse7_and_preserves_default8(self):
        previous=NativeBinding(50,mod1_inv_degree=7)
        candidate=NativeBinding(50,mod1_inv_degree=7,log_message_ratio=2)
        self.assertEqual(candidate.variant(32),previous.variant(32)+'-mr2')
        self.assertNotIn('bootstrap_log_message_ratio',previous.metadata())
        self.assertNotIn('bootstrap_log_message_ratio',NativeBinding().metadata())
        self.assertEqual(candidate.metadata()['bootstrap_log_message_ratio'],2)
        for kwargs in ({'precision_bits':50,'log_message_ratio':2},
                       {'precision_bits':43,'mod1_inv_degree':7,'log_message_ratio':2},
                       {'precision_bits':50,'mod1_inv_degree':7,'log_message_ratio':True},
                       {'precision_bits':50,'mod1_inv_degree':7,'log_message_ratio':8},
                       {'precision_bits':50,'mod1_degree':63,'log_message_ratio':2}):
            with self.assertRaises(ValueError):NativeBinding(**kwargs)
        with TemporaryDirectory() as temp:
            path,_,_,_=make_regression(Path(temp));source=json.loads(path.read_text())
            old,_=map_precision_plan(source,previous,bootstrap_cache=32)
            new,report=map_precision_plan(source,candidate,bootstrap_cache=32)
            self.assertEqual({k:v for k,v in old.items() if k!='target'},
                             {k:v for k,v in new.items() if k!='target'})
            self.assertIn('message-ratio2',report['mathematical_equations']['bootstrap'])


if __name__=='__main__':unittest.main()
