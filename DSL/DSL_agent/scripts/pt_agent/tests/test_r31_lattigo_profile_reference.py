"""Tiny deterministic clear checks for independent profile reference preparation."""
from array import array
import unittest

from pt_agent.experiments.r31_lattigo_binding import NativeBinding
from pt_agent.experiments.r31_lattigo_profile import make_profile_plan
from pt_agent.experiments.r31_lattigo_profile_reference import (
    _operation, compare_references, direct_profile_sum, replay_profile)


class ProfileReferenceChecks(unittest.TestCase):
    def test_nonzero_rotation_sign_and_clear_operations(self):
        x=array('d',[1.,-2.,3.,-4.])
        self.assertEqual(list(_operation('rotate',x,attrs={'steps':1})),[-2.,3.,-4.,1.])
        self.assertEqual(list(_operation('rotate',x,attrs={'steps':-1})),[-4.,1.,-2.,3.])
        self.assertEqual(list(_operation('mul_cc',x,x)),[1.,4.,9.,16.])
        copied=_operation('boot',x)
        copied[0]=7.
        self.assertEqual(x[0],1.)

    def test_direct_cells_match_setup_last_use_and_aggregation(self):
        def value(name,level,scale=50,components=2,kind='ciphertext'):
            return dict(id=name,context='profile-test',level=level,scale_log2=scale,
                components=components,kind=kind)
        source=dict(target={'target_id':'test'},values=[value('x',5),value('m',5,100,3),
            value('r',5,100),value('s',4),value('d',1),value('b',5),value('z',5)],
            execution=[dict(ordinal=1,op='mul_cc',inputs=['x','x'],output='m'),
                dict(ordinal=2,op='relinearize',inputs=['m'],output='r'),
                dict(ordinal=3,op='rescale',inputs=['r'],output='s',attrs={'target_level':4,'target_scale_log2':50}),
                dict(ordinal=4,op='mod_switch',inputs=['s'],output='d',attrs={'target_level':1}),
                dict(ordinal=5,op='boot',inputs=['d'],output='b',attrs={'target_level':5}),
                dict(ordinal=6,op='rotate',inputs=['b'],output='z',attrs={'steps':-1})])
        binding=NativeBinding(50,mod1_inv_degree=15,log_message_ratio=2,bootstrap_normalization_log2=3)
        plan,manifest,metadata=make_profile_plan(source,binding=binding)
        packed=array('d',[.25,-.5,.75,-1.])
        constants={x['content']:array('d',[x['value']])*4 for x in metadata['constant_vectors']}
        observed,summary=replay_profile(plan,constants,packed)
        expected=direct_profile_sum(metadata,packed,precision_bits=50)
        self.assertTrue(compare_references(expected,observed)['passed'])
        self.assertEqual(summary['execution_completed'],len(plan['execution']))
        self.assertEqual(metadata['cell_count'],6)
        self.assertEqual(metadata['actual_native_executions'],0)
        self.assertLessEqual(summary['peak_vector_payload_bytes'],len(plan['values'])*4*8)

    def test_original_math_criterion_and_nonfinite_rejection(self):
        self.assertFalse(compare_references([0.],[1e-11])['passed'])
        self.assertTrue(compare_references([1.],[1.+1e-13])['passed'])
        with self.assertRaises(ValueError):compare_references([0.],[float('nan')])
        with self.assertRaises(ValueError):compare_references([0.],[])


if __name__=='__main__':unittest.main()
