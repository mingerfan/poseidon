"""Source and public fixture checks for explicit C2S52/P61; no Go execution."""
from argparse import Namespace
from array import array
import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from pt_agent.experiments.r31_lattigo_binding import NativeBinding
from pt_agent.experiments import r31_lattigo_profile39 as profile
from pt_agent.experiments import r31_lattigo_runtime_plan as runtime
from pt_agent.experiments.r31_lattigo_profile_reference import (
    compare_references, direct_profile_sum, replay_profile,
)
from pt_agent.tests.test_r31_lattigo_profile39 import metadata_fixture, prepared_fixture


def selected():
    return NativeBinding(50,mod1_inv_degree=15,log_message_ratio=2,
        bootstrap_normalization_log2=3,auxiliary_prime_bits=61,c2s_bits=52)


def constructor_fixture(root):
    directory,record=prepared_fixture(root)
    record.update(variant=selected().variant(32),client_cache_protocol=10,supported_c2s_bits=[52])
    record['requested_native_parameters'].update(log_p=[61]*3,bootstrap_log_p=[61]*4,
        bootstrap_coeffs_to_slots_factorization_depth_and_log_scales=[[52]]*4)
    client=record['client'];client.update(variant=selected().variant(32),c2s_bits=52,bootstrap_cache_capacity=32)
    constructor=client['bootstrap_parameters']
    primes=[(1<<61)+i for i in (1,3,5,7)]
    q=[(1<<55)+1]+[(1<<50)+i for i in range(1,17,2)]+[(1<<39)+i for i in (1,3,5)]+[(1<<60)+i for i in range(1,25,2)]+[(1<<52)+i for i in (1,3,5,7)]
    constructor['ResidualParameters'].update(P=primes[:3],Q=q[:9])
    constructor['BootstrappingParameters'].update(P=primes,Q=q)
    constructor.update(CoeffsToSlotsParameters=dict(Levels=[1]*4),SlotsToCoeffsParameters=dict(Levels=[1]*3))
    (directory/'result.json').write_text(json.dumps(record)+'\n')
    return directory,record


class C2S52Checks(unittest.TestCase):
    def test_explicit_new_binding_and_previous_defaults_are_distinct(self):
        chosen=selected();old=NativeBinding(50,mod1_inv_degree=15,log_message_ratio=2,bootstrap_normalization_log2=3)
        self.assertTrue(chosen.variant(32).endswith('-c2s52x4'))
        self.assertNotIn('c2s',old.variant(32))
        self.assertEqual(chosen.metadata()['bootstrap_coeffs_to_slots_factorization_depth_and_log_scales'],[[52]]*4)
        self.assertNotIn('bootstrap_coeffs_to_slots_factorization_depth_and_log_scales',old.metadata())
        self.assertEqual(chosen.metadata()['log_p'],[61]*3)
        self.assertEqual(NativeBinding().precision_bits,43)
        self.assertEqual(profile.binding().auxiliary_prime_bits,57)

    def test_exact_policy_rejects_other_samplers_scales_or_decompositions(self):
        fields=dict(precision_bits=50,mod1_inv_degree=15,log_message_ratio=2,
            bootstrap_normalization_log2=3,auxiliary_prime_bits=61,c2s_bits=52)
        for changes in ({'precision_bits':43},{'mod1_inv_degree':7},{'log_message_ratio':None},
            {'bootstrap_normalization_log2':2},{'auxiliary_prime_bits':57},{'c2s_bits':51},
            {'c2s_bits':True},{'s2c_bits':50},{'mod1_degree':63}):
            with self.subTest(changes=changes),self.assertRaises(ValueError):NativeBinding(**dict(fields,**changes))

    def test_prepared_capability_gate_preserves_old_binary_default(self):
        self.assertEqual(runtime.c2s_arguments({},None),[])
        for record in ({},{'client_cache_protocol':9,'supported_c2s_bits':[52]},
                       {'client_cache_protocol':10,'supported_c2s_bits':[]}):
            with self.subTest(record=record),self.assertRaises(ValueError):runtime.c2s_arguments(record,52)
        self.assertEqual(runtime.c2s_arguments({'client_cache_protocol':10,'supported_c2s_bits':[52]},52),['-c2s-bits','52'])

    def test_same39_targets_and_independent_all_slot_arithmetic(self):
        base,missing=metadata_fixture();before=copy.deepcopy((base,missing))
        plan,manifest,metadata,inventory=profile.build_profile(base,missing,c2s_bits=52)
        self.assertEqual((base,missing),before)
        self.assertEqual((metadata['cell_count'],metadata['setup_and_aggregation_count'],inventory['execution_count'],inventory['encode_count']),(39,96,135,9))
        self.assertEqual(metadata['native_binding']['log_p'],[61]*3)
        self.assertEqual(metadata['native_binding']['bootstrap_coeffs_to_slots_factorization_depth_and_log_scales'],[[52]]*4)
        self.assertIn('c2s52',manifest['bundle_id']);self.assertTrue(plan['target']['adapter_variant'].endswith('-c2s52x4'))
        values=profile.deterministic_fixture()
        constants={row['content']:array('d',[row['value']])*profile.SLOTS for row in metadata['constant_vectors']}
        observed,summary=replay_profile(plan,constants,values)
        expected=direct_profile_sum(metadata,values,precision_bits=50)
        self.assertEqual(summary['execution_completed'],135)
        self.assertTrue(compare_references(expected,observed)['passed'])

    def test_exact_public_constructor_gate_and_foreign_group_rejection(self):
        with TemporaryDirectory() as temp:
            _,record=constructor_fixture(Path(temp))
            self.assertTrue(profile.validate_prepared(record,c2s_bits=52)['actual_constructor_validated'])
            for change in (lambda r:r.update(client_cache_protocol=9),
                lambda r:r['client'].update(c2s_bits=0),
                lambda r:r['client'].update(variant=profile.binding().variant(32)),
                lambda r:r['client']['bootstrap_parameters']['CoeffsToSlotsParameters'].update(Levels=[1]*3),
                lambda r:r['client']['bootstrap_parameters']['BootstrappingParameters']['Q'].__setitem__(-1,(1<<56)+1),
                lambda r:r['requested_native_parameters'].update(log_p=[57]*3)):
                bad=copy.deepcopy(record);change(bad)
                with self.subTest(change=change),self.assertRaises(ValueError):profile.validate_prepared(bad,c2s_bits=52)
            with self.assertRaises(ValueError):profile.validate_prepared(record)

    def test_actual_P61_nonprefix_subset_is_preserved(self):
        with TemporaryDirectory() as temp:
            _,record=constructor_fixture(Path(temp))
            raw=record['client']['bootstrap_parameters'];primes=raw['BootstrappingParameters']['P']
            raw['ResidualParameters']['P']=[primes[i] for i in (0,1,3)]
            observed=profile.validate_prepared(record,c2s_bits=52)
            self.assertEqual(observed['residual_P'],[primes[i] for i in (0,1,3)])
            self.assertEqual(observed['bootstrap_P'],primes)

    def test_selected_P61_subset_rejects_foreign_or_changed_P0(self):
        with TemporaryDirectory() as temp:
            _,record=constructor_fixture(Path(temp))
            raw=record['client']['bootstrap_parameters'];primes=raw['BootstrappingParameters']['P']
            for residue in ([primes[0],primes[1],(1<<61)+1001],
                            [primes[1],primes[0],primes[3]]):
                bad=copy.deepcopy(record)
                bad['client']['bootstrap_parameters']['ResidualParameters']['P']=residue
                with self.subTest(residue=residue),self.assertRaises(ValueError):
                    profile.validate_prepared(bad,c2s_bits=52)

    def test_execution_argv_keeps_new_group_and_old_P57_default(self):
        source=dict(output_plan='plan',output_bundle='bundle',input='input',expected='expected')
        old=profile.execution_command(source,out=Path('new'),prepared=Path('prepared'))
        new=profile.execution_command(source,out=Path('new'),prepared=Path('prepared'),c2s_bits=52)
        self.assertNotIn('--c2s-bits',old)
        self.assertEqual(old[old.index('--auxiliary-prime-bits')+1],'57')
        self.assertEqual(new[new.index('--auxiliary-prime-bits')+1],'61')
        self.assertEqual(new[new.index('--c2s-bits')+1],'52')
        self.assertEqual(new[new.index('--bootstrap-cache')+1],'32')

    def test_runtime_CLI_passes_explicit_value_without_starting_child(self):
        with patch.object(runtime,'execute',return_value=dict(status='ok',error=None)) as run:
            code=runtime.main(['prepare','--out','unused','--plan','plan','--bundle','bundle',
                '--c2s-bits','52','--precision-bits','50','--mod1-inv-degree','15',
                '--log-message-ratio','2','--bootstrap-normalization-log2','3'])
        self.assertEqual(code,0);self.assertEqual(run.call_args.args[0].c2s_bits,52)

    def test_new_profile_preparation_is_public_only_and_persisted(self):
        with TemporaryDirectory() as temp:
            root=Path(temp);prepared,_=constructor_fixture(root);base,missing=metadata_fixture()
            (root/'base.json').write_text(json.dumps(base));(root/'missing.json').write_text(json.dumps(missing))
            args=Namespace(prepared=prepared,base_metadata=root/'base.json',missing_metadata=root/'missing.json',
                out=root/'profile',native_out=root/'future',c2s_bits=52)
            with patch.object(profile.shutil,'disk_usage',return_value=Namespace(free=20*1024**3)):
                result=profile.prepare(args)
            self.assertEqual(result['status'],'prepared',result.get('error'))
            self.assertIn('c2s52',result['mode']);self.assertEqual(result['execution_count'],135)
            self.assertIn('--c2s-bits',result['future_native_argv'])
            self.assertEqual(result['native_executions'],0);self.assertEqual(result['keygen_runs'],0)
            self.assertEqual(result['artifact_bytes'],sum(x.stat().st_size for x in args.out.rglob('*') if x.is_file()))


if __name__=='__main__':unittest.main()
