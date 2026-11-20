"""Public39-cell protocol/reference tests; no native binary is executed."""
from argparse import Namespace
from array import array
import copy
import json
from pathlib import Path
import struct
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from pt_agent.experiments import r31_lattigo_profile39 as profile
from pt_agent.experiments.r31_lattigo_profile_reference import (
    compare_references,direct_profile_sum,replay_profile,
)


def metadata_fixture():
    base=dict(cell_count=36,cells=copy.deepcopy(profile._canonical_base()))
    missing=[]
    for original in profile.EXPECTED_CELLS:
        row=copy.deepcopy(original)
        row['representative_attrs']=dict(steps=1023) if row['op']=='rotate' else copy.deepcopy(row['attrs'])
        missing.append(row)
    return base,dict(cell_count=3,cells=missing)


def prepared_fixture(root):
    directory=root/'prepared';directory.mkdir();binary=directory/'never-executed-fixture-client'
    binary.write_text('Filesystem fixture only; no executable invocation\n')
    primes=[(1<<57)+n for n in (1,3,5,7)]
    constructor=dict(ResidualParameters=dict(LogN=16,LogDefaultScale=50,P=primes[:3]),
        BootstrappingParameters=dict(LogN=16,LogDefaultScale=50,P=primes,Xs=dict(Type='Ternary',H=192),
            Xe=dict(Type='DiscreteGaussian',Sigma=3.2)),Mod1ParametersLiteral=dict(Mod1InvDegree=15,
                LogMessageRatio=2,Mod1Degree=30,DoubleAngle=3,K=16))
    record=dict(status='ok',phase='prepare',variant=profile.binding().variant(32),
        client_cache_protocol=9,supported_auxiliary_prime_bits=[57,61],ready_binary=str(binary),
        requested_native_parameters=dict(log_p=[57]*3,bootstrap_log_p=[57]*4,actual_constructor_primes_pending=False),
        client=dict(status='ok',mode='parameters',precision_bits=50,bootstrap_normalization_log2=3,
            bootstrap_input_divisor=8,keygen_api_attempts=0,encryption_attempts=0,compute_attempts=0,
            final_decryption_attempts=0,bootstrap_parameters=constructor),
        cache_selftest=dict(checkpoint_policy=dict(unit='existing_save_request',interval=1,
            forced_requests=1,writes_completed=1,write_errors=0)))
    (directory/'result.json').write_text(json.dumps(record)+'\n')
    return directory,record


class Profile39Checks(unittest.TestCase):
    def test_exact39_union_preserves_sources_and_all_three_additions(self):
        base,missing=metadata_fixture();before=copy.deepcopy((base,missing))
        cells=profile.select_cells(base,missing)
        self.assertEqual(len(cells),39);self.assertEqual((base,missing),before)
        self.assertEqual(len({profile._key(x) for x in cells}),39)
        self.assertEqual(sum(x['op']=='boot' for x in cells),1)
        self.assertEqual(sum(x['op']=='rotate' for x in cells),5)

    def test_changed_cell_duplicate_missing_or_representative_rotation_is_rejected(self):
        changes=(lambda b,m:m['cells'].pop(),
            lambda b,m:m['cells'].__setitem__(0,copy.deepcopy(b['cells'][0])),
            lambda b,m:m['cells'][0]['output'].update(level=2),
            lambda b,m:next(x for x in b['cells'] if x['op']=='rotate')['representative_attrs'].update(steps=-4),
            lambda b,m:b['cells'][0]['inputs'][0].update(level=True))
        for change in changes:
            base,missing=metadata_fixture();change(base,missing)
            with self.subTest(change=change),self.assertRaises(ValueError):profile.select_cells(base,missing)

    def test135_operations39_targets96_setup9_plaintexts_and_new_group(self):
        plan,manifest,metadata,inventory=profile.build_profile(*metadata_fixture())
        self.assertEqual((metadata['cell_count'],metadata['setup_and_aggregation_count'],
            inventory['execution_count'],inventory['encode_count']),(39,96,135,9))
        self.assertEqual(inventory['op_counts'],profile.EXPECTED_OPS)
        self.assertEqual(inventory['bootstrap_count'],1)
        self.assertEqual(metadata['native_binding']['log_p'],[57]*3)
        self.assertEqual(metadata['native_binding']['bootstrap_log_p'],[57]*4)
        self.assertIn('-p57x3-',plan['target']['adapter_variant'])
        self.assertTrue(plan['target']['adapter_variant'].endswith('-bootp57x4'))
        self.assertEqual((metadata['bootstrap_cache_capacity'],metadata['residual_cache_capacity']),(32,1))
        self.assertEqual(len(metadata['target_operations']),39)
        self.assertEqual(len(set(row['target_ordinal'] for row in metadata['target_operations'])),39)
        self.assertFalse(metadata['old_target_durations_reused'])

    def test_all16384_public_slots_three_independent_arithmetic_paths(self):
        plan,_,metadata,_=profile.build_profile(*metadata_fixture())
        packed=profile.deterministic_fixture()
        constants={row['content']:array('d',[row['value']])*profile.SLOTS for row in metadata['constant_vectors']}
        replay,summary=replay_profile(plan,constants,packed)
        direct=direct_profile_sum(metadata,packed,precision_bits=50)
        # Independent closed form of the39 targets, before preparation ops:
        # eight x² terms,22.5*x, four .125 additions and five rotations.
        shifts=(4,1024,1024,16351,1023)
        explicit=array('d',(8*x*x+22.5*x+.5+sum(packed[(i+s)%len(packed)] for s in shifts)
                            for i,x in enumerate(packed)))
        self.assertTrue(compare_references(explicit,direct)['passed'])
        self.assertTrue(compare_references(direct,replay)['passed'])
        self.assertEqual(summary['execution_completed'],135)
        self.assertEqual(len(replay),16384);self.assertEqual(replay[0],direct[0])

    def test_actual_constructor_policy_gate_and_previousP61_rejection(self):
        with TemporaryDirectory() as temp:
            _,record=prepared_fixture(Path(temp));result=profile.validate_prepared(record)
            self.assertEqual(result['residual_P'],record['client']['bootstrap_parameters']['ResidualParameters']['P'])
            self.assertIsNone(result['security_acceptance'])
            changes=(lambda r:r['requested_native_parameters'].update(log_p=[61]*3),
                lambda r:r['client']['bootstrap_parameters']['BootstrappingParameters']['P'].__setitem__(0,(1<<61)+1),
                lambda r:r['client'].update(keygen_api_attempts=1),
                lambda r:r['client']['bootstrap_parameters']['Mod1ParametersLiteral'].update(Mod1InvDegree=7),
                lambda r:r['client']['bootstrap_parameters']['Mod1ParametersLiteral'].update(Mod1Degree=63),
                lambda r:r['cache_selftest']['checkpoint_policy'].update(writes_completed=0))
            for change in changes:
                bad=copy.deepcopy(record);change(bad)
                with self.subTest(change=change),self.assertRaises(ValueError):profile.validate_prepared(bad)

    def test_native_argv_is_explicit_group_and_finite_without_build_or_execution(self):
        preparation=dict(output_plan='new-plan.json',output_bundle='new-bundle',input='new-input.bin',expected='new-expected.json')
        argv=profile.execution_command(preparation,out=Path('new-native'),prepared=Path('accepted-P57'))
        for flag,value in (('--auxiliary-prime-bits','57'),('--bootstrap-cache','32'),
            ('--precision-bits','50'),('--mod1-inv-degree','15'),('--log-message-ratio','2'),
            ('--bootstrap-normalization-log2','3'),('--checkpoint-interval','1000'),('--seconds','900')):
            self.assertEqual(argv[argv.index(flag)+1],value)
        self.assertIn('execute',argv);self.assertIn('--allow-bootstrap',argv)
        self.assertNotIn('prepare',argv);self.assertNotIn('build',argv)

    def test_prepare_serializes_public_input_and_new_reference_without_native_actions(self):
        with TemporaryDirectory() as temp:
            root=Path(temp);prepared,_=prepared_fixture(root);base,missing=metadata_fixture()
            base_path=root/'base.json';missing_path=root/'missing.json'
            base_path.write_text(json.dumps(base));missing_path.write_text(json.dumps(missing))
            args=Namespace(prepared=prepared,base_metadata=base_path,missing_metadata=missing_path,
                out=root/'new-profile',native_out=root/'future-native')
            with patch.object(profile.shutil,'disk_usage',return_value=Namespace(free=20*1024**3)):
                result=profile.prepare(args)
            self.assertEqual(result['status'],'prepared',result.get('error'))
            packed=struct.unpack('<'+('d'*16384),(args.out/'input.bin').read_bytes())
            self.assertEqual(packed[:20],tuple((i%17-8)/32 for i in range(20)))
            expected=json.loads((args.out/'expected.json').read_text())
            self.assertEqual(expected['indices'],list(range(16384)))
            self.assertEqual(len(expected['values']),16384)
            self.assertTrue(expected['mathematical_comparison']['passed'])
            self.assertFalse(expected['model_quality_measurement'])
            self.assertTrue(all(result[k]==0 for k in ('native_executions','keygen_runs','encryption_runs',
                'bootstrap_runs','decryption_runs','model_forwards','calibration_reads','heldout_reads','report_reads','llm_calls')))
            self.assertEqual(result['artifact_bytes'],sum(p.stat().st_size for p in args.out.rglob('*') if p.is_file()))

    def test_prepare_error_is_retained_before_any_native_or_public_vector_execution(self):
        with TemporaryDirectory() as temp:
            root=Path(temp);prepared,record=prepared_fixture(root);record['status']='error'
            (prepared/'result.json').write_text(json.dumps(record))
            args=Namespace(prepared=prepared,base_metadata=root/'unread-base',missing_metadata=root/'unread-missing',
                out=root/'retained-failure',native_out=root/'never-native')
            with patch.object(profile,'build_profile') as builder:
                result=profile.prepare(args)
            self.assertEqual(result['status'],'error');builder.assert_not_called()
            self.assertTrue((args.out/'preparation.json').is_file())
            self.assertEqual(result['native_executions'],0)


if __name__=='__main__':unittest.main()
