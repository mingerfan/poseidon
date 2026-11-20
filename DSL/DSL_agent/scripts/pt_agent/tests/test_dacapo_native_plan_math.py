"""Synthetic, no-model checks of original DaCapo ideal RuntimePlan replay."""
from contextlib import ExitStack
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pt_agent.experiments import dacapo_native_binding as binding
from pt_agent.experiments import dacapo_native_plan_math as subject
from pt_agent.experiments.r10_paper_metrics import StageRecorder


def scalar_proof(plan,path):
    return dict(status='scalar_sites_verified',source_factory=binding.SOURCE_FACTORY,
        source_plan=str(path.resolve()),original_dsl_scalar_broadcast_interpretation=True,
        original_plan_modified=False,sites=[dict(value_id=op['output'],interpretation='full-slot-scalar',
            source_path='/synthetic/original/HEAAN_HEVM.cpp',source_line=289)
            for op in plan['initialization'] if op['payload']['kind']=='inline'])


def expected_inventory(metadata):
    return {key:metadata[key] for key in subject.EXPECTED}


def archive_record(bundle,source=None):
    source=source or bundle
    rows=[dict(path=source.name,kind='directory',bytes=0)]
    for path in sorted(bundle.rglob('*')):
        rows.append(dict(path=str(Path(source.name)/path.relative_to(bundle)),
            kind='directory' if path.is_dir() else 'file',bytes=0 if path.is_dir() else path.stat().st_size))
    return dict(files=[dict(kind='runtime_bundle',path=str(source),inventory=rows)])


class OriginalDaCapoIdealMathChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.np=subject._np()

    def fixture(self,base,*,bootstrap=True):
        np=self.np
        files=binding.write_public_fixture(base/'fixture',bootstrap=bootstrap)
        files={key:Path(value) for key,value in files.items() if key in ('plan','bundle','input','expected')}
        new_bundle=base/'original.bundle';files['bundle'].rename(new_bundle);files['bundle']=new_bundle
        new_plan=base/'original.runtime-plan.json';files['plan'].rename(new_plan);files['plan']=new_plan
        plan=json.loads(files['plan'].read_text());manifest=json.loads((files['bundle']/'manifest.json').read_text())
        proof=scalar_proof(plan,files['plan']);proof_path=base/'scalar.json';proof_path.write_text(json.dumps(proof))
        source=dict(split='train',indices=[0,1,2],sample_ids=['train:0','train:1','train:2'],labels=[0,1,2],
            normalization=dict(mean=[.485,.456,.406],std=[.229,.224,.225]))
        probes=base/'probes.npz';np.savez(probes,input=np.zeros((3,3,32,32),dtype=np.float64),
            labels=np.array([0,1,2],dtype=np.int64),indices=np.array([0,1,2],dtype=np.int64))
        probe_source=base/'probe-source.json';probe_source.write_text(json.dumps(source))
        original=base/'official';script=original/'examples/benchmarks/ResNet.py';script.parent.mkdir(parents=True)
        script.write_text('# synthetic source metadata only\n')
        checkpoint=original/'examples/data/resnet20.silu.model';checkpoint.parent.mkdir();checkpoint.write_text('never loaded')
        model_source=original/'python/poly/poly/models/ResNet.py';model_source.parent.mkdir(parents=True);model_source.write_text('# synthetic\n')
        r13_path=base/'r13.json';r24_path=base/'r24.json'
        r13=dict(round='R13',status='ok',model=binding.SOURCE_FACTORY,worker=dict(
            source=source,model_source=dict(checkpoint=str(checkpoint),original_source=str(model_source)),
            artifacts=dict(training_probes=str(probes),training_probe_source=str(probe_source))))
        native_root=base/'local-poly'
        r24=dict(round='R24',status='ok',model=binding.SOURCE_FACTORY,
            mode='orbit-native-polynomial-physical-scale-check',source_records=dict(records=dict(r13=str(r13_path))),
            worker=dict(original_reference_passed=True,native_polynomial_math_passed=True,
                physical_scaled_forward_completed=True,probe_source=source,
                original_forward_attempts=3,original_forward_completed=3,approx_forward_attempts=3,approx_forward_completed=3,
                activation_attempts=57,activation_completed=57,math_check_attempts=57,math_check_completed=57,
                native_metadata=dict(sources=[dict(path=str(native_root/'MPCB.py'))]),
                approximate_probes=[dict(probe_index=i,sample_id='train:'+str(i),approximate=[[-4.]*10]) for i in range(3)]))
        clean=dict(exit_code=0,timed_out=False,signal=None,signals_sent=[],remaining_pids=[])
        compiled=dict(round='DACAPO-ORIGINAL',model=subject.MODEL,status='ok',original_source_variant=True,r29_cheb63_variant=False,
            trace_attempts=1,trace_completed=1,compile_attempts=1,compile_completed=1,native_compile_attempts=1,native_compile_completed=1,
            bootstrap_count=int(bootstrap),export_contract_compatible=True,plan=dict(path=str(files['plan'])),
            trace_process=clean,compile_process=clean,compile_worker=dict(process=clean),sources=dict(
                script=dict(path=str(script)),saved_r13_result=dict(path=str(r13_path)),original_model=dict(
                    factory=binding.SOURCE_FACTORY,checkpoint=str(checkpoint),original_model_source=str(model_source))))
        alignment=dict(status='source_arithmetic_aligned',checks={key:True for key in
            ('MPCB.py:GenPoly','MPCB.py:fint','Poly.py:GenPoly')},coefficient_count=96,
            parsed_public_coefficient_values_equal=True,parsed_integer_tree_equal=True,all_coefficients_finite=True,
            silu_semantics=dict(verified_from_source=True),source_paths=[str(original/'python/poly/poly'),str(native_root)],
            source_reference=dict(r24_result=str(r24_path)))
        paths=dict(r13_result=r13_path,r24_result=r24_path,compile_result=base/'compile.json',
            poly_source_alignment=base/'alignment.json',archive_manifest=base/'archive-manifest.json')
        for name,value in [('r13_result',r13),('r24_result',r24),('compile_result',compiled),
                           ('poly_source_alignment',alignment),('archive_manifest',archive_record(files['bundle'],base/'original.bundle'))]:
            paths[name].write_text(json.dumps(value))
        args=SimpleNamespace(plan=files['plan'],bundle=files['bundle'],scalar_source_proof=proof_path,
            probes=probes,probe_source=probe_source,work_dir=base/'worker',seconds=900,memory_gib=4,case_gib=4,**paths)
        program=subject.load_program(args.plan,args.bundle/'manifest.json',proof)
        return args,program,expected_inventory(program['metadata'])

    def run_fixture(self,args,expected):
        args.work_dir.mkdir()
        result=subject.initial_result()
        with patch.object(subject,'EXPECTED',expected),patch.object(subject.shutil,'disk_usage',return_value=SimpleNamespace(free=1<<40)):
            subject.run_checks(args,StageRecorder(args.work_dir/'events.jsonl'),result)
        return result

    def test_all_ten_ops_match_an_independent_boot_branch_formula_and_preserve_input(self):
        np=self.np
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory);args,program,_=self.fixture(base)
            constants=subject.LazyPlaintexts(args.bundle,program)
            raw=np.array([.05+.1*i for i in range(8)],dtype=np.float64)
            packed=np.tile(raw,8192);before=packed.copy();result=subject.initial_result()
            actual=subject.execute_program(program,constants,packed,result)
            expected=-(np.roll(packed,-1)**2+.125+packed)
            np.testing.assert_array_equal(actual,expected)
            np.testing.assert_array_equal(packed,before)
            self.assertEqual(set(result['op_completed_by_kind']),set(binding.OPS))
            self.assertEqual(result['op_completed'],11)
            self.assertEqual(result['ideal_identity_nodes']['boot'],1)
            self.assertIsNone(result['bootstrap_input_rows'][0]['domain_acceptance'])
            self.assertTrue(all(result[k]==0 for k in subject.ZERO_FIELDS))
            self.assertLessEqual(result['maximum_live_ciphertext_values'],program['metadata']['last_use_ciphertext_peak_count'])

    def test_high_scale_bundle_formula_and_negative_rotation(self):
        np=self.np
        with tempfile.TemporaryDirectory() as directory:
            args,program,_=self.fixture(Path(directory),bootstrap=False)
            raw=np.tile(np.array([.05+.1*i for i in range(8)],dtype=np.float64),8192)
            constants=subject.LazyPlaintexts(args.bundle,program)
            actual=subject.execute_program(program,constants,raw,subject.initial_result())
            np.testing.assert_array_equal(actual,-(.25*raw*raw+.125))
            self.assertEqual(constants.summary()['blob_reads'],1)
            self.assertLessEqual(constants.summary()['peak_decoded_bytes'],65536*8)
            self.assertIn(127,program['metadata']['value_scales'])

    def test_lazy_cache_is_readonly_and_evicts_before_next_load(self):
        with tempfile.TemporaryDirectory() as directory:
            args,program,_=self.fixture(Path(directory),bootstrap=False)
            constants=subject.LazyPlaintexts(args.bundle,program)
            first=constants.get('half')
            self.assertFalse(first.flags.writeable)
            self.assertIs(constants.get('half'),first)
            second=constants.get('offset')
            self.assertFalse(second.flags.writeable)
            self.assertEqual(constants.summary()['resident_vectors'],1)
            self.assertEqual(constants.summary()['hits'],1)
            self.assertEqual(constants.summary()['evictions'],1)
            constants.clear();self.assertEqual(constants.summary()['resident_vectors'],0)

    def test_short_bundle_vector_fails_before_any_arithmetic(self):
        with tempfile.TemporaryDirectory() as directory:
            args,program,_=self.fixture(Path(directory),bootstrap=False)
            (args.bundle/'data/half.bin').write_bytes(b'12345678')
            with self.assertRaisesRegex(ValueError,'size/path'):
                subject.LazyPlaintexts(args.bundle,program).get('half')

    def test_plan_rejects_missing_scalar_source_proof_and_metadata_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            args,_,_=self.fixture(Path(directory))
            with self.assertRaisesRegex(ValueError,'singleton inline'):
                subject.load_program(args.plan,args.bundle/'manifest.json',None)
            plan=json.loads(args.plan.read_text());plan['values'][0]['level']=17
            args.plan.write_text(json.dumps(plan))
            with self.assertRaisesRegex(ValueError,'metadata'):
                subject.load_program(args.plan,args.bundle/'manifest.json',json.loads(args.scalar_source_proof.read_text()))

    def test_restored_inventory_checks_names_sizes_and_rejects_extra_files(self):
        with tempfile.TemporaryDirectory() as directory:
            args,program,_=self.fixture(Path(directory),bootstrap=False)
            result=subject.check_restored_bundle(program,args.bundle,args.archive_manifest)
            self.assertEqual(result['blob_count'],1)
            self.assertFalse(result['archive_payload_read'])
            (args.bundle/'unexpected').write_text('synthetic')
            with self.assertRaisesRegex(ValueError,'inventory differs'):
                subject.check_restored_bundle(program,args.bundle,args.archive_manifest)

    def test_restored_inventory_rejects_symbolic_payloads(self):
        with tempfile.TemporaryDirectory() as directory:
            args,program,_=self.fixture(Path(directory),bootstrap=False)
            original=args.bundle/'data/half.bin';original.unlink();original.symlink_to(args.probe_source)
            with self.assertRaisesRegex(ValueError,'links/special'):
                subject.check_restored_bundle(program,args.bundle,args.archive_manifest)

    def test_source_chain_accepts_only_the_original_clean_compile_and_r24(self):
        with tempfile.TemporaryDirectory() as directory:
            args,program,expected=self.fixture(Path(directory))
            with patch.object(subject,'EXPECTED',expected):
                summary,_,_,_=subject.check_sources(args)
                self.assertTrue(summary['original_plan_modified'] is False)
                record=json.loads(args.compile_result.read_text());record['compile_worker']['process']['signals_sent']=[dict(signal=15)]
                args.compile_result.write_text(json.dumps(record))
                with self.assertRaisesRegex(ValueError,'clean unchanged'):
                    subject.check_sources(args)

    def test_same_names_and_sizes_in_a_wrong_bundle_directory_are_rejected(self):
        import shutil
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory);args,program,expected=self.fixture(base)
            copied=base/'foreign.bundle';shutil.copytree(args.bundle,copied)
            args.bundle=copied
            with patch.object(subject,'EXPECTED',expected):
                with self.assertRaisesRegex(ValueError,'different original plan bundle'):
                    subject.check_sources(args)

    def test_complete_three_saved_synthetic_training_inputs_have_exact_counts_and_proof(self):
        with tempfile.TemporaryDirectory() as directory:
            args,program,expected=self.fixture(Path(directory));result=self.run_fixture(args,expected)
            self.assertEqual(result['status'],'ok')
            self.assertEqual((result['packed_forward_completed'],result['math_check_completed']),(3,3))
            self.assertEqual(result['op_completed'],33)
            self.assertEqual(result['op_completed_by_kind'],{k:v*3 for k,v in expected['op_counts'].items()})
            self.assertEqual(len(result['bootstrap_input_rows']),3)
            proof=json.loads((args.work_dir/'source-math-proof.json').read_text())
            self.assertEqual(proof['sample_ids'],['train:0','train:1','train:2'])
            self.assertTrue(proof['original_poly_preserved'])
            self.assertTrue(proof['original_pack_bb32_preserved'])
            self.assertTrue(result['full_execution_scope_ready'])
            self.assertTrue(all(result[k]==0 for k in subject.ZERO_FIELDS))
            self.assertTrue(all(result[k] is None for k in subject.FORMAL_FIELDS))
            self.assertEqual(len(result['probe_rows']),3)
            self.assertEqual(Path(result['probe_rows'][0]['packed_input']['path']).stat().st_size,65536*8)

    def test_first_strict_math_failure_retains_partial_counts_without_acceptance_proof(self):
        with tempfile.TemporaryDirectory() as directory:
            args,program,expected=self.fixture(Path(directory))
            r24=json.loads(args.r24_result.read_text());r24['worker']['approximate_probes'][0]['approximate']=[[-3.]*10]
            args.r24_result.write_text(json.dumps(r24));args.work_dir.mkdir();result=subject.initial_result()
            with patch.object(subject,'EXPECTED',expected),patch.object(subject.shutil,'disk_usage',return_value=SimpleNamespace(free=1<<40)):
                with self.assertRaisesRegex(ValueError,'strict mathematical'):
                    subject.run_checks(args,StageRecorder(args.work_dir/'events.jsonl'),result)
            self.assertEqual((result['packed_forward_attempts'],result['packed_forward_completed'],result['math_check_completed']),(1,1,1))
            self.assertEqual(result['op_completed'],11)
            self.assertFalse(result['original_training_math_verified'])
            self.assertFalse((args.work_dir/'source-math-proof.json').exists())
            self.assertTrue(all(result[k] is None for k in subject.FORMAL_FIELDS))

    def test_nonfinite_arithmetic_stops_at_its_actual_operation(self):
        np=self.np
        with tempfile.TemporaryDirectory() as directory:
            args,program,_=self.fixture(Path(directory));result=subject.initial_result()
            with self.assertRaises(FloatingPointError):
                subject.execute_program(program,subject.LazyPlaintexts(args.bundle,program),
                    np.full(65536,1e308,dtype=np.float64),result)
            self.assertGreater(result['op_attempts'],result['op_completed'])
            self.assertLess(result['op_completed'],11)
            self.assertFalse(result['original_training_math_verified'])

    def test_outer_missing_source_preserves_failure_registry_and_complete_endpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory);args,program,_=self.fixture(base)
            args.out=base/'fresh';args.registry=base/'registry.jsonl';args.python=Path('/no/python')
            args.plan=base/'missing-plan'
            with patch.object(subject,'run_process',side_effect=AssertionError('No child allowed')):
                result=subject.execute(args)
            self.assertEqual(result['status'],'error')
            self.assertEqual(result['worker_attempts'],0)
            self.assertIsNone(result['actual_source_reads']['op_completed'])
            rows=[json.loads(line) for line in args.registry.read_text().splitlines()]
            self.assertEqual(len(rows),1)
            self.assertEqual(set(rows[0]),{'run_id','round','model','mode','git_rev','metrics','artifacts_dir','ts'})
            marker=json.loads((args.out/'completion.json').read_text())
            self.assertTrue(marker['registry_persisted'])
            self.assertTrue(marker['required_artifacts_persisted'])
            self.assertGreaterEqual(marker['ended_ns'],marker['started_ns'])

    def test_interrupted_checkpoint_counts_remain_unknown_until_worker_terminal(self):
        import sys
        for terminal in (False,True):
            with self.subTest(terminal=terminal),tempfile.TemporaryDirectory() as directory:
                base=Path(directory);args,program,expected=self.fixture(base)
                args.out=base/'fresh';args.registry=base/'registry.jsonl';args.python=Path(sys.executable)
                def fake_process(argv,**kwargs):
                    if argv[0]=='git':
                        Path(kwargs['log']).write_text('synthetic-revision\n')
                        return dict(exit_code=0,timed_out=False,signal=None,signals_sent=[],remaining_pids=[])
                    target=args.out/'worker';target.mkdir()
                    saved=subject.initial_result();saved['op_completed']=17
                    if terminal:saved.update(status='error',elapsed_ns=123,error='intentional fake arithmetic stop')
                    (target/'worker-result.json').write_text(json.dumps(saved))
                    return dict(exit_code=1 if terminal else -15,timed_out=False,signal=None,signals_sent=[],remaining_pids=[])
                with patch.object(subject,'run_process',side_effect=fake_process),patch.object(subject.shutil,'disk_usage',return_value=SimpleNamespace(free=1<<40)):
                    result=subject.execute(args)
                self.assertEqual(result['status'],'error')
                self.assertEqual(result['partial_checkpoint_counts']['op_completed'],17)
                self.assertEqual(result['actual_source_reads']['op_completed'],17 if terminal else None)
                self.assertFalse(result['original_training_math_verified'])
                self.assertFalse(result['full_execution_scope_ready'])
                self.assertEqual(result['worker_attempts'],1)

    def test_native_environment_copy_is_controlled_and_parent_unchanged(self):
        import os
        with patch.dict(os.environ,{'LD_LIBRARY_PATH':'synthetic-bad','NIX_FAKE':'bad','HECATE':'bad','PYTHONHOME':'bad','KEEP_ME':'safe'}):
            before=dict(os.environ);env=subject._worker_env()
            self.assertEqual(dict(os.environ),before)
            self.assertNotIn('LD_LIBRARY_PATH',env);self.assertNotIn('NIX_FAKE',env)
            self.assertNotIn('HECATE',env);self.assertNotIn('PYTHONHOME',env)
            self.assertEqual(env['KEEP_ME'],'safe')
            self.assertEqual(env['OPENBLAS_NUM_THREADS'],'1')
            self.assertEqual(env['PYTHONPATH'],str(Path(subject.__file__).resolve().parents[2]))


if __name__=='__main__':unittest.main()
