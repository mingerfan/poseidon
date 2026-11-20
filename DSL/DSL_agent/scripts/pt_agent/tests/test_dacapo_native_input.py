"""Saved-math input/reference protocol fixtures; no image/model/native run."""
from argparse import Namespace
import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from pt_agent.experiments import dacapo_native_input as prep


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value)+'\n')


def fixture(root):
    run=root/'math';worker=run/'worker';worker.mkdir(parents=True)
    plan=root/'original'/'original.runtime-plan.json';save(plan,{})
    scalar_path=root/'scalar.json';proof_path=worker/'source-math-proof.json'
    ids=[str(i) for i in range(122)]
    scalar=dict(status='scalar_sites_verified',source_factory=prep.binding.SOURCE_FACTORY,source_plan=str(plan),
        original_dsl_scalar_broadcast_interpretation=True,original_plan_modified=False,
        sites=[dict(value_id=i,interpretation='full-slot-scalar',source_path='/trusted/source.py',source_line=72) for i in ids])
    proof=dict(status='strict_math_passed',source_factory=prep.binding.SOURCE_FACTORY,source_plan=str(plan),
        sample_ids=['train:0','train:1','train:2'],all_three_passed=True,original_poly_preserved=True,
        original_pack_bb32_preserved=True,original_plan_modified=False,absolute_tolerance=1e-12,relative_tolerance=1e-12)
    save(scalar_path,scalar);save(proof_path,proof)
    rows=[]
    for i in range(3):
        p=worker/('packed-input-train-'+str(i)+'.f64')
        with p.open('wb') as f:f.truncate(prep.INPUT_BYTES)
        logits=[[float(j-i) for j in range(10)]]
        rows.append(dict(probe_index=i,sample_id='train:'+str(i),decoded=logits,reference=copy.deepcopy(logits),
            packed_input=dict(path=str(p),bytes=prep.INPUT_BYTES),math=dict(passed=True,criterion='1e-12+1e-12*abs(reference)')))
    w=dict(status='ok',source_checked=True,source_factory=prep.binding.SOURCE_FACTORY,original_plan_modified=False,
        original_training_math_verified=True,full_execution_scope_ready=True,packed_forward_attempts=3,
        packed_forward_completed=3,math_check_attempts=3,math_check_completed=3,op_attempts=37983,op_completed=37983,
        source_math_proof=str(proof_path),probe_rows=rows,
        program_metadata=dict(execution_count=12661,bootstrap_count=19,original_plan_modified=False,
            inline_scalar_count=122,inline_scalar_ids=ids),
        source_summary=dict(paths=dict(plan=dict(path=str(plan)),scalar_source_proof=dict(path=str(scalar_path))),
            original_pack_bb32_preserved=True,original_poly_preserved=True,
            packing_provenance=dict(output_formula='first10 decoded slots times32, shape[1,10]',
                official_io_source='/trusted/official/ResNet.py',official_io_lines=[75,79])))
    for key in ('api_calls','llm_calls','model_forwards','torch_model_forwards','checkpoint_reads','raw_cifar_reads',
        'heldout_samples','confirm_samples','report_samples','compiler_runs','keygen_runs','ckks_executions','integrity_checks'):w[key]=0
    record=dict(round='DACAPO-NATIVE-MATH',model=prep.MODEL,mode='original-dacapo-decoded-ideal-three-probe-math',
        status='ok',original_training_math_verified=True,full_execution_scope_ready=True,worker=w,
        selected_paths=dict(plan=str(plan),scalar_source_proof=str(scalar_path),bundle=str(root/'original.bundle')))
    result=run/'result.json';save(result,record)
    return result,plan,record


class SavedOriginalReferenceChecks(unittest.TestCase):
    def test_same_original_plan_three_math_probes_selects_only_saved_train0(self):
        with TemporaryDirectory() as temp:
            root=Path(temp);result,plan,record=fixture(root)
            selected=prep.source_gate(result,plan)
            self.assertEqual(selected['sample_id'],'train:0')
            self.assertEqual(selected['packed_input'],record['worker']['probe_rows'][0]['packed_input']['path'])
            self.assertEqual(selected['reference'],record['worker']['probe_rows'][0]['reference'][0])
            self.assertFalse(selected['new_source_images_read']);self.assertFalse(selected['old_native_expected_values_used'])

    def test_foreign_plan_scalar_and_math_proofs_are_rejected(self):
        for which in ('plan','scalar','math'):
            with TemporaryDirectory() as temp:
                root=Path(temp);result,plan,record=fixture(root);foreign=root/'foreign.json';save(foreign,{})
                if which=='plan':record['selected_paths']['plan']=str(foreign);save(result,record)
                else:
                    path=Path(record['selected_paths']['scalar_source_proof'] if which=='scalar' else record['worker']['source_math_proof'])
                    evidence=json.loads(path.read_text());evidence['source_plan']=str(foreign);save(path,evidence)
                with self.subTest(which=which),self.assertRaises(ValueError):prep.source_gate(result,plan)

    def test_math_partial_nonfinite_report_or_new_actions_reject(self):
        changes=(lambda r:r['worker'].update(op_completed=37982),
            lambda r:r['worker']['probe_rows'][0].update(sample_id='test:0'),
            lambda r:r['worker']['probe_rows'][0]['decoded'][0].__setitem__(0,float('nan')),
            lambda r:r['worker']['probe_rows'][0]['decoded'][0].__setitem__(0,2.),
            lambda r:r['worker'].update(ckks_executions=1),
            lambda r:r['worker']['source_summary']['packing_provenance'].update(output_formula='unscaled'))
        for change in changes:
            with TemporaryDirectory() as temp:
                result,plan,record=fixture(Path(temp));change(record);save(result,record)
                with self.subTest(change=change),self.assertRaises(ValueError):prep.source_gate(result,plan)

    def test_expected_uses_independent_original_logits_and_explicit32_contract(self):
        with TemporaryDirectory() as temp:
            result,plan,_=fixture(Path(temp));source=prep.source_gate(result,plan)
            expected=prep.make_expected(source)
            self.assertEqual(expected['indices'],list(range(10)));self.assertEqual(expected['values'],source['reference'])
            self.assertEqual((expected['output_multiplier'],expected['output_decoder']),(32.,prep.DECODER))
            self.assertEqual(expected['source_plan'],str(plan));self.assertEqual(expected['source_math_proof'],source['source_math_proof'])
            self.assertEqual([x/32.*32. for x in expected['values']],expected['values'])

    def test_saved_packed_input_location_and_size_are_required(self):
        with TemporaryDirectory() as temp:
            result,plan,record=fixture(Path(temp));source=Path(record['worker']['probe_rows'][0]['packed_input']['path'])
            with source.open('wb') as f:f.truncate(8)
            with self.assertRaises(ValueError):prep.source_gate(result,plan)

    def test_original_binary_requires_observed_new_decoder_capability(self):
        client=dict(status='ok',mode='parameters',final_decoder_protocol=1,
            supported_final_output_multipliers=[1,32],original_final_decoder=prep.DECODER,
            keygen_api_attempts=0,encryption_attempts=0,compute_attempts=0,final_decryption_attempts=0)
        self.assertTrue(prep.check_native_decoder(client)['actual_parameter_metadata_verified'])
        for change in (lambda r:r.pop('final_decoder_protocol'),
            lambda r:r.update(original_final_decoder='unscaled'),
            lambda r:r.update(supported_final_output_multipliers=[1]),
            lambda r:r.update(keygen_api_attempts=1)):
            bad=copy.deepcopy(client);change(bad)
            with self.subTest(change=change),self.assertRaises(ValueError):prep.check_native_decoder(bad)

    def test_prepare_copies_once_without_forward_and_registers_complete_case(self):
        with TemporaryDirectory() as temp:
            root=Path(temp);result,plan,_=fixture(root);results=root/'results';results.mkdir()
            args=Namespace(out=results/'prepared',math_result=result,plan=plan)
            with (patch.object(prep.paths,'RESULTS_DIR',results),patch('pt_agent.registry.git_rev',return_value='fixture'),
                  patch.object(prep.shutil,'disk_usage',return_value=Namespace(free=20*1024**3))):
                observed=prep.execute(args)
            self.assertEqual(observed['status'],'ok',observed['error'])
            self.assertEqual((observed['saved_input_copy_attempts'],observed['saved_input_copy_completed']),(1,1))
            self.assertEqual((args.out/'input.bin').stat().st_size,prep.INPUT_BYTES)
            self.assertTrue(all(observed[key]==0 for key in prep.ZERO_FIELDS))
            self.assertEqual(observed['required_native_client']['final_decoder_protocol'],1)
            self.assertEqual(observed['required_native_client']['supported_final_output_multipliers'],[1,32])
            expected=json.loads((args.out/'expected.json').read_text());self.assertEqual(expected['output_multiplier'],32.)
            rows=[json.loads(x) for x in (results/'registry.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows),1);self.assertEqual(set(rows[0]),{'run_id','round','model','mode','git_rev','metrics','artifacts_dir','ts'})
            completion=json.loads((args.out/'completion.json').read_text());self.assertTrue(completion['registry_persisted'])
            spans=observed['timing']['spans'];roots=[r for r in spans if r['parent_id'] is None]
            self.assertEqual(len(roots),1);self.assertEqual(sum(r['exclusive_ns'] for r in spans),roots[0]['inclusive_ns'])

    def test_source_directory_is_protected_before_writes(self):
        with TemporaryDirectory() as temp:
            root=Path(temp);result,plan,_=fixture(root)
            with patch.object(prep.paths,'RESULTS_DIR',root):
                with self.assertRaises(ValueError):prep.execute(Namespace(out=result.parent,math_result=result,plan=plan))

    def test_copy_error_retains_partial_and_unique_registry_without_new_math(self):
        with TemporaryDirectory() as temp:
            root=Path(temp);result,plan,_=fixture(root);results=root/'results';results.mkdir()
            args=Namespace(out=results/'failed',math_result=result,plan=plan)
            with (patch.object(prep.paths,'RESULTS_DIR',results),patch('pt_agent.registry.git_rev',return_value='fixture'),
                  patch.object(prep.shutil,'disk_usage',return_value=Namespace(free=20*1024**3)),
                  patch.object(prep.shutil,'copyfile',side_effect=OSError('fixture copy failure'))):
                observed=prep.execute(args)
            self.assertEqual(observed['status'],'error');self.assertFalse(observed['reference_prepared'])
            self.assertEqual((observed['saved_input_copy_attempts'],observed['saved_input_copy_completed']),(1,0))
            self.assertEqual(len((results/'registry.jsonl').read_text().splitlines()),1)
            self.assertIsNone(observed['formal_latency']);self.assertEqual(observed['model_forwards'],0)


if __name__=='__main__':unittest.main()
