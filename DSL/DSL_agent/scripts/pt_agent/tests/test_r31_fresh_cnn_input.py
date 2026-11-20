"""Fresh-reference source/protocol checks and one isolated tiny numeric suite."""
from argparse import Namespace
import copy
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from pt_agent.experiments import r31_fresh_cnn_input as fresh
from pt_agent import paths
from pt_agent.tests._artifacts import require


def save(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value)+'\n')


def source_fixture(root):
    case=root/'accepted-current';case.mkdir();info=case/'work/intake/model_info/model_info.json'
    normal=dict(mean=[.485,.456,.406],std=[.229,.224,.225])
    provenance=dict(split='train',indices=[0,1,2],sample_ids=['train:0','train:1','train:2'],
        labels=[6,9,9],normalization=normal)
    sources={k:str(root/(k+'.fixture')) for k in ('source','checkpoint','inference_preprocessing_source')}
    record=dict(source=dict(factory='official-factory',normalization=normal,calibration_source=provenance,
        original_source=sources['source'],checkpoint=sources['checkpoint'],
        inference_preprocessing_source=sources['inference_preprocessing_source']),
        task='classification',metric='classification_top1',threshold={'acc_drop_pp':.5},
        inputs=[dict(name='image',shape=[1,3,32,32],dtype='float64')])
    gate=dict(source_case=str(case),model=fresh.MODEL,model_info=str(info),factory='official-factory',
        original_sources=sources,normalization=normal,program=str(case/'program.ptdsl'),
        probes=str(case/'training_probes.npz'),probe_source=str(case/'probe-source.json'))
    save(info,record);save(Path(gate['probe_source']),provenance)
    plan_path=case/'post-freeze/emit/lowered._hecate_golden.runtime-plan.json'
    plan=dict(format_version=1,values=[dict(id='x'),dict(id='y')],external_inputs=['x'],
        final_outputs=['y'],initialization=[],finalization=[],execution=[dict(ordinal=0,
            output='y',inputs=['x'],kind='compute',op='boot')])
    save(plan_path,plan)
    save(case/'result.json',dict(round='R30',status='ok',model=fresh.MODEL,
        phases=dict(compile=dict(runtime_plan=str(plan_path),compiled_bootstraps=1))))
    return gate,plan_path,record,provenance,plan


class SourceProtocolChecks(unittest.TestCase):
    def test_accepted_current_source_retains_own_probes_and_new_reference_policy(self):
        with TemporaryDirectory() as temp:
            gate,plan,_,_,_=source_fixture(Path(temp))
            with patch('pt_agent.experiments.r30_frozen_cnn_report.source_gate',return_value=gate) as checked:
                result=fresh.prepare_source(Path(gate['source_case']))
            checked.assert_called_once_with(Path(gate['source_case']),fresh.MODEL)
            self.assertEqual(result['gate']['probes'],gate['probes'])
            self.assertEqual(result['runtime_plan'],str(plan))
            self.assertEqual(result['sample_id'],'train:0')
            self.assertFalse(result['normalization_reapplied'])
            self.assertIn('New current frozen-program',result['expected_reference_policy'])

    def test_source_normalization_checkpoint_and_training_identity_rejected(self):
        with TemporaryDirectory() as temp:
            gate,plan,record,provenance,_=source_fixture(Path(temp))
            changes=(lambda r:r['source'].update(checkpoint='/unrelated/checkpoint'),
                lambda r:r['source']['normalization'].update(mean=[0,0,0]),
                lambda r:r.update(threshold={'acc_drop_pp':1}),
                lambda r:r['inputs'][0].update(dtype='float32'))
            for change in changes:
                bad=copy.deepcopy(record);change(bad);save(Path(gate['model_info']),bad)
                with self.subTest(change=change),self.assertRaises(ValueError):fresh._metadata_gate(gate,plan)
            save(Path(gate['model_info']),record)
            bad=copy.deepcopy(provenance);bad['sample_ids'][0]='test:0';save(Path(gate['probe_source']),bad)
            with self.assertRaises(ValueError):fresh._metadata_gate(gate,plan)

    def test_runtimeplan_ssa_and_compiled_boot_source_are_required(self):
        with TemporaryDirectory() as temp:
            _,_,_,_,plan=source_fixture(Path(temp))
            self.assertEqual(fresh.check_plan(plan,1)['bootstrap_count'],1)
            for change in (lambda p:p['execution'][0].update(output='x'),
                           lambda p:p['execution'][0].update(inputs=['future']),
                           lambda p:p['values'].append(dict(id='x')),
                           lambda p:p['execution'].append(copy.deepcopy(p['execution'][0]))):
                bad=copy.deepcopy(plan);change(bad)
                with self.subTest(change=change),self.assertRaises(ValueError):fresh.check_plan(bad,1)
            with self.assertRaises(ValueError):fresh.check_plan(plan,123)

    def test_explicit_nix_argv_and_finite_limits(self):
        with TemporaryDirectory() as temp:
            root=Path(temp);gate,plan,_,_,_=source_fixture(root)
            args=Namespace(out=root/'new',intake_python=root/'intake-python')
            source=dict(gate=gate,runtime_plan=str(plan),layout_proof=None)
            argv=fresh._command(args,source,args.out/'work')
            self.assertIn(str(fresh.NIX_SECONDS),argv);self.assertIn(str(args.intake_python),argv)
            for field in ('model-info','program','probes','probe-source','runtime-plan','source-record','work-dir'):
                self.assertEqual(argv.count('--'+field),1)
            self.assertNotIn('--expected',argv)
            self.assertEqual((fresh.OVERALL_SECONDS,fresh.WORKER_SECONDS,fresh.MEMORY_GIB),(600,480,4))

    def test_success_gate_requires_new_single_tensor_and_partial_is_preserved(self):
        record=fresh._initial_result();record.update(status='ok',source_checked=True,reference_prepared=True,
            training_samples_consumed=1,training_npz_rows_loaded=3)
        record.update({k:1 for k in fresh.COUNTERS})
        self.assertTrue(fresh._success(record))
        for key,value in [('tensor_completed',0),('tensor_attempts',2),('old_expected_values_read',True),
                          ('normalization_reapplied',True),('report_samples',1)]:
            bad=copy.deepcopy(record);bad[key]=value
            with self.subTest(key=key):self.assertFalse(fresh._success(bad))
        partial=copy.deepcopy(record);partial.update(status='error',reference_prepared=False)
        partial['tensor_completed']=0
        self.assertFalse(fresh._success(partial));self.assertEqual(partial['packing_completed'],1)

    def test_source_failure_registers_once_with_completion_and_unknown_worker_counts(self):
        from pt_agent.experiments import r30_frozen_cnn_report
        with TemporaryDirectory() as temp:
            root=Path(temp);results=root/'results';results.mkdir();source=root/'source';source.mkdir()
            args=Namespace(out=results/'new-source-failure',source_case=source,model=fresh.MODEL,
                layout_proof=None,intake_python=root/'python')
            with (patch.object(paths,'RESULTS_DIR',results),patch.object(fresh,'prepare_source',side_effect=ValueError('source fixture rejected')),
                 patch('pt_agent.registry.git_rev',return_value='fixture-rev'),patch.object(fresh.shutil,'disk_usage',return_value=Namespace(free=20*1024**3)),
                 patch.object(fresh,'run_process') as child):
                result=fresh.execute(args)
            child.assert_not_called();self.assertEqual(result['status'],'error')
            rows=[json.loads(x) for x in (results/'registry.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows),1);self.assertEqual(set(rows[0]),{'run_id','round','model','mode','git_rev','metrics','artifacts_dir','ts'})
            self.assertIsNone(rows[0]['metrics']['worker']['tensor_attempts'])
            completion=json.loads((args.out/'completion.json').read_text())
            self.assertTrue(completion['registry_persisted']);self.assertEqual(completion['status'],'error')
            spans=result['timing']['spans'];roots=[row for row in spans if row['parent_id'] is None]
            self.assertEqual(len(roots),1);self.assertEqual(roots[0]['stage_id'],'fresh_native_reference_preparation')
            self.assertEqual(roots[0]['status'],'error')
            self.assertEqual(sum(row['exclusive_ns'] for row in spans),roots[0]['inclusive_ns'])
            self.assertTrue(result['timing']['attribution_ready'])
            snapshots=json.loads((args.out/'source-snapshots/source-paths.json').read_text())
            self.assertEqual(len(snapshots['files']),4);self.assertEqual(snapshots['integrity_checks'],0)
            self.assertTrue(all(Path(row['snapshot']).is_file() for row in snapshots['files']))

    def test_new_output_artifact_gate_excludes_previous_expected_and_nonfinite_values(self):
        with TemporaryDirectory() as temp:
            work=Path(temp);(work/'input.bin').write_bytes(b'\0'*(16384*8))
            expected=dict(indices=list(range(10)),values=[float(x) for x in range(10)],
                tensor_values=[float(x) for x in range(10)],shape=[1,10],dtype='float64',sample_id='train:0',
                old_expected_values_used=False,criterion=fresh.CRITERION)
            save(work/'expected.json',expected)
            record=dict(input=str(work/'input.bin'),expected=str(work/'expected.json'),
                tensor_values=expected['values'],output_layout=dict(indices=list(range(10))))
            self.assertTrue(fresh.check_outputs(record,work)['passed'])
            expected['values'][0]=float('nan');save(work/'expected.json',expected)
            with self.assertRaises(ValueError):fresh.check_outputs(record,work)
            record['expected']=str(work.parent/'previous90-expected.json')
            with self.assertRaises(ValueError):fresh.check_outputs(record,work)


def numeric_suite():
    """Nested cases keep system discovery free of NumPy/Torch imports."""
    import numpy as np
    from pt_agent.dsl.parser import parse
    from pt_agent.dsl.typecheck import check
    from pt_agent.experiments.r10_paper_metrics import StageRecorder
    from pt_agent.tensor_sim import run_tensor
    program_text='''@fhe.program(slots=16384)
def tiny(image: Cipher[1,3,32,32]):
    x = pack(image, layout="mux", gap=1)
    x = conv2d(x, w=W["cw"], stride=1, pad=0, algo="mpcb")
    x = mean(x, axis=(2,3))
    x = relayout(x, layout="flat")
    y = linear(x, w=W["w"], b=W["b"], algo="bsgs")
    return y
'''
    class NumericCases(unittest.TestCase):
        def fixture(self):
            program=parse(program_text)
            weights=dict(cw=np.arange(48,dtype=np.float64).reshape(16,3,1,1)/128,
                         w=np.arange(160,dtype=np.float64).reshape(10,16)/512,b=np.arange(10,dtype=np.float64)/32)
            checked=check(program,{k:v.shape for k,v in weights.items()})
            self.assertTrue(checked.ok,str(checked.diags))
            image=np.arange(3072,dtype=np.float64).reshape(1,3,32,32)/4096
            return program,weights,checked,image

        def test_current_flat_head_decoder_and_historical_metadata_only(self):
            program,_,checked,_=self.fixture()
            first=fresh.output_layout(program,checked)
            self.assertEqual(first['indices'],list(range(10)))
            proof=dict(status='ok',mathematical_equivalence_verified=True,samples=[dict(status='ok',
                decode_indices=list(range(10)),output_layouts=[dict(kind='flat',n=16384,size=10)])])
            result=fresh.output_layout(program,checked,proof)
            self.assertFalse(result['historical_metadata_crosscheck']['historical_expected_values_used'])
            proof['samples'][0]['decode_indices'][0]=7
            with self.assertRaises(ValueError):fresh.output_layout(program,checked,proof)

        def test_single_tensor_reference_uses_current_math_and_preserves_image(self):
            program,weights,checked,image=self.fixture();before=image.copy()
            with TemporaryDirectory() as temp:
                work=Path(temp);result=fresh._initial_result()
                fresh.prepare_reference(program,weights,image,checked,work,StageRecorder(work/'events.jsonl'),result)
                expected=json.loads((work/'expected.json').read_text());packed=np.fromfile(work/'input.bin',dtype='<f8')
            direct=(image.mean(axis=(2,3))@weights['cw'].reshape(16,3).T)@weights['w'].T+weights['b']
            np.testing.assert_allclose(expected['values'],direct[0],atol=1e-12,rtol=1e-12)
            np.testing.assert_array_equal(image,before)
            self.assertEqual(packed.shape,(16384,));self.assertTrue(result['input_packing']['exact_input_roundtrip'])
            self.assertEqual((result['tensor_attempts'],result['tensor_completed']),(1,1))
            self.assertEqual(expected['indices'],list(range(10)));self.assertFalse(expected['old_expected_values_used'])

        def test_input_and_new_output_nonfinite_or_dtype_failures_remain_partial(self):
            program,weights,checked,image=self.fixture()
            with TemporaryDirectory() as temp:
                work=Path(temp);recorder=StageRecorder(work/'events.jsonl')
                for bad in (image.astype(np.float32),np.full_like(image,np.nan)):
                    with self.assertRaises(ValueError):fresh.prepare_reference(program,weights,bad,checked,work,recorder,fresh._initial_result())
                result=fresh._initial_result()
                with patch('pt_agent.tensor_sim.run_tensor',return_value=np.full((1,10),np.nan,dtype=np.float64)):
                    with self.assertRaises(ValueError):fresh.prepare_reference(program,weights,image,checked,work,recorder,result)
                self.assertEqual(result['packing_completed'],1);self.assertEqual(result['tensor_attempts'],1)
                self.assertEqual(result['tensor_completed'],0);self.assertFalse(result['reference_prepared'])

        def test_only_current_training_npz_first_row_is_consumed_without_normalization(self):
            _,_,_,image=self.fixture()
            with TemporaryDirectory() as temp:
                path=Path(temp)/'synthetic-training.npz'
                np.savez(path,input=np.concatenate([image,image+1,image+2]),indices=np.arange(3))
                loaded,metadata=fresh.load_training_probe(path,dict(sample_ids=['train:0','train:1','train:2'],labels=[6,9,9]))
                np.testing.assert_array_equal(loaded,image)
                self.assertEqual(metadata['label'],6);self.assertFalse(metadata['normalization_reapplied'])
                np.savez(path,input=np.concatenate([image,image+1,image+2]),indices=np.array([0,1,3]))
                with self.assertRaises(ValueError):fresh.load_training_probe(path,dict(sample_ids=['train:0','train:1','train:2']))

        def test_final_nonflat_or_nonlinear_layout_is_rejected(self):
            program,_,checked,_=self.fixture()
            checked.layouts[program.outputs[0]]='mux'
            with self.assertRaises(ValueError):fresh.output_layout(program,checked)
            checked.layouts[program.outputs[0]]='flat';checked.shapes[program.outputs[0]]=(1,11)
            with self.assertRaises(ValueError):fresh.output_layout(program,checked)
    return unittest.defaultTestLoader.loadTestsFromTestCase(NumericCases)


BOOTSTRAP = r'''import json,os,sys,tempfile,unittest
from pathlib import Path
work=Path(sys.argv[1]);work.mkdir(parents=True,exist_ok=True)
os.environ['TMPDIR']=str(work);tempfile.tempdir=str(work)
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
from pt_agent.tests.test_r31_fresh_cnn_input import numeric_suite
result=unittest.TextTestRunner(verbosity=2).run(numeric_suite())
record=dict(testsRun=result.testsRun,failures=len(result.failures),errors=len(result.errors),skips=len(result.skipped),success=result.wasSuccessful())
(work/'numeric-result.json').write_text(json.dumps(record)+'\n')
raise SystemExit(0 if record['success'] and record['testsRun']>0 and record['skips']==0 else 1)
'''


# The existing interpreter is a Nix-store symlink resolved inside the launcher.
# Require the local environment metadata and launcher; nixrun checks test -x
# in its pure environment before executing the unchanged numeric suite.
@require(paths.INTAKE_PYTHON.parent.parent/'pyvenv.cfg',
         Path(__file__).resolve().parents[1]/'tools/nixrun.py')
class NumericWrapperChecks(unittest.TestCase):
    def test_one_isolated_numeric_suite(self):
        from pt_agent.experiments.r10_paper_metrics import run_process
        with TemporaryDirectory() as temp:
            work=Path(temp)
            argv=[sys.executable,'-B','-m','pt_agent.tools.nixrun','--seconds','90','--python',
                str(paths.INTAKE_PYTHON),'--','-B','-c',BOOTSTRAP,str(work)]
            observation=run_process(argv,cwd=paths.REPO/'DSL/DSL_agent/scripts',log=work/'numeric.log',
                resources=work/'resources.jsonl',timeout=120,mem_gib=4,fsize_gib=8,
                env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
            self.assertEqual(observation['exit_code'],0,(work/'numeric.log').read_text())
            self.assertFalse(observation['timed_out']);self.assertIsNone(observation['signal'])
            self.assertEqual(observation['remaining_pids'],[])
            result=json.loads((work/'numeric-result.json').read_text())
            self.assertEqual(result,dict(testsRun=5,failures=0,errors=0,skips=0,success=True))


if __name__=='__main__':unittest.main()
