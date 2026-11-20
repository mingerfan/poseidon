"""Fresh-source protocol and isolated public small-plan replay checks."""
from argparse import Namespace
import copy
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from pt_agent import paths
from pt_agent.experiments import r31_fresh_plan_math as mathjob
from pt_agent.tests._artifacts import require


def save(path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value)+'\n')


def source_fixture(root):
    source=root/'fresh';source.mkdir();gate=dict(source_case=str(source),model_info=str(source/'model_info.json'),
        program=str(source/'program.ptdsl'),probes=str(source/'training.npz'),probe_source=str(source/'probe.json'),
        factory='pt_agent.experiments.r13_common_models:resnet20_silu',normalization={'mean':[.485,.456,.406]},original_sources={'checkpoint':'original'})
    plan=source/'post-freeze/emit/final.runtime-plan.json'
    current=dict(gate=gate,runtime_plan=str(plan),source_metadata=dict(plan=dict(execution_count=17863,bootstrap_count=123,op_counts={'boot':123})))
    prepared=root/'prepared';work=prepared/'work';work.mkdir(parents=True)
    with (work/'input.bin').open('wb') as f:f.truncate(16384*8)
    values=[float(x) for x in range(10)]
    save(work/'expected.json',dict(indices=list(range(10)),values=values,tensor_values=values,sample_id='train:0',
        shape=[1,10],dtype='float64',old_expected_values_used=False,criterion=mathjob.fresh.CRITERION))
    worker=mathjob.fresh._initial_result();worker.update(status='ok',source_checked=True,reference_prepared=True,
        training_samples_consumed=1,training_npz_rows_loaded=3,source=current,input=str(work/'input.bin'),expected=str(work/'expected.json'),
        output_layout=dict(indices=list(range(10))),tensor_values=values)
    worker.update({k:1 for k in mathjob.fresh.COUNTERS})
    record=dict(round=mathjob.fresh.ROUND,mode=mathjob.fresh.MODE,model=mathjob.fresh.MODEL,status='ok',worker=worker,source=current)
    result=prepared/'result.json';save(result,record)
    archive=root/'archive.json';bundle=plan.with_name('final.bundle')
    save(archive,dict(case_result=str(source/'result.json'),files=[dict(kind='runtime_bundle',path=str(bundle),inventory=[])]))
    return result,source,archive,current,record


class FreshMathSourceChecks(unittest.TestCase):
    def test_current_source_and_new_train0_reference_are_bound(self):
        with TemporaryDirectory() as temp:
            result,case,archive,current,_=source_fixture(Path(temp))
            with patch.object(mathjob.fresh,'prepare_source',return_value=current):selected=mathjob.source_gate(result,case,archive)
            self.assertEqual(selected['sample_id'],'train:0');self.assertEqual(selected['plan'],current['runtime_plan'])
            self.assertEqual(selected['program'],current['gate']['program']);self.assertEqual(selected['reference_values'],[float(i) for i in range(10)])
            self.assertNotIn('R28',selected['scope']);self.assertNotIn('R29',selected['scope'])

    def test_foreign_program_factory_normalization_and_partial_preparation_rejected(self):
        changes=(lambda r:r['source']['gate'].update(program='/foreign/program'),
            lambda r:r['source']['gate'].update(factory='different'),
            lambda r:r['source']['gate'].update(normalization={}),
            lambda r:r['worker'].update(tensor_completed=0),
            lambda r:r['worker'].update(input='/previous/input.bin'))
        for change in changes:
            with TemporaryDirectory() as temp:
                result,case,archive,current,record=source_fixture(Path(temp));bad=copy.deepcopy(record);change(bad);save(result,bad)
                with patch.object(mathjob.fresh,'prepare_source',return_value=current),self.subTest(change=change),self.assertRaises(ValueError):mathjob.source_gate(result,case,archive)

    def test_closed_bundle_requires_current_canonical_path_and_case(self):
        for key in ('path','case_result'):
            with TemporaryDirectory() as temp:
                result,case,archive,current,_=source_fixture(Path(temp));closed=json.loads(archive.read_text())
                if key=='path':closed['files'][0]['path']=str(Path(temp)/'same-name/final.bundle')
                else:closed['case_result']=str(Path(temp)/'old90/result.json')
                save(archive,closed)
                with patch.object(mathjob.fresh,'prepare_source',return_value=current),self.subTest(key=key),self.assertRaises(ValueError):mathjob.source_gate(result,case,archive)

    def test_old_reference_artifact_or_changed_expected_values_rejected(self):
        with TemporaryDirectory() as temp:
            result,case,archive,current,record=source_fixture(Path(temp));expected=Path(record['worker']['expected'])
            values=json.loads(expected.read_text());values['old_expected_values_used']=True;save(expected,values)
            with patch.object(mathjob.fresh,'prepare_source',return_value=current),self.assertRaises(ValueError):mathjob.source_gate(result,case,archive)

    def test_source_failure_saves_unknown_counts_and_one_complete_registry_row(self):
        with TemporaryDirectory() as temp:
            root=Path(temp);results=root/'results';results.mkdir()
            args=Namespace(out=results/'new',input_prepare=root/'input.json',source_case=root/'source',archive_manifest=root/'archive.json',python=root/'python')
            with (patch.object(paths,'RESULTS_DIR',results),patch.object(mathjob,'source_gate',side_effect=ValueError('source fixture rejected')),
                  patch('pt_agent.registry.git_rev',return_value='fixture'),patch.object(mathjob,'run_process') as child):
                result=mathjob.execute(args)
            child.assert_not_called();self.assertEqual(result['status'],'error');self.assertIsNone(result['worker'])
            rows=[json.loads(x) for x in (results/'registry.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows),1);self.assertEqual(set(rows[0]),{'run_id','round','model','mode','git_rev','metrics','artifacts_dir','ts'})
            completion=json.loads((args.out/'completion.json').read_text());self.assertTrue(completion['registry_persisted'])
            spans=result['timing']['spans'];self.assertEqual(sum(r['exclusive_ns'] for r in spans),spans[0]['inclusive_ns'])


def numeric_suite():
    import numpy as np
    from pt_agent.experiments import r31_lattigo_runtime_plan as runtime
    from pt_agent.experiments import r31_lattigo_math_worker as replay
    def fixture(root,mismatch=False,nonfinite=False):
        original=root/'original';original.mkdir();plan_path,bundle,input_path,_=runtime.make_regression(original)
        plan=json.loads(plan_path.read_text());previous=plan['final_outputs'][0]
        plan['values'].append(dict(plan['values'][-1],id='boot-final',level=5))
        plan['execution'].append(dict(kind='compute',op='boot',ordinal=len(plan['initialization'])+len(plan['execution']),inputs=[previous],output='boot-final',
            place=dict(kind='host',rank=0),attrs=dict(implementation='native',target_level=5,target_scale_log2=43,target_components=2)))
        plan['final_outputs']=['boot-final'];save(plan_path,plan)
        values=np.fromfile(input_path,dtype='<f8');reference=(-values[:10]**2-.25)+(1 if mismatch else 0)
        if nonfinite:values[0]=np.nan;values.tofile(input_path)
        expected=original/'new-reference.json';save(expected,dict(values=reference.tolist(),indices=list(range(10)),sample_id='train:0',old_expected_values_used=False))
        archive=root/'archive.json';files=[]
        for file in bundle.rglob('*'):
            if file.is_file():files.append(dict(kind='file',path=bundle.name+'/'+str(file.relative_to(bundle)),bytes=file.stat().st_size))
        save(archive,dict(files=[dict(kind='runtime_bundle',path=str(bundle),inventory=files)]))
        manifest=json.loads((bundle/'manifest.json').read_text());inventory=runtime.plan_inventory(plan,manifest)
        source=dict(plan=str(plan_path),bundle=str(bundle),input=str(input_path),expected=str(expected),archive_manifest=str(archive),program=str(original/'current.ptdsl'),
            inventory=inventory,input_prepare=str(root/'prepared/result.json'),source_case=str(root/'current'),factory='official',scope=mathjob.SCOPE,
            reference_values=reference.tolist(),decode_indices=list(range(10)))
        source_record=root/'source-record.json';save(source_record,source)
        args=Namespace(work_dir=root/'worker',source_record=source_record,**{k:Path(source[k]) for k in ('plan','bundle','input','expected','archive_manifest','program')})
        return args,len(plan['execution'])
    class NumericChecks(unittest.TestCase):
        def run_fixture(self,root,**kwargs):
            args,count=fixture(root,**kwargs)
            with (patch.object(mathjob,'EXPECTED_EXECUTION',count),patch.object(mathjob,'EXPECTED_BOOTS',1),
                  patch.object(mathjob.shutil,'disk_usage',return_value=Namespace(free=20*1024**3))):return mathjob.worker(args)
        def test_real_small_replay_duplicate_dependencies_boot_identity_and_neutral_scope(self):
            with TemporaryDirectory() as temp:result=self.run_fixture(Path(temp))
            self.assertEqual(result['status'],'ok',result['error']);self.assertTrue(result['comparison']['passed'])
            self.assertEqual(result['op_completed'],12);self.assertEqual(len(result['bootstrap_observations']),1)
            self.assertEqual(result['replay_summary']['reference_scope'],mathjob.SCOPE);self.assertNotIn('R28',result['scope'])
            self.assertTrue(all(result[k]==0 for k in mathjob.ZERO_FIELDS))
        def test_strict_mathematical_failure_retains_completed_replay_and_logits(self):
            with TemporaryDirectory() as temp:
                root=Path(temp);result=self.run_fixture(root,mismatch=True)
                self.assertEqual(result['status'],'error');self.assertFalse(result['math_passed'])
                self.assertEqual(result['op_completed'],12);self.assertTrue((root/'worker/selected-logits.json').is_file())
                self.assertFalse((root/'worker/source-math-proof.json').exists())
        def test_nonfinite_input_stops_before_replay_and_keeps_unknown_op_count(self):
            with TemporaryDirectory() as temp:result=self.run_fixture(Path(temp),nonfinite=True)
            self.assertEqual(result['status'],'error');self.assertEqual(result['replay_attempts'],0)
            self.assertEqual((result['input_read_attempts'],result['input_read_completed']),(1,0));self.assertIsNone(result['op_completed'])
        def test_interrupted_replay_keeps_actual_boot_observation_and_unknown_prefix_ops(self):
            def fail(*args,**kwargs):
                kwargs['bootstrap_observer'](dict(ordinal=9,absmax=.2));raise RuntimeError('fixture interrupted replay')
            with TemporaryDirectory() as temp,patch.object(replay,'replay_plan',side_effect=fail):result=self.run_fixture(Path(temp))
            self.assertEqual(result['status'],'error');self.assertEqual((result['replay_attempts'],result['replay_completed']),(1,0))
            self.assertEqual(len(result['bootstrap_observations']),1);self.assertIsNone(result['op_completed'])
    return unittest.defaultTestLoader.loadTestsFromTestCase(NumericChecks)


BOOTSTRAP=r'''import json,sys,tempfile,unittest
from pathlib import Path
work=Path(sys.argv[1]);work.mkdir(exist_ok=True);tempfile.tempdir=str(work)
from pt_agent.tests.test_r31_fresh_plan_math import numeric_suite
result=unittest.TextTestRunner(verbosity=2).run(numeric_suite())
(work/'numeric-result.json').write_text(json.dumps(dict(testsRun=result.testsRun,failures=len(result.failures),errors=len(result.errors),skips=len(result.skipped),success=result.wasSuccessful()))+'\n')
raise SystemExit(0 if result.wasSuccessful() and result.testsRun==4 and not result.skipped else 1)
'''


@require(paths.ORBIT_ROOT.parent/'orbit-venv/bin/python')
class NumericWrapperChecks(unittest.TestCase):
    def test_one_public_numeric_suite(self):
        from pt_agent.experiments.r10_paper_metrics import run_process
        with TemporaryDirectory() as temp:
            root=Path(temp)
            observed=run_process([str(paths.ORBIT_ROOT.parent/'orbit-venv/bin/python'),'-B','-c',BOOTSTRAP,str(root)],
                cwd=paths.REPO/'DSL/DSL_agent/scripts',log=root/'numeric.log',resources=root/'resources.jsonl',timeout=120,mem_gib=4,fsize_gib=1,
                env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1'))
            self.assertEqual(observed['exit_code'],0,(root/'numeric.log').read_text());self.assertEqual(observed['remaining_pids'],[])
            self.assertEqual(json.loads((root/'numeric-result.json').read_text()),dict(testsRun=4,failures=0,errors=0,skips=0,success=True))


if __name__=='__main__':unittest.main()
