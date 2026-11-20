"""R26 synthetic heldout records and tiny models, without original artifacts."""

from contextlib import ExitStack, nullcontext, redirect_stderr
import copy
import io
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


class QualityCases(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
            os.environ[name]='1'
        from pt_agent.experiments import r26_orbit_quality_worker as worker
        from pt_agent.experiments.r10_paper_metrics import StageRecorder
        cls.numeric_environment=worker._initialize_torch()
        cls.worker,cls.Recorder=worker,StageRecorder
        cls.np,cls.torch=worker._numeric_modules()

    def setUp(self):
        temporary=TemporaryDirectory(prefix='r26-inner-',dir=os.environ.get('R26_TEST_WORK'))
        self.addCleanup(temporary.cleanup);self.case=Path(temporary.name)
        self.work=self.case/'work/checks-worker';self.work.mkdir(parents=True)
        self.recorder=self.Recorder(self.work/'events.jsonl')
        self.records=self.case/'records';self.records.mkdir()
        self.args=SimpleNamespace(work_dir=self.work,records=self.records,model_info=self.case/'model_info.json',
            model_source=self.case/'model.py',checkpoint=self.case/'state.fixture',poly_dir=self.case/'poly',
            **{key+'_result':self.case/(key+'.json') for key in ('r13','r16','r22','r24','r25')})
        self.metadata=dict(format='poseidon-cifar10-binary-records-v1',record_bytes=3073,
            row_label_byte=0,preprocessing_applied=False,train_count=50000,test_count=10000,
            pixel_layout='uint8 planar RGB 3x32x32, C order',
            shards=[dict(file='test-09000.records',split='test',start=9000,count=1000)])
        self.info=dict(task='classification',metric='classification_top1',threshold=dict(acc_drop_pp=0.5),
            inputs=[dict(kind='features',dtype='float64',shape=[1,3,32,32])],
            split=dict(kind='legacy_cifar',heldout=list(range(9000,10000))),
            source=dict(normalization=dict(mean=[.485,.456,.406],std=[.229,.224,.225]),
                preprocessing_provenance='official inference test',
                calibration_source=dict(records=str(self.records)),records=str(self.records)))

    def _dataset(self,*,labels=None):
        self.np_data=self.np.zeros((1000,3073),dtype=self.np.uint8)
        self.np_data[:,0]=self.np.arange(1000,dtype=self.np.int64)%10 if labels is None else labels
        self.np_data[:,1:1025]=255;self.np_data[:,1025:2049]=128
        (self.records/'test-09000.records').write_bytes(self.np_data.tobytes())
        (self.records/'dataset.json').write_text(json.dumps(self.metadata))

    def _model(self,*,short_forward=False):
        torch=self.torch
        class Block(torch.nn.Module):
            def __init__(self):super().__init__();self.mish=torch.nn.SiLU()
        class FixtureModel(torch.nn.Module):
            def __init__(self):
                super().__init__();self.gain=torch.nn.Parameter(torch.ones((),dtype=torch.float64))
                self.bn=torch.nn.ModuleList([torch.nn.BatchNorm1d(1) for _ in range(19)])
                self.mish=torch.nn.SiLU();self.blocks=torch.nn.ModuleList([Block() for _ in range(9)])
                for i in range(20):self.register_buffer('fixture_'+str(i),torch.zeros((),dtype=torch.float64))
            def forward(self,images):
                value=images.mean(dim=(1,2,3)).reshape(-1,1)*self.gain
                value=self.mish(self.bn[0](value))
                for i,block in enumerate(self.blocks):
                    value=block.mish(value)
                    if not(short_forward and i==8):value=block.mish(value)
                return torch.cat((torch.zeros_like(value),value,torch.zeros_like(value).expand(-1,8)),dim=1)
        model=FixtureModel().double().eval()
        self.assertEqual(len(model.state_dict()),116)
        self.assertEqual(sum(value.dtype==torch.int64 for value in model.state_dict().values()),19)
        return model

    def _native(self,*,constant=False):
        metadata=dict(poly_closure_attempts=0,poly_closure_completed=0,closure_lengths=[],closure_scales=[])
        def native(name,values):
            metadata['poly_closure_attempts']+=1
            metadata['closure_lengths'].append(16);metadata['closure_scales'].append(1.0)
            output=self.np.full_like(values,0.5) if constant else values*0.5
            metadata['poly_closure_completed']+=1
            return output
        reference=(lambda coefficients,values:self.np.full_like(values,0.5)) if constant else (lambda coefficients,values:values*0.5)
        return native,self.np.zeros(96,dtype=self.np.float64),metadata,reference

    def test_single_heldout_shard_is_read_once_and_normalized_in_fixed_float64_order(self):
        self._dataset();opened=Mock(wraps=self.np.memmap)
        with patch.object(self.np,'memmap',opened):images,labels,source=self.worker.read_heldout(self.args,self.info)
        opened.assert_called_once()
        self.assertEqual(Path(opened.call_args.args[0]),self.records/'test-09000.records')
        self.assertEqual(opened.call_args.kwargs['mode'],'r')
        self.assertEqual(images.shape,(1000,3,32,32));self.assertEqual(images.dtype,self.np.float64)
        self.assertEqual(labels.dtype,self.np.int64)
        self.np.testing.assert_array_equal(labels,self.np.arange(1000)%10)
        expected=self.np_data[:,1:].reshape(1000,3,32,32).astype(self.np.float64)/255.0
        expected=(expected-self.np.array([.485,.456,.406])[None,:,None,None])/self.np.array([.229,.224,.225])[None,:,None,None]
        self.np.testing.assert_array_equal(images,expected)
        self.assertTrue(self.np.isfinite(images).all())
        self.assertEqual(source['sample_ids'][0],'test:9000');self.assertEqual(source['sample_ids'][-1],'test:9999')

    def test_heldout_metadata_rejects_other_split_start_counts_and_pixel_layout(self):
        self._dataset();original=copy.deepcopy(self.metadata)
        for variant in ('record_bytes','layout','train_count','split','start','count'):
            metadata=copy.deepcopy(original)
            if variant=='record_bytes':metadata['record_bytes']=3072
            elif variant=='layout':metadata['pixel_layout']='interleaved RGB'
            elif variant=='train_count':metadata['train_count']=40000
            elif variant=='split':metadata['shards'][0]['split']='train'
            elif variant=='start':metadata['shards'][0]['start']=8000
            else:metadata['shards'][0]['count']=999
            (self.records/'dataset.json').write_text(json.dumps(metadata))
            with self.subTest(variant=variant),self.assertRaises(self.worker.WorkerFailure):
                self.worker.read_heldout(self.args,self.info)

    def test_heldout_rejects_truncated_shard_and_labels_outside_original_classes(self):
        self._dataset(labels=self.np.full(1000,10,dtype=self.np.uint8))
        with self.assertRaises(self.worker.WorkerFailure):self.worker.read_heldout(self.args,self.info)
        self._dataset();path=self.records/'test-09000.records';path.write_bytes(path.read_bytes()[:-1])
        with self.assertRaises(self.worker.WorkerFailure):self.worker.read_heldout(self.args,self.info)

    def test_heldout_rejects_foreign_source_and_normalization_policy(self):
        self._dataset()
        for variant in ('source','mean'):
            info=copy.deepcopy(self.info)
            if variant=='source':
                info['source']['calibration_source']['records']=str(self.case/'other-records')
                info['source']['records']=str(self.case/'other-records')
            else:info['source']['normalization']['mean'][0]=0.5
            with self.subTest(variant=variant),self.assertRaises(self.worker.WorkerFailure):
                self.worker.read_heldout(self.args,info)

    def _state_and_report(self,model):
        state={name:value.clone() for name,value in model.state_dict().items()}
        report=dict(passed=True,strict_load=True,extra_state=[],
            structure=dict(passed=True,activation_occurrences=19),
            original_state=[dict(name=name,finite=True,numerically_equal=True) for name in state])
        return state,report

    def _fresh_sample(self,result):
        result.update(active_sample_id='synthetic:0',active_index=0,
            current_sample_stats=dict(self.worker._new_statistics(),module_calls={}))

    def test_original_loader_reuses_one_state_read_and_strict_build_with_integer_bn_inventory(self):
        model=self._model();state,report=self._state_and_report(model);result=self.worker._initial_result()
        with patch.object(self.worker.native,'_load_state',return_value=state) as loaded, \
             patch.object(self.worker.native,'_build_model',return_value=(model,report)) as built, \
             patch.object(self.worker.shutil,'disk_usage',return_value=SimpleNamespace(free=11*1024**3)):
            observed=self.worker.load_original_model(self.args,self.recorder,result)
        loaded.assert_called_once();built.assert_called_once()
        self.assertIs(observed,model);self.assertTrue(result['original_model_loaded'])
        self.assertEqual((result['checkpoint_load_attempts'],result['checkpoint_load_completed']),(1,1))
        self.assertEqual((result['model_build_attempts'],result['model_build_completed']),(1,1))
        self.assertTrue(all(not child.training for child in observed.modules()))

    def test_original_loader_rejects_state_count_integer_bn_and_non_strict_report(self):
        for variant in ('state_count','integer_count','strict'):
            model=self._model();state,report=self._state_and_report(model)
            if variant=='state_count':state.pop(next(reversed(state)))
            elif variant=='integer_count':
                name=next(name for name,value in state.items() if value.dtype==self.torch.int64)
                state[name]=state[name].to(self.torch.float64)
            else:report['strict_load']=False
            result=self.worker._initial_result()
            with self.subTest(variant=variant), \
                 patch.object(self.worker.native,'_load_state',return_value=state), \
                 patch.object(self.worker.native,'_build_model',return_value=(model,report)), \
                 patch.object(self.worker.shutil,'disk_usage',return_value=SimpleNamespace(free=11*1024**3)), \
                 self.assertRaises(self.worker.WorkerFailure):
                self.worker.load_original_model(self.args,self.recorder,result)
            self.assertFalse(result['original_model_loaded']);self.assertIsNone(result['validation_metrics'])

    def test_physical_scale_full_reference_and_ten_replacements_preserve_original_state(self):
        original=self._model();before={name:value.clone() for name,value in original.state_dict().items()}
        native,coefficients,metadata,reference=self._native();native=Mock(side_effect=native)
        reference=Mock(side_effect=reference);result=self.worker._initial_result();result['native_metadata']=metadata
        self._fresh_sample(result)
        with patch.object(self.worker.native,'odd_chebyshev_reference',reference), \
             patch.object(self.worker.shutil,'disk_usage',return_value=SimpleNamespace(free=11*1024**3)):
            approximate=self.worker.build_approximation(original,native,coefficients,self.args,self.recorder,result)
            with self.torch.no_grad():observed=approximate.mish(self.torch.tensor([[-64.0,16.0]],dtype=self.torch.float64))
        native.assert_called_once();reference.assert_called_once()
        self.np.testing.assert_array_equal(native.call_args.args[1],[[-2.0,.5]])
        self.np.testing.assert_array_equal(reference.call_args.args[1],[[-2.0,.5]])
        self.np.testing.assert_array_equal(observed.numpy(),[[-32.0,8.0]])
        self.assertEqual(result['silu_modules_replaced'],10)
        self.assertEqual((result['deepcopy_attempts'],result['deepcopy_completed']),(1,1))
        self.assertTrue(result['replacement_state_check']['passed'])
        self.assertEqual(result['module_stats']['mish']['outside_standard_interval_elements'],1)
        for name,value in original.state_dict().items():
            self.assertTrue(self.torch.equal(value,before[name]))
            self.assertTrue(self.torch.equal(approximate.state_dict()[name],before[name]))
        self.assertTrue(self.worker.native._state_isolated(original,approximate)['passed'])
        self.assertIsInstance(original.mish,self.torch.nn.SiLU)

    def test_activation_rejects_float32_before_native_or_reference_arithmetic(self):
        original=self._model();native=Mock();reference=Mock();result=self.worker._initial_result()
        self._fresh_sample(result)
        with patch.object(self.worker.native,'odd_chebyshev_reference',reference), \
             patch.object(self.worker.shutil,'disk_usage',return_value=SimpleNamespace(free=11*1024**3)):
            approximate=self.worker.build_approximation(original,native,self.np.zeros(96),self.args,self.recorder,result)
            with self.assertRaises(self.worker.WorkerFailure):
                approximate.mish(self.torch.zeros((1,1),dtype=self.torch.float32))
        native.assert_not_called();reference.assert_not_called()
        self.assertEqual((result['activation_attempts'],result['activation_completed']),(1,0))
        self.assertEqual(result['math_check_attempts'],0)

    def _run(self,*,rejected=False,fail_math_at=None,short_forward=False):
        model=self._model(short_forward=short_forward);self.last_model=model
        state,report=self._state_and_report(model)
        native,coefficients,metadata,reference=self._native(constant=rejected)
        reference_calls=0
        def checked_reference(coefficients,values):
            nonlocal reference_calls
            reference_calls+=1;observed=reference(coefficients,values)
            return observed+1.0 if reference_calls==fail_math_at else observed
        images=self.np.zeros((1000,3,32,32),dtype=self.np.float64)
        labels=self.np.zeros(1000,dtype=self.np.int64)
        source=dict(raw_cifar_reads=1,heldout_rows_read=1000,
            sample_ids=['test:'+str(i) for i in range(9000,10000)],labels=labels.tolist())
        result=self.worker._initial_result();self.last_result=result
        with ExitStack() as stack:
            checked=stack.enter_context(patch.object(self.worker,'_check_sources',return_value=({'fixture':True},self.info)))
            stack.enter_context(patch.object(self.worker,'_initialize_torch',return_value=self.numeric_environment))
            loaded=stack.enter_context(patch.object(self.worker.native,'_load_state',return_value=state))
            built=stack.enter_context(patch.object(self.worker.native,'_build_model',return_value=(model,report)))
            functions=stack.enter_context(patch.object(self.worker,'_native_functions',return_value=(native,coefficients,metadata)))
            heldout=stack.enter_context(patch.object(self.worker,'read_heldout',return_value=(images,labels,source)))
            stack.enter_context(patch.object(self.worker.native,'odd_chebyshev_reference',side_effect=checked_reference))
            stack.enter_context(patch.object(self.worker.shutil,'disk_usage',return_value=SimpleNamespace(free=11*1024**3)))
            self.worker.run_validation(self.args,self.recorder,result)
        return result,dict(checked=checked,loaded=loaded,built=built,functions=functions,heldout=heldout)

    def _assert_complete(self,result):
        self.assertTrue(result['paired_quality_completed']);self.assertTrue(result['native_activation_math_passed'])
        self.assertEqual((result['heldout_rows_read'],result['evaluated_pairs']),(1000,1000))
        for prefix in ('native_load','checkpoint_load','model_build','deepcopy','heldout_load'):
            self.assertEqual((result[prefix+'_attempts'],result[prefix+'_completed']),(1,1))
        for prefix in ('original_forward','approx_forward'):
            self.assertEqual((result[prefix+'_attempts'],result[prefix+'_completed']),(1000,1000))
        for prefix in ('activation','math_check','native_closure'):
            self.assertEqual((result[prefix+'_attempts'],result[prefix+'_completed']),(19000,19000))
        self.assertEqual(result['module_stats']['mish']['calls_completed'],1000)
        self.assertTrue(all(statistics['calls_completed']==2000 for name,statistics in result['module_stats'].items() if name!='mish'))
        self.assertEqual(len(result['sample_rows']),1000)
        self.assertTrue(all(row['activation_math']['calls_completed']==19 for row in result['sample_rows']))
        self.assertEqual(len((self.work/'samples.jsonl').read_text().splitlines()),1000)
        self.assertTrue(all(result[key]==0 for key in self.worker.ZERO_FIELDS))
        self.assertTrue(all(result[key] is False for key in self.worker.FALSE_FIELDS))
        self.assertTrue(all(result[key] is None for key in self.worker.UNMEASURED_FIELDS))
        self.assertEqual(result['native_packed_equivalence_scope'],self.worker.SCOPE)
        self.assertTrue(result['final_original_state_check']['passed'])
        self.assertTrue(result['final_state_check']['passed'])

    def test_full_synthetic_thousand_pairs_keep_original_compare_policy_and_native_counts(self):
        result,hooks=self._run()
        for hook in hooks.values():hook.assert_called_once()
        self._assert_complete(result)
        self.assertEqual(result['status'],'ok');self.assertTrue(result['paired_quality_policy_passed'])
        self.assertEqual(result['validation_metrics'],dict(agree=1000,total=1000,top1=100.0,exact_top1=100.0,acc_drop_pp=0.0))

    def test_complete_quality_rejection_keeps_all_math_and_quality_facts_without_tuning(self):
        result,hooks=self._run(rejected=True)
        for hook in hooks.values():hook.assert_called_once()
        self._assert_complete(result)
        self.assertEqual(result['status'],'quality_rejected');self.assertFalse(result['paired_quality_policy_passed'])
        self.assertEqual(result['validation_metrics'],dict(agree=0,total=1000,top1=0.0,exact_top1=100.0,acc_drop_pp=100.0))
        self.assertIsNone(result['accuracy_acceptance']);self.assertIsNone(result['domain_acceptance'])

    def test_first_independent_math_error_keeps_shard_rows_and_stops_before_third_pair(self):
        with self.assertRaises(self.worker.WorkerFailure) as error:self._run(fail_math_at=39)
        self.assertEqual(error.exception.status,'math_failed');result=self.last_result
        self.assertEqual((result['heldout_rows_read'],result['evaluated_pairs']),(1000,2))
        self.assertEqual((result['original_forward_attempts'],result['original_forward_completed']),(3,3))
        self.assertEqual((result['approx_forward_attempts'],result['approx_forward_completed']),(3,2))
        self.assertEqual((result['activation_attempts'],result['activation_completed']),(39,39))
        self.assertEqual((result['math_check_attempts'],result['math_check_completed']),(39,39))
        self.assertFalse(result['native_activation_math_passed']);self.assertFalse(result['paired_quality_completed'])
        self.assertIsNone(result['validation_metrics']);self.assertIsNone(result['paired_quality_policy_passed'])
        saved=json.loads((self.work/'phase_result.json').read_text())
        self.assertEqual(saved['active_sample_id'],'test:9002');self.assertEqual(len(saved['sample_rows']),2)

    def test_wrong_per_image_activation_count_stops_before_counting_a_pair(self):
        with self.assertRaises(self.worker.WorkerFailure):self._run(short_forward=True)
        result=self.last_result
        self.assertEqual(result['original_forward_completed'],1)
        self.assertEqual(result['approx_forward_completed'],1)
        self.assertEqual(result['evaluated_pairs'],0)
        self.assertEqual(result['activation_completed'],18)
        self.assertIsNone(result['validation_metrics'])

    def test_worker_main_returns_zero_for_completed_quality_rejection(self):
        work=self.case/'main/work/checks-worker'
        arguments=[]
        for name in ('work_dir','model_source','checkpoint','poly_dir','records','model_info',
                     'r13_result','r16_result','r22_result','r24_result','r25_result'):
            arguments+=['--'+name.replace('_','-'),str(work if name=='work_dir' else getattr(self.args,name))]
        def rejected(args,recorder,result):
            result.update(status='quality_rejected',paired_quality_completed=True,
                native_activation_math_passed=True,paired_quality_policy_passed=False,
                validation_metrics=dict(agree=0,total=1000,top1=0.0,exact_top1=100.0,acc_drop_pp=100.0))
        with patch.object(self.worker,'run_validation',side_effect=rejected) as validation, \
             patch.object(self.worker,'_budget_guard',side_effect=lambda *args:nullcontext()), \
             patch.object(self.worker.resource,'setrlimit'),patch.object(self.worker.signal,'signal'), \
             patch.object(self.worker.shutil,'disk_usage',return_value=SimpleNamespace(free=11*1024**3)), \
             redirect_stderr(io.StringIO()):
            code=self.worker.main(arguments)
        validation.assert_called_once();self.assertEqual(code,0)
        self.assertEqual(json.loads((work/'phase_result.json').read_text())['status'],'quality_rejected')


if __name__=='__main__':
    unittest.main()
