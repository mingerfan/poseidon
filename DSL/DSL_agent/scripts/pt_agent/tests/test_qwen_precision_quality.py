"""Small synthetic protocol/math tests, with no original models or samples."""

import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from pt_agent.experiments import qwen_precision_quality as quality
from pt_agent.experiments import qwen_diagnostic_jobs as jobs


class QualityProtocolCases(unittest.TestCase):
    def test_fixed_measurement_rejects_partial_and_retains_quality_rejection(self):
        record=dict(status='ok',tensor_forward_attempts=6,tensor_forward_completed=6,
            mock_forward_attempts=0,mock_forward_completed=0,source_checked=True,
            training_prefix_measurement_completed=True,all_finite=True,
            approximate_tokens_fed_back=False,training_quality_passed=False)
        jobs.check_completed(record,'quality-tensor','compact-rectangular')
        for key,value in (('tensor_forward_completed',5),('tensor_forward_attempts',7),
                          ('approximate_tokens_fed_back',True),('CKKS_latency',.1)):
            with self.subTest(key=key),self.assertRaises(ValueError):
                jobs.check_completed(dict(record,**{key:value}),'quality-tensor','compact-rectangular')

    def test_quality_commands_and_reference_only_resource_arguments(self):
        args=SimpleNamespace(job='quality-tensor',out=Path('/synthetic/out'),
            linear_variant='compact-rectangular',model_info=Path('/synthetic/model'),
            original_hf_reference=Path('/synthetic/hf.json'),count_result=Path('/synthetic/count.json'))
        definition=jobs.job_definition(args)
        self.assertEqual((definition['memory_gib'],definition['timeout']),(6,960))
        command=definition['command']
        self.assertIn('pt_agent.tools.nixrun',command)
        self.assertEqual(command[command.index('--phase')+1],'tensor')
        self.assertEqual(command[command.index('--count-result')+1],'/synthetic/count.json')
        from pt_agent.experiments import qwen_server_job as server
        with mock.patch.object(jobs,'main',return_value=0) as dispatch:
            self.assertEqual(server.main(['--out','/synthetic/out','--stage','quality-tensor',
                '--memory-gib','999','--phase-seconds','1']),0)
        forwarded=dispatch.call_args.args[0]
        self.assertNotIn('--memory-gib',forwarded)
        self.assertNotIn('--phase-seconds',forwarded)

    def test_single_packed_command_and_original_math_completion(self):
        args=SimpleNamespace(job='quality-mock',out=Path('/synthetic/out'),
            linear_variant='compact-rectangular',model_info=Path('/synthetic/model'),
            original_hf_reference=Path('/synthetic/hf.json'),count_result=Path('/synthetic/count.json'),
            tensor_result=Path('/synthetic/tensor.json'))
        definition=jobs.job_definition(args);command=definition['command']
        self.assertEqual((definition['memory_gib'],definition['timeout']),(12,5460))
        self.assertEqual(command[command.index('--seconds')+1],'5400')
        self.assertEqual(command[command.index('--phase')+1],'mock')
        self.assertEqual(command[command.index('--tensor-result')+1],'/synthetic/tensor.json')
        from pt_agent.experiments import qwen_server_job as server
        with mock.patch.object(jobs,'main',return_value=0) as dispatch:
            self.assertEqual(server.main(['--out','/synthetic/out','--stage','quality-mock',
                '--tensor-result','/synthetic/fixed-tensor.json']),0)
        forwarded=dispatch.call_args.args[0]
        self.assertEqual(forwarded[forwarded.index('--job')+1],'quality-mock')
        self.assertEqual(forwarded[forwarded.index('--tensor-result')+1],'/synthetic/fixed-tensor.json')
        record=dict(status='ok',source_checked=True,tensor_forward_attempts=0,tensor_forward_completed=0,
            mock_forward_attempts=1,mock_forward_completed=1,retained_constant_arrays=0,
            constant_identity_entries=0,independent_packed_math_passed=True,
            strict_tensor_mock_math={'passed':True},count_mock_schedule_matched=True)
        jobs.check_completed(record,'quality-mock','compact-rectangular')
        for key,value in (('mock_forward_completed',0),('mock_forward_attempts',2),
                          ('strict_tensor_mock_math',{'passed':False}),('count_mock_schedule_matched',False)):
            with self.subTest(key=key),self.assertRaises(ValueError):
                jobs.check_completed(dict(record,**{key:value}),'quality-mock','compact-rectangular')


class QualityMathCases(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
            os.environ[name]='1'
        import numpy as np
        cls.np=np

    def test_strict_arithmetic_and_dtype_rejection_keep_original_policy(self):
        np=self.np
        values=np.array([1.,2.],dtype=np.float64)
        self.assertTrue(quality.strict_math(values,values.copy(),np)['passed'])
        self.assertFalse(quality.strict_math(values,values+1e-7,np)['passed'])
        self.assertEqual(quality.strict_math(values,values,np)['criterion'],'1e-12+1e-12*abs(reference)')
        with self.assertRaises(ValueError):quality.strict_math(values,values.astype(np.float32),np)

    def test_domain_observations_keep_mask_and_nonfinite_separate(self):
        np=self.np
        value=np.array([.1,.9,np.nan],dtype=np.float64)
        attrs=dict(fn='inv',method='newton',domain=(.05,1.))
        record=quality.domain_record(value,attrs,np,np.array([True,True,False]))
        self.assertTrue(record['contained']);self.assertEqual(record['elements'],2)
        self.assertFalse(quality.domain_record(value,attrs,np)['all_finite'])
        self.assertFalse(quality.domain_record(np.array([1.1]),attrs,np)['contained'])

    def test_original_prefix_selector_requires_true_last_active_token(self):
        np=self.np
        class Embedding:
            shape=(151936,896)
            def __len__(self):return self.shape[0]
            def __getitem__(self,ids):return np.repeat(ids[:,None],896,axis=1).astype(np.float64)
        info=SimpleNamespace(graph=SimpleNamespace(weights={'original.model.embed_tokens.weight':Embedding()}))
        row=dict(prefix_token_ids=list(range(32)),prefix_length=9,
            attention_mask=[[1.]*9+[0.]*23],sel=[[0.]*8+[1.]+[0.]*23])
        arrays=quality.arrays_for(info,row,np)
        self.assertEqual(arrays['inputs_embeds'].shape,(1,32,896))
        self.assertEqual(arrays['inputs_embeds'][0,8,0],8.)
        changed=json.loads(json.dumps(row));changed['sel']=[[1.]+[0.]*31]
        with self.assertRaises(ValueError):quality.arrays_for(info,changed,np)

    def test_observed_mock_preserves_arithmetic_and_writes_small_progress(self):
        np=self.np
        from pt_agent.backends.streaming_mock import StreamingMockBackend
        from pt_agent.experiments.qwen_mock_progress import ObservedStreamingMockBackend,progress_records
        original=StreamingMockBackend(8,batch=1);observed=ObservedStreamingMockBackend(8,batch=1)
        inputs=[np.arange(8,dtype=np.float64)[None,:]]
        original.bind(inputs);observed.bind(inputs)
        def arithmetic(backend):
            x=backend.input(0)
            value=(x*2.+x.rotate(1))-x
            backend.output([value]);return value.value
        reference=arithmetic(original)
        with tempfile.TemporaryDirectory() as directory:
            result={};observed.op_context=7
            program=SimpleNamespace(ops=[SimpleNamespace(id=7,kind='linear',lineno=11)])
            with progress_records(observed,program,Path(directory),result):
                actual=arithmetic(observed)
            rows=[json.loads(line) for line in (Path(directory)/'mock-progress.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows),2);self.assertEqual(rows[-1]['status'],'stopped')
            self.assertEqual(rows[-1]['ops_total'],4)
            self.assertEqual(rows[-1]['active_op'],{'id':7,'kind':'linear','line':11})
            self.assertIn('worker_rss_kib',rows[-1]['memory'])
            self.assertLess(len(json.dumps(rows)),5000)
        self.assertTrue(quality.strict_math(reference,actual,np)['passed'])
        self.assertEqual(observed.stats()['counts'],dict(add=1,sub=1,neg=0,mul_cc=0,mul_cp=1,rot=1,boot=0))
        self.assertEqual(observed.stats()['ops_attempted'],4)
        self.assertEqual(observed.stats()['ops_total'],4)
        self.assertEqual(observed.stats()['rot_steps'],[1])
        self.assertTrue(all(item.data is None for item in observed._constants.values()))
        self.assertEqual(len(observed._constant_objects),0)
        original.release_batch();observed.release_batch()


if __name__=='__main__':
    unittest.main()
