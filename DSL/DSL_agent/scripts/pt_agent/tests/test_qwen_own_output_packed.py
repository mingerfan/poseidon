"""Synthetic source/state/resource checks for one cached packed continuation."""
import copy
import json
import os
from pathlib import Path
import signal
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from pt_agent.experiments import qwen_own_output_packed as packed
from pt_agent.experiments import qwen_diagnostic_jobs as jobs
from pt_agent.experiments.qwen_autoregressive_quality import advance_state


class ContinuationCases(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
            os.environ[key]='1'
        import numpy as np
        cls.np=np

    def fixture(self,directory):
        root=Path(directory)
        args=SimpleNamespace(own_output_result=root/'own/result.json',prefill_result=root/'prefill/result.json',
            prefill_logits=root/'prefill/worker/mock-logits.npy',count_result=root/'count.json',
            model_info=root/'model',reference=root/'original-reference.json',out=root/'out',
            linear_variant='compact-rectangular',job='quality-own-mock-step1',original_hf_reference=root/'original-reference.json')
        source=dict(count_result=str(args.count_result),model_info=str(args.model_info),original_hf_reference=str(args.reference))
        initial=dict(step=0,prefix_token_ids=[3,4]+[0]*30,prefix_length=2,
            attention_mask=[[1.,1.]+[0.]*30],sel=[[0.,1.]+[0.]*30])
        next_state=advance_state(initial,16,vocabulary_size=151936)
        rows=[dict(sample_id='train:'+str(i//2),step=i%2,input_state=copy.deepcopy(initial if i%2==0 else next_state),
            approximate_greedy_token=16 if i%2==0 else 15,logits_file='own-output-'+str(i)+'-logits.npy',
            all_domains_contained=True) for i in range(6)]
        own=dict(status='ok',job='quality-own-tensor',worker=dict(phase='own-output-tensor',
            own_output_generation_completed=True,approximate_tokens_fed_back=True,tensor_forward_attempts=6,
            tensor_forward_completed=6,state_update_attempts=3,state_update_completed=3,
            source=source,sample_rows=rows,training_quality_passed=True))
        cached=dict(status='ok',job='quality-mock',worker=dict(mock_forward_attempts=1,mock_forward_completed=1,
            independent_packed_math_passed=True,strict_tensor_mock_math=dict(passed=True),
            count_mock_schedule_matched=True,mock_sample=dict(sample_id='train:0',step=0),source=source))
        for path,record in ((args.own_output_result,own),(args.prefill_result,cached)):
            path.parent.mkdir(parents=True);path.write_text(json.dumps(record)+'\n')
        args.prefill_logits.parent.mkdir();args.prefill_logits.write_bytes(b'metadata only')
        reference=args.own_output_result.parent/'worker/own-output-1-logits.npy'
        reference.parent.mkdir();reference.write_bytes(b'metadata only')
        return args,own,cached

    def test_exact_accepted_sources_and_explicit_actual_packed_output_are_required(self):
        with tempfile.TemporaryDirectory() as directory:
            args,own,cached=self.fixture(directory)
            rows,reference,summary=packed.source_gate(args)
            self.assertEqual(reference,args.own_output_result.parent/'worker/own-output-1-logits.npy')
            self.assertEqual([row['step'] for row in rows],[0,1])
            self.assertEqual(summary['cached_prefill_logits'],str(args.prefill_logits.resolve()))
            for changed in (dict(cached,status='error'),dict(cached,job='quality-own-tensor')):
                args.prefill_result.write_text(json.dumps(changed))
                with self.assertRaises(ValueError):packed.source_gate(args)
            args.prefill_result.write_text(json.dumps(cached))
            changed=copy.deepcopy(own);changed['worker']['state_update_completed']=2
            args.own_output_result.write_text(json.dumps(changed))
            with self.assertRaises(ValueError):packed.source_gate(args)
            args.own_output_result.write_text(json.dumps(own))
            with self.assertRaises(ValueError):packed.source_gate(SimpleNamespace(**dict(vars(args),prefill_logits=Path(directory)/'fixed-step1.npy')))

    def test_actual_mock_argmax_drives_next_state_and_mismatch_is_preserved(self):
        np=self.np
        with tempfile.TemporaryDirectory() as directory:
            args,_,_=self.fixture(directory);rows,_,_=packed.source_gate(args)
            logits=np.zeros((1,1,151936),dtype=np.float64);logits[0,0,16]=1.
            current,metadata=packed.handoff(rows,logits,np)
            self.assertEqual(current['prefix_token_ids'][:4],[3,4,16,0])
            self.assertTrue(metadata['mock_argmax_matches_tensor'])
            self.assertTrue(metadata['actual_second_input_matches_tensor'])
            logits[0,0,17]=2.
            with self.assertRaises(ValueError):packed.handoff(rows,logits,np)
            logits[0,0,17]=0.
            changed=copy.deepcopy(rows);changed[1]['input_state']['prefix_token_ids'][2]=17
            with self.assertRaises(ValueError):packed.handoff(changed,logits,np)
            with self.assertRaises(ValueError):packed.handoff(rows,logits.astype(np.float32),np)

    def test_resident_guard_covers_supervisor_tree_and_stops_its_timer(self):
        from pt_agent.experiments import r10_paper_metrics as metrics
        current=os.getpid()
        with mock.patch.object(metrics,'_scan',return_value={
            123:dict(ppid=1,vmrss_kib=5),current:dict(ppid=123,vmrss_kib=7),
            456:dict(ppid=current,vmrss_kib=11),789:dict(ppid=1,vmrss_kib=999)}):
            self.assertEqual(packed._resident_snapshot(123),23)
            with self.assertRaises(ValueError):packed._resident_snapshot(789)
        before=signal.getsignal(signal.SIGALRM);observation={}
        with mock.patch.object(packed,'_resident_snapshot',side_effect=(10,6*1024**2)):
            with self.assertRaises(MemoryError):
                with packed.resident_guard(123,observation):signal.raise_signal(signal.SIGALRM)
        self.assertTrue(observation['stopped']);self.assertEqual(observation['resident_limit_gib'],5)
        self.assertEqual(signal.getitimer(signal.ITIMER_REAL),(0.,0.))
        self.assertEqual(signal.getsignal(signal.SIGALRM),before)

    def test_command_caps_and_completion_keep_cached_continuation_scope(self):
        with tempfile.TemporaryDirectory() as directory:
            args,_,_=self.fixture(directory);definition=jobs.job_definition(args);command=definition['command']
            self.assertEqual((definition['memory_gib'],definition['resident_limit_gib'],definition['timeout']),(12,5,5460))
            self.assertEqual(command[command.index('--seconds')+1],'5400')
            self.assertEqual(command[command.index('--prefill-logits')+1],str(args.prefill_logits))
            self.assertEqual(command[command.index('--supervisor-pid')+1],str(os.getpid()))
            record=dict(status='ok',source_checked=True,independent_packed_math_passed=True,
                cached_first_step_continuation_verified=True,strict_tensor_mock_math=dict(passed=True),
                count_mock_schedule_matched=True,handoff=dict(mock_argmax_matches_tensor=True,actual_second_input_matches_tensor=True),
                resident_guard=dict(stopped=False),mock_forward_attempts=1,mock_forward_completed=1,
                tensor_forward_attempts=0,tensor_forward_completed=0,retained_constant_arrays=0,
                constant_identity_entries=0,fresh_hf_forwards=0,approximation_selection_runs=0,api_calls=0,
                emit_runs=0,compiler_runs=0,keygen_runs=0,ckks_executions=0,heldout_samples=0,confirm_samples=0,report_samples=0)
            jobs.check_completed(record,args.job,args.linear_variant)
            for key,value in (('mock_forward_completed',0),('resident_guard',dict(stopped=True)),
                              ('handoff',dict(mock_argmax_matches_tensor=False,actual_second_input_matches_tensor=True)),
                              ('CKKS_latency',1.)):
                with self.subTest(key=key),self.assertRaises(ValueError):
                    jobs.check_completed(dict(record,**{key:value}),args.job,args.linear_variant)


if __name__=='__main__':unittest.main()
