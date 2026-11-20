"""Checkpoint semantic coverage fixtures; no real checkpoint or model imports."""
from argparse import Namespace
import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from pt_agent.experiments import bert_checkpoint_audit as audit


def row(shape,dtype='float32',equal=False):
    return dict(shape=list(shape),dtype=dtype,finite=True,**(dict(numerically_equal=True) if equal else {}))


def states():
    checkpoint={'bert.layer.weight':row([4,4]),'classifier.weight':row([2,4]),'classifier.bias':row([2])}
    loaded={key:dict(value,numerically_equal=True) for key,value in checkpoint.items()}
    info={key:[] for key in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')}
    return checkpoint,loaded,info


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value)+'\n')


def source_fixture(root):
    model=root/'model-source';source=dict(model_type='bert',architectures=['BertForSequenceClassification'],
        hidden_size=4,num_hidden_layers=2,num_labels=2)
    save(model/'config.json',source);(model/'pytorch_model.bin').write_text('synthetic metadata placeholder')
    target=root/'intake/model/model_info.json';save(target,dict(source=dict(local_path=str(model),config=source),
        task='classification',metric='classification_top1',threshold={'acc_drop_pp':.5}))
    prepared=root/'intake/phases/001-prepare/worker_result.json'
    result=dict(status='ok',phase='prepare',model=audit.MODEL,import_completed=True,model_dir=str(target.parent),
        model_metadata=dict(unchanged_original_model=str(model)),reference_checks=[dict(passed=True) for _ in range(3)])
    result.update(dict.fromkeys(('native_test_reads','report_samples','confirm_samples','llm_calls','compiler_runs',
        'keygen_runs','real_ckks_attempts','real_ckks_completed'),0));save(prepared,result)
    return model,prepared,target


class TrainedCheckpointCoverageChecks(unittest.TestCase):
    def test_every_original_state_and_classifier_key_is_covered(self):
        checkpoint,loaded,info=states()
        actual=audit.validate_coverage(checkpoint,loaded,info,hidden_size=4,num_labels=2)
        self.assertTrue(actual['passed']);self.assertEqual(actual['checkpoint_key_count'],3)
        self.assertTrue(actual['classifier']['both_original_checkpoint_keys'])
        self.assertEqual(actual['verified_nonpersistent_keys'],[])

    def test_missing_classifier_unknown_key_shape_dtype_finite_and_value_mismatches_reject(self):
        for kind in ('missing_head','unknown','shape','dtype','finite','value'):
            checkpoint,loaded,info=states()
            if kind=='missing_head':checkpoint.pop('classifier.weight')
            elif kind=='unknown':checkpoint['another_head.weight']=row([2,4])
            elif kind=='shape':checkpoint['classifier.weight']['shape']=[4,2]
            elif kind=='dtype':checkpoint['classifier.bias']['dtype']='float64'
            elif kind=='finite':checkpoint['bert.layer.weight']['finite']=False
            else:loaded['classifier.bias']['numerically_equal']=False
            with self.subTest(kind=kind),self.assertRaises(ValueError):
                audit.validate_coverage(checkpoint,loaded,info,hidden_size=4,num_labels=2)

    def test_unresolved_hf_loading_info_is_rejected_even_when_state_metadata_matches(self):
        for field in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs'):
            checkpoint,loaded,info=states();info[field]=['classifier.weight']
            with self.subTest(field=field),self.assertRaises(ValueError):
                audit.validate_coverage(checkpoint,loaded,info,hidden_size=4,num_labels=2)
        checkpoint,loaded,info=states();info.pop('missing_keys')
        with self.assertRaises(ValueError):audit.validate_coverage(checkpoint,loaded,info,hidden_size=4,num_labels=2)

    def test_legacy_nonpersistent_buffer_requires_existing_exact_shape_dtype_and_values(self):
        checkpoint,loaded,info=states();name='bert.embeddings.position_ids';checkpoint[name]=row([1,512],'int64')
        buffers={name:row([1,512],'int64',True)};info['unexpected_keys']=[name]
        result=audit.validate_coverage(checkpoint,loaded,info,hidden_size=4,num_labels=2,nonpersistent=buffers)
        self.assertEqual(result['verified_nonpersistent_keys'],[name])
        self.assertEqual(next(r for r in result['rows'] if r['name']==name)['role'],'verified_nonpersistent_buffer')
        buffers[name]['numerically_equal']=False
        with self.assertRaises(ValueError):audit.validate_coverage(checkpoint,loaded,info,hidden_size=4,num_labels=2,nonpersistent=buffers)

    def test_source_gate_requires_same_accepted_original_intake_and_preserves_weights_unread(self):
        with TemporaryDirectory() as temp:
            model,prepared,info=source_fixture(Path(temp))
            selected=audit.source_gate(model,prepared,info)
            self.assertEqual(selected['checkpoint'],str(model/'pytorch_model.bin'))
            self.assertFalse(selected['original_model_changed'])
            bad=json.loads(prepared.read_text());bad['model_metadata']['unchanged_original_model']=str(model.parent/'other')
            save(prepared,bad)
            with self.assertRaises(ValueError):audit.source_gate(model,prepared,info)

    def test_outer_runs_only_one_explicit_original_worker_and_registers_complete_case(self):
        with TemporaryDirectory() as temp:
            root=Path(temp);model,prepared,info=source_fixture(root);results=root/'results';results.mkdir()
            args=Namespace(model=model,prepare_result=prepared,model_info=info,out=results/'audit',intake_python=Path('/existing/intake/python'))
            def run(command,**kwargs):
                self.assertIn('--python',command);self.assertIn(str(args.intake_python),command)
                self.assertEqual(kwargs['mem_gib'],4);self.assertEqual(kwargs['timeout'],420)
                worker=dict(status='ok',checkpoint_semantic_coverage_passed=True,checkpoint_load_attempts=1,
                    checkpoint_load_completed=1,from_pretrained_attempts=1,from_pretrained_completed=1)
                worker.update(dict.fromkeys(audit.ZERO_FIELDS,0));save(args.out/'worker/worker-result.json',worker)
                return dict(exit_code=0,timed_out=False,signal=None,remaining_pids=[])
            with (patch.object(audit.paths,'RESULTS_DIR',results),patch.object(audit,'run_process',side_effect=run) as call,
                  patch.object(audit.shutil,'disk_usage',return_value=Namespace(free=20*1024**3)),
                  patch('pt_agent.registry.git_rev',return_value='synthetic')):record=audit.execute(args)
            self.assertEqual(record['status'],'ok',record['error']);self.assertEqual(call.call_count,1)
            rows=[json.loads(s) for s in (results/'registry.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows),1);self.assertEqual(set(rows[0]),{'run_id','round','model','mode','git_rev','metrics','artifacts_dir','ts'})
            self.assertTrue((args.out/'completion.json').is_file());self.assertEqual(record['api_calls'],0)

    def test_worker_failure_retains_zero_forward_partial_without_real_runtime(self):
        with TemporaryDirectory() as temp:
            work=Path(temp)/'worker';args=Namespace(work_dir=work,model=Path('/original'),
                prepare_result=Path('/prepare.json'),model_info=Path('/model_info.json'))
            with patch.object(audit,'source_gate',side_effect=ValueError('synthetic wrong source')):
                record=audit.worker(args)
            self.assertEqual(record['status'],'error');self.assertEqual(record['checkpoint_load_attempts'],0)
            self.assertEqual(record['model_forwards'],0);self.assertEqual(record['api_calls'],0)
            self.assertTrue((work/'worker-result.json').exists())


if __name__=='__main__':unittest.main()
