"""Standard-library job protocol checks; no original artifacts or children."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from pt_agent.experiments import qwen_diagnostic_jobs as jobs


class QwenDiagnosticJobChecks(unittest.TestCase):
    def args(self, root, job):
        return SimpleNamespace(job=job,out=root/'result',linear_variant='compact-rectangular',
            model_info=root/'model',local_model=root/'hf',token_record=root/'tokens.json',
            original_hf_reference=root/'original.json',framework_result=root/'framework.json',
            stable_calibration_result=root/'bridge.json')

    def test_bounded_commands_use_existing_interpreters_and_explicit_variant(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            embedding = jobs.job_definition(self.args(root,'embedding'))
            count = jobs.job_definition(self.args(root,'stable-count'))
        self.assertEqual((embedding['memory_gib'],embedding['timeout']), (4,900))
        self.assertEqual((count['memory_gib'],count['timeout']), (12,1860))
        self.assertNotIn('pt_agent.tools.nixrun',embedding['command'])
        self.assertIn('pt_agent.tools.nixrun',count['command'])
        for definition in (embedding,count):
            self.assertEqual(definition['command'][-2:],['--linear-variant','compact-rectangular'])
            self.assertFalse(any(item in definition['command'] for item in ('pip','install','download')))
        position = embedding['command'].index('--expected-token')
        self.assertEqual(embedding['command'][position+1],'8559')

    def test_worker_completion_rejects_partial_and_keeps_formal_results_unmeasured(self):
        worker = dict(status='ok',linear_variant='compact-rectangular',count_attempts=1,count_completed=1,
            math_attempts=1,math_completed=1,mapping_completed=1,original_weight_members_mapped=1,
            count_constant_identity_entries=0,count_retained_array_constants=0,
            token_probe=dict(token_id=8559),math=dict(passed=True))
        jobs.check_completed(worker,'embedding','compact-rectangular')
        for key,value in (('math_completed',0),('count_attempts',2),('CKKS_latency',.1)):
            changed = dict(worker,**{key:value})
            with self.subTest(key=key), self.assertRaises(ValueError):
                jobs.check_completed(changed,'embedding','compact-rectangular')

    def test_terminal_result_accounts_for_its_own_json(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root/'small.json').write_text('{}\n')
            record = dict(status='ok',full_ciphertext_model_binding_verified=False,candidate_frozen=False)
            size = jobs._write_terminal(root,record)
            self.assertEqual(size,sum(path.stat().st_size for path in root.iterdir()))
            self.assertEqual(json.loads((root/'result.json').read_text())['case_artifact_bytes'],size)

    def test_failed_child_preserves_partial_and_one_registry_record(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = self.args(root,'embedding')

            def stopped(command, **kwargs):
                work = args.out/'worker'
                (work/'worker-result.json').write_text(json.dumps(dict(status='error',
                    linear_variant='compact-rectangular',count_attempts=1,count_completed=0)))
                return dict(exit_code=1,timed_out=False,remaining_pids=[])

            with mock.patch.object(jobs.paths,'RESULTS_DIR',root), \
                    mock.patch.object(jobs,'run_process',side_effect=stopped) as launch, \
                    mock.patch.object(jobs.subprocess,'run',return_value=SimpleNamespace(stdout='synthetic-revision\n')):
                result = jobs.execute(args)
                with self.assertRaises(ValueError):
                    jobs.execute(args)
            self.assertEqual(launch.call_count,1)
            self.assertEqual(result['status'],'error')
            self.assertEqual(result['metrics']['count_attempts'],1)
            self.assertEqual(result['metrics']['count_completed'],0)
            self.assertFalse(result['full_ciphertext_model_binding_verified'])
            records = [json.loads(line) for line in (root/'registry.jsonl').read_text().splitlines()]
            self.assertEqual(len(records),1)
            self.assertEqual(len(records[0]),8)
            self.assertEqual(records[0]['metrics']['status'],'error')
            self.assertEqual(result['case_artifact_bytes'],sum(p.stat().st_size
                for p in args.out.rglob('*') if p.is_file()))

    def test_server_diagnostic_dispatch_preserves_variant_and_avoids_reference_stages(self):
        from pt_agent.experiments import qwen_server_job as server
        for stage in ('embedding','stable-count'):
            with self.subTest(stage=stage), mock.patch.object(jobs,'main',return_value=0) as run:
                result = server.main(['--out','/synthetic/new-result','--stage',stage,
                    '--saved-model-info','/synthetic/original-model','--linear-variant','compact-rectangular'])
            self.assertEqual(result,0)
            command = run.call_args.args[0]
            self.assertEqual(command[command.index('--job')+1],stage)
            self.assertEqual(command[command.index('--linear-variant')+1],'compact-rectangular')
            self.assertEqual(command[command.index('--model-info')+1],'/synthetic/original-model')


if __name__ == '__main__':
    unittest.main()
