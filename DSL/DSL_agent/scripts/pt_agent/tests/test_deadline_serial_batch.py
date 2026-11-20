"""Deadline behavior uses fake children only; no experiment or model is run."""
import json
import argparse
import fcntl
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pt_agent.experiments import autonomous_serial_batch as m


class DeadlineBatch(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.batch=self.root/'batch';self.batch.mkdir()
        self.calls=[]

    def job(self,name='job'):
        out=self.root/name
        return dict(id=name,out=str(out),seconds=7200,
            argv=['python3','-B','-m',m.PREFIX+'qwen_bounded_feedback','--out',str(out)])

    def fake(self,argv,**options):
        self.calls.append(options);out=Path(argv[argv.index('--out')+1]);out.mkdir()
        (out/'result.json').write_text(json.dumps({'status':'ok'}))
        return dict(exit_code=0,timed_out=False,signal=None,remaining_pids=[],signals_sent=[])

    def run_jobs(self,jobs,deadline=None,wall_clock=lambda:100):
        return m.execute_jobs(jobs,out=self.batch,results_root=self.root,runner=self.fake,
            free=lambda _:100*m.GIB,deadline=deadline,wall_clock=wall_clock)

    def test_timezone_required_and_offset_equivalent(self):
        self.assertIsNone(m.deadline_timestamp(None))
        self.assertEqual(m.deadline_timestamp('2026-11-17T18:00:00+08:00'),
            m.deadline_timestamp('2026-11-17T10:00:00+00:00'))
        self.assertEqual(m.deadline_timestamp('2026-11-17T10:00:00Z'),
            m.deadline_timestamp('2026-11-17T10:00:00+00:00'))
        for value in (True,123,'not-a-date','2026-11-17T18:00:00'):
            with self.assertRaises(ValueError):m.deadline_timestamp(value)

    def test_expired_starts_no_child(self):
        rows=self.run_jobs([self.job()],deadline=130)
        self.assertEqual(self.calls,[]);self.assertEqual(rows[0]['status'],'resource_deferred')

    def test_job_gets_remaining_window_with_persistence_reserve(self):
        rows=self.run_jobs([self.job()],deadline=500)
        self.assertEqual(self.calls[0]['timeout'],370)
        self.assertEqual(rows[0]['allowed_wall_seconds'],370)
        self.assertTrue(rows[0]['absolute_deadline_applied'])

    def test_later_jobs_deferred_after_first_uses_window(self):
        clocks=iter([100,100,501])
        rows=self.run_jobs([self.job('one'),self.job('two')],deadline=500,wall_clock=lambda:next(clocks))
        self.assertEqual(len(self.calls),1)
        self.assertEqual([x['status']for x in rows],['ok','resource_deferred'])

    def test_cutoff_during_metadata_preparation_never_launches(self):
        clocks=iter([100,501])
        rows=self.run_jobs([self.job()],deadline=500,wall_clock=lambda:next(clocks))
        self.assertEqual(self.calls,[])
        self.assertEqual(rows[0]['status'],'resource_deferred')
        self.assertNotIn('cleanup_unknown',rows[0])
        self.assertFalse((self.batch/'active.json').exists())
        self.assertTrue((self.batch/'job-unstarted-active.json').is_file())
        self.assertTrue((self.batch/'job-command.json').is_file())

    def test_default_preserves_existing_wall_allowance(self):
        rows=self.run_jobs([self.job()])
        self.assertEqual(self.calls[0]['timeout'],7200)
        self.assertEqual((self.calls[0]['mem_gib'],self.calls[0]['fsize_gib']),(24,16))
        self.assertFalse(rows[0]['absolute_deadline_applied'])

    def test_recorded_vgg_restore_uses_existing_parent_limits(self):
        job=self.job();job['argv'][3]=m.PREFIX+'dacapo_vgg_math_restore'
        self.assertEqual(m.check_command(job['argv'],job['out']),job['argv'])
        rows=self.run_jobs([job])
        self.assertEqual(rows[0]['status'],'ok')
        self.assertEqual((self.calls[0]['mem_gib'],self.calls[0]['fsize_gib']),(24,16))

    def test_explicit_heir_fused_pipeline_is_admitted_without_parent_expansion(self):
        job=self.job();job['argv'][3]=m.PREFIX+'heir_vgg_fused_pipeline';job['argv']+=['--run']
        self.assertEqual(m.check_command(job['argv'],job['out']),job['argv'])
        rows=self.run_jobs([job])
        self.assertEqual(rows[0]['status'],'ok')
        self.assertEqual((self.calls[0]['mem_gib'],self.calls[0]['fsize_gib']),(24,16))

    def test_only_explicit_trusted_new_modules_admitted(self):
        for module in ('qwen_bounded_feedback','r30_vgg_postfreeze','r31_common_cnn_native',
                       'r30_common_model_e2e','orbit_common_resource_retry'):
            job=self.job();job['argv'][3]=m.PREFIX+module
            self.assertEqual(m.check_command(job['argv'],job['out']),job['argv'])
        job=self.job();job['argv'][3]=m.PREFIX+'arbitrary_user_program'
        with self.assertRaises(ValueError):m.check_command(job['argv'],job['out'])

    def test_expanded_full_qwen_limits_reach_child_interface(self):
        job=self.job();job['argv'][3]=m.PREFIX+'qwen_prefill_compile'
        job.update(outer_address_space_gib=32,outer_file_gib=64)
        rows=self.run_jobs([job])
        self.assertEqual((self.calls[0]['mem_gib'],self.calls[0]['fsize_gib']),(32,64))
        self.assertEqual(rows[0]['parent_address_space_gib'],32)

    def test_parent_expansion_rejects_other_modules_and_implicit_values(self):
        for update in (dict(outer_address_space_gib=32,outer_file_gib=64),
                       dict(outer_address_space_gib=32.0),dict(outer_file_gib=True),
                       dict(outer_address_space_gib=64,outer_file_gib=64)):
            job=self.job();job.update(update)
            with self.assertRaises(ValueError):m.parent_limits(job)
        job=self.job();job['argv'][3]=m.PREFIX+'qwen_prefill_compile'
        for pair in ((32,16),(24,64),(64,64)):
            with self.assertRaises(ValueError):m.parent_limits(dict(job,outer_address_space_gib=pair[0],outer_file_gib=pair[1]))

    def test_source_wait_never_holds_serial_lease(self):
        manifest=self.root/'manifest.json'
        manifest.write_text(json.dumps(dict(schema_version=1,jobs=[],
            watch=dict(case=str(self.root/'source'),pid=42,starttime=3,seconds=120))))
        args=argparse.Namespace(manifest=manifest,out=self.root/'successor',lease_wait_seconds=0)
        lease_path=self.root/'autonomous-heavy-experiments.lock';sequence=[]
        def wait(_):
            with lease_path.open('a')as contender:
                fcntl.flock(contender,fcntl.LOCK_EX|fcntl.LOCK_NB)
                fcntl.flock(contender,fcntl.LOCK_UN)
            sequence.append('source_closed');return dict(status='ok')
        def execute(*a,**kw):
            with lease_path.open('a')as contender:
                with self.assertRaises(BlockingIOError):
                    fcntl.flock(contender,fcntl.LOCK_EX|fcntl.LOCK_NB)
            sequence.append('jobs_under_lease');return []
        with patch.object(m.paths,'RESULTS_DIR',self.root),patch.object(m,'wait_closed',side_effect=wait),\
                patch.object(m,'execute_jobs',side_effect=execute),\
                patch('pt_agent.registry.git_rev',return_value='fixture'):
            self.assertEqual(m.run(args),0)
        self.assertEqual(sequence,['source_closed','jobs_under_lease'])
        self.assertTrue((args.out/'completion.json').is_file())

    def test_saved_trace_dacapo_expansion_is_explicit_and_bounded(self):
        job=self.job();job['argv'][3]=m.PREFIX+'dacapo_compile_continuation'
        job.update(outer_address_space_gib=32,outer_file_gib=16)
        with self.assertRaises(ValueError):m.parent_limits(job)
        job['argv']+=['--memory-gib','32','--rss-gib','16','--file-gib','16']
        rows=self.run_jobs([job])
        self.assertEqual((self.calls[0]['mem_gib'],self.calls[0]['fsize_gib']),(32,16))
        self.assertEqual(rows[0]['parent_address_space_gib'],32)
        for flag,value in (('--memory-gib','48'),('--rss-gib','17'),('--file-gib','64')):
            wrong=dict(job,argv=list(job['argv']));wrong['argv'][wrong['argv'].index(flag)+1]=value
            with self.assertRaises(ValueError):m.parent_limits(wrong)


if __name__=='__main__':unittest.main()
