"""Synthetic closed probes and small saved resource traces only."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from pt_agent.experiments import heir_initializer_terminal_audit as entry


class TerminalAuditChecks(unittest.TestCase):
    def setUp(self):
        temp=TemporaryDirectory();self.addCleanup(temp.cleanup);self.root=Path(temp.name)
        self.case=self.root/'probe';self.case.mkdir();self.registry=self.root/'registry.jsonl';self.ready=self.root/'readiness.json'
        self.result=dict(round='HEIR-INIT-PROBE',status='memory_failure',attempts=dict(build=1,initialize=1),
            completed=dict(build=1,initialize=0),limits=dict(native_AS_gib=12,build_AS_gib=4),
            processes=[dict(stage='build',observation=self.observation()),
                dict(stage='initialize',observation=self.observation(exit_code=2))],worker=None,
            **{k:0 for k in entry.ZERO})
        self.write(self.ready,dict(source_static_estimates=dict(full_encoded_coefficients_GiB=103.39404296875,
            probe_preprocessing_coefficient_GiB=103.388671875,largest_single_LT_GiB=40,
            current_package_global_inventory=[dict(source='original generated source',source_float_payload_bytes=160938590288)])))
        (self.case/'initialize.log').write_text('fatal error: runtime: out of memory\nheircnn/heircnn_utils.init.0()\n')

    def observation(self,**changes):
        row=dict(exit_code=0,timed_out=False,signal=None,signals_sent=[],remaining_pids=[],started_ns=10,ended_ns=1000000010,
            limits=dict(memory_gib=12,wall_seconds=300));row.update(changes);return row

    @staticmethod
    def write(path,value):path.write_text(json.dumps(value))

    def close(self):
        self.write(self.case/'result.json',self.result)
        self.write(self.case/'completion.json',dict(status=self.result['status'],registry_persisted=True,
            required_artifacts_persisted=True,started_ns=1,ended_ns=9000000001,complete_case_elapsed_seconds=9,case_artifact_bytes=1))
        self.write(self.registry,dict(run_id=self.case.name,round='HEIR-INIT-PROBE',model='resnet20_silu',mode='probe',git_rev=None,
            metrics=dict(status=self.result['status']),artifacts_dir=str(self.case),ts='synthetic'))

    def run_audit(self):self.close();return entry.audit(self.case,self.registry,self.ready)

    def test_init_oom_stack_and_static_volume_remain_distinct_from_actual_resources(self):
        def row(hwm,rss,vsz,outer):
            return dict(processes=[dict(pid=7,starttime=10,vmhwm_kib=hwm,vmrss_kib=rss,vmsize_kib=vsz)],
                outer_process=dict(pid=8,vmhwm_kib=80,vmrss_kib=outer))
        path=self.case/'initialize-resources.jsonl'
        path.write_text('\n'.join(json.dumps(x) for x in (row(500,400,900,50),row(600,300,1000,60)))+'\n')
        audit=self.run_audit();self.assertEqual(audit['audited_classification'],'memory_failure')
        self.assertEqual(audit['failure_or_progress_stage'],'original-generated-package-initialization')
        self.assertFalse(audit['startup_markers']['main_entered']);self.assertIsNone(audit['actual_encoding_entry_observed'])
        observed=audit['resource_observations']['initialize'];self.assertEqual(observed['samples'],2)
        self.assertEqual(observed['single_PID_sampled_HWM_KiB'],600);self.assertEqual(observed['single_PID_sampled_maxVmSize_KiB'],1000)
        self.assertEqual(observed['simultaneous_child_subtree_RSS_KiB'],400)
        self.assertEqual(observed['simultaneous_subtree_plus_outer_supervisor_RSS_KiB'],450)
        self.assertEqual(audit['original_limits']['native_AS_gib'],12)
        self.assertEqual(audit['static_source_estimates']['preprocessing_encoded_coefficient_GiB'],103.388671875)
        self.assertTrue(audit['zero_crypto_contract_compatible']);self.assertIsNone(audit['CKKS_latency'])

    def test_absent_marker_and_empty_samples_leave_location_and_memory_unobserved(self):
        (self.case/'initialize.log').write_text('fatal error: runtime: out of memory\n')
        audit=self.run_audit()
        self.assertEqual(audit['audited_classification'],'memory_failure')
        self.assertEqual(audit['failure_or_progress_stage'],'initialization-entry-location-unobserved')
        self.assertIsNone(audit['startup_markers']['main_entered'])
        self.assertIsNone(audit['resource_observations']['initialize']['single_PID_sampled_HWM_KiB'])
        self.assertIsNone(audit['actual_encoding_completed'])

    def test_recorded_encoding_marker_locates_failure_after_main(self):
        self.result['worker']=dict(main_entered=True,public_parameters_completed=True,preprocessing_attempts=1,preprocessing_completed=0)
        (self.case/'initialize.log').write_text('fatal error: runtime: out of memory\n')
        audit=self.run_audit();self.assertEqual(audit['failure_or_progress_stage'],'original-preprocessing-encoding')
        self.assertTrue(audit['actual_encoding_entry_observed']);self.assertFalse(audit['actual_encoding_completed'])
        self.assertTrue(audit['startup_markers']['main_entered'])

    def test_build_oom_is_classified_as_build_without_initialization_or_encoding_attempt(self):
        self.result.update(status='build_error',attempts=dict(build=1,initialize=0),completed=dict(build=0,initialize=0),
            processes=[dict(stage='build',observation=self.observation(exit_code=2,limits=dict(memory_gib=4,wall_seconds=300)))])
        (self.case/'build.log').write_text('runtime: cannot allocate memory\n')
        (self.case/'initialize.log').unlink()
        audit=self.run_audit();self.assertEqual(audit['audited_classification'],'memory_failure')
        self.assertEqual(audit['failure_or_progress_stage'],'build');self.assertEqual(audit['processes']['initialize']['attempts'],0)
        self.assertEqual(audit['processes']['build']['limits']['memory_gib'],4)
        self.assertEqual(audit['processes']['build']['observed_process_elapsed_seconds'],1)

    def test_resource_timeout_incoming_signal_and_cleanup_keep_distinct_causes(self):
        examples=[(dict(guard_trigger=dict(kind='disk'),guard_stop_signal=15),'resource_stopped'),
                  (dict(timeout_signal=14),'timeout'),(dict(incoming_signal=15),'interrupted')]
        for extra,expected in examples:
            with self.subTest(expected=expected):
                self.result.update(extra);audit=self.run_audit();self.assertEqual(audit['audited_classification'],expected)
                for key in extra:self.result.pop(key)
        self.result['processes'][1]['observation']=self.observation(signals_sent=[dict(pid=10,signal=15)])
        audit=self.run_audit();self.assertEqual(audit['audited_classification'],'supervision_error')
        self.assertIsNone(audit['processes']['initialize']['observation']['signal'])

    def test_partial_trace_and_reused_pid_are_retained_without_summing_peaks(self):
        path=self.root/'resources.jsonl'
        path.write_text(json.dumps(dict(processes=[dict(pid=1,starttime=1,vmrss_kib=20,vmhwm_kib=30,vmsize_kib=50)]))+'\n'+
            json.dumps(dict(processes=[dict(pid=1,starttime=2,vmrss_kib=40,vmhwm_kib=60,vmsize_kib=70)]))+'\n{"partial":')
        summary=entry.resource_summary(path)
        self.assertEqual(summary['samples'],2);self.assertEqual(summary['malformed_or_partial_rows'],1)
        self.assertEqual(len(summary['sampled_processes']),2);self.assertEqual(summary['simultaneous_child_subtree_RSS_KiB'],40)

    def test_terminal_and_unique_registry_contracts_are_required(self):
        self.close();self.registry.write_text(self.registry.read_text()+'\n'+self.registry.read_text())
        with self.assertRaisesRegex(ValueError,'One matching'):entry.audit(self.case,self.registry,self.ready)
        self.close();self.result['status']='running';self.close()
        with self.assertRaisesRegex(ValueError,'closed original'):entry.audit(self.case,self.registry,self.ready)

    def test_audit_output_must_not_modify_original_case(self):
        with patch.object(entry,'audit',side_effect=AssertionError('Invalid output must fail before reading a case')):
            with self.assertRaises(SystemExit):entry.main(['--case',str(self.case),'--readiness',str(self.ready),'--out',str(self.case/'audit.json')])


if __name__=='__main__':unittest.main()
