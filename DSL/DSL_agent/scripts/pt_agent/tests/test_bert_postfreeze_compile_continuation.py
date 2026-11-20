"""Frozen-only synthetic state/protocol fixtures, no model or tools."""
import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from pt_agent.experiments import bert_postfreeze_compile_continuation as entry
from pt_agent.tests import test_bert_serial_continuation as fixtures

base=entry.base


class FrozenOnlyContinuationChecks(unittest.TestCase):
    def run_fixture(self,root,*,phase='done',failed=None,reuse_audit=True,reuse_report=False,
                    partial_report=False,acceptance=True,child_error=None):
        args=fixtures.fixture(root);args.overall_seconds=21600
        if phase=='done':fixtures.frozen(args.source_case,error=failed)
        else:
            state=base.read(args.source_case/'session/SESSION.json');state.update(phase=phase,failed=failed)
            fixtures.save(args.source_case/'session/SESSION.json',state)
        with patch.object(base.paths,'RESULTS_DIR',args.out.parent):
            # This fixture source is metadata-only, supplied to fake audit rows.
            source=base.source_gate(args)
        if reuse_audit:fixtures.checkpoint(args,source)
        if reuse_report:fixtures.report(args,acceptance)
        if partial_report:fixtures.save(args.source_case/'session/REPORT_ACCESS.json',dict(complete=False))
        def child(argv,**kwargs):
            phase='checkpoint' if argv[3]==base.CHECKPOINT else 'compile' if argv[3]==base.COMPILE else argv[argv.index('--phase')+1]
            self.assertNotEqual(phase,'step')
            self.assertEqual(kwargs['mem_gib'],base.PHASE_LIMITS[phase][1])
            if phase==child_error:return fixtures.observed(exit_code=1)
            if phase=='checkpoint':fixtures.checkpoint(args,source)
            elif phase=='report':fixtures.report(args,acceptance)
            else:
                provenance=base.accepted_report(args)['source']
                fixtures.save(args.compile_out/'result.json',dict(round='BERT-FROZEN-COMPILE',status='ok',source=provenance,
                    completed=dict(emit=1,trace=1,compile=1),compiled_bootstraps=7,predicted_latency_us=None))
            return fixtures.observed()
        with (patch.object(base.paths,'RESULTS_DIR',args.out.parent),
                patch.object(base,'foreign_heavy_processes',return_value=[]),
                patch.object(base,'run_process',side_effect=child) as calls,
                patch.object(base.shutil,'disk_usage',return_value=fixtures.Namespace(free=20*base.GIB)),
                patch('pt_agent.registry.git_rev',return_value='fixture')):
            result=entry.execute(args)
        return args,result,calls

    def assert_closed(self,args,result):
        row=base.read(args.out/'completion.json')
        self.assertEqual(row['round'],entry.ROUND);self.assertEqual(row['status'],result['status'])
        self.assertTrue(row['registry_persisted']);self.assertTrue(row['required_artifacts_persisted'])
        self.assertEqual(row['case_artifact_bytes'],base.artifact_bytes(args.out))
        registry=[json.loads(line) for line in (args.out.parent/'registry.jsonl').read_text().splitlines()]
        self.assertEqual(len(registry),1);self.assertEqual(set(registry[0]),base.REGISTRY_FIELDS)
        self.assertEqual(registry[0]['mode'],entry.MODE)
        for key in entry.ZERO:self.assertEqual(result[key],0)
        self.assertEqual(result['attempted']['step'],0);self.assertEqual(result['max_step_commands'],0)

    def test_inflight_tune_is_observed_before_original_source_or_commands(self):
        for phase in ('evaluate','tune','confirm'):
            with self.subTest(phase=phase),TemporaryDirectory() as tmp:
                args=fixtures.fixture(Path(tmp));args.overall_seconds=21600
                state=base.read(args.source_case/'session/SESSION.json');state['phase']=phase
                fixtures.save(args.source_case/'session/SESSION.json',state)
                old=(args.source_case/'session/SESSION.json').read_text()
                with (patch.object(base.paths,'RESULTS_DIR',args.out.parent),
                        patch.object(base,'source_gate',side_effect=AssertionError('original source admission forbidden')) as source,
                        patch.object(base,'frozen_state',side_effect=AssertionError('freeze not started')),
                        patch.object(base,'run_process',side_effect=AssertionError('no child commands')) as child,
                        patch.object(base.shutil,'disk_usage',return_value=fixtures.Namespace(free=20*base.GIB)),
                        patch('pt_agent.registry.git_rev',return_value='fixture')):
                    result=entry.execute(args)
                self.assertEqual(result['status'],'ok');self.assertEqual(result['outcome'],'observed_not_eligible')
                self.assertFalse(result['eligible']);self.assertFalse(result['postfreeze_compile_completed'])
                source.assert_not_called();child.assert_not_called()
                self.assertEqual((args.source_case/'session/SESSION.json').read_text(),old)
                self.assertFalse(args.compile_out.exists());self.assertFalse(args.checkpoint_out.exists())
                self.assert_closed(args,result)

    def test_done_with_failed_state_is_observed_without_recovery(self):
        with TemporaryDirectory() as tmp:
            args,result,calls=self.run_fixture(Path(tmp),failed='retained error')
            self.assertEqual(result['status'],'ok');self.assertEqual(result['outcome'],'observed_not_eligible')
            self.assertFalse(result['eligible']);calls.assert_not_called()
            self.assertFalse(result['compile_completed']);self.assert_closed(args,result)

    def test_complete_source_absent_report_runs_report_once_then_compile(self):
        with TemporaryDirectory() as tmp:
            args,result,calls=self.run_fixture(Path(tmp))
            self.assertEqual(result['status'],'ok',result['error']);self.assertEqual(calls.call_count,2)
            self.assertEqual([row['phase'] for row in result['processes']],['report','compile'])
            self.assertEqual(result['attempted'],dict(checkpoint=0,step=0,report=1,compile=1))
            self.assertTrue(result['frozen']);self.assertTrue(result['compile_completed'])
            self.assertEqual(result['compiled_bootstraps'],7)
            self.assertTrue((args.out/'driver-source.py').is_file())
            self.assertTrue((args.out/'original-serial-driver-source.py').is_file())
            self.assert_closed(args,result)

    def test_complete_report_exact_reuse_and_report_policy_rejection_still_compiles(self):
        with TemporaryDirectory() as tmp:
            args,result,calls=self.run_fixture(Path(tmp),reuse_report=True,acceptance=False)
            self.assertEqual(result['status'],'ok',result['error']);self.assertEqual(calls.call_count,1)
            self.assertEqual(result['processes'][0]['phase'],'compile')
            self.assertEqual(result['attempted']['report'],0);self.assertFalse(result['task_report_acceptance'])
            self.assert_closed(args,result)

    def test_missing_semantic_audit_runs_only_after_frozen_admission(self):
        with TemporaryDirectory() as tmp:
            args,result,calls=self.run_fixture(Path(tmp),reuse_audit=False)
            self.assertEqual(result['status'],'ok',result['error'])
            self.assertEqual([row['phase'] for row in result['processes']],['checkpoint','report','compile'])
            self.assertEqual(calls.call_count,3);self.assert_closed(args,result)

    def test_incomplete_report_refuses_before_checkpoint_or_report_commands(self):
        with TemporaryDirectory() as tmp:
            args,result,calls=self.run_fixture(Path(tmp),reuse_audit=False,partial_report=True)
            self.assertEqual(result['status'],'error');calls.assert_not_called()
            self.assertIn('incomplete',result['error'])
            self.assertFalse(base.read(args.source_case/'session/REPORT_ACCESS.json')['complete'])
            self.assert_closed(args,result)

    def test_changed_selected_program_is_rejected_before_old_source_gate(self):
        with TemporaryDirectory() as tmp:
            args=fixtures.fixture(Path(tmp));args.overall_seconds=21600;fixtures.frozen(args.source_case)
            (args.source_case/'session/program.ptdsl').write_text('altered')
            with (patch.object(base.paths,'RESULTS_DIR',args.out.parent),
                    patch.object(base,'source_gate',side_effect=AssertionError('no original gate')),
                    patch.object(base,'run_process',side_effect=AssertionError('no child')),
                    patch.object(base.shutil,'disk_usage',return_value=fixtures.Namespace(free=20*base.GIB)),
                    patch('pt_agent.registry.git_rev',return_value='fixture')):
                result=entry.execute(args)
            self.assertEqual(result['status'],'error');self.assertIn('C2 frozen metadata differ',result['error'])
            self.assert_closed(args,result)

    def test_command_barrier_rejects_steps_and_does_not_change_original_globals(self):
        with TemporaryDirectory() as tmp:
            args=fixtures.fixture(Path(tmp));fixtures.frozen(args.source_case);accepted=entry.admission(args)
            original=base.command
            with patch.object(base,'command',side_effect=AssertionError('step cannot reach original command')) as command:
                for phase in ('step','tune','confirm','run'):
                    with self.assertRaises(ValueError):entry.command(args,{},phase,accepted=accepted)
                command.assert_not_called()
            private=entry.namespace(accepted,{})
            self.assertIs(base.command,original);self.assertIsNot(private['execute'],base.execute)
            self.assertIs(private['serial_lease'],base.serial_lease)
            self.assertEqual(base.ROUND,'BERT-SERIAL-CONTINUATION')
            self.assertEqual(private['ROUND'],entry.ROUND)

    def test_candidate_change_between_commands_is_rejected(self):
        with TemporaryDirectory() as tmp:
            args=fixtures.fixture(Path(tmp));fixtures.frozen(args.source_case);accepted=entry.admission(args)
            state=base.read(args.source_case/'session/SESSION.json');state['phase']='tune'
            fixtures.save(args.source_case/'session/SESSION.json',state)
            with patch.object(base,'command',side_effect=AssertionError('no child argv')):
                with self.assertRaises(ValueError):entry.command(args,{},'checkpoint',accepted=accepted)

    def test_child_failure_stops_and_keeps_original_frozen_program(self):
        with TemporaryDirectory() as tmp:
            args,result,calls=self.run_fixture(Path(tmp),child_error='report')
            self.assertEqual(result['status'],'error');self.assertEqual(calls.call_count,1)
            self.assertEqual(result['attempted']['compile'],0)
            self.assertTrue(base.frozen_state(args.source_case)['frozen']);self.assert_closed(args,result)

    def test_explicit_fresh_outputs_and_finite_allowance(self):
        for field in ('out','compile_out'):
            with TemporaryDirectory() as tmp:
                args=fixtures.fixture(Path(tmp));args.overall_seconds=21600
                getattr(args,field).mkdir(parents=True)
                with patch.object(base.paths,'RESULTS_DIR',args.out.parent),self.assertRaises(ValueError):entry.validate(args)
        for value in (True,0,21601,float('nan')):
            with TemporaryDirectory() as tmp:
                args=fixtures.fixture(Path(tmp));args.overall_seconds=value
                with patch.object(base.paths,'RESULTS_DIR',args.out.parent),self.assertRaises(ValueError):entry.validate(args)
        with patch('sys.stderr'),self.assertRaises(SystemExit):
            entry.parser().parse_args(['--source-case','source','--out','outer','--checkpoint-out','audit',
                '--compile-out','compile','--max-steps','1'])

    def test_main_argv_preserves_original_paths_no_max_steps_interface(self):
        with TemporaryDirectory() as tmp:
            args=fixtures.fixture(Path(tmp))
            argv=[]
            for name in ('source_case','out','checkpoint_out','compile_out','intake_python','data_python','ledger',
                         'allowance','metadata_template','profile','frontend'):
                argv+=['--'+name.replace('_','-'),str(getattr(args,name))]
            argv+=['--overall-seconds','21600']
            with patch.object(entry,'execute',return_value=dict(status='ok',outcome='observed_not_eligible')) as execute, \
                    patch.object(base.paths,'RESULTS_DIR',args.out.parent),patch('builtins.print'):
                self.assertEqual(entry.main(argv),0)
            actual=execute.call_args.args[0]
            self.assertEqual(actual.max_steps,0);self.assertEqual(actual.overall_seconds,21600)
            self.assertEqual(actual.compile_out,args.compile_out)


if __name__=='__main__':unittest.main()
