"""Small source/argv checks for original common-model compiler preparation."""

from argparse import Namespace
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pt_agent.experiments import r29_local_compile_matrix as matrix


class CommonCompileSourceCases(unittest.TestCase):
    def test_format1_boot_output_and_legacy_outputs_keep_placement_metadata(self):
        values = [dict(id='input', level=1), dict(id='booted', level=5)]
        profile = dict(latencyTable={'earth.bootstrap_single': [0, 17]})
        for output_field in (dict(output='booted'), dict(outputs=['booted'])):
            with self.subTest(output_field=output_field):
                step = dict(kind='compute', op='boot', ordinal=7, inputs=['input'], **output_field)
                observed = matrix.plan_observation(dict(values=values, execution=[step]), profile)
                self.assertEqual(observed['compiled_bootstraps'], 1)
                self.assertEqual(observed['bootstrap_positions'][0]['outputs'], ['booted'])
                self.assertEqual(observed['bootstrap_positions'][0]['input_levels'], [1])
                self.assertEqual(observed['bootstrap_positions'][0]['output_levels'], [5])
                self.assertEqual(observed['predicted_latency_us'], 17)

    def fixture(self, root, factory):
        info = root/'original/model_info.json'
        info.parent.mkdir(parents=True)
        original = dict(factory=factory, checkpoint=str(root/'original-checkpoint'),
                        normalization=dict(mean=[.485,.456,.406],std=[.229,.224,.225]))
        info.write_text(json.dumps(dict(source=original)))
        r13 = root/'original/result.json'
        r13.write_text(json.dumps(dict(round='R13',status='ok',model=factory,
            worker=dict(artifacts=dict(model_info=str(info.parent))))))
        prior = root/'accepted/result.json';prior.parent.mkdir()
        program = prior.parent/'worker/program.ptdsl';program.parent.mkdir()
        program.write_text('synthetic saved program\n')
        worker = dict(model=factory,status='ok',source_checked=True,evaluation_completed=True,
            task_passed=True,domain_passed=True,heldout_acceptance=True,all_finite=True,
            completed_samples=1000,heldout_samples=1000,model_source=original,
            source_preparation=dict(model_info=str(info.parent),r13_result=str(r13)),
            artifacts=dict(program=str(program)),validation_metrics=dict(agree=1000,total=1000),
            max_u=.85,candidate_frozen=False,confirm_samples=0,report_samples=0)
        record = dict(round='R17',status='ok',model=factory,worker=worker)
        prior.write_text(json.dumps(record))
        return info,prior,record

    def test_both_rule_factories_keep_original_quality_and_internal_scope(self):
        for factory in matrix.MODEL_FACTORIES[1:]:
            with self.subTest(factory=factory),tempfile.TemporaryDirectory() as directory:
                info,prior,record=self.fixture(Path(directory),factory)
                result=matrix.validate_rule_source(record,result_path=prior,model_info=info,
                    model_info_record=json.loads(info.read_text()),model_factory=factory)
                self.assertTrue(result['independent_source_checked'])
                self.assertEqual(result['model_factory'],factory)
                self.assertFalse(result['source_quality']['candidate_frozen'])
                self.assertIn('internal fixed-rule',result['scope'])

    def test_foreign_factory_quality_source_or_program_is_rejected(self):
        factory=matrix.MODEL_FACTORIES[1]
        with tempfile.TemporaryDirectory() as directory:
            info,prior,original=self.fixture(Path(directory),factory);metadata=json.loads(info.read_text())
            changes=[lambda x:x.update(model=matrix.MODEL_FACTORIES[2]),
                lambda x:x['worker'].update(model=matrix.MODEL),
                lambda x:x['worker'].update(domain_passed=False),
                lambda x:x['worker'].update(completed_samples=999),
                lambda x:x['worker']['source_preparation'].update(model_info=str(info.parent/'foreign')),
                lambda x:x['worker']['model_source'].update(checkpoint='foreign-checkpoint'),
                lambda x:x['worker']['artifacts'].update(program=str(info))]
            for change in changes:
                record=deepcopy(original);change(record)
                with self.assertRaises(ValueError):
                    matrix.validate_rule_source(record,result_path=prior,model_info=info,
                        model_info_record=metadata,model_factory=factory)

    def test_original_r13_record_factory_is_checked_before_lowering(self):
        factory=matrix.MODEL_FACTORIES[2]
        with tempfile.TemporaryDirectory() as directory:
            info,prior,record=self.fixture(Path(directory),factory)
            r13=Path(record['worker']['source_preparation']['r13_result'])
            changed=json.loads(r13.read_text());changed['model']=matrix.MODEL;r13.write_text(json.dumps(changed))
            with self.assertRaises(ValueError):
                matrix.validate_rule_source(record,result_path=prior,model_info=info,
                    model_info_record=json.loads(info.read_text()),model_factory=factory)

    def test_agent_freeze_audit_uses_selected_short_model_and_rejects_rule(self):
        factory=matrix.MODEL_FACTORIES[1];short=factory.split(':')[1]
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);info,prior,rule=self.fixture(root,factory)
            program=rule['worker']['artifacts']['program']
            worker=dict(model=short,status='ok',frozen=True,selected_program=program,
                source_model_info=str(info),quality_acceptance=dict(passed=True),
                strict_math_probes=[dict(sample_id='test:'+str(9000+i),
                    exact=dict(passed=True),approximate=dict(passed=True)) for i in range(3)])
            record=dict(round='R28',status='ok',model=short,worker=worker)
            audit=dict(status='quality_accepted',formal_frozen=True,quality_acceptance_passed=True,
                strict_math_passed=True,model=short,source_result=str(prior),selected_program=program)
            result=matrix.validate_agent_source(record,audit,result_path=prior,model_info=info,model_factory=factory)
            self.assertTrue(result['independent_audit_passed'])
            for invalid in (dict(audit,formal_frozen=False),dict(audit,model='resnet20_silu')):
                with self.assertRaises(ValueError):
                    matrix.validate_agent_source(record,invalid,result_path=prior,model_info=info,model_factory=factory)
            with self.assertRaises(ValueError):
                matrix.validate_agent_source(rule,audit,result_path=prior,model_info=info,model_factory=factory)

    def test_model_cli_defaults_and_worker_argv_are_explicit(self):
        from pt_agent import paths
        with tempfile.TemporaryDirectory() as directory,patch.object(paths,'RESULTS_DIR',Path(directory)):
            for factory in matrix.MODEL_FACTORIES:
                with patch.object(matrix,'outer',return_value=0) as outer:
                    self.assertEqual(matrix.main(['--model-factory',factory,'--systems','dacapo-depth-dp']),0)
                args=outer.call_args.args[0]
                self.assertEqual(args.model_factory,factory)
                self.assertIn(factory.split(':')[1],str(args.rule_result))
                argv,timeout=matrix._launch_worker(args,'dacapo-depth-dp','compile',Path('/explicit/program'),Path('/explicit/work'))
                self.assertEqual(argv[argv.index('--model-factory')+1],factory)
                self.assertEqual(timeout,7440)
                self.assertEqual(argv[argv.index('--compile-seconds')+1],'7200')
            with patch.object(matrix,'outer',return_value=0) as outer:
                matrix.main(['--systems','dacapo-depth-dp'])
            self.assertEqual(outer.call_args.args[0].model_factory,matrix.MODEL)
            self.assertEqual(outer.call_args.args[0].case_gib,8)
            with patch.object(matrix,'outer',return_value=0) as outer:
                matrix.main(['--model-factory',matrix.MODEL_FACTORIES[1],
                             '--systems','dacapo-depth-dp','--case-gib','32'])
            self.assertEqual(outer.call_args.args[0].case_gib,32)
            self.assertEqual(matrix.selected_model(Namespace()),matrix.MODEL)
            with patch.object(matrix,'outer') as outer,patch('sys.stderr',new=io.StringIO()),self.assertRaises(SystemExit):
                matrix.main(['--model-factory','foreign:factory'])
            outer.assert_not_called()

    def run_fake_case(self, root, *, wrong_worker_model=False):
        from pt_agent import paths
        factory=matrix.MODEL_FACTORIES[2]
        info,prior,_=self.fixture(root,factory)
        profile=root/'profile.json';profile.write_text('{}\n')
        template=root/'compiler-argv.json';template.write_text('[]\n')
        args=Namespace(model_factory=factory,rule_result=prior,model_info=info,profile=profile,
            metadata_template=template,git_rev='synthetic',case_gib=16,memory_gib=12,
            compile_seconds=7200,trace_seconds=1800,frontend=root/'frontend.py')
        calls=[]
        def fake_process(argv,**kwargs):
            phase=argv[argv.index('--worker-phase')+1];work=Path(argv[argv.index('--work-dir')+1])
            self.assertEqual(argv[argv.index('--model-factory')+1],factory)
            record=dict(status='ok',model=matrix.MODEL if wrong_worker_model else factory,
                source_model_info=str(info),source_program=str(work/'program.ptdsl'))
            if phase=='compile':record.update(compiled_bootstraps=2,predicted_latency_us=123.)
            (work/(phase+'-result.json')).write_text(json.dumps(record));calls.append(phase)
            return dict(exit_code=0,timed_out=False,signal=None,remaining_pids=[])
        with patch.object(paths,'RESULTS_DIR',root),patch.object(matrix,'run_process',side_effect=fake_process),\
                patch.object(matrix.shutil,'disk_usage',return_value=SimpleNamespace(free=100*matrix.GIB)):
            result=matrix.run_case(args,'dacapo-depth-dp',root/'new-case')
        return result,calls

    def test_fake_case_records_original_model_and_unique_registry(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);result,calls=self.run_fake_case(root)
            self.assertEqual(result['status'],'ok');self.assertEqual(calls,['emit','trace','compile'])
            self.assertEqual(result['model'],matrix.MODEL_FACTORIES[2])
            self.assertFalse(result['source']['source_quality']['candidate_frozen'])
            self.assertIsNone(result['accuracy_acceptance']);self.assertEqual(result['real_ckks_runs'],0)
            rows=[json.loads(line) for line in (root/'registry.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows),1);self.assertEqual(len(rows[0]),8)
            self.assertEqual(rows[0]['model'],matrix.MODEL_FACTORIES[2])
            self.assertEqual(result['completion']['case_artifact_bytes'],matrix._size(root/'new-case'))

    def test_worker_identity_mismatch_stops_before_trace(self):
        with tempfile.TemporaryDirectory() as directory:
            result,calls=self.run_fake_case(Path(directory),wrong_worker_model=True)
            self.assertEqual(result['status'],'error');self.assertEqual(calls,['emit'])
            self.assertEqual(result['completed'],{})
            self.assertIn('different original model',result['error'])


if __name__=='__main__':
    unittest.main()
